#include "attention.hpp"
#include "celeg/model/position.hpp"
#include "celeg/model/weights/quantization.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace celeg {

void apply_cpu_attention_qk(const AttentionSpec& layout,
                            const CpuCompiledModel::AttentionWeights& weights,
                            float* query,
                            float* key,
                            int scalar_position,
                            const std::array<int32_t, 3>& rope_position) {
    const int q_width = layout.query_width();
    const bool has_key = key != nullptr && !weights.k.segments.empty();
    const RopePositionSpec* rope = layout.rope_position();

    // The attention kernels (cpu_gqa_decode_paged_parallel and the chunk
    // equivalent) already fold 1/sqrt(head_dim) into the scores, so every path
    // here must premultiply the query by the *ratio* between the model's own
    // query scale and that default -- never by layout.query_scale itself, which
    // would scale a second time. The query-key-norm branches used to do exactly
    // that, making every score 1/sqrt(head_dim) too small (8x for head_dim 64).
    // Softmax over a single position hides it, so it only shows up once a
    // sequence has more than one token.
    const float kernel_scale = 1.0f / std::sqrt(static_cast<float>(layout.head_dim));
    const float query_scale_ratio = layout.query_scale / kernel_scale;

    const auto apply_norm_only = [&](float* data, const float* norm_weight,
                                     int heads, const NormSpec& norm) {
        if (norm.granularity == NormGranularity::PerHead) {
            cpu_qk_norm_only(data, norm_weight, heads, layout.head_dim, norm.epsilon);
        } else {
            cpu_qk_norm_only(data, norm_weight, 1, heads * layout.head_dim, norm.epsilon);
        }
    };

    if (rope == nullptr) {
        if (layout.has_query_key_norm()) {
            apply_norm_only(query, weights.q_norm.data(), layout.query_heads,
                            *layout.query_norm);
            if (has_key) {
                apply_norm_only(key, weights.k_norm.data(), layout.key_value_heads,
                                *layout.key_norm);
            }
            for (int i = 0; i < q_width; ++i) query[i] *= query_scale_ratio;
            return;
        }
        for (int i = 0; i < q_width; ++i) query[i] *= query_scale_ratio;
        return;
    }

    if (layout.has_query_key_norm()) {
        const auto apply_norm_and_rope = [&](float* data, const float* norm_weight,
                                             int heads, const NormSpec& norm) {
            if (norm.granularity == NormGranularity::PerHead) {
                if (const auto* multi = layout.multi_axis_position()) {
                    cpu_qk_norm_rope_mrope(
                        data, norm_weight, heads, layout.head_dim, rope_position,
                        multi->sections, multi->interleaved, *rope, norm.epsilon);
                } else {
                    cpu_qk_norm_rope(data, norm_weight, heads, layout.head_dim,
                                     scalar_position, *rope, norm.epsilon);
                }
                return;
            }

            cpu_qk_norm_only(data, norm_weight, 1, heads * layout.head_dim,
                             norm.epsilon);
            if (const auto* multi = layout.multi_axis_position()) {
                cpu_rope_mrope(data, heads, layout.head_dim, rope_position,
                               multi->sections, multi->interleaved, *rope);
            } else {
                cpu_rope(data, heads, layout.head_dim, scalar_position, *rope);
            }
        };

        apply_norm_and_rope(query, weights.q_norm.data(), layout.query_heads,
                            *layout.query_norm);
        if (has_key) {
            apply_norm_and_rope(key, weights.k_norm.data(), layout.key_value_heads,
                                *layout.key_norm);
        }
        const float query_scale = query_scale_ratio *
            rope_attention_scale(*rope, scalar_position);
        for (int i = 0; i < q_width; ++i) query[i] *= query_scale;
        return;
    }

    if (const auto* multi = layout.multi_axis_position()) {
        cpu_rope_mrope(query, layout.query_heads, layout.head_dim, rope_position,
                       multi->sections, multi->interleaved,
                       *rope);
        if (has_key) {
            cpu_rope_mrope(key, layout.key_value_heads, layout.head_dim, rope_position,
                           multi->sections, multi->interleaved,
                           *rope);
        }
    } else {
        cpu_rope(query, layout.query_heads, layout.head_dim, scalar_position,
                 *rope);
        if (has_key) {
            cpu_rope(key, layout.key_value_heads, layout.head_dim, scalar_position,
                     *rope);
        }
    }
    const float query_scale = query_scale_ratio *
        rope_attention_scale(*rope, scalar_position);
    for (int i = 0; i < q_width; ++i) query[i] *= query_scale;
}

void load_cpu_attention_current_value(
    const CpuCompiledModel::Shared& shared,
    const CpuCompiledModel::AttentionState& state,
    int position, float* output, size_t width) {
    if (position < 0 || !output || width == 0) {
        throw std::invalid_argument("invalid CPU attention current-value request");
    }
    const CpuKvPagePool& pool = *shared.kv_pools.at(state.pool_index);
    const size_t position_value = static_cast<size_t>(position);
    if (position_value >= state.token_count) {
        throw std::out_of_range("CPU attention current value is not committed");
    }
    const size_t page_index = position_value / pool.page_tokens();
    const size_t token_offset = position_value % pool.page_tokens();
    if (page_index >= state.pages.size()) {
        throw std::out_of_range("CPU attention current value page is missing");
    }
    if (pool.mode() == CpuKvCacheMode::Fp32) {
        std::copy_n(pool.value_fp32(state.pages[page_index], token_offset),
                    width, output);
        return;
    }
    const uint16_t* value = pool.value_bf16(state.pages[page_index], token_offset);
    for (size_t index = 0; index < width; ++index) {
        output[index] = bf16_bits_to_float(value[index]);
    }
}

void apply_cpu_attention_output_transform(const AttentionSpec& layout,
                                          float* output,
                                          const float* current_value) {
    const auto* transform = std::get_if<OrthogonalizeCurrentValueSpec>(
        &layout.output_transform);
    if (!transform) return;
    if (!output || !current_value) {
        throw std::invalid_argument(
            "attention current-value orthogonalization needs output and value data");
    }
    if (layout.query_heads <= 0 || layout.key_value_heads <= 0 ||
        layout.head_dim <= 0 || layout.query_heads % layout.key_value_heads != 0) {
        throw std::invalid_argument(
            "attention current-value orthogonalization has invalid head geometry");
    }
    if (!(transform->minimum_norm_squared > 0.0f) ||
        !std::isfinite(transform->minimum_norm_squared)) {
        throw std::invalid_argument(
            "attention current-value orthogonalization has invalid norm floor");
    }

    const int query_heads_per_value = layout.query_heads / layout.key_value_heads;
    for (int value_head = 0; value_head < layout.key_value_heads; ++value_head) {
        const float* value = current_value +
            static_cast<size_t>(value_head) * layout.head_dim;
        float norm_squared = 0.0f;
        for (int d = 0; d < layout.head_dim; ++d) {
            norm_squared += value[d] * value[d];
        }
        norm_squared = std::max(norm_squared, transform->minimum_norm_squared);

        for (int repetition = 0; repetition < query_heads_per_value; ++repetition) {
            const int query_head = value_head * query_heads_per_value + repetition;
            float* head_output = output +
                static_cast<size_t>(query_head) * layout.head_dim;
            float projection = 0.0f;
            for (int d = 0; d < layout.head_dim; ++d) {
                projection += head_output[d] * value[d];
            }
            const float coefficient = projection / norm_squared;
            for (int d = 0; d < layout.head_dim; ++d) {
                head_output[d] -= coefficient * value[d];
            }
        }
    }
}

void apply_cpu_attention_output_gate(float* output, const float* gate, size_t width) {
    for (size_t i = 0; i < width; ++i) {
        output[i] *= 1.0f / (1.0f + std::exp(-gate[i]));
    }
}

void apply_cpu_attention_output_gate(float* output, const float* gate,
                                     size_t width,
                                     AttentionGateGranularity granularity,
                                     int heads, int head_dim) {
    if (granularity != AttentionGateGranularity::HeadWise) {
        apply_cpu_attention_output_gate(output, gate, width);
        return;
    }
    if (heads <= 0 || head_dim <= 0 || width != static_cast<size_t>(heads * head_dim)) {
        throw std::invalid_argument("invalid head-wise attention gate geometry");
    }
    for (int head = 0; head < heads; ++head) {
        const float scale = 1.0f / (1.0f + std::exp(-gate[head]));
        for (int d = 0; d < head_dim; ++d) {
            output[static_cast<size_t>(head * head_dim + d)] *= scale;
        }
    }
}

void apply_cpu_latent_attention_positions(
    const AttentionSpec& layout,
    float* query_rope, float* key_rope, int scalar_position,
    const std::array<int32_t, 3>& rope_position) {
    const auto* latent = layout.latent_state();
    if (!latent || !latent->decoupled_rope || latent->rope_head_dim == 0) return;
    const RopePositionSpec* rope = layout.rope_position();
    if (!rope) return;
    if (const auto* multi = layout.multi_axis_position()) {
        cpu_rope_mrope(query_rope, layout.query_heads, latent->rope_head_dim,
                       rope_position, multi->sections, multi->interleaved, *rope);
        cpu_rope_mrope(key_rope, 1, latent->rope_head_dim, rope_position,
                       multi->sections, multi->interleaved, *rope);
    } else {
        cpu_rope(query_rope, layout.query_heads, latent->rope_head_dim,
                 scalar_position, *rope);
        cpu_rope(key_rope, 1, latent->rope_head_dim, scalar_position, *rope);
    }
}

}
