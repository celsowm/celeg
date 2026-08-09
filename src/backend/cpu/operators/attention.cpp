#include "attention.hpp"
#include "celeg/model/position.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace celeg {

void apply_cpu_attention_qk(const RuntimeTopology& shape,
                            const AttentionSpec& layout,
                            const CpuCompiledModel::AttentionWeights& weights,
                            float* query,
                            float* key,
                            int scalar_position,
                            const std::array<int32_t, 3>& rope_position) {
    const int q_width = layout.query_width();
    const bool has_key = key != nullptr && !weights.k.segments.empty();
    const RopePositionSpec* rope = layout.rope_position();
    if (rope == nullptr) {
        const float ratio = shape.numerical_policy.attention_multiplier /
            (1.0f / std::sqrt(static_cast<float>(layout.head_dim)));
        for (int i = 0; i < q_width; ++i) query[i] *= ratio;
        return;
    }

    if (layout.query_key_norm) {
        if (const auto* multi = layout.multi_axis_position()) {
            cpu_qk_norm_rope_mrope(query, weights.q_norm.data(), layout.query_heads,
                layout.head_dim, rope_position, multi->sections, multi->interleaved,
                *rope, shape.numerical_policy.norm_eps);
            if (has_key) {
                cpu_qk_norm_rope_mrope(key, weights.k_norm.data(), layout.key_value_heads,
                    layout.head_dim, rope_position, multi->sections, multi->interleaved,
                    *rope, shape.numerical_policy.norm_eps);
            }
        } else {
            cpu_qk_norm_rope(query, weights.q_norm.data(), layout.query_heads,
                             layout.head_dim, scalar_position,
                             *rope, shape.numerical_policy.norm_eps);
            if (has_key) {
                cpu_qk_norm_rope(key, weights.k_norm.data(), layout.key_value_heads,
                                 layout.head_dim, scalar_position,
                                 *rope, shape.numerical_policy.norm_eps);
            }
        }
        const float query_scale = layout.query_scale * rope_attention_scale(*rope, scalar_position);
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
    const float ratio = shape.numerical_policy.attention_multiplier /
        (1.0f / std::sqrt(static_cast<float>(layout.head_dim)));
    const float query_scale = ratio * rope_attention_scale(*rope, scalar_position);
    for (int i = 0; i < q_width; ++i) query[i] *= query_scale;
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

void apply_cpu_query_gate(float* output, const float* gate, size_t width) {
    for (size_t i = 0; i < width; ++i) {
        output[i] *= 1.0f / (1.0f + std::exp(-gate[i]));
    }
}

void apply_cpu_latent_attention_positions(
    const RuntimeTopology& shape, const AttentionSpec& layout,
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
    (void)shape;
}

} // namespace celeg
