#include "attention.hpp"
#include "celeg/model/position.hpp"

#include <cmath>

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
