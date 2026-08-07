#include "attention.hpp"

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
    if (layout.positional_encoding == PositionalEncodingKind::None) {
        const float ratio = shape.numerical_policy.attention_multiplier /
            (1.0f / std::sqrt(static_cast<float>(layout.head_dim)));
        for (int i = 0; i < q_width; ++i) query[i] *= ratio;
        return;
    }

    if (layout.query_key_norm) {
        if (shape.mrope_interleaved) {
            cpu_qk_norm_rope_mrope(query, weights.q_norm.data(), layout.query_heads,
                layout.head_dim, rope_position, shape.mrope_section, true,
                static_cast<float>(layout.rope_theta),
                shape.numerical_policy.norm_eps,
                static_cast<float>(layout.rotary_fraction));
            if (has_key) {
                cpu_qk_norm_rope_mrope(key, weights.k_norm.data(), layout.key_value_heads,
                    layout.head_dim, rope_position, shape.mrope_section, true,
                    static_cast<float>(layout.rope_theta),
                    shape.numerical_policy.norm_eps,
                    static_cast<float>(layout.rotary_fraction));
            }
        } else {
            cpu_qk_norm_rope(query, weights.q_norm.data(), layout.query_heads,
                             layout.head_dim, scalar_position,
                             static_cast<float>(layout.rope_theta),
                             shape.numerical_policy.norm_eps,
                             static_cast<float>(layout.rotary_fraction));
            if (has_key) {
                cpu_qk_norm_rope(key, weights.k_norm.data(), layout.key_value_heads,
                                 layout.head_dim, scalar_position,
                                 static_cast<float>(layout.rope_theta),
                                 shape.numerical_policy.norm_eps,
                                 static_cast<float>(layout.rotary_fraction));
            }
        }
        for (int i = 0; i < q_width; ++i) query[i] *= layout.query_scale;
        return;
    }

    if (shape.mrope_interleaved) {
        cpu_rope_mrope(query, layout.query_heads, layout.head_dim, rope_position,
                       shape.mrope_section, true,
                       static_cast<float>(layout.rope_theta),
                       static_cast<float>(layout.rotary_fraction));
        if (has_key) {
            cpu_rope_mrope(key, layout.key_value_heads, layout.head_dim, rope_position,
                           shape.mrope_section, true,
                           static_cast<float>(layout.rope_theta),
                           static_cast<float>(layout.rotary_fraction));
        }
    } else {
        cpu_rope(query, layout.query_heads, layout.head_dim, scalar_position,
                 static_cast<float>(layout.rope_theta),
                 static_cast<float>(layout.rotary_fraction));
        if (has_key) {
            cpu_rope(key, layout.key_value_heads, layout.head_dim, scalar_position,
                     static_cast<float>(layout.rope_theta),
                     static_cast<float>(layout.rotary_fraction));
        }
    }
    const float ratio = shape.numerical_policy.attention_multiplier /
        (1.0f / std::sqrt(static_cast<float>(layout.head_dim)));
    for (int i = 0; i < q_width; ++i) query[i] *= ratio;
}

void apply_cpu_query_gate(float* output, const float* gate, size_t width) {
    for (size_t i = 0; i < width; ++i) {
        output[i] *= 1.0f / (1.0f + std::exp(-gate[i]));
    }
}

} // namespace celeg
