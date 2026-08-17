#pragma once

#include "celeg/model/program.hpp"

#include <algorithm>
#include <cstddef>

namespace celeg {

struct CudaWorkspacePlan {
    size_t hidden = 0;
    size_t attention_output = 0;
    size_t attention_projection = 0;
    size_t latent_query_content = 0;
    size_t latent_query_rope = 0;
    size_t latent_rank = 0;
    size_t latent_rope = 0;
    size_t latent_projection = 0;
    size_t latent_decompressed = 0;
    int attention_query_heads = 0;
    int attention_head_dim = 0;
    size_t mamba_projection = 0;
    size_t mamba_inner = 0;
    size_t gated_delta_qkv = 0;
    size_t gated_delta_z = 0;
    size_t gated_delta_b = 0;
    size_t gated_delta_a = 0;
    size_t gated_delta_output = 0;
    size_t ffn_intermediate = 0;
    int conv_cache = 0;

    static CudaWorkspacePlan from_program(const CompiledModelProgram& program) {
        CudaWorkspacePlan plan;
        plan.hidden = static_cast<size_t>(program.hidden);
        for (const CompiledLayerProgram& layer : program.layers) {
            std::visit([&](const auto& mixer) {
                using Mixer = std::decay_t<decltype(mixer)>;
                if constexpr (std::is_same_v<Mixer, CompiledAttentionProgram>) {
                    const AttentionSpec& attn = mixer.semantics;
                    plan.attention_query_heads = std::max(plan.attention_query_heads, attn.query_heads);
                    plan.attention_head_dim = std::max(plan.attention_head_dim, attn.head_dim);
                    const size_t out_width = attn.uses_latent_state()
                        ? static_cast<size_t>(attn.latent_query_content_width())
                        : static_cast<size_t>(attn.query_width());
                    plan.attention_output = std::max(plan.attention_output, out_width);
                    plan.attention_projection = std::max(
                        plan.attention_projection, static_cast<size_t>(attn.projection_width()));
                    if (const auto* latent = attn.latent_state()) {
                        plan.latent_query_content = std::max(
                            plan.latent_query_content, static_cast<size_t>(attn.latent_query_content_width()));
                        plan.latent_query_rope = std::max(
                            plan.latent_query_rope, static_cast<size_t>(attn.latent_query_rope_width()));
                        plan.latent_rank = std::max(
                            plan.latent_rank, static_cast<size_t>(latent->latent_rank));
                        plan.latent_rope = std::max(
                            plan.latent_rope, static_cast<size_t>(latent->decoupled_rope ? latent->rope_head_dim : 0));
                        plan.latent_projection = std::max(
                            plan.latent_projection, static_cast<size_t>(attn.latent_query_projection_width()));
                        plan.latent_decompressed = std::max(
                            plan.latent_decompressed, static_cast<size_t>(attn.latent_output_width()));
                    }
                } else if constexpr (std::is_same_v<Mixer, ShortConvolutionSpec>) {
                    plan.conv_cache = std::max(plan.conv_cache, mixer.cache_length);
                } else if constexpr (std::is_same_v<Mixer, GatedDeltaNetSpec>) {
                    plan.gated_delta_qkv = std::max(
                        plan.gated_delta_qkv, static_cast<size_t>(mixer.qkv_width()));
                    plan.gated_delta_z = std::max(
                        plan.gated_delta_z, static_cast<size_t>(mixer.value_width()));
                    plan.gated_delta_b = std::max(
                        plan.gated_delta_b, static_cast<size_t>(std::max(mixer.value_heads, mixer.decay_width())));
                    plan.gated_delta_a = std::max(
                        plan.gated_delta_a, static_cast<size_t>(std::max(mixer.value_heads, mixer.decay_width())));
                    plan.gated_delta_output = std::max(
                        plan.gated_delta_output, static_cast<size_t>(mixer.value_width()));
                    plan.attention_output = std::max(
                        plan.attention_output, static_cast<size_t>(mixer.value_width()));
                } else if constexpr (std::is_same_v<Mixer, Mamba2Spec>) {
                    plan.mamba_inner = std::max(
                        plan.mamba_inner, static_cast<size_t>(mixer.intermediate_size));
                    plan.mamba_projection = std::max(
                        plan.mamba_projection,
                        static_cast<size_t>(2 * mixer.intermediate_size +
                                            2 * mixer.group_count * mixer.state_size + mixer.num_heads));
                    plan.attention_output = std::max(
                        plan.attention_output, static_cast<size_t>(mixer.intermediate_size));
                } else if constexpr (std::is_same_v<Mixer, MlpBlockSpec>) {
                    plan.ffn_intermediate = std::max(
                        plan.ffn_intermediate, static_cast<size_t>(mixer.intermediate_size));
                }
            }, layer.mixer);

            std::visit([&](const auto& ff) {
                using FF = std::decay_t<decltype(ff)>;
                if constexpr (std::is_same_v<FF, std::monostate>) {
                    return;
                } else if constexpr (std::is_same_v<FF, CompiledDenseFeedForwardProgram>) {
                    plan.ffn_intermediate = std::max(
                        plan.ffn_intermediate, static_cast<size_t>(ff.intermediate_size));
                } else if constexpr (std::is_same_v<FF, MoeLayerProgram>) {
                    plan.ffn_intermediate = std::max(
                        plan.ffn_intermediate, static_cast<size_t>(ff.routed.mlp.intermediate_size));
                }
            }, layer.feed_forward);
        }
        return plan;
    }
};

}
