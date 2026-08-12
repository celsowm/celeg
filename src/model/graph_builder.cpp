#include "celeg/model/graph_builder.hpp"

#include <utility>
#include <stdexcept>

namespace celeg {

void build_dense_transformer_graph(ResolvedModel& model,
                                  const RuntimeTopology& t) {
    model.graph.hidden = t.hidden;
    model.graph.embedding_transform.multiplier = t.numerical_policy.embedding_multiplier;
    model.graph.logits_divisor = t.numerical_policy.logits_divisor;
    model.graph.logits_multiplier = t.numerical_policy.logits_multiplier;
    model.graph.final_norm = {t.numerical_policy.norm_eps, NormWeightKind::Scale};
    model.graph.final_logit_softcap = 0.0f;
    model.graph.norm_after_layers.clear();
    model.graph.layers.clear();
    model.graph.layers.reserve(static_cast<size_t>(t.num_hidden_layers));
    for (int i = 0; i < t.num_hidden_layers; ++i) {
        LayerSpec layer;
        layer.operator_norm = {t.numerical_policy.norm_eps, NormWeightKind::Scale};
        layer.feed_forward_norm = {t.numerical_policy.norm_eps, NormWeightKind::Scale};
        if (t.mixer_kinds[static_cast<size_t>(i)] == MixerKind::Attention) {
            if (i >= static_cast<int>(t.attention_layouts.size())) {
                throw std::runtime_error("attention layer has no resolved layout");
            }
            layer.mixer = t.attention_layout(i);
        } else if (t.mixer_kinds[static_cast<size_t>(i)] == MixerKind::ShortConvolution) {
            layer.mixer = ShortConvolutionSpec{t.conv_cache, t.conv_dim, false};
        } else if (t.mixer_kinds[static_cast<size_t>(i)] == MixerKind::GatedDeltaNet) {
            layer.mixer = t.gated_delta_net_layouts.at(static_cast<size_t>(i));
        } else if (t.mixer_kinds[static_cast<size_t>(i)] == MixerKind::Mamba2) {
            layer.mixer = t.mamba2_layouts.at(static_cast<size_t>(i));
        } else if (t.mixer_kinds[static_cast<size_t>(i)] == MixerKind::MlpOnly) {
            layer.mixer = t.mlp_only_layouts.at(static_cast<size_t>(i));
        } else {
            throw std::runtime_error("unknown automatic mixer kind");
        }
        const int intermediate = t.feed_forward_intermediates.at(static_cast<size_t>(i));
        if (t.feed_forward_kinds[static_cast<size_t>(i)] == FeedForwardKind::MixtureOfExperts) {
            layer.feed_forward = MixtureOfExpertsSpec{
                .intermediate_size = t.moe_intermediate,
                .num_experts = t.num_experts,
                .experts_per_token = t.experts_per_token,
                .normalize_topk = t.normalize_topk,
                .use_expert_bias = t.use_expert_bias,
                .routed_scaling_factor = t.routed_scaling_factor,
                .routing_group_count = t.moe_routing_group_count,
                .routing_experts_per_group = t.moe_routing_experts_per_group,
                .routing_groups_per_token = t.moe_routing_groups_per_token,
                .routing_group_score_top_k = t.moe_routing_group_score_top_k,
                .has_shared_expert = t.shared_expert_intermediate > 0,
                .shared_intermediate_size = t.shared_expert_intermediate,
                .shared_before_routed = false,
                .router_softmax = t.moe_router_softmax};
        } else {
            layer.feed_forward = DenseFeedForwardSpec{intermediate, ActivationKind::SwiGLU};
        }
        layer.residual.multiplier = t.numerical_policy.residual_multiplier;
        if (!t.execute_feed_forward.empty()) {
            layer.execute_feed_forward = t.execute_feed_forward.at(static_cast<size_t>(i));
        }
        model.graph.layers.push_back(std::move(layer));
    }
}

} // namespace celeg
