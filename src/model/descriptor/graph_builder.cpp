#include "detail.hpp"

namespace celeg::descriptor_detail {
namespace {

RopePairingKind rope_pairing_kind(std::string_view value) {
    if (value.empty() || value == "split_half") return RopePairingKind::SplitHalf;
    if (value == "adjacent_pairs") return RopePairingKind::AdjacentPairs;
    throw std::invalid_argument("descriptor has unsupported RoPE pairing: " +
                                std::string(value));
}

void apply_rope_pairing(PositionSpec& position, RopePairingKind pairing) {
    if (auto* rope = std::get_if<RopePositionSpec>(&position)) {
        rope->pairing = pairing;
        return;
    }
    if (auto* multi = std::get_if<MultiAxisRopeSpec>(&position)) {
        if (pairing != RopePairingKind::SplitHalf) {
            throw std::invalid_argument(
                "multi-axis RoPE currently supports split-half pairing only");
        }
        multi->base.pairing = pairing;
    }
}

} // namespace

ModelGraph build_descriptor_graph(const Descriptor& descriptor,
                                  const RuntimeTopology& topology,
                                  const NumericalPolicy& numerical_policy,
                                  const CheckpointMetadata& metadata) {
    ModelGraph graph;
    graph.hidden = topology.hidden;
    graph.embedding_transform.multiplier =
        numerical_policy.embedding_multiplier;
    if (descriptor.embedding_post_norm_kind) {
        graph.embedding_transform.post_norm = NormSpec{
            numerical_policy.norm_eps, *descriptor.embedding_post_norm_kind};
    }
    graph.logits_divisor = numerical_policy.logits_divisor;
    graph.logits_multiplier = numerical_policy.logits_multiplier;
    graph.final_norm = {numerical_policy.norm_eps,
                              descriptor.final_norm_kind};
    graph.final_logit_softcap = descriptor.final_logit_softcap.has_value()
        ? static_cast<float>(number_value(metadata, *descriptor.final_logit_softcap)) : 0.0f;
    if (descriptor.norm_after_physical_block) {
        const int physical_layer_count = integer_value(
            metadata, descriptor.dimensions.at("layers"));
        graph.norm_after_layers = {physical_layer_count - 1};
    } else {
        graph.norm_after_layers.clear();
    }
    const bool orthogonalize_current_value = boolean_value(
        metadata, descriptor.orthogonalize_current_value, false);
    const float orthogonalize_minimum_norm_squared =
        descriptor.orthogonalize_current_value_minimum_norm_squared.has_value()
            ? static_cast<float>(number_value(
                metadata, *descriptor.orthogonalize_current_value_minimum_norm_squared))
            : 1.0e-6f;
    const auto query_scale_field = descriptor.numbers.find("query_scale");
    const bool has_explicit_query_scale = query_scale_field != descriptor.numbers.end();
    const float explicit_query_scale = has_explicit_query_scale
        ? static_cast<float>(number_value(metadata, query_scale_field->second)) : 1.0f;
    const RopePairingKind rope_pairing = rope_pairing_kind(descriptor.rope_pairing);
    const int shared_layers = descriptor.shared_kv_suffix_layers.has_value()
        ? integer_value(metadata, *descriptor.shared_kv_suffix_layers) : 0;
    const int shared_start = topology.num_hidden_layers - shared_layers;

    graph.layers.clear();
    graph.layers.reserve(static_cast<size_t>(topology.num_hidden_layers));
    for (int layer_index = 0; layer_index < topology.num_hidden_layers; ++layer_index) {
        LayerSpec layer;
        const float epsilon = numerical_policy.norm_eps;
        const float post_epsilon = numerical_policy.post_norm_eps;
        layer.operator_norm = {epsilon, descriptor.operator_norm_kind};
        layer.feed_forward_norm = {epsilon, descriptor.feed_forward_norm_kind};
        if (descriptor.split_attention_norms) {
            layer.post_attention_norm = {post_epsilon, descriptor.operator_norm_kind};
            layer.pre_feed_forward_norm = {epsilon, descriptor.feed_forward_norm_kind};
            layer.post_feed_forward_norm = {post_epsilon, descriptor.feed_forward_norm_kind};
        }
        if (topology.mixer_kinds[static_cast<size_t>(layer_index)] == MixerKind::Attention) {
            AttentionSpec attention = topology.attention_layout(layer_index);
            apply_rope_pairing(attention.position, rope_pairing);
            if (has_explicit_query_scale) {
                attention.query_scale = explicit_query_scale;
            }
            if (orthogonalize_current_value) {
                attention.output_transform = OrthogonalizeCurrentValueSpec{
                    orthogonalize_minimum_norm_squared};
            }
            layer.mixer = std::move(attention);
        } else if (topology.mixer_kinds[static_cast<size_t>(layer_index)] ==
                   MixerKind::ShortConvolution) {
            layer.mixer = ShortConvolutionSpec{topology.conv_cache, topology.conv_dim, false};
        } else if (topology.mixer_kinds[static_cast<size_t>(layer_index)] ==
                   MixerKind::GatedDeltaNet) {
            layer.mixer = topology.gated_delta_net_layouts.at(static_cast<size_t>(layer_index));
        } else if (topology.mixer_kinds[static_cast<size_t>(layer_index)] == MixerKind::Mamba2) {
            layer.mixer = topology.mamba2_layouts.at(static_cast<size_t>(layer_index));
        } else if (topology.mixer_kinds[static_cast<size_t>(layer_index)] == MixerKind::MlpOnly) {
            layer.mixer = topology.mlp_only_layouts.at(static_cast<size_t>(layer_index));
        } else {
            throw std::invalid_argument("descriptor has unsupported hybrid mixer kind");
        }
        int intermediate = topology.feed_forward_intermediates.at(
            static_cast<size_t>(layer_index));
        if (descriptor.double_wide_shared_suffix && layer_index >= shared_start &&
            topology.feed_forward_kinds[static_cast<size_t>(layer_index)] ==
                FeedForwardKind::Dense) {
            intermediate *= 2;
        }
        if (topology.feed_forward_kinds[static_cast<size_t>(layer_index)] ==
            FeedForwardKind::MixtureOfExperts) {
            layer.feed_forward = MixtureOfExpertsSpec{
                .intermediate_size = topology.moe_intermediate,
                .num_experts = topology.num_experts,
                .experts_per_token = topology.experts_per_token,
                .normalize_topk = topology.normalize_topk,
                .use_expert_bias = topology.use_expert_bias,
                .routed_scaling_factor = topology.routed_scaling_factor,
                .routing_group_count = topology.moe_routing_group_count,
                .routing_experts_per_group = topology.moe_routing_experts_per_group,
                .routing_groups_per_token = topology.moe_routing_groups_per_token,
                .routing_group_score_top_k = topology.moe_routing_group_score_top_k,
                .has_shared_expert = topology.shared_expert_intermediate > 0,
                .shared_intermediate_size = topology.shared_expert_intermediate,
                .shared_before_routed = false,
                .router_softmax = topology.moe_router_softmax};
        } else {
            layer.feed_forward = DenseFeedForwardSpec{
                intermediate, topology.feed_forward_activations.at(
                    static_cast<size_t>(layer_index))};
        }
        layer.residual.multiplier = numerical_policy.residual_multiplier;
        layer.execute_feed_forward = topology.mixer_kinds[static_cast<size_t>(layer_index)] !=
            MixerKind::MlpOnly;
        layer.per_layer_input = {topology.per_layer_input_size, ActivationKind::GeluTanh,
                                 topology.has_per_layer_input};
        graph.layers.push_back(std::move(layer));
    }
    graph.validate();
    return graph;
}

} // namespace celeg::descriptor_detail
