#include "detail.hpp"

namespace celeg::descriptor_detail {

void build_descriptor_graph(ResolvedModel& model, const Descriptor& descriptor,
                            const CheckpointMetadata& metadata) {
    const RuntimeTopology& topology = model.topology;
    model.graph.embedding_multiplier = topology.numerical_policy.embedding_multiplier;
    model.graph.logits_divisor = topology.numerical_policy.logits_divisor;
    model.graph.final_norm.epsilon = topology.numerical_policy.norm_eps;
    model.graph.final_logit_softcap = descriptor.final_logit_softcap.has_value()
        ? static_cast<float>(number_value(metadata, *descriptor.final_logit_softcap)) : 0.0f;
    if (descriptor.norm_after_physical_block) {
        const int physical_layer_count = integer_value(
            metadata, descriptor.dimensions.at("layers"));
        model.graph.norm_after_layers = {physical_layer_count - 1};
    } else {
        model.graph.norm_after_layers.clear();
    }
    model.graph.layers.clear();
    model.graph.layers.reserve(static_cast<size_t>(topology.num_hidden_layers));
    for (int layer_index = 0; layer_index < topology.num_hidden_layers; ++layer_index) {
        LayerSpec layer;
        const float epsilon = topology.numerical_policy.norm_eps;
        layer.operator_norm.epsilon = epsilon;
        layer.feed_forward_norm.epsilon = epsilon;
        if (descriptor.split_attention_norms) {
            layer.post_attention_norm.epsilon = epsilon;
            layer.pre_feed_forward_norm.epsilon = epsilon;
            layer.post_feed_forward_norm.epsilon = epsilon;
        }
        if (topology.mixer_kinds[static_cast<size_t>(layer_index)] == MixerKind::Attention) {
            layer.mixer = topology.attention_layout(layer_index);
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
        const int intermediate = topology.feed_forward_intermediates.empty()
            ? topology.intermediate
            : topology.feed_forward_intermediates.at(static_cast<size_t>(layer_index));
        if (topology.feed_forward_kinds[static_cast<size_t>(layer_index)] ==
            FeedForwardKind::MixtureOfExperts) {
            layer.feed_forward = MixtureOfExpertsSpec{
                topology.moe_intermediate, topology.num_experts, topology.experts_per_token,
                topology.normalize_topk, topology.use_expert_bias, topology.routed_scaling_factor,
                0, 0, topology.shared_expert_intermediate > 0,
                topology.shared_expert_intermediate, false, topology.moe_router_softmax};
        } else {
            layer.feed_forward = DenseFeedForwardSpec{
                intermediate, topology.feed_forward_activations.empty()
                    ? descriptor.feed_forward_activation
                    : topology.feed_forward_activations.at(static_cast<size_t>(layer_index))};
        }
        layer.residual.multiplier = topology.numerical_policy.residual_multiplier;
        layer.execute_feed_forward = topology.mixer_kinds[static_cast<size_t>(layer_index)] !=
            MixerKind::MlpOnly;
        layer.per_layer_input = {topology.per_layer_input_size, ActivationKind::GeluTanh,
                                 topology.has_per_layer_input};
        model.graph.layers.push_back(std::move(layer));
    }
}


} // namespace celeg::descriptor_detail
