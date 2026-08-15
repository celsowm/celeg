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

}

ModelGraph finalize_descriptor_graph(ModelGraph graph, const Descriptor& descriptor,
                                     const NumericalPolicy& numerical_policy,
                                     const CheckpointMetadata& metadata) {
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
    for (int layer_index = 0; layer_index < static_cast<int>(graph.layers.size()); ++layer_index) {
        LayerSpec& layer = graph.layers[static_cast<size_t>(layer_index)];
        const float epsilon = numerical_policy.norm_eps;
        const float post_epsilon = numerical_policy.post_norm_eps;
        layer.operator_norm = {epsilon, descriptor.operator_norm_kind};
        layer.feed_forward_norm = {epsilon, descriptor.feed_forward_norm_kind};
        if (descriptor.split_attention_norms) {
            layer.post_attention_norm = {post_epsilon, descriptor.operator_norm_kind};
            layer.pre_feed_forward_norm = {epsilon, descriptor.feed_forward_norm_kind};
            layer.post_feed_forward_norm = {post_epsilon, descriptor.feed_forward_norm_kind};
        }
        if (const auto* current_attention = std::get_if<AttentionSpec>(&layer.mixer)) {
            AttentionSpec attention = *current_attention;
            apply_rope_pairing(attention.position, rope_pairing);
            if (has_explicit_query_scale) {
                attention.query_scale = explicit_query_scale;
            }
            if (orthogonalize_current_value) {
                attention.output_transform = OrthogonalizeCurrentValueSpec{
                    orthogonalize_minimum_norm_squared};
            }
            layer.mixer = std::move(attention);
        }
        layer.residual.multiplier = numerical_policy.residual_multiplier;
    }
    if (descriptor.per_layer_input_size.has_value()) {
        graph.per_layer_input = PerLayerInputPolicy{
            integer_value(metadata, *descriptor.per_layer_input_size),
            ActivationKind::GeluTanh,
            NormSpec{numerical_policy.norm_eps, NormWeightKind::Scale}};
    } else {
        graph.per_layer_input = std::nullopt;
    }
    graph.validate();
    return graph;
}

}
