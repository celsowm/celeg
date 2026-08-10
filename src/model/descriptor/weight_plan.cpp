#include "detail.hpp"
#include "weight_requirements.hpp"

#include <type_traits>
#include <utility>

namespace celeg::descriptor_detail {
namespace {

void add_descriptor_request(ResolvedModel& model, TensorRole role, int layer, int expert,
                            std::vector<int64_t> shape, int physical_layer = -1) {
    model.weight_plan.requests.push_back({role, layer, expert, std::move(shape),
                                          std::nullopt, physical_layer});
}

void add_norm_request(ResolvedModel& model, TensorRole role, int layer,
                      const NormSpec& spec, std::vector<int64_t> shape,
                      int physical_layer) {
    if (!spec.enabled() || spec.weightless()) return;
    model.weight_plan.requests.push_back({role, layer, -1, std::move(shape),
                                          std::nullopt, physical_layer,
                                          spec.weight_kind});
}

} // namespace

void build_weight_plan(ResolvedModel& model, const Descriptor&,
                       const ITensorNamingPolicy& naming_policy) {
    const RuntimeTopology& topology = model.topology;
    model.weight_plan.requests.clear();
    add_descriptor_request(model, TensorRole::TokenEmbedding, -1, -1,
                           {topology.vocab_size, topology.hidden});
    add_descriptor_request(model, TensorRole::LanguageModelHead, -1, -1,
                           {topology.vocab_size, topology.hidden});
    add_norm_request(model, TensorRole::FinalNorm, -1, model.graph.final_norm,
                     {topology.hidden}, -1);
    if (topology.has_per_layer_input) {
        const int width = topology.num_hidden_layers * topology.per_layer_input_size;
        add_descriptor_request(model, TensorRole::PerLayerEmbedding, -1, -1,
                               {topology.vocab_size, width});
        add_descriptor_request(model, TensorRole::PerLayerContextProjection, -1, -1,
                               {width, topology.hidden});
        add_descriptor_request(model, TensorRole::PerLayerProjectionNorm, -1, -1,
                               {topology.per_layer_input_size});
    }
    for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
        const LayerSpec& semantic_layer = model.graph.layers.at(
            static_cast<size_t>(layer));
        const int physical_layer = topology.checkpoint_layer_for_layer.empty()
            ? layer : topology.checkpoint_layer_for_layer.at(static_cast<size_t>(layer));
        const int intermediate = std::visit([](const auto& feed_forward) {
            using FeedForward = std::decay_t<decltype(feed_forward)>;
            if constexpr (std::is_same_v<FeedForward, DenseFeedForwardSpec> ||
                          std::is_same_v<FeedForward, MlpBlockSpec>) {
                return feed_forward.intermediate_size;
            } else {
                return feed_forward.intermediate_size;
            }
        }, semantic_layer.feed_forward);
        add_norm_request(model, TensorRole::AttentionInputNorm, layer,
                         semantic_layer.operator_norm, {topology.hidden}, physical_layer);
        if (semantic_layer.mixer_kind() == MixerKind::Attention) {
            append_attention_weight_requirements(model, topology, layer, physical_layer);
        } else if (semantic_layer.mixer_kind() == MixerKind::ShortConvolution) {
            append_short_conv_weight_requirements(model, topology, layer, physical_layer);
        } else if (semantic_layer.mixer_kind() == MixerKind::GatedDeltaNet) {
            append_gated_delta_net_weight_requirements(model, topology, layer, physical_layer);
        } else if (semantic_layer.mixer_kind() == MixerKind::Mamba2) {
            append_mamba2_weight_requirements(model, topology, layer, physical_layer);
        }
        const bool mlp_only = semantic_layer.mixer_kind() == MixerKind::MlpOnly;
        if (mlp_only) {
            append_mlp_only_weight_requirements(model, topology, layer, physical_layer);
        } else {
            add_norm_request(model, TensorRole::FfnInputNorm, layer,
                             semantic_layer.feed_forward_norm, {topology.hidden}, physical_layer);
        }
        if (!mlp_only && semantic_layer.feed_forward_kind() ==
                FeedForwardKind::MixtureOfExperts) {
            append_moe_weight_requirements(model, topology, layer, physical_layer);
        } else if (!mlp_only) {
            append_dense_ffn_weight_requirements(model, topology, layer, physical_layer,
                                                 intermediate);
        }
        if (topology.has_per_layer_input) {
            add_descriptor_request(model, TensorRole::PerLayerInputGate, layer, -1,
                                   {topology.per_layer_input_size, topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::PerLayerProjection, layer, -1,
                                   {topology.hidden, topology.per_layer_input_size}, physical_layer);
            add_descriptor_request(model, TensorRole::PerLayerInputNorm, layer, -1,
                                   {topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::LayerScalar, layer, -1, {1}, physical_layer);
        }
    }
    resolve_weight_plan(model, naming_policy);
}

} // namespace celeg::descriptor_detail
