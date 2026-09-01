#include "celeg/model/resolved.hpp"
#include "celeg/model/program.hpp"
#include "support/assertions.hpp"

#include <iostream>
#include <stdexcept>

namespace {

celeg::ModelGraph attention_graph() {
    celeg::ModelGraph graph;
    graph.hidden = 8;
    celeg::LayerSpec layer;
    celeg::AttentionSpec attention;
    attention.query_heads = 2;
    attention.key_value_heads = 1;
    attention.head_dim = 4;
    attention.position = celeg::NoPositionEncodingSpec{};
    layer.mixer = attention;
    layer.feed_forward = celeg::DenseFeedForwardSpec{
        16, celeg::ActivationKind::SwiGLU};
    graph.layers.push_back(std::move(layer));
    return graph;
}

bool graph_rejected(celeg::ModelGraph graph) {
    try {
        graph.validate();
    } catch (const std::runtime_error&) {
        return true;
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}

int main() {
    celeg::TokenPolicy tokens;
    tokens.bos_token_id = 1;
    tokens.eos_token_ids = {2};
    tokens.pad_token_id = 0;
    tokens.validate();

    celeg::NumericalPolicy numerical;
    numerical.norm_eps = 1.0e-5f;
    numerical.logits_divisor = 1.0f;
    numerical.validate();

    bool rejected_tokens = false;
    try {
        celeg::TokenPolicy invalid;
        invalid.validate();
    } catch (const std::runtime_error&) {
        rejected_tokens = true;
    }
    CELEG_TEST_CHECK(rejected_tokens);

    bool rejected_numerical = false;
    try {
        celeg::NumericalPolicy invalid;
        invalid.norm_eps = 0.0f;
        invalid.validate();
    } catch (const std::runtime_error&) {
        rejected_numerical = true;
    }
    CELEG_TEST_CHECK(rejected_numerical);

    celeg::ResolvedModel resolved;
    resolved.graph.hidden = 8;
    resolved.graph.layers.resize(2);
    for (auto& layer : resolved.graph.layers) {
        layer.mixer = celeg::AttentionSpec{};
        layer.feed_forward = celeg::DenseFeedForwardSpec{
            16, celeg::ActivationKind::SwiGLU};
    }
    resolved.graph.per_layer_input = celeg::PerLayerInputPolicy{
        4, celeg::ActivationKind::GeluTanh, celeg::NormSpec{1.0e-5f}};
    resolved.topology = celeg::compose_runtime_topology(
        celeg::CheckpointDimensions{32, 0, {}, {}, 0}, resolved.graph);
    const auto per_layer = celeg::PerLayerInputPlan::derive(resolved);
    CELEG_TEST_CHECK(per_layer.enabled && per_layer.packed_width == 8);
    CELEG_TEST_CHECK(per_layer.checked_elements(3) == 24);

    celeg::ResolvedModel overflow = resolved;
    overflow.graph.hidden = 0;
    bool rejected_overflow = false;
    try {
        (void)celeg::PerLayerInputPlan::derive(overflow);
    } catch (const std::invalid_argument&) {
        rejected_overflow = true;
    }
    CELEG_TEST_CHECK(rejected_overflow);

    celeg::RuntimeTopology token_topology = resolved.topology;
    token_topology.dims.token_policy = {1, {2}, 32};
    bool rejected_token_range = false;
    try {
        token_topology.validate();
    } catch (const std::runtime_error&) {
        rejected_token_range = true;
    }
    CELEG_TEST_CHECK(rejected_token_range);

    celeg::ModelGraph packed_elementwise = attention_graph();
    auto& packed_elementwise_attention = std::get<celeg::AttentionSpec>(
        packed_elementwise.layers[0].mixer);
    packed_elementwise_attention.output_gate = celeg::SigmoidAttentionGateSpec{
        true, celeg::AttentionGateGranularity::ElementWise};
    packed_elementwise.validate();

    celeg::ModelGraph packed_headwise = attention_graph();
    auto& packed_headwise_attention = std::get<celeg::AttentionSpec>(
        packed_headwise.layers[0].mixer);
    packed_headwise_attention.output_gate = celeg::SigmoidAttentionGateSpec{
        true, celeg::AttentionGateGranularity::HeadWise};
    CELEG_TEST_CHECK(graph_rejected(std::move(packed_headwise)));

    celeg::ModelGraph packed_latent = attention_graph();
    auto& packed_latent_attention = std::get<celeg::AttentionSpec>(
        packed_latent.layers[0].mixer);
    packed_latent_attention.state = celeg::LatentAttentionStateSpec{
        4, 0, 4, false};
    packed_latent_attention.output_gate = celeg::SigmoidAttentionGateSpec{
        true, celeg::AttentionGateGranularity::ElementWise};
    CELEG_TEST_CHECK(graph_rejected(std::move(packed_latent)));

    celeg::ModelGraph external = attention_graph();
    auto& external_attention = std::get<celeg::AttentionSpec>(
        external.layers[0].mixer);
    external_attention.key_value_source = celeg::ExternalMemorySource{2};
    external.validate();

    celeg::ModelGraph negative_external_slot = external;
    std::get<celeg::AttentionSpec>(negative_external_slot.layers[0].mixer)
        .key_value_source = celeg::ExternalMemorySource{-1};
    CELEG_TEST_CHECK(graph_rejected(std::move(negative_external_slot)));

    celeg::ModelGraph shared_external = external;
    std::get<celeg::AttentionSpec>(shared_external.layers[0].mixer).kv_sharing =
        celeg::SharedKvPublisher{0};
    CELEG_TEST_CHECK(graph_rejected(std::move(shared_external)));

    celeg::ModelGraph latent_external = external;
    std::get<celeg::AttentionSpec>(latent_external.layers[0].mixer).state =
        celeg::LatentAttentionStateSpec{4, 0, 4, false};
    CELEG_TEST_CHECK(graph_rejected(std::move(latent_external)));

    celeg::ModelGraph transformed_external = external;
    std::get<celeg::AttentionSpec>(transformed_external.layers[0].mixer)
        .output_transform = celeg::OrthogonalizeCurrentValueSpec{1.0e-6f};
    CELEG_TEST_CHECK(graph_rejected(std::move(transformed_external)));

    std::cout << "policy_test: ok\n";
    return 0;
}
