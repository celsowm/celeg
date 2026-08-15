#include "celeg/model/resolved.hpp"
#include "celeg/model/program.hpp"
#include "support/assertions.hpp"

#include <iostream>
#include <stdexcept>

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
        layer.per_layer_input = {4, celeg::ActivationKind::GeluTanh, true};
        layer.per_layer_input_norm = celeg::NormSpec{1.0e-5f};
        layer.feed_forward = celeg::DenseFeedForwardSpec{
            16, celeg::ActivationKind::SwiGLU};
    }
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

    std::cout << "policy_test: ok\n";
    return 0;
}
