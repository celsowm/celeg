#include "celeg/model/resolved.hpp"
#include "celeg/model/program.hpp"
#include "support/assertions.hpp"

#include <iostream>
#include <limits>
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
    resolved.topology.hidden = 8;
    resolved.topology.intermediate = 16;
    resolved.topology.checkpoint.vocab_size = 32;
    resolved.topology.num_hidden_layers = 2;
    resolved.topology.mixer_kinds = {
        celeg::MixerKind::Attention, celeg::MixerKind::Attention};
    resolved.topology.feed_forward_kinds = {
        celeg::FeedForwardKind::Dense, celeg::FeedForwardKind::Dense};
    resolved.topology.attention_layer_count = 2;
    resolved.topology.conv_layer_count = 0;
    resolved.topology.attention_layouts.resize(2);
    resolved.topology.has_per_layer_input = true;
    resolved.topology.per_layer_input_size = 4;
    resolved.graph.layers.resize(2);
    for (auto& layer : resolved.graph.layers) {
        layer.per_layer_input = {4, celeg::ActivationKind::GeluTanh, true};
        layer.per_layer_input_norm.epsilon = 1.0e-5f;
    }
    const auto per_layer = celeg::PerLayerInputPlan::derive(resolved);
    CELEG_TEST_CHECK(per_layer.enabled && per_layer.packed_width == 8);
    CELEG_TEST_CHECK(per_layer.checked_elements(3) == 24);

    celeg::ResolvedModel overflow = resolved;
    overflow.topology.num_hidden_layers = std::numeric_limits<int>::max();
    bool rejected_overflow = false;
    try {
        (void)celeg::PerLayerInputPlan::derive(overflow);
    } catch (const std::invalid_argument&) {
        rejected_overflow = true;
    }
    CELEG_TEST_CHECK(rejected_overflow);

    celeg::RuntimeTopology token_topology;
    token_topology.hidden = 8;
    token_topology.intermediate = 16;
    token_topology.checkpoint.vocab_size = 32;
    token_topology.num_hidden_layers = 1;
    token_topology.mixer_kinds = {celeg::MixerKind::Attention};
    token_topology.feed_forward_kinds = {celeg::FeedForwardKind::Dense};
    token_topology.attention_layer_count = 1;
    token_topology.attention_layouts.resize(1);
    token_topology.checkpoint.token_policy = {1, {2}, 32};
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
