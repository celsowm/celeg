#include "celeg/model/program.hpp"
#include "celeg/model/position.hpp"
#include "support/assertions.hpp"

#include <iostream>
#include <stdexcept>

namespace {

celeg::ResolvedModel make_model() {
    celeg::ResolvedModel model;
    model.provenance.identity = "program-fingerprint-fixture";
    model.graph.hidden = 8;

    celeg::LayerSpec layer;
    celeg::AttentionSpec attention;
    attention.query_heads = 1;
    attention.key_value_heads = 1;
    attention.head_dim = 8;
    attention.position = celeg::RopePositionSpec{10000.0, 1.0, {}};
    layer.mixer = attention;
    layer.feed_forward = celeg::DenseFeedForwardSpec{
        16, celeg::ActivationKind::SwiGLU};
    model.graph.layers.push_back(layer);

    model.weight_plan.requests.push_back(
        {celeg::TensorRole::AttentionInputNorm, 0, -1, {}});
    return model;
}

void expect_fingerprint_change(const celeg::ResolvedModel& baseline,
                               const celeg::ResolvedModel& changed) {
    const auto original = celeg::build_model_program(baseline);
    const auto modified = celeg::build_model_program(changed);
    CELEG_TEST_CHECK(original.semantic_fingerprint != modified.semantic_fingerprint);
}

} // namespace

int main() {
    const celeg::ResolvedModel baseline = make_model();
    const auto compiled = celeg::build_model_program(baseline);
    CELEG_TEST_CHECK(!compiled.semantic_fingerprint.empty());
    CELEG_TEST_CHECK(compiled.semantic_fingerprint.size() <= 16);

    celeg::ResolvedModel head_geometry = baseline;
    std::get<celeg::AttentionSpec>(head_geometry.graph.layers[0].mixer).head_dim = 16;
    expect_fingerprint_change(baseline, head_geometry);

    celeg::ResolvedModel rope = baseline;
    std::get<celeg::AttentionSpec>(rope.graph.layers[0].mixer).position =
        celeg::RopePositionSpec{500000.0, 1.0, {}};
    expect_fingerprint_change(baseline, rope);

    celeg::ResolvedModel feed_forward = baseline;
    std::get<celeg::DenseFeedForwardSpec>(feed_forward.graph.layers[0].feed_forward)
        .intermediate_size = 32;
    expect_fingerprint_change(baseline, feed_forward);

    celeg::ResolvedModel residual = baseline;
    residual.graph.layers[0].residual.multiplier = 0.5f;
    expect_fingerprint_change(baseline, residual);

    celeg::ResolvedModel logits = baseline;
    logits.graph.final_logit_softcap = 30.0f;
    expect_fingerprint_change(baseline, logits);

    celeg::CompiledModelProgram ambiguous = compiled;
    ambiguous.layers[0].mamba2 = celeg::Mamba2Spec{};
    bool ambiguous_rejected = false;
    try {
        ambiguous.validate();
    } catch (const std::invalid_argument&) {
        ambiguous_rejected = true;
    }
    CELEG_TEST_CHECK(ambiguous_rejected);

    std::cout << "program_fingerprint_test: ok\n";
    return 0;
}
