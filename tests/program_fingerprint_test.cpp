#include "celeg/model/program.hpp"
#include "celeg/model/position.hpp"
#include "support/assertions.hpp"

#include <iostream>
#include <utility>

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
    CELEG_TEST_CHECK(compiled.layers[0].mixer_kind() ==
                     celeg::CompiledMixer::Attention);
    CELEG_TEST_CHECK(compiled.layers[0].attention() != nullptr);
    CELEG_TEST_CHECK(compiled.layers[0].state_layout() != nullptr);
    CELEG_TEST_CHECK(compiled.layers[0].mamba2() == nullptr);
    CELEG_TEST_CHECK(compiled.layers[0].feed_forward_kind() ==
                     celeg::CompiledFeedForward::Dense);
    CELEG_TEST_CHECK(compiled.layers[0].execute_feed_forward);
    CELEG_TEST_CHECK(compiled.layers[0].feed_forward_intermediate == 16);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::CompiledDenseFeedForwardProgram>(
        compiled.layers[0].feed_forward.storage()));

    celeg::CompiledModelProgram copied = compiled;
    CELEG_TEST_CHECK(copied.layers[0].attention() != nullptr);
    CELEG_TEST_CHECK(copied.layers[0].attention() != compiled.layers[0].attention());
    copied.layers[0].attention()->head_dim = 16;
    CELEG_TEST_CHECK(compiled.layers[0].attention()->head_dim == 8);
    CELEG_TEST_CHECK(copied.layers[0].attention()->head_dim == 16);
    copied.layers[0].feed_forward_intermediate = 32;
    CELEG_TEST_CHECK(compiled.layers[0].feed_forward_intermediate == 16);
    CELEG_TEST_CHECK(copied.layers[0].feed_forward_intermediate == 32);

    celeg::CompiledModelProgram moved = std::move(copied);
    CELEG_TEST_CHECK(moved.layers[0].attention() != nullptr);
    CELEG_TEST_CHECK(moved.layers[0].attention()->head_dim == 16);
    CELEG_TEST_CHECK(moved.layers[0].feed_forward_intermediate == 32);

    celeg::ResolvedModel no_feed_forward = baseline;
    no_feed_forward.graph.layers[0].execute_feed_forward = false;
    const auto no_feed_forward_program =
        celeg::build_model_program(no_feed_forward);
    CELEG_TEST_CHECK(no_feed_forward_program.layers[0].feed_forward_kind() ==
                     celeg::CompiledFeedForward::None);
    CELEG_TEST_CHECK(!no_feed_forward_program.layers[0].execute_feed_forward);
    CELEG_TEST_CHECK(!no_feed_forward_program.layers[0].moe.has_value());
    CELEG_TEST_CHECK(std::holds_alternative<std::monostate>(
        no_feed_forward_program.layers[0].feed_forward.storage()));

    celeg::CompiledLayerProgram mlp_only_layer;
    mlp_only_layer.mixer = celeg::MlpBlockSpec{
        24, celeg::ActivationKind::GeluTanh};
    mlp_only_layer.feed_forward = celeg::CompiledFeedForward::None;
    CELEG_TEST_CHECK(!mlp_only_layer.execute_feed_forward);
    CELEG_TEST_CHECK(mlp_only_layer.feed_forward_kind() ==
                     celeg::CompiledFeedForward::None);
    CELEG_TEST_CHECK(mlp_only_layer.feed_forward_intermediate == 24);
    CELEG_TEST_CHECK(mlp_only_layer.feed_forward_activation ==
                     celeg::ActivationKind::GeluTanh);
    CELEG_TEST_CHECK(std::holds_alternative<std::monostate>(
        mlp_only_layer.feed_forward.storage()));

    celeg::CompiledLayerProgram copied_mlp_only = mlp_only_layer;
    CELEG_TEST_CHECK(copied_mlp_only.feed_forward_intermediate == 24);
    CELEG_TEST_CHECK(copied_mlp_only.feed_forward_activation ==
                     celeg::ActivationKind::GeluTanh);

    celeg::ResolvedModel moe_model = baseline;
    moe_model.graph.layers[0].feed_forward = celeg::MixtureOfExpertsSpec{
        16, 4, 2, true, false, 1.0f};
    const auto moe_program = celeg::build_model_program(moe_model);
    CELEG_TEST_CHECK(moe_program.layers[0].feed_forward_kind() ==
                     celeg::CompiledFeedForward::MixtureOfExperts);
    CELEG_TEST_CHECK(moe_program.layers[0].moe.has_value());
    CELEG_TEST_CHECK(moe_program.layers[0].moe->router.expert_count == 4);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::MoeLayerProgram>(
        moe_program.layers[0].feed_forward.storage()));

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

    std::cout << "program_fingerprint_test: ok\n";
    return 0;
}