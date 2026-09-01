#include "celeg/backend/cpu/compiler.hpp"
#include "support/assertions.hpp"

#include <stdexcept>

namespace {

celeg::ResolvedModel base_model() {
    celeg::ResolvedModel model;
    model.provenance.identity = "cpu-attention-semantic-fixture";
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
    model.graph.layers.push_back(std::move(layer));
    model.weight_plan.requests.push_back(
        {celeg::TensorRole::AttentionInputNorm, 0, -1, {}});
    return model;
}

void configure_external(celeg::AttentionSpec& attention) {
    attention.key_value_source = celeg::ExternalMemorySource{3};
    attention.pattern = celeg::BidirectionalPattern{};
    attention.position = celeg::NoPositionEncodingSpec{};
    attention.bias = celeg::NoAttentionBiasSpec{};
    attention.query_norm.reset();
    attention.key_norm.reset();
    attention.output_transform = celeg::NoAttentionOutputTransformSpec{};
}

void configure_factorized(celeg::AttentionSpec& attention) {
    celeg::LatentAttentionStateSpec latent;
    latent.latent_rank = 4;
    latent.rope_head_dim = 0;
    latent.nope_head_dim = 4;
    latent.decoupled_rope = false;
    celeg::FactorizedLatentProjection projection;
    projection.query_rank = 4;
    projection.value_head_dim = 4;
    projection.query_latent_norm = celeg::NormSpec{1.0e-5f};
    projection.key_latent_norm = celeg::NormSpec{1.0e-5f};
    latent.projection = projection;
    attention.state = latent;
    attention.position = celeg::NoPositionEncodingSpec{};
}

template <typename Mutator>
bool rejects(Mutator mutate) {
    celeg::ResolvedModel model = base_model();
    auto& attention = std::get<celeg::AttentionSpec>(model.graph.layers[0].mixer);
    mutate(attention);
    try {
        (void)celeg::CpuModelCompiler{}.compile(model);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}

int main() {
    CELEG_TEST_CHECK(!rejects([](auto&) {}));
    CELEG_TEST_CHECK(!rejects([](auto& attention) {
        celeg::SigmoidAttentionGateSpec gate;
        gate.granularity = celeg::AttentionGateGranularity::HeadWise;
        attention.output_gate = gate;
    }));
    CELEG_TEST_CHECK(rejects([](auto& attention) {
        celeg::SigmoidAttentionGateSpec gate;
        gate.packed_with_query = true;
        gate.granularity = celeg::AttentionGateGranularity::HeadWise;
        attention.output_gate = gate;
    }));
    CELEG_TEST_CHECK(rejects([](auto& attention) {
        auto& ordinary = std::get<celeg::OrdinaryKvStateSpec>(attention.state);
        ordinary.storage.key = celeg::StateScalarType::INT8;
    }));
    CELEG_TEST_CHECK(rejects([](auto& attention) {
        attention.state = celeg::LatentAttentionStateSpec{16, 2, 6, true};
        celeg::MultiAxisRopeSpec multi;
        multi.base = celeg::RopePositionSpec{10000.0, 1.0, {}};
        multi.sections = {2, 1, 1};
        multi.interleaved = true;
        multi.axes = 3;
        attention.position = multi;
    }));
    CELEG_TEST_CHECK(rejects([](auto& attention) {
        attention.state = celeg::LatentAttentionStateSpec{16, 2, 6, true};
        attention.output_gate = celeg::SigmoidAttentionGateSpec{};
    }));
    CELEG_TEST_CHECK(rejects([](auto& attention) {
        attention.state = celeg::LatentAttentionStateSpec{16, 2, 6, true};
        attention.output_transform = celeg::OrthogonalizeCurrentValueSpec{1.0e-6f};
    }));

    CELEG_TEST_CHECK(!rejects([](auto& attention) {
        configure_factorized(attention);
    }));
    CELEG_TEST_CHECK(!rejects([](auto& attention) {
        configure_factorized(attention);
        celeg::SigmoidAttentionGateSpec gate;
        gate.granularity = celeg::AttentionGateGranularity::HeadWise;
        attention.output_gate = gate;
    }));
    CELEG_TEST_CHECK(rejects([](auto& attention) {
        configure_factorized(attention);
        celeg::SigmoidAttentionGateSpec gate;
        gate.packed_with_query = true;
        gate.granularity = celeg::AttentionGateGranularity::ElementWise;
        attention.output_gate = gate;
    }));

    CELEG_TEST_CHECK(!rejects([](auto& attention) {
        configure_external(attention);
    }));
    CELEG_TEST_CHECK(!rejects([](auto& attention) {
        configure_external(attention);
        celeg::SigmoidAttentionGateSpec gate;
        gate.granularity = celeg::AttentionGateGranularity::HeadWise;
        attention.output_gate = gate;
    }));
    CELEG_TEST_CHECK(rejects([](auto& attention) {
        configure_external(attention);
        attention.pattern = celeg::FullCausalPattern{};
    }));
    CELEG_TEST_CHECK(rejects([](auto& attention) {
        configure_external(attention);
        attention.position = celeg::RopePositionSpec{10000.0, 1.0, {}};
    }));
    CELEG_TEST_CHECK(rejects([](auto& attention) {
        configure_external(attention);
        attention.bias = celeg::AlibiBiasSpec{{1.0f}};
    }));
    CELEG_TEST_CHECK(rejects([](auto& attention) {
        configure_external(attention);
        attention.query_norm = celeg::NormSpec{};
    }));
    CELEG_TEST_CHECK(rejects([](auto& attention) {
        configure_external(attention);
        celeg::SigmoidAttentionGateSpec gate;
        gate.packed_with_query = true;
        gate.granularity = celeg::AttentionGateGranularity::ElementWise;
        attention.output_gate = gate;
    }));
    CELEG_TEST_CHECK(rejects([](auto& attention) {
        configure_external(attention);
        attention.kv_sharing = celeg::SharedKvPublisher{0};
    }));
    CELEG_TEST_CHECK(rejects([](auto& attention) {
        configure_external(attention);
        attention.state = celeg::LatentAttentionStateSpec{16, 2, 6, true};
    }));
    CELEG_TEST_CHECK(rejects([](auto& attention) {
        configure_external(attention);
        attention.output_transform = celeg::OrthogonalizeCurrentValueSpec{1.0e-6f};
    }));
    CELEG_TEST_CHECK(rejects([](auto& attention) {
        configure_external(attention);
        attention.key_value_source = celeg::ExternalMemorySource{-1};
    }));

    celeg::ResolvedModel external_model = base_model();
    configure_external(std::get<celeg::AttentionSpec>(
        external_model.graph.layers[0].mixer));
    const celeg::CompiledModelProgram external_program =
        celeg::CpuModelCompiler{}.compile(external_model);
    const auto& compiled_external = std::get<celeg::CompiledAttentionProgram>(
        external_program.layers[0].mixer);
    CELEG_TEST_CHECK(!compiled_external.execution.has_key_value);
    CELEG_TEST_CHECK(compiled_external.execution.kind ==
                     celeg::AttentionExecutionKind::Standard);

    return 0;
}
