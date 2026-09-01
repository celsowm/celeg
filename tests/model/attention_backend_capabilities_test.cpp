#include "celeg/backend/metal/attention_capabilities.hpp"
#include "support/assertions.hpp"

#include <stdexcept>

namespace {

celeg::CompiledModelProgram program_with(celeg::AttentionSpec attention) {
    celeg::CompiledAttentionProgram compiled;
    compiled.semantics = std::move(attention);
    compiled.execution.kind = celeg::AttentionExecutionKind::Standard;
    celeg::CompiledLayerProgram layer;
    layer.mixer = std::move(compiled);
    celeg::CompiledModelProgram program;
    program.layers.push_back(std::move(layer));
    return program;
}

template <typename Mutator>
bool metal_rejects(Mutator mutate) {
    celeg::AttentionSpec attention;
    attention.query_heads = 1;
    attention.key_value_heads = 1;
    attention.head_dim = 8;
    attention.query_norm = celeg::NormSpec{};
    attention.key_norm = celeg::NormSpec{};
    attention.query_norm->granularity = celeg::NormGranularity::PerHead;
    attention.key_norm->granularity = celeg::NormGranularity::PerHead;
    mutate(attention);
    try {
        celeg::validate_metal_attention_capabilities(program_with(std::move(attention)));
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

celeg::MultiAxisRopeSpec valid_mrope() {
    celeg::MultiAxisRopeSpec multi;
    multi.base.theta = 10000.0;
    multi.base.rotary_fraction = 1.0;
    multi.base.pairing = celeg::RopePairingKind::SplitHalf;
    multi.sections = {2, 1, 1};
    multi.interleaved = true;
    multi.axes = 3;
    return multi;
}

}

int main() {
    CELEG_TEST_CHECK(!metal_rejects([](auto&) {}));
    CELEG_TEST_CHECK(!metal_rejects([](auto& attention) {
        attention.query_norm.reset();
        attention.key_norm.reset();
    }));
    CELEG_TEST_CHECK(!metal_rejects([](auto& attention) {
        attention.query_norm->granularity = celeg::NormGranularity::WholeVector;
        attention.key_norm->granularity = celeg::NormGranularity::WholeVector;
    }));
    CELEG_TEST_CHECK(!metal_rejects([](auto& attention) {
        attention.query_norm.reset();
        attention.key_norm->granularity = celeg::NormGranularity::PerHead;
    }));
    CELEG_TEST_CHECK(!metal_rejects([](auto& attention) {
        attention.query_norm->granularity = celeg::NormGranularity::WholeVector;
        attention.key_norm->granularity = celeg::NormGranularity::PerHead;
    }));
    CELEG_TEST_CHECK(!metal_rejects([](auto& attention) {
        attention.query_norm->weight_kind = celeg::NormWeightKind::None;
        attention.key_norm->weight_kind = celeg::NormWeightKind::None;
    }));
    CELEG_TEST_CHECK(!metal_rejects([](auto& attention) {
        attention.query_norm->granularity = celeg::NormGranularity::WholeVector;
        attention.query_norm->weight_kind = celeg::NormWeightKind::None;
        attention.key_norm->granularity = celeg::NormGranularity::WholeVector;
        attention.key_norm->weight_kind = celeg::NormWeightKind::None;
    }));
    CELEG_TEST_CHECK(!metal_rejects([](auto& attention) {
        attention.position = celeg::NoPositionEncodingSpec{};
    }));
    CELEG_TEST_CHECK(!metal_rejects([](auto& attention) {
        auto& rope = std::get<celeg::RopePositionSpec>(attention.position);
        rope.pairing = celeg::RopePairingKind::SplitHalf;
    }));
    CELEG_TEST_CHECK(!metal_rejects([](auto& attention) {
        auto& rope = std::get<celeg::RopePositionSpec>(attention.position);
        rope.pairing = celeg::RopePairingKind::AdjacentPairs;
    }));
    CELEG_TEST_CHECK(!metal_rejects([](auto& attention) {
        attention.position = valid_mrope();
    }));
    CELEG_TEST_CHECK(metal_rejects([](auto& attention) {
        auto multi = valid_mrope();
        multi.sections = {1, 1, 1};
        attention.position = multi;
    }));
    CELEG_TEST_CHECK(metal_rejects([](auto& attention) {
        auto multi = valid_mrope();
        multi.base.pairing = celeg::RopePairingKind::AdjacentPairs;
        attention.position = multi;
    }));
    CELEG_TEST_CHECK(metal_rejects([](auto& attention) {
        auto& rope = std::get<celeg::RopePositionSpec>(attention.position);
        rope.rotary_fraction = 0.5;
    }));
    CELEG_TEST_CHECK(metal_rejects([](auto& attention) {
        auto& rope = std::get<celeg::RopePositionSpec>(attention.position);
        rope.scaling = celeg::LinearRopeScaling{2.0};
    }));
    CELEG_TEST_CHECK(!metal_rejects([](auto& attention) {
        attention.pattern = celeg::SlidingWindowPattern{128};
    }));
    CELEG_TEST_CHECK(!metal_rejects([](auto& attention) {
        attention.bias = celeg::AlibiBiasSpec{{1.0f}};
    }));
    CELEG_TEST_CHECK(!metal_rejects([](auto& attention) {
        attention.pattern = celeg::SlidingWindowPattern{128};
        attention.bias = celeg::AlibiBiasSpec{{1.0f}};
    }));
    CELEG_TEST_CHECK(!metal_rejects([](auto& attention) {
        attention.bias = celeg::RelativePositionBiasSpec{32, 128, false};
    }));
    CELEG_TEST_CHECK(!metal_rejects([](auto& attention) {
        attention.pattern = celeg::SlidingWindowPattern{128};
        attention.bias = celeg::RelativePositionBiasSpec{32, 128, false};
    }));
    CELEG_TEST_CHECK(!metal_rejects([](auto& attention) {
        attention.bias = celeg::RelativePositionBiasSpec{32, 128, true};
    }));
    CELEG_TEST_CHECK(!metal_rejects([](auto& attention) {
        attention.output_transform = celeg::OrthogonalizeCurrentValueSpec{1.0e-6f};
    }));
    CELEG_TEST_CHECK(metal_rejects([](auto& attention) {
        attention.output_transform = celeg::OrthogonalizeCurrentValueSpec{0.0f};
    }));
    CELEG_TEST_CHECK(!metal_rejects([](auto& attention) {
        attention.output_gate = celeg::SigmoidAttentionGateSpec{};
    }));
    CELEG_TEST_CHECK(!metal_rejects([](auto& attention) {
        celeg::SigmoidAttentionGateSpec gate;
        gate.granularity = celeg::AttentionGateGranularity::ElementWise;
        attention.output_gate = gate;
    }));
    CELEG_TEST_CHECK(!metal_rejects([](auto& attention) {
        celeg::SigmoidAttentionGateSpec gate;
        gate.granularity = celeg::AttentionGateGranularity::HeadWise;
        attention.output_gate = gate;
    }));
    CELEG_TEST_CHECK(!metal_rejects([](auto& attention) {
        celeg::SigmoidAttentionGateSpec gate;
        gate.packed_with_query = true;
        gate.granularity = celeg::AttentionGateGranularity::OutputWise;
        attention.output_gate = gate;
    }));
    CELEG_TEST_CHECK(!metal_rejects([](auto& attention) {
        celeg::SigmoidAttentionGateSpec gate;
        gate.packed_with_query = true;
        gate.granularity = celeg::AttentionGateGranularity::ElementWise;
        attention.output_gate = gate;
        attention.output_transform = celeg::OrthogonalizeCurrentValueSpec{1.0e-6f};
    }));
    CELEG_TEST_CHECK(metal_rejects([](auto& attention) {
        celeg::SigmoidAttentionGateSpec gate;
        gate.packed_with_query = true;
        gate.granularity = celeg::AttentionGateGranularity::HeadWise;
        attention.output_gate = gate;
    }));
    CELEG_TEST_CHECK(metal_rejects([](auto& attention) {
        attention.pattern = celeg::SlidingWindowPattern{0};
    }));
    CELEG_TEST_CHECK(metal_rejects([](auto& attention) {
        attention.pattern = celeg::BidirectionalPattern{};
    }));
    CELEG_TEST_CHECK(metal_rejects([](auto& attention) {
        attention.pattern = celeg::PrefixLmPattern{4};
    }));
    CELEG_TEST_CHECK(metal_rejects([](auto& attention) {
        attention.pattern = celeg::BlockSparsePattern{16, 2, 1};
    }));
    CELEG_TEST_CHECK(metal_rejects([](auto& attention) {
        attention.pattern = celeg::DynamicSparsePattern{16, 4};
    }));
    CELEG_TEST_CHECK(metal_rejects([](auto& attention) {
        attention.key_value_source = celeg::ExternalMemorySource{0};
    }));
    CELEG_TEST_CHECK(metal_rejects([](auto& attention) {
        attention.kv_sharing = celeg::SharedKvPublisher{1};
    }));
    CELEG_TEST_CHECK(metal_rejects([](auto& attention) {
        attention.position = celeg::MultiAxisRopeSpec{};
    }));
    CELEG_TEST_CHECK(metal_rejects([](auto& attention) {
        auto& ordinary = std::get<celeg::OrdinaryKvStateSpec>(attention.state);
        ordinary.storage.key = celeg::StateScalarType::INT8;
    }));

    return 0;
}
