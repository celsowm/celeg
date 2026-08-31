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
    mutate(attention);
    try {
        celeg::validate_metal_attention_capabilities(program_with(std::move(attention)));
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}

int main() {
    CELEG_TEST_CHECK(!metal_rejects([](auto&) {}));
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
        attention.position = celeg::NoPositionEncodingSpec{};
    }));
    CELEG_TEST_CHECK(metal_rejects([](auto& attention) {
        attention.position = celeg::MultiAxisRopeSpec{};
    }));
    CELEG_TEST_CHECK(metal_rejects([](auto& attention) {
        auto& ordinary = std::get<celeg::OrdinaryKvStateSpec>(attention.state);
        ordinary.storage.key = celeg::StateScalarType::INT8;
    }));
    CELEG_TEST_CHECK(metal_rejects([](auto& attention) {
        attention.output_gate = celeg::SigmoidAttentionGateSpec{};
    }));
    CELEG_TEST_CHECK(metal_rejects([](auto& attention) {
        attention.output_transform = celeg::OrthogonalizeCurrentValueSpec{1.0e-6f};
    }));

    return 0;
}
