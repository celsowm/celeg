#include "backend/cuda/attention_semantic_capability.hpp"
#include "support/assertions.hpp"

#include <stdexcept>

namespace {

template <typename Mutator>
bool rejected(Mutator mutate) {
    celeg::CompiledAttentionProgram compiled;
    compiled.semantics.query_heads = 1;
    compiled.semantics.key_value_heads = 1;
    compiled.semantics.head_dim = 8;
    compiled.execution.kind = celeg::AttentionExecutionKind::Standard;
    mutate(compiled);
    try {
        celeg::validate_cuda_attention_semantics(compiled);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}

int main() {
    CELEG_TEST_CHECK(rejected([](auto& compiled) {
        compiled.semantics.pattern = celeg::PrefixLmPattern{0};
    }));

    CELEG_TEST_CHECK(rejected([](auto& compiled) {
        compiled.semantics.pattern = celeg::BlockSparsePattern{0, 1, 0};
    }));

    CELEG_TEST_CHECK(rejected([](auto& compiled) {
        compiled.semantics.pattern = celeg::DynamicSparsePattern{16, 33};
    }));

    CELEG_TEST_CHECK(rejected([](auto& compiled) {
        compiled.semantics.pattern = celeg::BidirectionalPattern{};
        compiled.semantics.bias = celeg::AlibiBiasSpec{{1.0f}};
    }));

    CELEG_TEST_CHECK(rejected([](auto& compiled) {
        compiled.semantics.pattern = celeg::PrefixLmPattern{4};
        compiled.semantics.state = celeg::LatentAttentionStateSpec{32, 16, 16, true};
        compiled.execution.kind = celeg::AttentionExecutionKind::Latent;
    }));

    CELEG_TEST_CHECK(rejected([](auto& compiled) {
        compiled.semantics.state = celeg::LatentAttentionStateSpec{513, 16, 16, true};
        compiled.execution.kind = celeg::AttentionExecutionKind::Latent;
    }));

    CELEG_TEST_CHECK(rejected([](auto& compiled) {
        compiled.semantics.state = celeg::LatentAttentionStateSpec{32, 16, 16, true};
        compiled.semantics.output_gate = celeg::SigmoidAttentionGateSpec{};
        compiled.execution.kind = celeg::AttentionExecutionKind::Latent;
    }));

    CELEG_TEST_CHECK(rejected([](auto& compiled) {
        compiled.semantics.state = celeg::LatentAttentionStateSpec{32, 16, 16, true};
        compiled.semantics.position = celeg::MultiAxisRopeSpec{};
        compiled.execution.kind = celeg::AttentionExecutionKind::Latent;
    }));

    CELEG_TEST_CHECK(rejected([](auto& compiled) {
        compiled.semantics.position = celeg::RopePositionSpec{
            10000.0, 1.0, celeg::LongRopeScaling{32, {}, {}}};
    }));

    celeg::CompiledAttentionProgram supported;
    supported.semantics.query_heads = 1;
    supported.semantics.key_value_heads = 1;
    supported.semantics.head_dim = 8;
    supported.semantics.pattern = celeg::PrefixLmPattern{4};
    supported.execution.kind = celeg::AttentionExecutionKind::Standard;
    celeg::validate_cuda_attention_semantics(supported);

    supported.semantics.pattern = celeg::FullCausalPattern{};
    supported.semantics.state = celeg::LatentAttentionStateSpec{32, 16, 16, true};
    supported.semantics.output_gate = celeg::SigmoidAttentionGateSpec{};
    supported.execution.kind = celeg::AttentionExecutionKind::FactorizedLatent;
    celeg::validate_cuda_attention_semantics(supported);

    return 0;
}
