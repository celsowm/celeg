#include "backend/cuda/packed/attention_capability.hpp"
#include "backend/cuda/packed/layer_program.hpp"
#include "support/assertions.hpp"

#include <stdexcept>
#include <utility>

namespace {

template <typename Pattern>
celeg::CompiledModelProgram program_with(Pattern pattern) {
    celeg::CompiledAttentionProgram attention;
    attention.semantics.pattern = std::move(pattern);
    celeg::CompiledLayerProgram layer;
    layer.mixer = std::move(attention);
    celeg::CompiledModelProgram program;
    program.layers.push_back(std::move(layer));
    return program;
}

template <typename Pattern>
bool policy_rejects(Pattern pattern) {
    celeg::AttentionSpec attention;
    attention.pattern = std::move(pattern);
    try {
        celeg::validate_cuda_packed_attention(attention);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

template <typename Pattern>
bool program_rejects(Pattern pattern) {
    try {
        (void)celeg::PackedLayerProgram::compile(program_with(std::move(pattern)));
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

template <typename Pattern>
void check_pattern(Pattern pattern, bool expected_rejection) {
    CELEG_TEST_CHECK(policy_rejects(pattern) == expected_rejection);
    CELEG_TEST_CHECK(program_rejects(std::move(pattern)) == expected_rejection);
}

}

int main() {
    check_pattern(celeg::FullCausalPattern{}, false);
    check_pattern(celeg::SlidingWindowPattern{128}, false);
    check_pattern(celeg::BidirectionalPattern{}, true);
    check_pattern(celeg::PrefixLmPattern{32}, true);
    check_pattern(celeg::BlockSparsePattern{16, 2, 1}, true);
    check_pattern(celeg::DynamicSparsePattern{16, 8}, true);
    return 0;
}
