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

bool prefill_span_rejects(const celeg::AttentionSpec& attention,
                          int start,
                          std::size_t count) {
    try {
        celeg::validate_cuda_packed_prefill_span(attention, start, count);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

bool decode_position_rejects(const celeg::AttentionSpec& attention,
                             int position) {
    try {
        celeg::validate_cuda_packed_decode_position(attention, position);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void prefix_lm_lifecycle_is_explicit() {
    celeg::AttentionSpec attention;
    attention.pattern = celeg::PrefixLmPattern{4};

    CELEG_TEST_CHECK(!prefill_span_rejects(attention, 0, 4));
    CELEG_TEST_CHECK(!prefill_span_rejects(attention, 0, 6));
    CELEG_TEST_CHECK(!prefill_span_rejects(attention, 4, 2));
    CELEG_TEST_CHECK(prefill_span_rejects(attention, 0, 3));
    CELEG_TEST_CHECK(prefill_span_rejects(attention, 2, 2));

    CELEG_TEST_CHECK(!decode_position_rejects(attention, 4));
    CELEG_TEST_CHECK(!decode_position_rejects(attention, 8));
    CELEG_TEST_CHECK(decode_position_rejects(attention, 3));
    CELEG_TEST_CHECK(celeg::cuda_packed_prefix_length(attention) == 4);
}

}

int main() {
    check_pattern(celeg::FullCausalPattern{}, false);
    check_pattern(celeg::SlidingWindowPattern{128}, false);
    check_pattern(celeg::PrefixLmPattern{32}, false);
    check_pattern(celeg::BidirectionalPattern{}, true);
    check_pattern(celeg::BlockSparsePattern{16, 2, 1}, true);
    check_pattern(celeg::DynamicSparsePattern{16, 8}, true);
    prefix_lm_lifecycle_is_explicit();
    return 0;
}
