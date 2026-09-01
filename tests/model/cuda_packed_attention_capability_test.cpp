#include "backend/cuda/packed/attention_capability.hpp"
#include "backend/cuda/packed/layer_program.hpp"
#include "backend/cuda/runtime/prefill_policy.hpp"
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

void compiled_program_owns_required_initial_span() {
    celeg::CompiledModelProgram program;
    for (const int prefix_length : {4, 0, 7}) {
        celeg::CompiledAttentionProgram attention;
        attention.semantics.pattern = prefix_length == 0
            ? celeg::AttentionPatternSpec{celeg::FullCausalPattern{}}
            : celeg::AttentionPatternSpec{celeg::PrefixLmPattern{prefix_length}};
        celeg::CompiledLayerProgram layer;
        layer.mixer = std::move(attention);
        program.layers.push_back(std::move(layer));
    }
    CELEG_TEST_CHECK(program.required_initial_attention_span() == 7);

    celeg::CompiledModelProgram causal;
    causal.layers.push_back(program_with(celeg::FullCausalPattern{}).layers.front());
    CELEG_TEST_CHECK(causal.required_initial_attention_span() == 0);
}

void full_prompt_policy_identifies_bidirectional_attention() {
    const auto causal = program_with(celeg::FullCausalPattern{});
    CELEG_TEST_CHECK(!celeg::cuda_requires_full_prompt_prefill(causal));

    const auto prefix = program_with(celeg::PrefixLmPattern{8});
    CELEG_TEST_CHECK(!celeg::cuda_requires_full_prompt_prefill(prefix));

    const auto bidirectional = program_with(celeg::BidirectionalPattern{});
    CELEG_TEST_CHECK(celeg::cuda_requires_full_prompt_prefill(bidirectional));
}

void prefill_span_policy_preserves_atomic_prefix() {
    auto decision = celeg::plan_cuda_prefill_span(20, 0, 8, 4, 4, false);
    CELEG_TEST_CHECK(decision.count == 8);
    CELEG_TEST_CHECK(decision.requires_packed);
    CELEG_TEST_CHECK(!decision.defer);

    decision = celeg::plan_cuda_prefill_span(20, 0, 8, 16, 6, true);
    CELEG_TEST_CHECK(decision.count == 0);
    CELEG_TEST_CHECK(decision.requires_packed);
    CELEG_TEST_CHECK(decision.defer);

    decision = celeg::plan_cuda_prefill_span(12, 8, 8, 4, 4, true);
    CELEG_TEST_CHECK(decision.count == 4);
    CELEG_TEST_CHECK(!decision.requires_packed);
    CELEG_TEST_CHECK(!decision.defer);

    decision = celeg::plan_cuda_prefill_span(20, 0, 0, 6, 5, false);
    CELEG_TEST_CHECK(decision.count == 5);
    CELEG_TEST_CHECK(!decision.requires_packed);

    bool short_prefix_rejected = false;
    try {
        (void)celeg::plan_cuda_prefill_span(3, 0, 4, 8, 8, false);
    } catch (const std::invalid_argument&) {
        short_prefix_rejected = true;
    }
    CELEG_TEST_CHECK(short_prefix_rejected);
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
    compiled_program_owns_required_initial_span();
    full_prompt_policy_identifies_bidirectional_attention();
    prefill_span_policy_preserves_atomic_prefix();
    return 0;
}
