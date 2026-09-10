#include "celeg/attention/pattern_semantics.hpp"
#include "celeg/backend/cpu/paged_kv.hpp"
#include "support/assertions.hpp"

namespace {

namespace semantics = celeg::attention_semantics;

void causal_and_sliding() {
    CELEG_TEST_CHECK(semantics::causal_visible(8, 8));
    CELEG_TEST_CHECK(semantics::causal_visible(8, 0));
    CELEG_TEST_CHECK(!semantics::causal_visible(8, 9));
    CELEG_TEST_CHECK(semantics::sliding_window_first_candidate(8, 4) == 5);
    CELEG_TEST_CHECK(semantics::sliding_window_visible(8, 5, 4));
    CELEG_TEST_CHECK(!semantics::sliding_window_visible(8, 4, 4));
    CELEG_TEST_CHECK(!semantics::sliding_window_visible(8, 9, 4));
}

void prefix_semantics() {
    CELEG_TEST_CHECK(semantics::prefix_lm_visible(2, 5, 6));
    CELEG_TEST_CHECK(!semantics::prefix_lm_visible(2, 6, 6));
    CELEG_TEST_CHECK(semantics::prefix_lm_visible(8, 6, 6));
    CELEG_TEST_CHECK(!semantics::prefix_lm_visible(8, 9, 6));
    CELEG_TEST_CHECK(semantics::prefix_lm_may_read_future(2, 10, 6));
    CELEG_TEST_CHECK(!semantics::prefix_lm_may_read_future(8, 10, 6));
}

void sparse_semantics() {
    CELEG_TEST_CHECK(semantics::block_sparse_visible(63, 0, 16, 2, 1));
    CELEG_TEST_CHECK(semantics::block_sparse_visible(63, 32, 16, 2, 1));
    CELEG_TEST_CHECK(!semantics::block_sparse_visible(63, 16, 16, 2, 1));
    CELEG_TEST_CHECK(!semantics::block_sparse_visible(63, 64, 16, 2, 1));
}

void cpu_matches_canonical() {
    const celeg::CpuAttentionPattern causal = celeg::CpuAttentionPattern::lower(
        celeg::FullCausalPattern{});
    CELEG_TEST_CHECK(causal.allows(8, 7) ==
        semantics::causal_visible(8, 7));
    CELEG_TEST_CHECK(causal.allows(8, 9) ==
        semantics::causal_visible(8, 9));

    const celeg::CpuAttentionPattern sliding = celeg::CpuAttentionPattern::lower(
        celeg::SlidingWindowPattern{4});
    for (int key = 0; key < 12; ++key) {
        CELEG_TEST_CHECK(sliding.allows(8, key) ==
            semantics::sliding_window_visible(8, key, 4));
    }
    CELEG_TEST_CHECK(sliding.first_candidate(8) ==
        semantics::sliding_window_first_candidate(8, 4));

    const celeg::CpuAttentionPattern block = celeg::CpuAttentionPattern::lower(
        celeg::BlockSparsePattern{16, 2, 1});
    for (int key = 0; key < 80; ++key) {
        CELEG_TEST_CHECK(block.allows(63, key) ==
            semantics::block_sparse_visible(63, key, 16, 2, 1));
    }

    const celeg::CpuAttentionPattern dynamic = celeg::CpuAttentionPattern::lower(
        celeg::DynamicSparsePattern{16, 2});
    CELEG_TEST_CHECK(!dynamic.may_read_future(63, 80));
    CELEG_TEST_CHECK(dynamic.first_candidate(63) == 0);
}

}

int main() {
    causal_and_sliding();
    prefix_semantics();
    sparse_semantics();
    cpu_matches_canonical();
    return 0;
}
