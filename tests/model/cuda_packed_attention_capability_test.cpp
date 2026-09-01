#include "backend/cuda/packed/attention_capability.hpp"
#include "support/assertions.hpp"

#include <stdexcept>

namespace {

template <typename Pattern>
bool rejects(Pattern pattern) {
    celeg::AttentionSpec attention;
    attention.pattern = std::move(pattern);
    try {
        celeg::validate_cuda_packed_prefill_attention(attention);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}

int main() {
    CELEG_TEST_CHECK(!rejects(celeg::FullCausalPattern{}));
    CELEG_TEST_CHECK(!rejects(celeg::SlidingWindowPattern{128}));
    CELEG_TEST_CHECK(rejects(celeg::BidirectionalPattern{}));
    CELEG_TEST_CHECK(rejects(celeg::PrefixLmPattern{32}));
    CELEG_TEST_CHECK(rejects(celeg::BlockSparsePattern{16, 2, 1}));
    CELEG_TEST_CHECK(rejects(celeg::DynamicSparsePattern{16, 8}));
    return 0;
}
