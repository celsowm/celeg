#include "lfm/runtime/concurrency/policy_types.hpp"

#include <cassert>
#include <iostream>

int main() {
    using lfm::AttentionMode;
    using lfm::select_segmented_attention;

    assert(!select_segmented_attention(AttentionMode::Single, 0, 4096));
    assert(!select_segmented_attention(AttentionMode::Single, 8192, 4096));
    assert(select_segmented_attention(AttentionMode::Segmented, 0, 4096));
    assert(select_segmented_attention(AttentionMode::Segmented, 8192, 4096));
    assert(!select_segmented_attention(AttentionMode::Auto, 4095, 4096));
    assert(select_segmented_attention(AttentionMode::Auto, 4096, 4096));
    assert(select_segmented_attention(AttentionMode::Auto, 8192, 4096));

    std::cout << "policy_test: ok\n";
    return 0;
}
