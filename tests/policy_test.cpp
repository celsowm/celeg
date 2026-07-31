#include "celeg/runtime/concurrency/policy_types.hpp"
#include "support/assertions.hpp"
#include <iostream>

int main() {
    using celeg::AttentionMode;
    using celeg::select_segmented_attention;

    CELEG_TEST_CHECK(!select_segmented_attention(AttentionMode::Single, 0, 4096));
    CELEG_TEST_CHECK(!select_segmented_attention(AttentionMode::Single, 8192, 4096));
    CELEG_TEST_CHECK(select_segmented_attention(AttentionMode::Segmented, 0, 4096));
    CELEG_TEST_CHECK(select_segmented_attention(AttentionMode::Segmented, 8192, 4096));
    CELEG_TEST_CHECK(!select_segmented_attention(AttentionMode::Auto, 4095, 4096));
    CELEG_TEST_CHECK(select_segmented_attention(AttentionMode::Auto, 4096, 4096));
    CELEG_TEST_CHECK(select_segmented_attention(AttentionMode::Auto, 8192, 4096));

    std::cout << "policy_test: ok\n";
    return 0;
}
