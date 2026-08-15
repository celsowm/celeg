#include "operators/attention.hpp"
#include "celeg/backend/cpu/kernels.hpp"
#include "support/assertions.hpp"

#include <cmath>

namespace {

bool close(float actual, float expected, float tolerance = 1.0e-5f) {
    return std::abs(actual - expected) <= tolerance;
}

void test_current_value_orthogonalization() {
    celeg::AttentionSpec layout;
    layout.query_heads = 4;
    layout.key_value_heads = 2;
    layout.head_dim = 2;
    layout.output_transform = celeg::OrthogonalizeCurrentValueSpec{1.0e-6f};

    float output[] = {
        1.0f, 1.0f,
        2.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 2.0f,
    };
    const float current_value[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };

    celeg::apply_cpu_attention_output_transform(layout, output, current_value);

    CELEG_TEST_CHECK(close(output[0], 0.0f));
    CELEG_TEST_CHECK(close(output[1], 1.0f));
    CELEG_TEST_CHECK(close(output[2], 0.0f));
    CELEG_TEST_CHECK(close(output[3], 0.0f));
    CELEG_TEST_CHECK(close(output[4], 1.0f));
    CELEG_TEST_CHECK(close(output[5], 0.0f));
    CELEG_TEST_CHECK(close(output[6], 0.0f));
    CELEG_TEST_CHECK(close(output[7], 0.0f));

    for (int query_head = 0; query_head < layout.query_heads; ++query_head) {
        const int value_head = query_head / (layout.query_heads / layout.key_value_heads);
        const float* value = current_value + value_head * layout.head_dim;
        const float* transformed = output + query_head * layout.head_dim;
        const float dot = transformed[0] * value[0] + transformed[1] * value[1];
        CELEG_TEST_CHECK(close(dot, 0.0f));
    }
}

void test_adjacent_pair_rope() {
    celeg::RopePositionSpec rope;
    rope.theta = 10000.0;
    rope.rotary_fraction = 1.0;
    rope.pairing = celeg::RopePairingKind::AdjacentPairs;

    float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    celeg::cpu_rope(values, 1, 4, 1, rope);

    const float c0 = std::cos(1.0f);
    const float s0 = std::sin(1.0f);
    const float second_frequency = std::pow(10000.0f, -0.5f);
    const float c1 = std::cos(second_frequency);
    const float s1 = std::sin(second_frequency);

    CELEG_TEST_CHECK(close(values[0], 1.0f * c0 - 2.0f * s0));
    CELEG_TEST_CHECK(close(values[1], 2.0f * c0 + 1.0f * s0));
    CELEG_TEST_CHECK(close(values[2], 3.0f * c1 - 4.0f * s1));
    CELEG_TEST_CHECK(close(values[3], 4.0f * c1 + 3.0f * s1));
}

}

int main() {
    test_current_value_orthogonalization();
    test_adjacent_pair_rope();
    return 0;
}
