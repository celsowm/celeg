#include "celeg/model/reference.hpp"
#include "support/assertions.hpp"
#include <cmath>
#include <iostream>
#include <vector>

namespace {

void near(float a, float b, float tolerance = 0.02f) {
    if (std::fabs(a - b) > tolerance) {
        std::cerr << "expected " << b << ", got " << a << '\n';
        std::abort();
    }
}

} // namespace

int main() {
    using celeg::reference::conv_decode_bf16;
    using celeg::reference::gqa_decode_strict_bf16;
    using celeg::reference::rmsnorm_bf16;
    using celeg::reference::rope_bf16_inplace;
    using celeg::reference::round_bf16;

    CELEG_TEST_CHECK(round_bf16(1.0f) == 1.0f);
    CELEG_TEST_CHECK(std::isnan(round_bf16(std::nanf(""))));

    const auto normalized = rmsnorm_bf16({1, 2, 3, 4}, {1, 1, 1, 1}, 1e-5f);
    const float inv = 1.0f / std::sqrt(7.5f + 1e-5f);
    near(normalized[0], round_bf16(round_bf16(inv)));

    std::vector<float> vector = {1, 2, 3, 4};
    rope_bf16_inplace(vector, {1, 0}, {0, 1});
    near(vector[0], 1.0f);
    near(vector[2], 3.0f);
    near(vector[1], -4.0f);
    near(vector[3], 2.0f);

    const std::vector<float> q = {1, 0, 1, 0}; // 2 query heads, dim 2
    const std::vector<float> keys = {
        1, 0, // token 0, kv head 0
        0, 1, // token 1, kv head 0
    };
    const std::vector<float> values = {
        2, 4,
        6, 8,
    };
    const auto attention = gqa_decode_strict_bf16(q, keys, values, 2, 2, 1, 2);
    CELEG_TEST_CHECK(attention.size() == 4);
    near(attention[0], attention[2]);
    near(attention[1], attention[3]);
    CELEG_TEST_CHECK(attention[0] > 2.0f && attention[0] < 6.0f);

    std::vector<float> state(6, 0.0f);
    const std::vector<float> weights = {1, 2, 3, 1, 2, 3};
    const auto conv0 = conv_decode_bf16(
        {1, 1, 1, 1, 2, 4}, weights, state, 2, 3, 0);
    near(conv0[0], 6.0f);
    near(conv0[1], 12.0f);

    std::cout << "reference_test: ok\n";
    return 0;
}
