#include "celeg/backend/cpu/kernels.hpp"
#include "support/assertions.hpp"

#include <cmath>
#include <vector>

int main() {
    std::vector<float> weight(8, 1.0f);
    std::vector<float> output(8);
    const float input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    celeg::cpu_rmsnorm(input, weight.data(), output.data(), 8, 1e-5f);

    float mean_square = 0.0f;
    for (float value : output) mean_square += value * value;
    CELEG_TEST_CHECK(std::abs(mean_square / 8.0f - 1.0f) < 1e-4f);

    const float relu2_input[4] = {-2.0f, -0.5f, 2.0f, 3.0f};
    float relu2_output[4]{};
    celeg::cpu_relu2(relu2_input, relu2_output, 4);
    CELEG_TEST_CHECK(relu2_output[0] == 0.0f);
    CELEG_TEST_CHECK(relu2_output[1] == 0.0f);
    CELEG_TEST_CHECK(std::abs(relu2_output[2] - 4.0f) < 1e-6f);
    CELEG_TEST_CHECK(std::abs(relu2_output[3] - 9.0f) < 1e-6f);
}
