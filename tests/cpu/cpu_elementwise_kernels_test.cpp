#include "backend/cpu/kernels/math.hpp"
#include "celeg/backend/cpu/elementwise.hpp"
#include "celeg/backend/cpu/kernel_backend.hpp"
#include "celeg/backend/cpu/normalization.hpp"
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

    const celeg::CpuMathEngine scalar_math(
        celeg::cpu_kernel_backend(celeg::CpuIsa::Scalar));
    CELEG_TEST_CHECK(scalar_math.isa() == celeg::CpuIsa::Scalar);
    std::vector<float> scalar_output(8);
    scalar_math.rmsnorm(input, weight.data(), scalar_output.data(), 8, 1e-5f);
    for (size_t i = 0; i < scalar_output.size(); ++i) {
        CELEG_TEST_CHECK(std::abs(scalar_output[i] - output[i]) < 1e-6f);
    }

    const float gate_up[8] = {-2.0f, -0.5f, 2.0f, 3.0f,
                              0.5f, 1.5f, -2.0f, 0.25f};
    float reference_swiglu[4]{};
    float scalar_swiglu[4]{};
    celeg::cpu_swiglu(gate_up, reference_swiglu, 4);
    scalar_math.swiglu(gate_up, scalar_swiglu, 4);
    for (size_t i = 0; i < 4; ++i) {
        CELEG_TEST_CHECK(std::abs(scalar_swiglu[i] - reference_swiglu[i]) < 1e-6f);
    }

    const float relu2_input[4] = {-2.0f, -0.5f, 2.0f, 3.0f};
    float relu2_output[4]{};
    celeg::cpu_relu2(relu2_input, relu2_output, 4);
    CELEG_TEST_CHECK(relu2_output[0] == 0.0f);
    CELEG_TEST_CHECK(relu2_output[1] == 0.0f);
    CELEG_TEST_CHECK(std::abs(relu2_output[2] - 4.0f) < 1e-6f);
    CELEG_TEST_CHECK(std::abs(relu2_output[3] - 9.0f) < 1e-6f);
}
