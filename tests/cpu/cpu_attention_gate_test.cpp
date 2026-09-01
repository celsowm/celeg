#include "backend/cpu/operators/attention.hpp"
#include "support/assertions.hpp"

#include <cmath>

int main() {
    {
        float output[] = {2.0f, 4.0f, 6.0f, 8.0f};
        const float gate[] = {0.0f, std::log(3.0f)};
        celeg::apply_cpu_attention_output_gate(
            output, gate, 4,
            celeg::AttentionGateGranularity::HeadWise, 2, 2);
        CELEG_TEST_CHECK(std::abs(output[0] - 1.0f) < 1.0e-6f);
        CELEG_TEST_CHECK(std::abs(output[1] - 2.0f) < 1.0e-6f);
        CELEG_TEST_CHECK(std::abs(output[2] - 4.5f) < 1.0e-6f);
        CELEG_TEST_CHECK(std::abs(output[3] - 6.0f) < 1.0e-6f);
    }
    {
        float output[] = {2.0f, 4.0f};
        const float gate[] = {0.0f, std::log(3.0f)};
        celeg::apply_cpu_attention_output_gate(
            output, gate, 2,
            celeg::AttentionGateGranularity::ElementWise, 1, 2);
        CELEG_TEST_CHECK(std::abs(output[0] - 1.0f) < 1.0e-6f);
        CELEG_TEST_CHECK(std::abs(output[1] - 3.0f) < 1.0e-6f);
    }
    return 0;
}
