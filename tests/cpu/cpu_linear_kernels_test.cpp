#include "celeg/backend/cpu/kernels.hpp"
#include "celeg/quantization/scalars.hpp"
#include "support/assertions.hpp"

#include <cmath>
#include <vector>

namespace {

std::vector<float> run_gemv(const celeg::Q4GroupMatrix& weight,
                            const std::vector<float>& input,
                            celeg::CpuIsa isa) {
    celeg::CpuThreadPool pool(4);
    celeg::CpuLinearEngine engine(isa, pool);
    std::vector<float> output(weight.rows);
    engine.gemv(weight, input.data(), output.data());
    return output;
}

void check_close(const std::vector<float>& actual,
                 const std::vector<float>& expected,
                 float tolerance) {
    CELEG_TEST_CHECK(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        CELEG_TEST_CHECK(std::abs(actual[i] - expected[i]) < tolerance);
    }
}

}

int main() {
    constexpr size_t rows = 37;
    constexpr size_t cols = 128;
    std::vector<float> weights(rows * cols);
    std::vector<float> input(cols);
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = std::sin(static_cast<float>(i) * 0.03f);
    }
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::cos(static_cast<float>(i) * 0.07f) * 1.3f;
    }

    const auto q4 = celeg::quantize_float_groupwise_q4(
        weights.data(), rows, cols, 32);
    const std::vector<float> scalar = run_gemv(q4, input, celeg::CpuIsa::Scalar);

    const celeg::CpuCapabilities caps = celeg::detect_cpu_capabilities();
    if (caps.supports(celeg::CpuIsa::Avx2) && caps.fma &&
        celeg::cpu_isa_compiled(celeg::CpuIsa::Avx2)) {
        const auto avx2 = run_gemv(q4, input, celeg::CpuIsa::Avx2);
        check_close(avx2, scalar, 0.08f);

        const auto activation = celeg::quantize_float_groupwise_q8(
            input.data(), cols, 32);
        const auto q8_kernel = celeg::select_q4_q8_dot_kernel(celeg::CpuIsa::Avx2);
        CELEG_TEST_CHECK(q8_kernel != nullptr);
        for (size_t row = 0; row < rows; ++row) {
            const float value = q8_kernel(
                q4.values.data() + row * q4.packed_values_per_row(),
                q4.scales_bf16.data() + row * q4.groups_per_row,
                activation.values.data(), activation.scales.data(),
                activation.sums.data(), cols, q4.group_size,
                q4.groups_per_row);
            CELEG_TEST_CHECK(std::abs(value - scalar[row]) < 0.08f);
        }
    }
    if (caps.supports(celeg::CpuIsa::AvxVnni) &&
        celeg::cpu_isa_compiled(celeg::CpuIsa::AvxVnni)) {
        check_close(run_gemv(q4, input, celeg::CpuIsa::AvxVnni), scalar, 0.08f);
    }
    if (caps.supports(celeg::CpuIsa::Avx512Vnni) &&
        celeg::cpu_isa_compiled(celeg::CpuIsa::Avx512Vnni)) {
        check_close(run_gemv(q4, input, celeg::CpuIsa::Avx512Vnni), scalar, 0.08f);
    }

    constexpr size_t batch = 5;
    std::vector<float> batch_input(batch * cols);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < cols; ++c) {
            batch_input[b * cols + c] = input[c] *
                (1.0f + 0.05f * static_cast<float>(b));
        }
    }

    celeg::CpuThreadPool pool(6);
    celeg::CpuLinearEngine best_engine(caps.best_isa(), pool);
    std::vector<float> batch_output(batch * rows);
    best_engine.gemm(q4, batch_input.data(), batch_output.data(), batch);
    for (size_t b = 0; b < batch; ++b) {
        std::vector<float> one(rows);
        best_engine.gemv(q4, batch_input.data() + b * cols, one.data());
        for (size_t row = 0; row < rows; ++row) {
            CELEG_TEST_CHECK(std::abs(one[row] - batch_output[b * rows + row]) < 1e-5f);
        }
    }

    std::vector<float> accumulated = batch_output;
    best_engine.gemm(q4, batch_input.data(), accumulated.data(), batch, 0.5f);
    for (size_t i = 0; i < accumulated.size(); ++i) {
        CELEG_TEST_CHECK(std::abs(accumulated[i] - 1.5f * batch_output[i]) < 1e-4f);
    }

    celeg::CpuBf16Matrix bf16;
    bf16.rows = static_cast<uint32_t>(rows);
    bf16.cols = static_cast<uint32_t>(cols);
    bf16.values->resize(rows * cols);
    for (size_t i = 0; i < bf16.values->size(); ++i) {
        (*bf16.values)[i] = celeg::float_to_bf16_bits(weights[i]);
    }
    const celeg::CpuLinearWeight bf16_weight = celeg::CpuLinearWeight::from_bf16(
        std::move(bf16));
    std::vector<float> bf16_output(rows);
    best_engine.gemv(bf16_weight, input.data(), bf16_output.data());
    for (size_t row = 0; row < rows; ++row) {
        float reference = 0.0f;
        for (size_t col = 0; col < cols; ++col) {
            reference += celeg::bf16_bits_to_float(
                celeg::float_to_bf16_bits(weights[row * cols + col])) * input[col];
        }
        CELEG_TEST_CHECK(std::abs(bf16_output[row] - reference) < 1e-4f);
    }

    std::vector<float> bf16_batch(batch * rows);
    best_engine.gemm(bf16_weight, batch_input.data(), bf16_batch.data(), batch);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t row = 0; row < rows; ++row) {
            float reference = 0.0f;
            for (size_t col = 0; col < cols; ++col) {
                reference += celeg::bf16_bits_to_float(
                    celeg::float_to_bf16_bits(weights[row * cols + col])) *
                    batch_input[b * cols + col];
            }
            CELEG_TEST_CHECK(std::abs(bf16_batch[b * rows + row] - reference) < 1e-4f);
        }
    }
}
