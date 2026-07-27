#include "lfm/backend/cpu/kernels.hpp"
#include "support/assertions.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {
std::vector<float> run_gemv(const lfm::Q4GroupMatrix& q4,
                            const std::vector<float>& input,
                            lfm::CpuIsa isa) {
    lfm::CpuThreadPool pool(4);
    lfm::CpuLinearEngine linear(isa, pool);
    std::vector<float> output(q4.rows);
    linear.gemv(q4, input.data(), output.data());
    return output;
}
}

int main() {
    constexpr size_t rows = 37, cols = 128;
    std::vector<float> weights(rows * cols), input(cols), expected(rows);
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = std::sin(static_cast<float>(i) * 0.03f);
    }
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::cos(static_cast<float>(i) * 0.07f) * 1.3f;
    }
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            expected[r] += weights[r * cols + c] * input[c];
        }
    }
    const auto q4 = lfm::quantize_float_groupwise_q4(
        weights.data(), rows, cols, 32);
    const std::vector<float> scalar = run_gemv(q4, input, lfm::CpuIsa::Scalar);
    float max_q4_error = 0.0f;
    for (size_t r = 0; r < rows; ++r) {
        max_q4_error = std::max(max_q4_error, std::abs(scalar[r] - expected[r]));
    }
    LFM_TEST_CHECK(max_q4_error < 1.6f);

    const lfm::CpuCapabilities caps = lfm::detect_cpu_capabilities();
    if (caps.avx2 && caps.fma) {
        const auto avx2 = run_gemv(q4, input, lfm::CpuIsa::Avx2);
        for (size_t r = 0; r < rows; ++r) {
            LFM_TEST_CHECK(std::abs(avx2[r] - scalar[r]) < 2e-4f);
        }
    }
    if (caps.avx_vnni) {
        const auto vnni = run_gemv(q4, input, lfm::CpuIsa::AvxVnni);
        for (size_t r = 0; r < rows; ++r) {
            LFM_TEST_CHECK(std::abs(vnni[r] - scalar[r]) < 0.08f);
        }
    }
    if (caps.avx512f && caps.avx512_vnni) {
        const auto vnni512 = run_gemv(q4, input, lfm::CpuIsa::Avx512Vnni);
        for (size_t r = 0; r < rows; ++r) {
            LFM_TEST_CHECK(std::abs(vnni512[r] - scalar[r]) < 0.08f);
        }
    }

    // Validate the M>1 path and flattened two-dimensional scheduler.
    constexpr size_t batch = 5;
    std::vector<float> batch_input(batch * cols);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < cols; ++c) {
            batch_input[b * cols + c] = input[c] * (1.0f + 0.05f * static_cast<float>(b));
        }
    }
    lfm::CpuThreadPool pool(6);
    const lfm::CpuIsa best = caps.best_isa();
    lfm::CpuLinearEngine linear(best, pool);
    std::vector<float> batch_output(batch * rows);
    linear.gemm(q4, batch_input.data(), batch_output.data(), batch);
    for (size_t b = 0; b < batch; ++b) {
        std::vector<float> one(rows);
        linear.gemv(q4, batch_input.data() + b * cols, one.data());
        for (size_t r = 0; r < rows; ++r) {
            LFM_TEST_CHECK(std::abs(one[r] - batch_output[b * rows + r]) < 1e-5f);
        }
    }

    std::vector<float> norm_weight(8, 1.0f), norm_out(8);
    const float norm_in[8] = {1,2,3,4,5,6,7,8};
    lfm::cpu_rmsnorm(norm_in, norm_weight.data(), norm_out.data(), 8, 1e-5f);
    float mean_square = 0.0f;
    for (float v : norm_out) mean_square += v * v;
    LFM_TEST_CHECK(std::abs(mean_square / 8.0f - 1.0f) < 1e-4f);

    const int hidden = 4, cache = 3;
    const float bcx[12] = {1,1,1,1, 2,2,2,2, 3,3,3,3};
    const float conv_weight[12] = {1,0,0, 1,0,0, 1,0,0, 1,0,0};
    float state[12]{}; float conv_out[4]{};
    lfm::cpu_conv_decode(bcx, conv_weight, state, conv_out, hidden, cache, 0);
    for (float v : conv_out) LFM_TEST_CHECK(std::isfinite(v));
    std::cout << "cpu_kernels_test: isa=" << lfm::cpu_isa_name(best)
              << " max_q4_error=" << max_q4_error << '\n';
}
