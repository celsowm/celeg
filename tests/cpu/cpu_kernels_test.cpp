#include "celeg/backend/cpu/kernels.hpp"
#include "support/assertions.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {
std::vector<float> run_gemv(const celeg::Q4GroupMatrix& q4,
                            const std::vector<float>& input,
                            celeg::CpuIsa isa) {
    celeg::CpuThreadPool pool(4);
    celeg::CpuLinearEngine linear(isa, pool);
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
    const auto q4 = celeg::quantize_float_groupwise_q4(
        weights.data(), rows, cols, 32);
    const std::vector<float> scalar = run_gemv(q4, input, celeg::CpuIsa::Scalar);
    float max_q4_error = 0.0f;
    for (size_t r = 0; r < rows; ++r) {
        max_q4_error = std::max(max_q4_error, std::abs(scalar[r] - expected[r]));
    }
    CELEG_TEST_CHECK(max_q4_error < 1.6f);

    const celeg::CpuCapabilities caps = celeg::detect_cpu_capabilities();
    if (caps.avx2 && caps.fma) {
        const auto avx2 = run_gemv(q4, input, celeg::CpuIsa::Avx2);
        for (size_t r = 0; r < rows; ++r) {
            CELEG_TEST_CHECK(std::abs(avx2[r] - scalar[r]) < 0.08f);
        }

        const auto activation = celeg::quantize_float_groupwise_q8(
            input.data(), cols, 32);
        const auto q8_kernel = celeg::select_q4_q8_dot_kernel(celeg::CpuIsa::Avx2);
        CELEG_TEST_CHECK(q8_kernel != nullptr);
        for (size_t r = 0; r < rows; ++r) {
            const float value = q8_kernel(
                q4.values.data() + r * q4.packed_values_per_row(),
                q4.scales_bf16.data() + r * q4.groups_per_row,
                activation.values.data(), activation.scales.data(),
                activation.sums.data(), cols, q4.group_size,
                q4.groups_per_row);
            CELEG_TEST_CHECK(std::abs(value - scalar[r]) < 0.08f);
        }
    }
    if (caps.avx_vnni) {
        const auto vnni = run_gemv(q4, input, celeg::CpuIsa::AvxVnni);
        for (size_t r = 0; r < rows; ++r) {
            CELEG_TEST_CHECK(std::abs(vnni[r] - scalar[r]) < 0.08f);
        }
    }
    if (caps.avx512f && caps.avx512_vnni) {
        const auto vnni512 = run_gemv(q4, input, celeg::CpuIsa::Avx512Vnni);
        for (size_t r = 0; r < rows; ++r) {
            CELEG_TEST_CHECK(std::abs(vnni512[r] - scalar[r]) < 0.08f);
        }
    }

    constexpr size_t batch = 5;
    std::vector<float> batch_input(batch * cols);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < cols; ++c) {
            batch_input[b * cols + c] = input[c] * (1.0f + 0.05f * static_cast<float>(b));
        }
    }
    celeg::CpuThreadPool pool(6);
    const celeg::CpuIsa best = caps.best_isa();
    celeg::CpuLinearEngine linear(best, pool);
    std::vector<float> batch_output(batch * rows);
    linear.gemm(q4, batch_input.data(), batch_output.data(), batch);
    for (size_t b = 0; b < batch; ++b) {
        std::vector<float> one(rows);
        linear.gemv(q4, batch_input.data() + b * cols, one.data());
        for (size_t r = 0; r < rows; ++r) {
            CELEG_TEST_CHECK(std::abs(one[r] - batch_output[b * rows + r]) < 1e-5f);
        }
    }
    std::vector<float> accumulated = batch_output;
    linear.gemm(q4, batch_input.data(), accumulated.data(), batch, 0.5f);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t r = 0; r < rows; ++r) {
            CELEG_TEST_CHECK(std::abs(accumulated[b * rows + r] -
                                    1.5f * batch_output[b * rows + r]) < 1e-4f);
        }
    }

    const size_t split_rows = 13;
    const auto q4_first = celeg::quantize_float_groupwise_q4(
        weights.data(), split_rows, cols, 32);
    const auto q4_second = celeg::quantize_float_groupwise_q4(
        weights.data() + split_rows * cols, rows - split_rows, cols, 32);
    celeg::CpuLinearWeight segmented;
    segmented.rows = rows;
    segmented.cols = cols;
    segmented.segments.emplace_back(q4_first);
    segmented.segments.emplace_back(q4_second);
    segmented.validate();
    std::vector<float> segmented_gemv(rows);
    linear.gemv(segmented, input.data(), segmented_gemv.data());
    std::vector<float> segmented_expected(rows);
    linear.gemv(q4_first, input.data(), segmented_expected.data());
    linear.gemv(q4_second, input.data(), segmented_expected.data() + split_rows);
    for (size_t r = 0; r < rows; ++r) {
        CELEG_TEST_CHECK(std::abs(segmented_gemv[r] - segmented_expected[r]) < 1e-5f);
    }
    std::vector<float> segmented_gemm(batch * rows);
    linear.gemm(segmented, batch_input.data(), segmented_gemm.data(), batch);
    std::vector<float> segmented_expected_gemm(batch * rows);
    std::vector<float> first_gemm(batch * split_rows);
    std::vector<float> second_gemm(batch * (rows - split_rows));
    linear.gemm(q4_first, batch_input.data(), first_gemm.data(), batch);
    linear.gemm(q4_second, batch_input.data(), second_gemm.data(), batch);
    for (size_t b = 0; b < batch; ++b) {
        std::copy(first_gemm.begin() + b * split_rows,
                  first_gemm.begin() + (b + 1) * split_rows,
                  segmented_expected_gemm.begin() + b * rows);
        std::copy(second_gemm.begin() + b * (rows - split_rows),
                  second_gemm.begin() + (b + 1) * (rows - split_rows),
                  segmented_expected_gemm.begin() + b * rows + split_rows);
    }
    for (size_t b = 0; b < batch; ++b) {
        for (size_t r = 0; r < rows; ++r) {
            CELEG_TEST_CHECK(std::abs(segmented_gemm[b * rows + r] -
                                      segmented_expected_gemm[b * rows + r]) < 1e-5f);
        }
    }

    std::vector<float> norm_weight(8, 1.0f), norm_out(8);
    const float norm_in[8] = {1,2,3,4,5,6,7,8};
    celeg::cpu_rmsnorm(norm_in, norm_weight.data(), norm_out.data(), 8, 1e-5f);
    float mean_square = 0.0f;
    for (float v : norm_out) mean_square += v * v;
    CELEG_TEST_CHECK(std::abs(mean_square / 8.0f - 1.0f) < 1e-4f);

    const float relu2_input[4] = {-2.0f, -0.5f, 2.0f, 3.0f};
    float relu2_output[4]{};
    celeg::cpu_relu2(relu2_input, relu2_output, 4);
    CELEG_TEST_CHECK(relu2_output[0] == 0.0f && relu2_output[1] == 0.0f);
    CELEG_TEST_CHECK(std::abs(relu2_output[2] - 4.0f) < 1e-6f);
    CELEG_TEST_CHECK(std::abs(relu2_output[3] - 9.0f) < 1e-6f);

    const int hidden = 4, cache = 3;
    const float bcx[12] = {1,1,1,1, 2,2,2,2, 3,3,3,3};
    const float conv_weight[12] = {1,1,1,1, 0,0,0,0, 0,0,0,0};
    float state[12]{}; float conv_out[4]{};
    celeg::cpu_conv_decode(bcx, conv_weight, state, conv_out, hidden, cache, 0);
    for (float v : conv_out) CELEG_TEST_CHECK(std::isfinite(v));
    constexpr size_t conv_rows = 5;
    std::vector<float> conv_input(conv_rows * 3 * hidden);
    for (size_t i = 0; i < conv_input.size(); ++i) {
        conv_input[i] = 0.1f * static_cast<float>(i + 1);
    }
    std::vector<float> conv_reference(conv_rows * hidden), conv_batched(conv_rows * hidden);
    std::vector<float> state_reference(cache * hidden), state_batched(cache * hidden);
    for (size_t row = 0; row < conv_rows; ++row) {
        celeg::cpu_conv_decode(conv_input.data() + row * 3 * hidden, conv_weight,
                             state_reference.data(), conv_reference.data() + row * hidden,
                             hidden, cache, static_cast<int>(row));
    }
    celeg::cpu_conv_prefill(conv_input.data(), conv_weight, state_batched.data(),
                          conv_batched.data(), conv_rows, hidden, cache, 0, pool);
    for (size_t i = 0; i < conv_batched.size(); ++i) {
        CELEG_TEST_CHECK(std::abs(conv_batched[i] - conv_reference[i]) < 1e-6f);
    }
    for (size_t i = 0; i < state_batched.size(); ++i) {
        CELEG_TEST_CHECK(std::abs(state_batched[i] - state_reference[i]) < 1e-6f);
    }

    constexpr int gdn_kernel = 3;
    constexpr int gdn_key_dim = 3;
    constexpr int gdn_value_dim = 2;
    constexpr int gdn_key_heads = 2;
    constexpr int gdn_value_heads = 4;
    constexpr int gdn_qkv_width = 2 * gdn_key_dim * gdn_key_heads +
        gdn_value_dim * gdn_value_heads;
    constexpr int gdn_value_width = gdn_value_dim * gdn_value_heads;
    std::vector<float> gdn_qkv(conv_rows * gdn_qkv_width);
    std::vector<float> gdn_z(conv_rows * gdn_value_width);
    std::vector<float> gdn_b(conv_rows * gdn_value_heads);
    std::vector<float> gdn_a(conv_rows * gdn_value_heads);
    std::vector<float> gdn_conv(static_cast<size_t>(gdn_qkv_width) * gdn_kernel);
    std::vector<float> gdn_dt(gdn_value_heads), gdn_alog(gdn_value_heads);
    std::vector<float> gdn_norm(gdn_value_dim, 1.0f);
    for (size_t i = 0; i < gdn_qkv.size(); ++i) gdn_qkv[i] = std::sin(0.11f * static_cast<float>(i));
    for (size_t i = 0; i < gdn_z.size(); ++i) gdn_z[i] = std::cos(0.07f * static_cast<float>(i));
    for (size_t i = 0; i < gdn_b.size(); ++i) gdn_b[i] = -0.3f + 0.02f * static_cast<float>(i);
    for (size_t i = 0; i < gdn_a.size(); ++i) gdn_a[i] = 0.1f * std::sin(static_cast<float>(i));
    for (size_t i = 0; i < gdn_conv.size(); ++i) gdn_conv[i] = 0.03f * std::cos(static_cast<float>(i));
    for (int i = 0; i < gdn_value_heads; ++i) {
        gdn_dt[i] = 0.8f + 0.03f * static_cast<float>(i);
        gdn_alog[i] = -0.2f + 0.04f * static_cast<float>(i);
    }
    std::vector<float> gdn_prefill_state_conv(gdn_conv.size());
    std::vector<float> gdn_decode_state_conv(gdn_conv.size());
    const size_t recurrent_size = static_cast<size_t>(gdn_value_heads) * gdn_key_dim * gdn_value_dim;
    std::vector<float> gdn_prefill_state(recurrent_size);
    std::vector<float> gdn_decode_state(recurrent_size);
    std::vector<float> gdn_prefill_output(conv_rows * gdn_value_width);
    std::vector<float> gdn_decode_output(conv_rows * gdn_value_width);
    celeg::cpu_gated_delta_net_prefill(
        gdn_qkv.data(), gdn_z.data(), gdn_b.data(), gdn_a.data(), gdn_conv.data(),
        gdn_dt.data(), gdn_alog.data(), gdn_norm.data(), gdn_prefill_state_conv.data(),
        gdn_prefill_state.data(), gdn_prefill_output.data(), conv_rows, gdn_kernel,
        gdn_key_dim, gdn_value_dim, gdn_key_heads, gdn_value_heads, 1e-6f,
        false, false, -5.0f, false);
    for (size_t row = 0; row < conv_rows; ++row) {
        celeg::cpu_gated_delta_net_decode(
            gdn_qkv.data() + row * gdn_qkv_width, gdn_z.data() + row * gdn_value_width,
            gdn_b.data() + row * gdn_value_heads, gdn_a.data() + row * gdn_value_heads,
            gdn_conv.data(), gdn_dt.data(), gdn_alog.data(), gdn_norm.data(),
            gdn_decode_state_conv.data(), gdn_decode_state.data(),
            gdn_decode_output.data() + row * gdn_value_width, gdn_kernel, gdn_key_dim,
            gdn_value_dim, gdn_key_heads, gdn_value_heads, 1e-6f,
            false, false, -5.0f, false);
    }
    for (size_t i = 0; i < gdn_decode_output.size(); ++i) {
        CELEG_TEST_CHECK(std::abs(gdn_decode_output[i] - gdn_prefill_output[i]) < 1e-5f);
    }
    for (size_t i = 0; i < gdn_decode_state_conv.size(); ++i) {
        CELEG_TEST_CHECK(std::abs(gdn_decode_state_conv[i] - gdn_prefill_state_conv[i]) < 1e-5f);
    }
    for (size_t i = 0; i < gdn_decode_state.size(); ++i) {
        CELEG_TEST_CHECK(std::abs(gdn_decode_state[i] - gdn_prefill_state[i]) < 1e-5f);
    }
    std::cout << "cpu_kernels_test: isa=" << celeg::cpu_isa_name(best)
              << " max_q4_error=" << max_q4_error << '\n';
}
