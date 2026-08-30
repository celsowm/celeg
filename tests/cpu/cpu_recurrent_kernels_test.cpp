#include "celeg/backend/cpu/kernels.hpp"
#include "support/assertions.hpp"

#include <cmath>
#include <vector>

int main() {
    celeg::CpuThreadPool pool(4);
    constexpr int hidden = 4;
    constexpr int cache = 3;
    constexpr size_t rows = 5;
    const float conv_weight[12] = {
        1, 1, 1, 1,
        0, 0, 0, 0,
        0, 0, 0, 0,
    };
    std::vector<float> conv_input(rows * 3 * hidden);
    for (size_t i = 0; i < conv_input.size(); ++i) {
        conv_input[i] = 0.1f * static_cast<float>(i + 1);
    }
    std::vector<float> reference(rows * hidden), batched(rows * hidden);
    std::vector<float> reference_state(cache * hidden), batched_state(cache * hidden);
    for (size_t row = 0; row < rows; ++row) {
        celeg::cpu_conv_decode(
            conv_input.data() + row * 3 * hidden, conv_weight,
            reference_state.data(), reference.data() + row * hidden,
            hidden, cache, static_cast<int>(row));
    }
    celeg::cpu_conv_prefill(
        conv_input.data(), conv_weight, batched_state.data(), batched.data(),
        rows, hidden, cache, 0, pool);
    for (size_t i = 0; i < batched.size(); ++i) {
        CELEG_TEST_CHECK(std::abs(batched[i] - reference[i]) < 1e-6f);
    }
    for (size_t i = 0; i < batched_state.size(); ++i) {
        CELEG_TEST_CHECK(std::abs(batched_state[i] - reference_state[i]) < 1e-6f);
    }

    constexpr int kernel = 3;
    constexpr int key_dim = 3;
    constexpr int value_dim = 2;
    constexpr int key_heads = 2;
    constexpr int value_heads = 4;
    constexpr int qkv_width = 2 * key_dim * key_heads + value_dim * value_heads;
    constexpr int value_width = value_dim * value_heads;
    std::vector<float> qkv(rows * qkv_width);
    std::vector<float> z(rows * value_width);
    std::vector<float> b(rows * value_heads);
    std::vector<float> a(rows * value_heads);
    std::vector<float> conv(static_cast<size_t>(qkv_width) * kernel);
    std::vector<float> dt(value_heads), alog(value_heads);
    std::vector<float> norm(value_dim, 1.0f);
    for (size_t i = 0; i < qkv.size(); ++i) qkv[i] = std::sin(0.11f * static_cast<float>(i));
    for (size_t i = 0; i < z.size(); ++i) z[i] = std::cos(0.07f * static_cast<float>(i));
    for (size_t i = 0; i < b.size(); ++i) b[i] = -0.3f + 0.02f * static_cast<float>(i);
    for (size_t i = 0; i < a.size(); ++i) a[i] = 0.1f * std::sin(static_cast<float>(i));
    for (size_t i = 0; i < conv.size(); ++i) conv[i] = 0.03f * std::cos(static_cast<float>(i));
    for (int i = 0; i < value_heads; ++i) {
        dt[i] = 0.8f + 0.03f * static_cast<float>(i);
        alog[i] = -0.2f + 0.04f * static_cast<float>(i);
    }

    std::vector<float> prefill_conv_state(conv.size());
    std::vector<float> decode_conv_state(conv.size());
    const size_t recurrent_size =
        static_cast<size_t>(value_heads) * key_dim * value_dim;
    std::vector<float> prefill_state(recurrent_size);
    std::vector<float> decode_state(recurrent_size);
    std::vector<float> prefill_output(rows * value_width);
    std::vector<float> decode_output(rows * value_width);

    celeg::cpu_gated_delta_net_prefill(
        qkv.data(), z.data(), b.data(), a.data(), conv.data(), dt.data(),
        alog.data(), norm.data(), prefill_conv_state.data(), prefill_state.data(),
        prefill_output.data(), rows, kernel, key_dim, value_dim, key_heads,
        value_heads, 1e-6f, false, false, -5.0f, false);
    for (size_t row = 0; row < rows; ++row) {
        celeg::cpu_gated_delta_net_decode(
            qkv.data() + row * qkv_width,
            z.data() + row * value_width,
            b.data() + row * value_heads,
            a.data() + row * value_heads,
            conv.data(), dt.data(), alog.data(), norm.data(),
            decode_conv_state.data(), decode_state.data(),
            decode_output.data() + row * value_width,
            kernel, key_dim, value_dim, key_heads, value_heads, 1e-6f,
            false, false, -5.0f, false);
    }
    for (size_t i = 0; i < decode_output.size(); ++i) {
        CELEG_TEST_CHECK(std::abs(decode_output[i] - prefill_output[i]) < 1e-5f);
    }
    for (size_t i = 0; i < decode_conv_state.size(); ++i) {
        CELEG_TEST_CHECK(std::abs(decode_conv_state[i] - prefill_conv_state[i]) < 1e-5f);
    }
    for (size_t i = 0; i < decode_state.size(); ++i) {
        CELEG_TEST_CHECK(std::abs(decode_state[i] - prefill_state[i]) < 1e-5f);
    }
}
