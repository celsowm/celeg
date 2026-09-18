#include "celeg/backend/cpu/convolution.hpp"
#include "celeg/backend/cpu/gated_delta.hpp"
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

    /// GQA-style head sharing tiles key/query heads across value heads
    /// (value head h reads key/query head h % key_heads). Zeroing key head 1
    /// of a 2-key/4-value layer must therefore silence exactly value heads 1
    /// and 3 (whose state can never leave zero), while heads 0 and 2 stay
    /// live. The previous whole-block repeat (h / repeat) instead silenced
    /// heads 2 and 3, derailing every unequal-heads checkpoint.
    {
        constexpr int kernel1 = 1;
        constexpr int kdim = 2;
        constexpr int vdim = 2;
        constexpr int kheads = 2;
        constexpr int vheads = 4;
        constexpr int qkvw = 2 * kdim * kheads + vdim * vheads;
        constexpr int vw = vdim * vheads;
        std::vector<float> qkv1(qkvw, 0.0f);
        qkv1[0] = 0.5f;
        qkv1[1] = -0.3f;
        qkv1[2] = 0.7f;
        qkv1[3] = 0.2f;
        qkv1[kdim * kheads + 0] = 0.4f;
        qkv1[kdim * kheads + 1] = -0.6f;
        for (int i = 0; i < vw; ++i) {
            qkv1[2 * kdim * kheads + i] = 0.1f * static_cast<float>(i + 1);
        }
        std::vector<float> z1(vw, 0.3f);
        std::vector<float> b1(vheads, -0.2f);
        std::vector<float> a1(vheads, 0.1f);
        std::vector<float> conv1(static_cast<size_t>(qkvw) * kernel1, 1.0f);
        std::vector<float> dt1(vheads, 0.5f);
        std::vector<float> alog1(vheads, -0.3f);
        std::vector<float> norm1(vdim, 1.0f);
        std::vector<float> conv_state1(conv1.size(), 0.0f);
        std::vector<float> rec_state1(
            static_cast<size_t>(vheads) * kdim * vdim, 0.0f);
        std::vector<float> out1(vw, 0.0f);
        celeg::cpu_gated_delta_net_decode(
            qkv1.data(), z1.data(), b1.data(), a1.data(), conv1.data(),
            dt1.data(), alog1.data(), norm1.data(), conv_state1.data(),
            rec_state1.data(), out1.data(), kernel1, kdim, vdim, kheads,
            vheads, 1e-6f, false, false, -5.0f, false);
        for (int d = 0; d < vdim; ++d) {
            CELEG_TEST_CHECK(out1[static_cast<size_t>(1) * vdim + d] == 0.0f);
            CELEG_TEST_CHECK(out1[static_cast<size_t>(3) * vdim + d] == 0.0f);
            CELEG_TEST_CHECK(
                std::abs(out1[static_cast<size_t>(0) * vdim + d]) > 1e-6f);
            CELEG_TEST_CHECK(
                std::abs(out1[static_cast<size_t>(2) * vdim + d]) > 1e-6f);
        }
    }
}
