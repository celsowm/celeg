#include "lfm/backend/cpu/gguf.hpp"
#include "lfm/backend/cpu/kernels.hpp"
#include "lfm/backend/cpu/model.hpp"
#include "support/assertions.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

#pragma pack(push, 1)
struct BlockQ4K {
    uint16_t d = 0x3c00;
    uint16_t dmin = 0;
    uint8_t scales[12]{};
    uint8_t qs[128]{};
};

struct BlockQ6K {
    uint8_t ql[128]{};
    uint8_t qh[64]{};
    int8_t scales[16]{};
    uint16_t d = 0x3c00;
};
#pragma pack(pop)

static_assert(sizeof(BlockQ4K) == 144);
static_assert(sizeof(BlockQ6K) == 210);

BlockQ4K unit_q4k() {
    BlockQ4K block;
    for (int i = 0; i < 4; ++i) block.scales[i] = 1;
    for (int i = 8; i < 12; ++i) block.scales[i] = 1;
    for (uint8_t& value : block.qs) value = 0x11;
    return block;
}

BlockQ6K unit_q6k() {
    BlockQ6K block;
    for (uint8_t& value : block.ql) value = 0x11;
    for (uint8_t& value : block.qh) value = 0xaa;
    for (int8_t& value : block.scales) value = 1;
    return block;
}

template <typename T>
const std::byte* bytes(const std::vector<T>& values) {
    return reinterpret_cast<const std::byte*>(values.data());
}

} // namespace

int main() {
    std::vector<BlockQ4K> q4_rows(2, unit_q4k());
    std::vector<BlockQ6K> q6_rows(2, unit_q6k());
    for (int8_t& value : q6_rows[1].scales) value = 2;

    lfm::CpuGgufMatrix q4{
        lfm::GgmlType::Q4_K, 2, 256, bytes(q4_rows),
        q4_rows.size() * sizeof(BlockQ4K)};
    lfm::CpuGgufMatrix q6{
        lfm::GgmlType::Q6_K, 2, 256, bytes(q6_rows),
        q6_rows.size() * sizeof(BlockQ6K)};
    q4.validate();
    q6.validate();

    std::vector<float> row(256);
    lfm::cpu_gguf_dequantize_row(q4, 1, row.data());
    for (float value : row) LFM_TEST_CHECK(std::abs(value - 1.0f) < 1e-6f);
    lfm::cpu_gguf_dequantize_row(q6, 0, row.data());
    for (float value : row) LFM_TEST_CHECK(std::abs(value - 1.0f) < 1e-6f);
    lfm::cpu_gguf_dequantize_row(q6, 1, row.data());
    for (float value : row) LFM_TEST_CHECK(std::abs(value - 2.0f) < 1e-6f);

    std::vector<float> input(256);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.07f);
    }
    const auto activation =
        lfm::cpu_quantize_q8k(input.data(), input.size(), lfm::CpuIsa::Scalar);
    const lfm::CpuIsa isa = lfm::detect_cpu_capabilities().best_isa();
    const auto optimized_activation =
        lfm::cpu_quantize_q8k(input.data(), input.size(), isa);
    LFM_TEST_CHECK(optimized_activation.size() == activation.size());
    for (size_t block = 0; block < activation.size(); ++block) {
        LFM_TEST_CHECK(optimized_activation[block].d == activation[block].d);
        LFM_TEST_CHECK(optimized_activation[block].qs == activation[block].qs);
        LFM_TEST_CHECK(
            optimized_activation[block].bsums == activation[block].bsums);
    }
    for (const auto* matrix : {&q4, &q6}) {
        const float scalar = lfm::cpu_gguf_dot_scalar(
            matrix->data, matrix->type, activation.data(), matrix->cols);
        const auto selected =
            lfm::select_cpu_gguf_dot_kernel(isa);
        const float optimized = selected(
            matrix->data, matrix->type, activation.data(), matrix->cols);
        LFM_TEST_CHECK(std::abs(optimized - scalar) < 1e-4f);
        if (const auto dot4 = lfm::select_cpu_gguf_dot4_kernel(isa)) {
            std::array<lfm::CpuQ8KBlock, 4> batch{};
            for (size_t lane = 0; lane < batch.size(); ++lane) batch[lane] = activation[0];
            std::array<float, 4> values{};
            dot4(matrix->data, matrix->type, batch.data(), matrix->cols, values.data());
            for (float value : values) LFM_TEST_CHECK(std::abs(value - scalar) < 1e-4f);
        }
    }

    BlockQ4K known_q4;
    known_q4.d = 0x3800;
    known_q4.dmin = 0x3400;
    for (size_t i = 0; i < std::size(known_q4.scales); ++i) {
        known_q4.scales[i] = static_cast<uint8_t>(17 + i * 19);
    }
    for (size_t i = 0; i < std::size(known_q4.qs); ++i) {
        known_q4.qs[i] = static_cast<uint8_t>(i * 37 + 11);
    }
    BlockQ6K known_q6;
    known_q6.d = 0x3800;
    for (size_t i = 0; i < std::size(known_q6.ql); ++i) {
        known_q6.ql[i] = static_cast<uint8_t>(i * 29 + 7);
    }
    for (size_t i = 0; i < std::size(known_q6.qh); ++i) {
        known_q6.qh[i] = static_cast<uint8_t>(i * 43 + 3);
    }
    for (size_t i = 0; i < std::size(known_q6.scales); ++i) {
        known_q6.scales[i] = static_cast<int8_t>(
            static_cast<int>(i) * 7 - 48);
    }
    for (const auto matrix : {
             lfm::CpuGgufMatrix{
                 lfm::GgmlType::Q4_K, 1, 256,
                 reinterpret_cast<const std::byte*>(&known_q4),
                 sizeof(known_q4)},
             lfm::CpuGgufMatrix{
                 lfm::GgmlType::Q6_K, 1, 256,
                 reinterpret_cast<const std::byte*>(&known_q6),
                 sizeof(known_q6)}}) {
        std::vector<float> dequantized(256);
        lfm::cpu_gguf_dequantize_row(matrix, 0, dequantized.data());
        float reference = 0.0f;
        for (size_t i = 0; i < dequantized.size(); ++i) {
            reference += dequantized[i] * activation[0].d *
                         static_cast<float>(activation[0].qs[i]);
        }
        const float scalar = lfm::cpu_gguf_dot_scalar(
            matrix.data, matrix.type, activation.data(), matrix.cols);
        const float optimized = lfm::select_cpu_gguf_dot_kernel(isa)(
            matrix.data, matrix.type, activation.data(), matrix.cols);
        const float tolerance =
            1e-4f * std::max(1.0f, std::abs(reference));
        LFM_TEST_CHECK(std::abs(scalar - reference) < tolerance);
        LFM_TEST_CHECK(std::abs(optimized - reference) < tolerance);
    }

    lfm::CpuLinearWeight composite;
    composite.rows = 4;
    composite.cols = 256;
    composite.segments.emplace_back(q4);
    composite.segments.emplace_back(q6);
    composite.validate();
    LFM_TEST_CHECK(composite.gguf_native());

    lfm::CpuThreadPool pool(4);
    lfm::CpuLinearEngine linear(isa, pool);
    std::vector<float> gemv(4);
    linear.gemv(composite, input.data(), gemv.data());
    LFM_TEST_CHECK(std::abs(gemv[1] - gemv[0]) < 1e-4f);
    LFM_TEST_CHECK(std::abs(gemv[2] - gemv[0]) < 1e-4f);
    LFM_TEST_CHECK(std::abs(gemv[3] - 2.0f * gemv[0]) < 1e-3f);

    std::vector<float> batch_input(2 * input.size());
    std::copy(input.begin(), input.end(), batch_input.begin());
    for (size_t i = 0; i < input.size(); ++i) {
        batch_input[input.size() + i] = input[i] * 0.5f;
    }
    std::vector<float> gemm(8, 2.0f);
    linear.gemm(composite, batch_input.data(), gemm.data(), 2, 0.25f);
    for (size_t r = 0; r < 2; ++r) {
        LFM_TEST_CHECK(std::abs(gemm[r * 4 + 1] - gemm[r * 4]) < 1e-4f);
        LFM_TEST_CHECK(std::abs(gemm[r * 4 + 2] - gemm[r * 4]) < 1e-4f);
        LFM_TEST_CHECK(
            std::abs(gemm[r * 4 + 3] -
                     (2.0f * gemm[r * 4] - 0.5f)) < 1e-3f);
    }

    lfm::CpuLinearWeight q4_weight = lfm::CpuLinearWeight::from_gguf(q4);
    lfm::CpuLinearWeight q6_weight = lfm::CpuLinearWeight::from_gguf(q6);
    std::vector<float> grouped_input(4 * input.size());
    for (size_t row_index = 0; row_index < 4; ++row_index) {
        for (size_t col = 0; col < input.size(); ++col) {
            grouped_input[row_index * input.size() + col] =
                input[col] * static_cast<float>(row_index + 1) * 0.25f;
        }
    }
    std::vector<float> grouped_expected(8), grouped_actual(8);
    linear.gemm(q4_weight, grouped_input.data(), grouped_expected.data(), 2);
    linear.gemm(q6_weight, grouped_input.data() + 2 * input.size(),
                grouped_expected.data() + 4, 2);
    const std::array<lfm::CpuGroupedGemmJob, 2> grouped_jobs{{
        {&q4_weight, 0, 2}, {&q6_weight, 2, 2},
    }};
    linear.gemm_grouped(grouped_jobs, grouped_input.data(), grouped_actual.data());
    for (size_t value = 0; value < grouped_actual.size(); ++value) {
        LFM_TEST_CHECK(std::abs(grouped_actual[value] - grouped_expected[value]) < 1e-4f);
    }

    const char* real_gguf = std::getenv("LFM_GGUF_TEST_FILE");
    if (real_gguf && *real_gguf) {
        lfm::CpuModelOptions options;
        options.threads = 4;
        options.use_pack_cache = true;
        lfm::CpuModel model(real_gguf, 16, options);
        LFM_TEST_CHECK(model.diagnostics().pack_path().empty());
        LFM_TEST_CHECK(
            model.diagnostics().backend_description().find("gguf-native") !=
            std::string::npos);
        model.session().prefill({1});
        const std::vector<float> first = model.diagnostics().copy_logits();
        model.session().prefill({1});
        const std::vector<float> second = model.diagnostics().copy_logits();
        LFM_TEST_CHECK(first.size() == second.size());
        for (size_t i = 0; i < first.size(); ++i) {
            LFM_TEST_CHECK(std::isfinite(first[i]));
            LFM_TEST_CHECK(first[i] == second[i]);
        }
    } else {
        std::cout << "real_gguf SKIP (set LFM_GGUF_TEST_FILE)\n";
    }

    std::cout << "cpu_gguf_kernels_test: isa=" << lfm::cpu_isa_name(isa)
              << " ok\n";
    return 0;
}
