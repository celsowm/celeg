#include "celeg/backend/cpu/gguf.hpp"
#include "celeg/backend/cpu/kernels.hpp"
#include "celeg/backend/cpu/model.hpp"
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

struct BlockQ2K {
    uint8_t scales[16]{};
    uint8_t qs[64]{};
    uint16_t d = 0x3c00;
    uint16_t dmin = 0;
};

struct BlockQ3K {
    uint8_t hmask[32]{};
    uint8_t qs[64]{};
    uint8_t scales[12]{};
    uint16_t d = 0x3c00;
};
struct BlockQ4_0 {
    uint16_t d = 0x3c00;
    uint8_t qs[16]{};
};
struct BlockQ5_0 {
    uint16_t d = 0x3c00;
    uint8_t qh[4]{};
    uint8_t qs[16]{};
};
struct BlockQ4_1 {
    uint16_t d = 0x3c00;
    uint16_t dmin = 0x3c00;
    uint8_t qs[16]{};
};
struct BlockQ8_0 {
    uint16_t d = 0x3c00;
    int8_t qs[32]{};
};
struct BlockQ5K {
    uint16_t d = 0x3c00;
    uint16_t dmin = 0;
    uint8_t scales[12]{};
    uint8_t qh[32]{};
    uint8_t qs[128]{};
};
#pragma pack(pop)

static_assert(sizeof(BlockQ4K) == 144);
static_assert(sizeof(BlockQ6K) == 210);
static_assert(sizeof(BlockQ2K) == 84);
static_assert(sizeof(BlockQ3K) == 110);
static_assert(sizeof(BlockQ4_0) == 18);
static_assert(sizeof(BlockQ5_0) == 22);
static_assert(sizeof(BlockQ4_1) == 20);
static_assert(sizeof(BlockQ8_0) == 34);
static_assert(sizeof(BlockQ5K) == 176);

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

BlockQ5K unit_q5k() {
    BlockQ5K block;
    for (uint8_t& value : block.scales) value = 1;
    for (uint8_t& value : block.qh) value = 0;
    for (uint8_t& value : block.qs) value = 0x11;
    return block;
}

BlockQ8_0 unit_q8_0() {
    BlockQ8_0 block;
    for (int8_t& value : block.qs) value = 1;
    return block;
}

BlockQ4_1 unit_q4_1() {
    BlockQ4_1 block;
    for (uint8_t& value : block.qs) value = 0x11;
    return block;
}

template <typename T>
const std::byte* bytes(const std::vector<T>& values) {
    return reinterpret_cast<const std::byte*>(values.data());
}

}

int main() {
    std::vector<BlockQ4K> q4_rows(2, unit_q4k());
    std::vector<BlockQ6K> q6_rows(2, unit_q6k());
    for (int8_t& value : q6_rows[1].scales) value = 2;

    celeg::CpuGgufMatrix q4{
        celeg::GgmlType::Q4_K, 2, 256, bytes(q4_rows),
        q4_rows.size() * sizeof(BlockQ4K)};
    celeg::CpuGgufMatrix q6{
        celeg::GgmlType::Q6_K, 2, 256, bytes(q6_rows),
        q6_rows.size() * sizeof(BlockQ6K)};
    BlockQ2K q2_block;
    for (uint8_t& value : q2_block.scales) value = 1;
    BlockQ3K q3_block;
    celeg::CpuGgufMatrix q2{
        celeg::GgmlType::Q2_K, 1, 256,
        reinterpret_cast<const std::byte*>(&q2_block), sizeof(q2_block)};
    celeg::CpuGgufMatrix q3{
        celeg::GgmlType::Q3_K, 1, 256,
        reinterpret_cast<const std::byte*>(&q3_block), sizeof(q3_block)};
    q4.validate();
    q6.validate();
    q2.validate();
    q3.validate();

    std::vector<float> row(256);
    celeg::cpu_gguf_dequantize_row(q4, 1, row.data());
    for (float value : row) CELEG_TEST_CHECK(std::abs(value - 1.0f) < 1e-6f);
    celeg::cpu_gguf_dequantize_row(q6, 0, row.data());
    for (float value : row) CELEG_TEST_CHECK(std::abs(value - 1.0f) < 1e-6f);
    celeg::cpu_gguf_dequantize_row(q6, 1, row.data());
    for (float value : row) CELEG_TEST_CHECK(std::abs(value - 2.0f) < 1e-6f);
    celeg::cpu_gguf_dequantize_row(q2, 0, row.data());
    for (float value : row) CELEG_TEST_CHECK(std::abs(value) < 1e-6f);
    celeg::cpu_gguf_dequantize_row(q3, 0, row.data());
    for (float value : row) CELEG_TEST_CHECK(std::abs(value - 128.0f) < 1e-6f);

    std::vector<float> input(256);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.07f);
    }
    const auto activation =
        celeg::cpu_quantize_q8k(input.data(), input.size(), celeg::CpuIsa::Scalar);
    const celeg::CpuIsa isa = celeg::detect_cpu_capabilities().best_isa();
    const auto optimized_activation =
        celeg::cpu_quantize_q8k(input.data(), input.size(), isa);
    CELEG_TEST_CHECK(optimized_activation.size() == activation.size());
    for (size_t block = 0; block < activation.size(); ++block) {
        CELEG_TEST_CHECK(optimized_activation[block].d == activation[block].d);
        CELEG_TEST_CHECK(optimized_activation[block].qs == activation[block].qs);
        CELEG_TEST_CHECK(
            optimized_activation[block].bsums == activation[block].bsums);
    }
    for (const auto* matrix : {&q4, &q6}) {
        const float scalar = celeg::cpu_gguf_dot_scalar(
            matrix->data, matrix->type, activation.data(), matrix->cols);
        const auto selected =
            celeg::select_cpu_gguf_dot_kernel(isa);
        const float optimized = selected(
            matrix->data, matrix->type, activation.data(), matrix->cols);
        CELEG_TEST_CHECK(std::abs(optimized - scalar) < 1e-4f);
        if (const auto dot4 = celeg::select_cpu_gguf_dot4_kernel(isa)) {
            std::array<celeg::CpuQ8KBlock, 4> batch{};
            std::array<float, 4> expected{};
            for (size_t lane = 0; lane < batch.size(); ++lane) {
                std::vector<float> lane_input(input.size());
                for (size_t col = 0; col < input.size(); ++col) {
                    lane_input[col] = input[col] * static_cast<float>(lane + 1) +
                        std::cos(static_cast<float>(col) * 0.13f + static_cast<float>(lane));
                }
                const auto lane_activation = celeg::cpu_quantize_q8k(
                    lane_input.data(), lane_input.size(), isa);
                batch[lane] = lane_activation[0];
                expected[lane] = selected(matrix->data, matrix->type,
                    lane_activation.data(), matrix->cols);
            }
            std::array<float, 4> values{};
            dot4(matrix->data, matrix->type, batch.data(), matrix->cols, values.data());
            for (size_t lane = 0; lane < values.size(); ++lane) {
                CELEG_TEST_CHECK(std::abs(values[lane] - expected[lane]) < 1e-4f);
            }
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
    BlockQ2K known_q2k;
    known_q2k.d = 0x3800;
    known_q2k.dmin = 0x3400;
    for (size_t i = 0; i < std::size(known_q2k.scales); ++i) {
        known_q2k.scales[i] = static_cast<uint8_t>(13 + i * 23);
    }
    for (size_t i = 0; i < std::size(known_q2k.qs); ++i) {
        known_q2k.qs[i] = static_cast<uint8_t>(i * 31 + 5);
    }
    BlockQ3K known_q3k;
    known_q3k.d = 0x3800;
    for (size_t i = 0; i < std::size(known_q3k.hmask); ++i) {
        known_q3k.hmask[i] = static_cast<uint8_t>(i * 17 + 2);
    }
    for (size_t i = 0; i < std::size(known_q3k.qs); ++i) {
        known_q3k.qs[i] = static_cast<uint8_t>(i * 19 + 9);
    }
    for (size_t i = 0; i < std::size(known_q3k.scales); ++i) {
        known_q3k.scales[i] = static_cast<uint8_t>(7 + i * 11);
    }
    BlockQ5K known_q5k = unit_q5k();
    for (uint8_t& value : known_q5k.qs) value = 0x11;
    std::vector<BlockQ8_0> known_q8_0(8, unit_q8_0());
    for (BlockQ8_0& block : known_q8_0) {
        for (int8_t& value : block.qs) value = 1;
    }
    std::vector<BlockQ4_1> known_q4_1(8, unit_q4_1());
    for (BlockQ4_1& block : known_q4_1) {
        for (uint8_t& value : block.qs) value = 0x11;
    }
    std::vector<float> dequant_q4_1(32);
    celeg::cpu_gguf_dequantize_row(
        celeg::CpuGgufMatrix{
            celeg::GgmlType::Q4_1, 1, 32,
            reinterpret_cast<const std::byte*>(&known_q4_1[0]), sizeof(known_q4_1[0])},
        0, dequant_q4_1.data());
    for (size_t i = 0; i < dequant_q4_1.size(); ++i) {
        const float expected = 1.0f * static_cast<float>(0x11 & 0x0f) + 1.0f;
        CELEG_TEST_CHECK(std::abs(dequant_q4_1[i] - expected) < 1e-6f);
    }
    for (const auto matrix : {
             celeg::CpuGgufMatrix{
                 celeg::GgmlType::Q4_K, 1, 256,
                 reinterpret_cast<const std::byte*>(&known_q4),
                 sizeof(known_q4)},
             celeg::CpuGgufMatrix{
                 celeg::GgmlType::Q6_K, 1, 256,
                 reinterpret_cast<const std::byte*>(&known_q6),
                 sizeof(known_q6)},
             celeg::CpuGgufMatrix{
                 celeg::GgmlType::Q2_K, 1, 256,
                 reinterpret_cast<const std::byte*>(&known_q2k),
                 sizeof(known_q2k)},
             celeg::CpuGgufMatrix{
                  celeg::GgmlType::Q3_K, 1, 256,
                  reinterpret_cast<const std::byte*>(&known_q3k),
                  sizeof(known_q3k)},
             celeg::CpuGgufMatrix{
                  celeg::GgmlType::Q5_K, 1, 256,
                  reinterpret_cast<const std::byte*>(&known_q5k),
                  sizeof(known_q5k)},
             celeg::CpuGgufMatrix{
                  celeg::GgmlType::Q8_0, 1, 256,
                  reinterpret_cast<const std::byte*>(known_q8_0.data()),
                  known_q8_0.size() * sizeof(BlockQ8_0)},
             celeg::CpuGgufMatrix{
                  celeg::GgmlType::Q4_1, 1, 256,
                  reinterpret_cast<const std::byte*>(known_q4_1.data()),
                  known_q4_1.size() * sizeof(BlockQ4_1)}}) {
        std::vector<float> dequantized(256);
        celeg::cpu_gguf_dequantize_row(matrix, 0, dequantized.data());
        float reference = 0.0f;
        for (size_t i = 0; i < dequantized.size(); ++i) {
            reference += dequantized[i] * activation[0].d *
                         static_cast<float>(activation[0].qs[i]);
        }
        const float scalar = celeg::cpu_gguf_dot_scalar(
            matrix.data, matrix.type, activation.data(), matrix.cols);
        const float optimized = celeg::select_cpu_gguf_dot_kernel(isa)(
            matrix.data, matrix.type, activation.data(), matrix.cols);
        const float tolerance =
            1e-4f * std::max(1.0f, std::abs(reference));
        CELEG_TEST_CHECK(std::abs(scalar - reference) < tolerance);
        CELEG_TEST_CHECK(std::abs(optimized - reference) < tolerance);
    }

    std::vector<BlockQ4_0> known_q4_0(8);
    for (BlockQ4_0& block : known_q4_0) {
        block.d = 0x3800;
        for (size_t i = 0; i < std::size(block.qs); ++i) {
            block.qs[i] = static_cast<uint8_t>(i * 37 + 11);
        }
    }
    std::vector<BlockQ5_0> known_q5_0(8);
    for (BlockQ5_0& block : known_q5_0) {
        block.d = 0x3800;
        for (size_t i = 0; i < std::size(block.qs); ++i) {
            block.qs[i] = static_cast<uint8_t>(i * 29 + 7);
        }
        for (size_t i = 0; i < std::size(block.qh); ++i) {
            block.qh[i] = static_cast<uint8_t>(i * 43 + 3);
        }
    }
    for (const auto matrix : {
             celeg::CpuGgufMatrix{
                 celeg::GgmlType::Q4_0, 1, 256,
                 reinterpret_cast<const std::byte*>(known_q4_0.data()),
                 known_q4_0.size() * sizeof(BlockQ4_0)},
             celeg::CpuGgufMatrix{
                 celeg::GgmlType::Q5_0, 1, 256,
                 reinterpret_cast<const std::byte*>(known_q5_0.data()),
                 known_q5_0.size() * sizeof(BlockQ5_0)}}) {
        std::vector<float> dequantized(256);
        celeg::cpu_gguf_dequantize_row(matrix, 0, dequantized.data());
        float reference = 0.0f;
        for (size_t i = 0; i < dequantized.size(); ++i) {
            reference += dequantized[i] * activation[0].d *
                         static_cast<float>(activation[0].qs[i]);
        }
        const float scalar = celeg::cpu_gguf_dot_scalar(
            matrix.data, matrix.type, activation.data(), matrix.cols);
        const float optimized = celeg::select_cpu_gguf_dot_kernel(isa)(
            matrix.data, matrix.type, activation.data(), matrix.cols);
        const float tolerance =
            1e-4f * std::max(1.0f, std::abs(reference));
        CELEG_TEST_CHECK(std::abs(scalar - reference) < tolerance);
        CELEG_TEST_CHECK(std::abs(optimized - reference) < tolerance);
        CELEG_TEST_CHECK(std::abs(optimized - scalar) < 1e-4f);
    }

    celeg::CpuLinearWeight composite;
    composite.rows = 4;
    composite.cols = 256;
    composite.segments.emplace_back(q4);
    composite.segments.emplace_back(q6);
    composite.validate();
    CELEG_TEST_CHECK(composite.gguf_native());

    celeg::CpuThreadPool pool(4);
    celeg::CpuLinearEngine linear(isa, pool);
    std::vector<float> gemv(4);
    linear.gemv(composite, input.data(), gemv.data());
    CELEG_TEST_CHECK(std::abs(gemv[1] - gemv[0]) < 1e-4f);
    CELEG_TEST_CHECK(std::abs(gemv[2] - gemv[0]) < 1e-4f);
    CELEG_TEST_CHECK(std::abs(gemv[3] - 2.0f * gemv[0]) < 1e-3f);

    std::vector<float> batch_input(2 * input.size());
    std::copy(input.begin(), input.end(), batch_input.begin());
    for (size_t i = 0; i < input.size(); ++i) {
        batch_input[input.size() + i] = input[i] * 0.5f;
    }
    std::vector<float> gemm(8, 2.0f);
    linear.gemm(composite, batch_input.data(), gemm.data(), 2, 0.25f);
    for (size_t r = 0; r < 2; ++r) {
        CELEG_TEST_CHECK(std::abs(gemm[r * 4 + 1] - gemm[r * 4]) < 1e-4f);
        CELEG_TEST_CHECK(std::abs(gemm[r * 4 + 2] - gemm[r * 4]) < 1e-4f);
        CELEG_TEST_CHECK(
            std::abs(gemm[r * 4 + 3] -
                     (2.0f * gemm[r * 4] - 0.5f)) < 1e-3f);
    }

    celeg::CpuLinearWeight q4_weight = celeg::CpuLinearWeight::from_gguf(q4);
    celeg::CpuLinearWeight q6_weight = celeg::CpuLinearWeight::from_gguf(q6);
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
    const std::array<celeg::CpuGroupedGemmJob, 2> grouped_jobs{{
        {&q4_weight, 0, 2}, {&q6_weight, 2, 2},
    }};
    linear.gemm_grouped(grouped_jobs, grouped_input.data(), grouped_actual.data());
    for (size_t value = 0; value < grouped_actual.size(); ++value) {
        CELEG_TEST_CHECK(std::abs(grouped_actual[value] - grouped_expected[value]) < 1e-4f);
    }

    const char* real_gguf = std::getenv("CELEG_GGUF_TEST_FILE");
    if (real_gguf && *real_gguf) {
        celeg::CpuModelOptions options;
        options.threads = 4;
        options.use_pack_cache = true;
        celeg::CpuModel model(real_gguf, 16, options);
        CELEG_TEST_CHECK(model.diagnostics().pack_path().empty());
        CELEG_TEST_CHECK(
            model.diagnostics().backend_description().find("gguf-native") !=
            std::string::npos);
        model.session().prefill({1});
        const std::vector<float> first = model.diagnostics().copy_logits();
        model.session().prefill({1});
        const std::vector<float> second = model.diagnostics().copy_logits();
        CELEG_TEST_CHECK(first.size() == second.size());
        for (size_t i = 0; i < first.size(); ++i) {
            CELEG_TEST_CHECK(std::isfinite(first[i]));
            CELEG_TEST_CHECK(first[i] == second[i]);
        }
    } else {
        std::cout << "real_gguf SKIP (set CELEG_GGUF_TEST_FILE)\n";
    }

    std::cout << "cpu_gguf_kernels_test: isa=" << celeg::cpu_isa_name(isa)
              << " ok\n";
    return 0;
}
