#pragma once

#include "celeg/backend/cpu/isa.hpp"
#include "celeg/backend/cpu/quantization.hpp"
#include "celeg/checkpoint/formats/gguf.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

namespace celeg {

struct CpuGgufMatrix {
    GgmlType type = GgmlType::Unknown;
    uint32_t rows = 0;
    uint32_t cols = 0;
    const std::byte* data = nullptr;
    size_t bytes = 0;

    size_t row_bytes() const;
    size_t memory_bytes() const { return bytes; }
    void validate() const;
};

struct CpuInt8Matrix {
    uint32_t rows = 0;
    uint32_t cols = 0;
    std::shared_ptr<std::vector<int8_t>> values =
        std::make_shared<std::vector<int8_t>>();
    std::shared_ptr<std::vector<float>> scales =
        std::make_shared<std::vector<float>>();

    size_t memory_bytes() const {
        return values->size() * sizeof(int8_t) + scales->size() * sizeof(float);
    }
    const int8_t* data() const { return values->data(); }
    const float* scale_data() const { return scales->data(); }
    void validate() const;
};

using CpuLinearMatrix = std::variant<Q4GroupMatrix, CpuGgufMatrix, CpuInt8Matrix>;

// A logical [rows, cols] linear weight. Most weights have one segment.
// GGUF concatenations may retain multiple row-contiguous Q4_K/Q6_K segments
// so no dequantization or requantization is needed.
struct CpuLinearWeight {
    uint32_t rows = 0;
    uint32_t cols = 0;
    std::vector<CpuLinearMatrix> segments;

    static CpuLinearWeight from_q4(Q4GroupMatrix matrix);
    static CpuLinearWeight from_gguf(CpuGgufMatrix matrix);
    static CpuLinearWeight from_int8(CpuInt8Matrix matrix);
    size_t memory_bytes() const;
    bool gguf_native() const;
    void validate() const;
};

// GGML Q8_K activation block: one scale for 256 signed values plus sums for
// each 16-value group. The sums let Q4_K's per-subblock minimum term be
// applied without reconstructing floating-point weights.
struct CpuQ8KBlock {
    float d = 0.0f;
    std::array<int8_t, 256> qs{};
    std::array<int16_t, 16> bsums{};
};

using CpuGgufDotFunction = float (*)(const std::byte* packed_row,
                                     GgmlType type,
                                     const CpuQ8KBlock* activation,
                                     size_t cols);
using CpuGgufDot4Function = void (*)(const std::byte* packed_row,
                                     GgmlType type,
                                     const CpuQ8KBlock* activation,
                                     size_t cols, float* output4);

std::vector<CpuQ8KBlock> cpu_quantize_q8k(const float* input, size_t cols,
                                          CpuIsa isa);
void cpu_quantize_q8k_into(const float* input, size_t cols, CpuIsa isa,
                           CpuQ8KBlock* output);
float cpu_gguf_dot_scalar(const std::byte* packed_row, GgmlType type,
                          const CpuQ8KBlock* activation, size_t cols);
CpuGgufDotFunction select_cpu_gguf_dot_kernel(CpuIsa isa);
CpuGgufDot4Function select_cpu_gguf_dot4_kernel(CpuIsa isa);
void cpu_gguf_dequantize_row(const CpuGgufMatrix& matrix, size_t row,
                             float* output);

} // namespace celeg
