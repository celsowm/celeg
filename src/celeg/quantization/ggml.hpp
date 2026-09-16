#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace celeg {

enum class GgmlType : std::int32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q8_0 = 8,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    IQ3_XXS = 18,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    BF16 = 30,
    Unknown = -1,
};

struct GgmlTypeTrait {
    std::uint32_t block_size = 0;
    std::uint32_t type_size = 0;
};

struct GgmlMatrixView {
    GgmlType type = GgmlType::Unknown;
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    const std::byte* data = nullptr;
    std::size_t bytes = 0;

    std::size_t row_bytes() const;
    std::size_t memory_bytes() const { return bytes; }
    void validate() const;
};

using GgmlDecodeRowFunction = void (*)(const GgmlMatrixView&, std::size_t,
                                       float*);

GgmlTypeTrait ggml_type_trait(GgmlType type);
const char* ggml_type_name(GgmlType type);
GgmlType ggml_type_from_ordinal(std::int32_t raw);
std::optional<GgmlDecodeRowFunction> ggml_row_decoder(GgmlType type);
void ggml_decode_row(const GgmlMatrixView& matrix, std::size_t row,
                     float* output);

/// Host-side Q4_K encoder: packs a row-major float matrix into llama.cpp
/// BlockQ4K layout (256-value superblocks, 8 sub-blocks of 32, 6-bit
/// packed scale/min pair, fp16 super-scales). Columns must be a multiple
/// of the 256-value superblock size -- the native CUDA MMQ kernels impose
/// the same constraint, so loaders gate on it before calling. Returns one
/// packed block stream of rows * (cols/256) * 144 bytes. The encoder is
/// the exact structural inverse of the Q4_K branch of ggml_decode_row;
/// tests roundtrip through it.
std::vector<std::uint8_t> quantize_f32_q4k(std::span<const float> values,
                                           std::size_t rows,
                                           std::size_t cols);

}
