#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

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

}
