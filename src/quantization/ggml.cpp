#include "celeg/quantization/ggml.hpp"
#include "celeg/model/weights/quantization.hpp"
#include "celeg/quantization/gguf_blocks.hpp"
#include "celeg/checkpoint/gguf_iq.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace celeg {
namespace {

struct GgmlTypeEntry {
    std::int32_t ordinal;
    GgmlType type;
    const char* name;
    GgmlTypeTrait trait;
};

constexpr GgmlTypeEntry kGgmlTypes[] = {
    {0, GgmlType::F32, "F32", {1, 4}},
    {1, GgmlType::F16, "F16", {1, 2}},
    {30, GgmlType::BF16, "BF16", {1, 2}},
    {2, GgmlType::Q4_0, "Q4_0", {32, 18}},
    {3, GgmlType::Q4_1, "Q4_1", {32, 20}},
    {6, GgmlType::Q5_0, "Q5_0", {32, 22}},
    {8, GgmlType::Q8_0, "Q8_0", {32, 34}},
    {10, GgmlType::Q2_K, "Q2_K", {256, 84}},
    {11, GgmlType::Q3_K, "Q3_K", {256, 110}},
    {12, GgmlType::Q4_K, "Q4_K", {256, 144}},
    {13, GgmlType::Q5_K, "Q5_K", {256, 176}},
    {14, GgmlType::Q6_K, "Q6_K", {256, 210}},
    {18, GgmlType::IQ3_XXS, "IQ3_XXS", {256, 98}},
    {20, GgmlType::IQ4_NL, "IQ4_NL", {32, 18}},
    {21, GgmlType::IQ3_S, "IQ3_S", {256, 110}},
    {22, GgmlType::IQ2_S, "IQ2_S", {256, 82}},
    {23, GgmlType::IQ4_XS, "IQ4_XS", {256, 136}},
};

const GgmlTypeEntry* find_entry(GgmlType type) {
    for (const GgmlTypeEntry& entry : kGgmlTypes) {
        if (entry.type == type) return &entry;
    }
    return nullptr;
}

}

GgmlTypeTrait ggml_type_trait(GgmlType type) {
    const GgmlTypeEntry* entry = find_entry(type);
    return entry ? entry->trait : GgmlTypeTrait{};
}

const char* ggml_type_name(GgmlType type) {
    const GgmlTypeEntry* entry = find_entry(type);
    return entry ? entry->name : "Unknown";
}

GgmlType ggml_type_from_ordinal(std::int32_t raw) {
    for (const GgmlTypeEntry& entry : kGgmlTypes) {
        if (entry.ordinal == raw) return entry.type;
    }
    return GgmlType::Unknown;
}

std::optional<GgmlDecodeRowFunction> ggml_row_decoder(GgmlType type) {
    const GgmlTypeTrait trait = ggml_type_trait(type);
    if (trait.block_size == 1 || trait.block_size == 0) return std::nullopt;
    return &ggml_decode_row;
}

std::size_t GgmlMatrixView::row_bytes() const {
    const GgmlTypeTrait trait = ggml_type_trait(type);
    if (trait.block_size == 0 || trait.type_size == 0 ||
        cols % trait.block_size != 0) {
        return 0;
    }
    return static_cast<std::size_t>(cols / trait.block_size) * trait.type_size;
}

void GgmlMatrixView::validate() const {
    if (!ggml_row_decoder(type).has_value() || rows == 0 || cols == 0 ||
        data == nullptr || row_bytes() == 0 || bytes != rows * row_bytes()) {
        throw std::invalid_argument("invalid GGML matrix");
    }
}

using namespace ggml_detail;

void ggml_decode_row(const GgmlMatrixView& matrix, size_t row,
                             float* output) {
    matrix.validate();
    if (row >= matrix.rows || !output) {
        throw std::invalid_argument("invalid GGML row decode");
    }
    const std::byte* packed = matrix.data + row * matrix.row_bytes();
    const size_t blocks = matrix.cols / 256;
    if (matrix.type == GgmlType::Q2_K) {
        const auto* weights = reinterpret_cast<const BlockQ2K*>(packed);
        for (size_t b = 0; b < blocks; ++b) {
            const BlockQ2K& weight = weights[b];
            const float d = fp16_bits_to_float(weight.d);
            const float dmin = fp16_bits_to_float(weight.dmin);
            for (int col = 0; col < 256; ++col) {
                const int sub = col / 16;
                output[b * 256 + static_cast<size_t>(col)] =
                    d * static_cast<float>(weight.scales[sub] & 0x0f) *
                        q2k_value(weight, col) -
                    dmin * static_cast<float>(weight.scales[sub] >> 4);
            }
        }
        return;
    }
    if (matrix.type == GgmlType::Q3_K) {
        const auto* weights = reinterpret_cast<const BlockQ3K*>(packed);
        for (size_t b = 0; b < blocks; ++b) {
            const BlockQ3K& weight = weights[b];
            int8_t scales[16]{};
            q3k_scales(weight, scales);
            const float d = fp16_bits_to_float(weight.d);
            for (int col = 0; col < 256; ++col) {
                const int sub = col / 16;
                output[b * 256 + static_cast<size_t>(col)] =
                    d * static_cast<float>(scales[sub]) *
                    q3k_value(weight, col);
            }
        }
        return;
    }
    if (matrix.type == GgmlType::Q4_0) {
        const auto* weights = reinterpret_cast<const BlockQ4_0*>(packed);
        for (size_t b = 0; b < matrix.cols / 32; ++b) {
            const BlockQ4_0& weight = weights[b];
            const float d = fp16_bits_to_float(weight.d);
            /// GGML packs each block as two halves, not interleaved pairs:
            /// qs[j] holds element j in its low nibble and element j+16 in
            /// its high nibble (j in [0,16)).
            for (int j = 0; j < 16; ++j) {
                const uint8_t packed_value = weight.qs[j];
                output[b * 32 + static_cast<size_t>(j)] =
                    d * static_cast<float>((packed_value & 0x0f) - 8);
                output[b * 32 + static_cast<size_t>(j + 16)] =
                    d * static_cast<float>((packed_value >> 4) - 8);
            }
        }
        return;
    }
    if (matrix.type == GgmlType::Q5_0) {
        const auto* weights = reinterpret_cast<const BlockQ5_0*>(packed);
        for (size_t b = 0; b < matrix.cols / 32; ++b) {
            const BlockQ5_0& weight = weights[b];
            const float d = fp16_bits_to_float(weight.d);
            uint32_t qh;
            std::memcpy(&qh, weight.qh, sizeof(qh));
            /// Same split-half nibble layout as Q4_0; the high (5th) bit of
            /// element j lives at bit j of qh, and of element j+16 at bit
            /// j+16 of qh.
            for (int j = 0; j < 16; ++j) {
                const uint8_t packed_value = weight.qs[j];
                const int high0 = (qh >> j) & 1;
                const int high1 = (qh >> (j + 16)) & 1;
                output[b * 32 + static_cast<size_t>(j)] =
                    d * static_cast<float>(((packed_value & 0x0f) | (high0 << 4)) - 16);
                output[b * 32 + static_cast<size_t>(j + 16)] =
                    d * static_cast<float>(((packed_value >> 4) | (high1 << 4)) - 16);
            }
        }
        return;
    }
    if (matrix.type == GgmlType::Q4_1) {
        const auto* weights = reinterpret_cast<const BlockQ4_1*>(packed);
        for (size_t b = 0; b < matrix.cols / 32; ++b) {
            const BlockQ4_1& weight = weights[b];
            const float d = fp16_bits_to_float(weight.d);
            const float m = fp16_bits_to_float(weight.dmin);
            for (int j = 0; j < 16; ++j) {
                const uint8_t packed_value = weight.qs[j];
                output[b * 32 + static_cast<size_t>(j)] =
                    d * static_cast<float>(packed_value & 0x0f) + m;
                output[b * 32 + static_cast<size_t>(j + 16)] =
                    d * static_cast<float>(packed_value >> 4) + m;
            }
        }
        return;
    }
    if (matrix.type == GgmlType::Q8_0) {
        const auto* weights = reinterpret_cast<const BlockQ8_0*>(packed);
        for (size_t b = 0; b < matrix.cols / 32; ++b) {
            const BlockQ8_0& weight = weights[b];
            const float d = fp16_bits_to_float(weight.d);
            for (int col = 0; col < 32; ++col) {
                output[b * 32 + static_cast<size_t>(col)] =
                    d * static_cast<float>(weight.qs[col]);
            }
        }
        return;
    }
    if (matrix.type == GgmlType::Q4_K) {
        const auto* weights = reinterpret_cast<const BlockQ4K*>(packed);
        for (size_t b = 0; b < blocks; ++b) {
            const BlockQ4K& weight = weights[b];
            const float d = fp16_bits_to_float(weight.d);
            const float dmin = fp16_bits_to_float(weight.dmin);
            for (int sub = 0; sub < 8; ++sub) {
                uint8_t scale = 0, minimum = 0;
                q4k_scale_min(sub, weight.scales, scale, minimum);
                for (int i = 0; i < 32; ++i) {
                    const int col = sub * 32 + i;
                    output[b * 256 + static_cast<size_t>(col)] =
                        d * static_cast<float>(scale * q4k_value(weight, col)) -
                        dmin * static_cast<float>(minimum);
                }
            }
        }
        return;
    }
    if (matrix.type == GgmlType::Q5_K) {
        const auto* weights = reinterpret_cast<const BlockQ5K*>(packed);
        for (size_t b = 0; b < blocks; ++b) {
            const BlockQ5K& weight = weights[b];
            const float d = fp16_bits_to_float(weight.d);
            const float dmin = fp16_bits_to_float(weight.dmin);
            for (int sub = 0; sub < 8; ++sub) {
                uint8_t scale = 0, minimum = 0;
                q4k_scale_min(sub, weight.scales, scale, minimum);
                for (int i = 0; i < 32; ++i) {
                    const int col = sub * 32 + i;
                    output[b * 256 + static_cast<size_t>(col)] =
                        d * static_cast<float>(scale * q5k_value(weight, col)) -
                        dmin * static_cast<float>(minimum);
                }
            }
        }
        return;
    }
    if (matrix.type == GgmlType::IQ2_S) {
        const auto* weights = reinterpret_cast<const gguf_iq::BlockIq2S*>(packed);
        for (size_t b = 0; b < blocks; ++b) {
            const gguf_iq::BlockIq2S& weight = weights[b];
            const float d = fp16_bits_to_float(weight.d);
            for (int col = 0; col < 256; ++col) {
                const float scale =
                    gguf_iq::iq2s_sub_scale(weight, col / 32, (col % 32) / 16);
                output[b * 256 + static_cast<size_t>(col)] =
                    d * scale * static_cast<float>(gguf_iq::iq2s_value(weight, col));
            }
        }
        return;
    }
    if (matrix.type == GgmlType::IQ3_XXS) {
        const auto* weights = reinterpret_cast<const gguf_iq::BlockIq3XXS*>(packed);
        for (size_t b = 0; b < blocks; ++b) {
            const gguf_iq::BlockIq3XXS& weight = weights[b];
            const float d = fp16_bits_to_float(weight.d);
            for (int col = 0; col < 256; ++col) {
                const float scale =
                    gguf_iq::iq3xxs_sub_scale(gguf_iq::iq3xxs_aux(weight, col / 32));
                output[b * 256 + static_cast<size_t>(col)] =
                    d * scale * static_cast<float>(gguf_iq::iq3xxs_value(weight, col));
            }
        }
        return;
    }
    if (matrix.type == GgmlType::IQ3_S) {
        const auto* weights = reinterpret_cast<const gguf_iq::BlockIq3S*>(packed);
        for (size_t b = 0; b < blocks; ++b) {
            const gguf_iq::BlockIq3S& weight = weights[b];
            const float d = fp16_bits_to_float(weight.d);
            for (int col = 0; col < 256; ++col) {
                output[b * 256 + static_cast<size_t>(col)] =
                    d * gguf_iq::iq3s_sub_scale(weight, col / 32) *
                    static_cast<float>(gguf_iq::iq3s_value(weight, col));
            }
        }
        return;
    }
    if (matrix.type == GgmlType::IQ4_XS) {
        const auto* weights = reinterpret_cast<const gguf_iq::BlockIq4XS*>(packed);
        for (size_t b = 0; b < blocks; ++b) {
            const gguf_iq::BlockIq4XS& weight = weights[b];
            const float d = fp16_bits_to_float(weight.d);
            for (int col = 0; col < 256; ++col) {
                output[b * 256 + static_cast<size_t>(col)] =
                    d * gguf_iq::iq4xs_sub_scale(weight, col / 32) *
                    static_cast<float>(gguf_iq::iq4xs_value(weight, col));
            }
        }
        return;
    }
    if (matrix.type == GgmlType::IQ4_NL) {
        const auto* weights = reinterpret_cast<const gguf_iq::BlockIq4NL*>(packed);
        for (size_t b = 0; b < matrix.cols / 32; ++b) {
            const gguf_iq::BlockIq4NL& weight = weights[b];
            const float d = fp16_bits_to_float(weight.d);
            for (int col = 0; col < 32; ++col) {
                output[b * 32 + static_cast<size_t>(col)] =
                    d * static_cast<float>(gguf_iq::iq4nl_value(weight, col));
            }
        }
        return;
    }
    if (matrix.type != GgmlType::Q6_K) {
        throw std::invalid_argument(std::string("unsupported GGML dequantize type: ") +
                                    ggml_type_name(matrix.type));
    }
    const auto* weights = reinterpret_cast<const BlockQ6K*>(packed);
    for (size_t b = 0; b < blocks; ++b) {
        const BlockQ6K& weight = weights[b];
        const float d = fp16_bits_to_float(weight.d);
        for (int sub = 0; sub < 16; ++sub) {
            for (int i = 0; i < 16; ++i) {
                const int col = sub * 16 + i;
                output[b * 256 + static_cast<size_t>(col)] =
                    d * static_cast<float>(weight.scales[sub]) *
                    static_cast<float>(q6k_value(weight, col) - 32);
            }
        }
    }
}

}

namespace celeg {

std::vector<uint8_t> quantize_f32_q4k(std::span<const float> values,
                                      std::size_t rows,
                                      std::size_t cols) {
    constexpr size_t kSuperBlock = 256;
    constexpr size_t kSubBlocks = 8;
    constexpr size_t kSubSize = kSuperBlock / kSubBlocks;
    constexpr size_t kBlockBytes = sizeof(BlockQ4K);
    if (rows == 0 || cols == 0 || cols % kSuperBlock != 0) {
        throw std::invalid_argument(
            "Q4_K pack requires a column count that is a multiple of 256");
    }
    if (values.size() != rows * cols) {
        throw std::invalid_argument("Q4_K pack input size mismatch");
    }
    std::vector<uint8_t> result(rows * (cols / kSuperBlock) * kBlockBytes);
    BlockQ4K* blocks = reinterpret_cast<BlockQ4K*>(result.data());
    for (size_t row = 0; row < rows; ++row) {
        const float* source = values.data() + row * cols;
        for (size_t sb = 0; sb < cols / kSuperBlock; ++sb) {
            const float* block_in = source + sb * kSuperBlock;
            BlockQ4K& block = blocks[row * (cols / kSuperBlock) + sb];
            block = BlockQ4K{};

            /// Per-sub ranges; a sub with all-positive values quantizes from
            /// zero (its codes never reach into negatives), a negative-mix
            /// sub keeps its true minimum.
            float sub_scale[kSubBlocks] = {};
            float sub_low[kSubBlocks] = {};
            float sub_min[kSubBlocks] = {};
            float sub_max[kSubBlocks] = {};
            for (size_t j = 0; j < kSubBlocks; ++j) {
                const float* s = block_in + j * kSubSize;
                float lo = s[0], hi = s[0];
                for (size_t i = 1; i < kSubSize; ++i) {
                    lo = std::min(lo, s[i]);
                    hi = std::max(hi, s[i]);
                }
                sub_min[j] = lo;
                sub_max[j] = hi;
            }
            float max_scale = 0.0f;
            float max_min = 0.0f;
            for (size_t j = 0; j < kSubBlocks; ++j) {
                const float low = std::min(sub_min[j], 0.0f);
                sub_low[j] = low;
                sub_scale[j] = (sub_max[j] - low) / 15.0f;
                max_scale = std::max(max_scale, sub_scale[j]);
                max_min = std::max(max_min, -low);
            }
            const float d_model = max_scale > 0.0f ? max_scale / 63.0f : 0.0f;
            block.d = float_to_fp16_bits(d_model);
            const float d = fp16_bits_to_float(block.d);
            /// A superblock with no per-sub spread at all (e.g. a constant
            /// row) carries d==0 but may still hold a non-zero minimum;
            /// dmin stays independent of d so "all mins, no spread" decodes
            /// to the constant value instead of 0.
            const float dmin_model = max_min > 0.0f ? max_min / 63.0f : 0.0f;
            block.dmin = float_to_fp16_bits(dmin_model);
            const float dmin = fp16_bits_to_float(block.dmin);
            if (d <= 0.0f && dmin <= 0.0f) continue;

            uint8_t sc[kSubBlocks] = {};
            uint8_t m[kSubBlocks] = {};
            for (size_t j = 0; j < kSubBlocks; ++j) {
                /// fp16 rounding of d/dmin can push an extreme ratio one
                /// step past the 6-bit code range; clamp instead of letting
                /// the 7th bit bleed into a neighbor's packed lane.
                sc[j] = d > 0.0f
                    ? static_cast<uint8_t>(
                          std::clamp(std::lround(sub_scale[j] / d), 0L, 63L))
                    : uint8_t{0};
                m[j] = dmin > 0.0f
                    ? static_cast<uint8_t>(
                        std::clamp(std::lround(-sub_low[j] / dmin), 0L, 63L))
                    : uint8_t{0};
                /// Packed layout, decoding forwards: bytes 0-3 carry the
                /// low 6 bits of sc_0..3; bytes 4-7 carry the low 6 bits
                /// of m_0..3; bytes 8-11 interleave the low nibbles of
                /// sc_4..7 and m_4..7; the missing top 2 bits of the subs
                /// 4..7 live in the high 2 bits of bytes 0..3 (sc) and
                /// bytes 4..7 (m) -- so those bytes are only ever |=d
                /// here, never assigned twice.
                if (j < 4) {
                    block.scales[j] = sc[j];
                    block.scales[j + 4] = m[j];
                } else {
                    block.scales[j + 4] = static_cast<uint8_t>((sc[j] & 0x0f) |
                        ((m[j] & 0x0f) << 4));
                    block.scales[j - 4] |= static_cast<uint8_t>(
                        ((sc[j] >> 4) & 0x03) << 6);
                    block.scales[j] |= static_cast<uint8_t>(
                        ((m[j] >> 4) & 0x03) << 6);
                }
                const float step = d * static_cast<float>(sc[j]);
                if (step <= 0.0f) continue;
                const float base = -dmin * static_cast<float>(m[j]);
                for (size_t i = 0; i < kSubSize; ++i) {
                    const size_t col = j * kSubSize + i;
                    const float nominal =
                        block_in[col] / step - base / step;  /// (x-lo)/scale
                    const long q = std::lround(nominal);
                    const int code =
                        static_cast<int>(q < 0 ? 0 : (q > 15 ? 15 : q));
                    const size_t within = col & (kSubSize - 1);
                    uint8_t& byte = block.qs[(j >> 1) * kSubSize + within];
                    if ((j & 1) == 0) {
                        byte = static_cast<uint8_t>((byte & 0xF0) | code);
                    } else {
                        byte = static_cast<uint8_t>((byte & 0x0F) | (code << 4));
                    }
                }
            }
        }
    }
    return result;
}

}
