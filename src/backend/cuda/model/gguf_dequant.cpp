#include "celeg/backend/cuda/weights_loader.hpp"
#include "celeg/backend/cuda/kernels/gguf.cuh"
#include "celeg/checkpoint/gguf_blocks.hpp"
#include "celeg/checkpoint/gguf_iq.hpp"
#include "celeg/model/weights/quantization.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace celeg {

namespace {
struct Q4KHost { __half d; __half dmin; uint8_t scales[12]; uint8_t qs[128]; };
struct Q5KHost { __half d; __half dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
struct Q2KHost { uint8_t scales[16]; uint8_t qs[64]; __half d; __half dmin; };
struct Q3KHost { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; __half d; };
struct Q6KHost { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; __half d; };
struct Q5_0Host { __half d; uint8_t qh[4]; uint8_t qs[16]; };
struct Q8_0Host { __half d; int8_t qs[32]; };
struct Q4_0Host { __half d; uint8_t qs[16]; };
struct Q4_1Host { __half d; __half m; uint8_t qs[16]; };

/// The shared IQ block structs store the scale as raw fp16 bits, since they
/// are compiled for the CPU backend too and cannot depend on __half.
float half_bits_to_float(std::uint16_t bits) { return fp16_bits_to_float(bits); }

void q4_0_decode(const Q4_0Host* blk, int col, float& out) {
    // GGML packs each 32-element block as two halves, not interleaved
    // pairs: qs[j] holds element j in its low nibble and element j+16 in
    // its high nibble (j in [0,16)).
    const uint8_t packed = blk->qs[col & 15];
    const int q = (col < 16) ? (packed & 0x0f) : (packed >> 4);
    out = __half2float(blk->d) * static_cast<float>(q - 8);
}

void q4_1_decode(const Q4_1Host* blk, int col, float& out) {
    // Same split-half nibble layout as Q4_0, but the quants are unsigned and
    // the block carries an explicit minimum instead of the implicit -8 bias.
    const uint8_t packed = blk->qs[col & 15];
    const int q = (col < 16) ? (packed & 0x0f) : (packed >> 4);
    out = __half2float(blk->d) * static_cast<float>(q) + __half2float(blk->m);
}

void q5_0_decode(const Q5_0Host* blk, int col, float& out) {
    // Same split-half nibble layout as Q4_0; the high (5th) bit of every
    // element lives at bit `col` of qh.
    const uint8_t packed = blk->qs[col & 15];
    const int low = (col < 16) ? (packed & 0x0f) : (packed >> 4);
    const uint32_t high_bits = static_cast<uint32_t>(blk->qh[0]) |
        (static_cast<uint32_t>(blk->qh[1]) << 8) |
        (static_cast<uint32_t>(blk->qh[2]) << 16) |
        (static_cast<uint32_t>(blk->qh[3]) << 24);
    const int q = low | (((high_bits >> col) & 1u) << 4);
    out = __half2float(blk->d) * static_cast<float>(q - 16);
}

void q8_0_decode(const Q8_0Host* blk, int col, float& out) {
    out = __half2float(blk->d) * static_cast<float>(blk->qs[col]);
}

void q4k_decode(const Q4KHost* blk, int col, float& out) {
    const float d = __half2float(blk->d);
    const float dmin = __half2float(blk->dmin);
    const int sub = col >> 5, within = col & 31;
    uint8_t sc, m;
    celeg::gguf_blocks::q4k_scale_min(sub, blk->scales, sc, m);
    const uint8_t* qs = blk->qs + (sub >> 1) * 32;
    const int q = (sub & 1) ? (qs[within] >> 4) : (qs[within] & 0xF);
    out = d * sc * static_cast<float>(q) - dmin * m;
}

void q5k_decode(const Q5KHost* blk, int col, float& out) {
    const int sub = col >> 5, within = col & 31;
    uint8_t scale = 0, minimum = 0;
    celeg::gguf_blocks::q4k_scale_min(sub, blk->scales, scale, minimum);
    const uint8_t packed = blk->qs[(sub >> 1) * 32 + within];
    const int low = (sub & 1) ? (packed >> 4) : (packed & 0x0f);
    const int high = (blk->qh[within] >> sub) & 1;
    out = __half2float(blk->d) * static_cast<float>(scale * (low | (high << 4))) -
          __half2float(blk->dmin) * static_cast<float>(minimum);
}

int q2k_value(const Q2KHost* blk, int col) {
    const int half = col / 128;
    const int local = col % 128;
    const int lane = local % 32;
    const int shift = (local / 32) * 2;
    return (blk->qs[half * 32 + lane] >> shift) & 3;
}

void q2k_decode(const Q2KHost* blk, int col, float& out) {
    const int sub = col / 16;
    out = __half2float(blk->d) * static_cast<float>(blk->scales[sub] & 0x0f) *
              static_cast<float>(q2k_value(blk, col)) -
          __half2float(blk->dmin) * static_cast<float>(blk->scales[sub] >> 4);
}

void q3k_scales(const Q3KHost* blk, int8_t* output) {
    uint32_t words[4]{};
    uint32_t source[3]{};
    std::memcpy(source, blk->scales, sizeof(blk->scales));
    constexpr uint32_t low_nibble = 0x0f0f0f0f;
    constexpr uint32_t high_bits = 0x03030303;
    const uint32_t packed = source[2];
    words[2] = ((source[0] >> 4) & low_nibble) |
               (((packed >> 4) & high_bits) << 4);
    words[3] = ((source[1] >> 4) & low_nibble) |
               (((packed >> 6) & high_bits) << 4);
    words[0] = (source[0] & low_nibble) |
               (((packed >> 0) & high_bits) << 4);
    words[1] = (source[1] & low_nibble) |
               (((packed >> 2) & high_bits) << 4);
    for (int i = 0; i < 16; ++i) {
        output[i] = static_cast<int8_t>(
            reinterpret_cast<const uint8_t*>(words)[i]) - 32;
    }
}

int q3k_value(const Q3KHost* blk, int col) {
    const int half = col / 128;
    const int local = col % 128;
    const int lane = local % 32;
    const int shift = (local / 32) * 2;
    const int high = (blk->hmask[lane] & (1u << (local / 32))) ? 0 : 4;
    return ((blk->qs[half * 32 + lane] >> shift) & 3) - high;
}
void q6k_decode(const Q6KHost* blk, int col, float& out) {
    const float d = __half2float(blk->d);
    const int half = col >> 7, idx = col & 127;
    const int n = idx & 31, grp = idx >> 5;
    const uint8_t* ql = blk->ql + half * 64;
    const uint8_t* qh = blk->qh + half * 32;
    int q;
    if (grp == 0) q = (ql[n] & 0xF) | (((qh[n]) & 3) << 4);
    else if (grp == 1) q = (ql[n + 32] & 0xF) | (((qh[n] >> 2) & 3) << 4);
    else if (grp == 2) q = (ql[n] >> 4) | (((qh[n] >> 4) & 3) << 4);
    else q = (ql[n + 32] >> 4) | (((qh[n] >> 6) & 3) << 4);
    const int is = half * 8 + grp * 2 + (n >> 4);
    out = d * static_cast<float>(blk->scales[is]) * static_cast<float>(q - 32);
}

/// Decodes one whole block. Working per block rather than per element is
/// what keeps loading a multi-billion-parameter checkpoint tolerable: the
/// type dispatch happens once per block instead of once per weight, and
/// per-block setup such as Q3_K's scale expansion is computed once rather
/// than re-derived for all 256 elements.
using BlockDecoder = void (*)(const uint8_t* block, float* out);

template <typename Block, void (*Decode)(const Block*, int, float&)>
void decode_uniform(const uint8_t* block, float* out) {
    const auto* typed = reinterpret_cast<const Block*>(block);
    constexpr int elements = std::is_same_v<Block, Q4_0Host> ||
                             std::is_same_v<Block, Q4_1Host> ||
                             std::is_same_v<Block, Q5_0Host> ||
                             std::is_same_v<Block, Q8_0Host> ? 32 : 256;
    for (int i = 0; i < elements; ++i) Decode(typed, i, out[i]);
}

void decode_q3k(const uint8_t* block, float* out) {
    const auto* typed = reinterpret_cast<const Q3KHost*>(block);
    int8_t scales[16]{};
    q3k_scales(typed, scales);
    const float d = __half2float(typed->d);
    for (int i = 0; i < 256; ++i) {
        out[i] = d * static_cast<float>(scales[i / 16]) *
                 static_cast<float>(q3k_value(typed, i));
    }
}

// The IQ decoders are shared with the CPU kernels rather than reimplemented
// here; they take the raw fp16 scale, so the conversion is the only
// backend-specific part.

void decode_iq2s(const uint8_t* block, float* out) {
    const auto& iq = *reinterpret_cast<const gguf_iq::BlockIq2S*>(block);
    const float d = half_bits_to_float(iq.d);
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        for (int half = 0; half < 2; ++half) {
            const float scale = d * gguf_iq::iq2s_sub_scale(iq, ib32, half);
            for (int i = 0; i < 16; ++i) {
                const int col = ib32 * 32 + half * 16 + i;
                out[col] = scale * static_cast<float>(gguf_iq::iq2s_value(iq, col));
            }
        }
    }
}

void decode_iq3xxs(const uint8_t* block, float* out) {
    const auto& iq = *reinterpret_cast<const gguf_iq::BlockIq3XXS*>(block);
    const float d = half_bits_to_float(iq.d);
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const float scale = d * gguf_iq::iq3xxs_sub_scale(gguf_iq::iq3xxs_aux(iq, ib32));
        for (int i = 0; i < 32; ++i) {
            const int col = ib32 * 32 + i;
            out[col] = scale * static_cast<float>(gguf_iq::iq3xxs_value(iq, col));
        }
    }
}

void decode_iq3s(const uint8_t* block, float* out) {
    const auto& iq = *reinterpret_cast<const gguf_iq::BlockIq3S*>(block);
    const float d = half_bits_to_float(iq.d);
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const float scale = d * gguf_iq::iq3s_sub_scale(iq, ib32);
        for (int i = 0; i < 32; ++i) {
            const int col = ib32 * 32 + i;
            out[col] = scale * static_cast<float>(gguf_iq::iq3s_value(iq, col));
        }
    }
}

void decode_iq4xs(const uint8_t* block, float* out) {
    const auto& iq = *reinterpret_cast<const gguf_iq::BlockIq4XS*>(block);
    const float d = half_bits_to_float(iq.d);
    for (int ib = 0; ib < 8; ++ib) {
        const float scale = d * gguf_iq::iq4xs_sub_scale(iq, ib);
        for (int i = 0; i < 32; ++i) {
            const int col = ib * 32 + i;
            out[col] = scale * static_cast<float>(gguf_iq::iq4xs_value(iq, col));
        }
    }
}

void decode_iq4nl(const uint8_t* block, float* out) {
    const auto& iq = *reinterpret_cast<const gguf_iq::BlockIq4NL*>(block);
    const float d = half_bits_to_float(iq.d);
    for (int i = 0; i < 32; ++i) {
        out[i] = d * static_cast<float>(gguf_iq::iq4nl_value(iq, i));
    }
}

/// Resolved once per tensor. The switch is exhaustive on purpose: an
/// unhandled type must throw here rather than fall into a neighbour's
/// decoder, which is what a bare `else` branch used to allow.
BlockDecoder select_block_decoder(GgmlType type) {
    switch (type) {
        case GgmlType::Q2_K: return decode_uniform<Q2KHost, q2k_decode>;
        case GgmlType::Q3_K: return decode_q3k;
        case GgmlType::Q4_0: return decode_uniform<Q4_0Host, q4_0_decode>;
        case GgmlType::Q4_1: return decode_uniform<Q4_1Host, q4_1_decode>;
        case GgmlType::Q4_K: return decode_uniform<Q4KHost, q4k_decode>;
        case GgmlType::Q5_0: return decode_uniform<Q5_0Host, q5_0_decode>;
        case GgmlType::Q5_K: return decode_uniform<Q5KHost, q5k_decode>;
        case GgmlType::Q6_K: return decode_uniform<Q6KHost, q6k_decode>;
        case GgmlType::Q8_0: return decode_uniform<Q8_0Host, q8_0_decode>;
        case GgmlType::IQ2_S: return decode_iq2s;
        case GgmlType::IQ3_XXS: return decode_iq3xxs;
        case GgmlType::IQ3_S: return decode_iq3s;
        case GgmlType::IQ4_XS: return decode_iq4xs;
        case GgmlType::IQ4_NL: return decode_iq4nl;
        default:
            throw std::runtime_error(
                std::string("no CUDA host dequantizer for GGUF type ") +
                ggml_type_name(type));
    }
}

void dequantize_gguf_to_bf16_impl(const HostTensorView& tensor,
                                  std::vector<__nv_bfloat16>& out) {
    const GgmlType ggml_type = ggml_type_from_block_encoding(tensor.block_encoding);
    if (!ggml_type_support(ggml_type).cuda_dequantize) {
        throw std::runtime_error(
            std::string("unsupported GGUF quantization for CUDA dequantization: ") +
            ggml_type_name(ggml_type));
    }
    const int rows = static_cast<int>(tensor.shape[0]);
    const int cols = static_cast<int>(tensor.shape[1]);
    out.resize(static_cast<size_t>(rows) * cols);
    const GgmlTypeTrait trait = ggml_type_trait(ggml_type);
    if (cols % trait.block_size != 0) {
        throw std::runtime_error("GGUF tensor width is not block-aligned for dequantization");
    }
    const BlockDecoder decode = select_block_decoder(ggml_type);
    const int blocks_per_row = cols / trait.block_size;
    const size_t row_bytes = static_cast<size_t>(blocks_per_row) * trait.type_size;
    std::vector<float> decoded(static_cast<size_t>(trait.block_size));
    for (int r = 0; r < rows; ++r) {
        const uint8_t* row_blocks =
            reinterpret_cast<const uint8_t*>(tensor.data) + static_cast<size_t>(r) * row_bytes;
        __nv_bfloat16* row_out = out.data() + static_cast<size_t>(r) * cols;
        for (int b = 0; b < blocks_per_row; ++b) {
            decode(row_blocks + static_cast<size_t>(b) * trait.type_size, decoded.data());
            for (int i = 0; i < trait.block_size; ++i) {
                row_out[b * trait.block_size + i] = __float2bfloat16(decoded[i]);
            }
        }
    }
}
}

void dequantize_gguf_to_bf16(const HostTensorView& tensor,
                             std::vector<__nv_bfloat16>& out) {
    dequantize_gguf_to_bf16_impl(tensor, out);
}

}
