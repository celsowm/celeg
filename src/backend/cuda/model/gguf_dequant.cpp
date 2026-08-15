#include "celeg/backend/cuda/weights_loader.hpp"
#include "celeg/backend/cuda/kernels/gguf.cuh"
#include "celeg/checkpoint/gguf_blocks.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstring>
#include <stdexcept>
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

void q4_0_decode(const Q4_0Host* blk, int col, float& out) {
    const uint8_t packed = blk->qs[col >> 1];
    const int q = (col & 1) ? (packed >> 4) : (packed & 0x0f);
    out = __half2float(blk->d) * static_cast<float>(q - 8);
}

void q5_0_decode(const Q5_0Host* blk, int col, float& out) {
    const uint8_t packed = blk->qs[col >> 1];
    const int low = (col & 1) ? (packed >> 4) : (packed & 0x0f);
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

void dequantize_gguf_to_bf16_impl(const HostTensorView& tensor,
                                  std::vector<__nv_bfloat16>& out) {
    const GgmlType ggml_type = ggml_type_from_block_encoding(tensor.block_encoding);
    if (ggml_type != GgmlType::Q2_K && ggml_type != GgmlType::Q3_K &&
        ggml_type != GgmlType::Q4_0 && ggml_type != GgmlType::Q4_K && ggml_type != GgmlType::Q5_0 &&
        ggml_type != GgmlType::Q5_K && ggml_type != GgmlType::Q6_K &&
        ggml_type != GgmlType::Q8_0) {
        throw std::runtime_error("unsupported GGUF quantization for CUDA dequantization");
    }
    const int rows = static_cast<int>(tensor.shape[0]);
    const int cols = static_cast<int>(tensor.shape[1]);
    out.resize(static_cast<size_t>(rows) * cols);
    const GgmlTypeTrait trait = ggml_type_trait(ggml_type);
    const int blocks_per_row = cols / trait.block_size;
    const size_t row_bytes = static_cast<size_t>(blocks_per_row) * trait.type_size;
    for (int r = 0; r < rows; ++r) {
        const uint8_t* row_blocks =
            reinterpret_cast<const uint8_t*>(tensor.data) + static_cast<size_t>(r) * row_bytes;
        for (int c = 0; c < cols; ++c) {
            const int b = c / trait.block_size;
            const int within = c % trait.block_size;
            float v = 0.0f;
            if (ggml_type == GgmlType::Q2_K) {
                q2k_decode(reinterpret_cast<const Q2KHost*>(row_blocks) + b, within, v);
            } else if (ggml_type == GgmlType::Q3_K) {
                const auto* block = reinterpret_cast<const Q3KHost*>(row_blocks) + b;
                int8_t scales[16]{};
                q3k_scales(block, scales);
                v = __half2float(block->d) * static_cast<float>(scales[within / 16]) *
                    static_cast<float>(q3k_value(block, within));
            } else if (ggml_type == GgmlType::Q4_0) {
                q4_0_decode(reinterpret_cast<const Q4_0Host*>(row_blocks) + b, within, v);
            } else if (ggml_type == GgmlType::Q4_K) {
                q4k_decode(reinterpret_cast<const Q4KHost*>(row_blocks) + b, within, v);
            } else if (ggml_type == GgmlType::Q5_0) {
                q5_0_decode(reinterpret_cast<const Q5_0Host*>(row_blocks) + b, within, v);
            } else if (ggml_type == GgmlType::Q5_K) {
                q5k_decode(reinterpret_cast<const Q5KHost*>(row_blocks) + b, within, v);
            } else if (ggml_type == GgmlType::Q6_K) {
                q6k_decode(reinterpret_cast<const Q6KHost*>(row_blocks) + b, within, v);
            } else {
                q8_0_decode(reinterpret_cast<const Q8_0Host*>(row_blocks) + b, within, v);
            }
            out[static_cast<size_t>(r) * cols + c] = __float2bfloat16(v);
        }
    }
}
}

void dequantize_gguf_to_bf16(const HostTensorView& tensor,
                             std::vector<__nv_bfloat16>& out) {
    dequantize_gguf_to_bf16_impl(tensor, out);
}

}
