#include "celeg/backend/cuda/weights_loader.hpp"
#include "celeg/backend/cuda/kernels/gguf.cuh"
#include "celeg/checkpoint/gguf_blocks.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <vector>

namespace celeg {

namespace {
// as a fallback when a concatenation mixes quant formats (e.g. attention q/k/v
// where value projection is Q6_K while query/key are Q4_K) and cannot stay
// packed. The block decode mirrors the device kernel in gguf_kernels.cu.
struct Q4KHost { __half d; __half dmin; uint8_t scales[12]; uint8_t qs[128]; };
struct Q6KHost { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; __half d; };

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
    if (ggml_type != GgmlType::Q4_K && ggml_type != GgmlType::Q6_K) {
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
            if (ggml_type == GgmlType::Q4_K) {
                q4k_decode(reinterpret_cast<const Q4KHost*>(row_blocks) + b, within, v);
            } else {
                q6k_decode(reinterpret_cast<const Q6KHost*>(row_blocks) + b, within, v);
            }
            out[static_cast<size_t>(r) * cols + c] = __float2bfloat16(v);
        }
    }
}
} // namespace

void dequantize_gguf_to_bf16(const HostTensorView& tensor,
                             std::vector<__nv_bfloat16>& out) {
    dequantize_gguf_to_bf16_impl(tensor, out);
}

} // namespace celeg
