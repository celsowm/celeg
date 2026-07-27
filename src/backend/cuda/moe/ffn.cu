#include "lfm/runtime/moe.hpp"
#include "lfm/backend/cuda/utils.cuh"

#include <cuda_runtime.h>

namespace lfm {

namespace {

__device__ inline float bf16_to_f(const __nv_bfloat16 v) {
    return __bfloat162float(v);
}
__device__ inline __nv_bfloat16 f_to_bf16(float v) {
    return __float2bfloat16(v);
}

// One block per (token, selected-expert) pair. Computes the SwiGLU FFN for
// that token through the selected expert and atomically accumulates the
// routing-weighted result into the token's FP32 output accumulator.
//
// Improvements over the original scalar slide-rule version:
//   * The token hidden vector is staged into shared memory once per block
//     (after a single cooperative BF16 -> FP32 cast) and reused by every
//     gate_up channel. Down-projection reuses the per-pair `act_out` buffer
//     which is already shared via __syncthreads.
//   * Expert contributions accumulate into an FP32 buffer via atomicAdd,
//     removing the previous BF16 CAS round-after-every-contribution reduce
//     and matching the audit's "reduce in FP32, cast once" recommendation.
//     The caller is responsible for invoking launch_finalize_moe_output to
//     cast the accumulator back into the BF16 output buffer.
// Channel-tiled MoE FFN kernels:
//   1. moe_gate_up_swiglu_tiled_kernel: Grid (pairs, num_inter_tiles).
//      Computes gate and up projections for each output channel tile in parallel,
//      fuses the SwiGLU activation, and writes directly into scratch_activated.
//   2. moe_down_tiled_kernel: Grid (pairs, num_hidden_tiles).
//      Computes down-projection for each output hidden channel tile in parallel
//      and atomically accumulates scaled contributions into output_accum.
__global__ void moe_gate_up_swiglu_tiled_kernel(
    const __nv_bfloat16* gate_up,
    const __nv_bfloat16* const* gate_up_ptrs,
    int num_experts, int inter, int hidden_dim,
    size_t gate_up_stride,
    const int* selected_experts,
    const __nv_bfloat16* hidden,
    int rows, int K,
    __nv_bfloat16* scratch_activated) {
    const int pair = blockIdx.x;
    if (pair >= rows * K) return;
    const int row = pair / K;
    const int expert = selected_experts[pair];
    if (expert < 0 || expert >= num_experts) return;

    const __nv_bfloat16* gu = (gate_up_ptrs != nullptr)
        ? gate_up_ptrs[expert]
        : (gate_up + static_cast<size_t>(expert) * gate_up_stride);
    if (gu == nullptr) return;

    const __nv_bfloat16* token_hidden = hidden + static_cast<size_t>(row) * hidden_dim;
    __nv_bfloat16* act_out = scratch_activated + static_cast<size_t>(pair) * inter;

    const int i = blockIdx.y * blockDim.x + threadIdx.x;

    constexpr int ONE_BLOCK_H = 1024;
    extern __shared__ float smem[];
    float* shared_hidden = smem;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    const __nv_bfloat16* gate_col = (i < inter) ? (gu + static_cast<size_t>(i) * hidden_dim) : nullptr;
    const __nv_bfloat16* up_col = (i < inter) ? (gu + static_cast<size_t>(inter + i) * hidden_dim) : nullptr;

    for (int h_base = 0; h_base < hidden_dim; h_base += ONE_BLOCK_H) {
        for (int h_idx = threadIdx.x; h_idx < ONE_BLOCK_H && (h_base + h_idx) < hidden_dim; h_idx += blockDim.x) {
            shared_hidden[h_idx] = bf16_to_f(token_hidden[h_base + h_idx]);
        }
        __syncthreads();

        if (i < inter) {
            const int chunk_len = min(ONE_BLOCK_H, hidden_dim - h_base);
            for (int h = 0; h < chunk_len; ++h) {
                const float h_val = shared_hidden[h];
                gate_acc += h_val * bf16_to_f(gate_col[h_base + h]);
                up_acc += h_val * bf16_to_f(up_col[h_base + h]);
            }
        }
        __syncthreads();
    }

    if (i < inter) {
        const float silu = gate_acc / (1.0f + expf(-gate_acc));
        act_out[i] = f_to_bf16(silu * up_acc);
    }
}

__global__ void moe_down_tiled_kernel(
    const __nv_bfloat16* down,
    const __nv_bfloat16* const* down_ptrs,
    int num_experts, int inter, int hidden_dim,
    size_t down_stride,
    const int* selected_experts,
    const float* routing_weights,
    float* output_accum,
    int rows, int K,
    const __nv_bfloat16* scratch_activated) {
    const int pair = blockIdx.x;
    if (pair >= rows * K) return;
    const int row = pair / K;
    const int expert = selected_experts[pair];
    if (expert < 0 || expert >= num_experts) return;
    const float rw = routing_weights[pair];

    const __nv_bfloat16* dw = (down_ptrs != nullptr)
        ? down_ptrs[expert]
        : (down + static_cast<size_t>(expert) * down_stride);
    if (dw == nullptr) return;

    const __nv_bfloat16* act_out = scratch_activated + static_cast<size_t>(pair) * inter;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;

    if (h < hidden_dim) {
        float acc = 0.0f;
        const __nv_bfloat16* col = dw + static_cast<size_t>(h) * inter;
        for (int i = 0; i < inter; ++i) {
            acc += bf16_to_f(act_out[i]) * bf16_to_f(col[i]);
        }
        const float contrib = acc * rw;
        atomicAdd(output_accum + static_cast<size_t>(row) * hidden_dim + h, contrib);
    }
}

__global__ void cast_float_to_bf16_kernel(const float* input,
                                          __nv_bfloat16* output, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    output[i] = f_to_bf16(input[i]);
}

} // namespace

void launch_moe_ffn(const MoeFfnDevice& device,
                    const int* selected_experts,
                    const float* routing_weights,
                    const __nv_bfloat16* hidden,
                    float* output_accum,        // [rows * hidden], FP32, caller-zeroed
                    int rows, int K,
                    __nv_bfloat16* scratch_gate_up,
                    __nv_bfloat16* scratch_activated,
                    cudaStream_t stream) {
    const bool indirect =
        device.gate_up_ptrs != nullptr || device.down_ptrs != nullptr;
    if (device.num_experts <= 0 || device.inter <= 0 || device.hidden_dim <= 0) {
        throw std::invalid_argument("invalid MoE FFN device configuration");
    }
    if (indirect) {
        // Indirect mode resolves experts through the pointer tables; both must
        // be present. Per-expert strides are irrelevant here.
        if (device.gate_up_ptrs == nullptr || device.down_ptrs == nullptr) {
            throw std::invalid_argument(
                "MoE FFN offload requires both gate_up_ptrs and down_ptrs");
        }
    } else if (device.expert_gate_up_stride == 0 ||
               device.expert_down_stride == 0) {
        throw std::invalid_argument("invalid MoE FFN device configuration");
    }
    if (rows <= 0 || K <= 0) {
        throw std::invalid_argument("invalid MoE FFN dimensions");
    }
    if (selected_experts == nullptr || routing_weights == nullptr ||
        hidden == nullptr || output_accum == nullptr) {
        throw std::invalid_argument("null MoE FFN pointer");
    }

    const int tile_size = 128;
    const int pairs = rows * K;
    constexpr int ONE_BLOCK_H = 1024;
    const size_t smem_bytes = ONE_BLOCK_H * sizeof(float);

    dim3 grid_gu(pairs, (device.inter + tile_size - 1) / tile_size);
    moe_gate_up_swiglu_tiled_kernel<<<grid_gu, tile_size, smem_bytes, stream>>>(
        device.gate_up, device.gate_up_ptrs,
        device.num_experts, device.inter, device.hidden_dim,
        device.expert_gate_up_stride, selected_experts, hidden,
        rows, K, scratch_activated);
    LFM_KERNEL_CHECK();

    dim3 grid_dw(pairs, (device.hidden_dim + tile_size - 1) / tile_size);
    moe_down_tiled_kernel<<<grid_dw, tile_size, 0, stream>>>(
        device.down, device.down_ptrs,
        device.num_experts, device.inter, device.hidden_dim,
        device.expert_down_stride, selected_experts, routing_weights,
        output_accum, rows, K, scratch_activated);
    LFM_KERNEL_CHECK();
}

void launch_finalize_moe_output(const float* accum,
                                __nv_bfloat16* output,
                                int count,
                                cudaStream_t stream) {
    if (count <= 0) return;
    if (accum == nullptr || output == nullptr) {
        throw std::invalid_argument("null finalize MoE output pointer");
    }
    const int block = 256;
    cast_float_to_bf16_kernel<<<static_cast<unsigned>((count + block - 1) / block), block, 0, stream>>>(
        accum, output, count);
    LFM_KERNEL_CHECK();
}

__global__ void cast_bf16_to_float_kernel(const __nv_bfloat16* input,
                                          float* output, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    output[i] = bf16_to_f(input[i]);
}

void launch_cast_bf16_to_float(const __nv_bfloat16* input,
                                float* output,
                                int count,
                                cudaStream_t stream) {
    if (count <= 0) return;
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("null cast pointer");
    }
    const int block = 256;
    cast_bf16_to_float_kernel<<<static_cast<unsigned>((count + block - 1) / block), block, 0, stream>>>(
        input, output, count);
    LFM_KERNEL_CHECK();
}

} // namespace lfm
