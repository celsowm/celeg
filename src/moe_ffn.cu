#include "lfm/moe.hpp"
#include "lfm/cuda_utils.cuh"

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
__global__ void moe_ffn_kernel(const __nv_bfloat16* gate_up,
                                const __nv_bfloat16* down,
                                const __nv_bfloat16* const* gate_up_ptrs,
                                const __nv_bfloat16* const* down_ptrs,
                                int num_experts, int inter, int hidden_dim,
                                size_t gate_up_stride, size_t down_stride,
                                const int* selected_experts,
                                const float* routing_weights,
                                const __nv_bfloat16* hidden,
                                float* output_accum,
                                int rows, int K,
                                __nv_bfloat16* scratch_gate_up,
                                __nv_bfloat16* scratch_activated) {
    // Block index encodes (token * K + k).
    const int pair = blockIdx.x;
    if (pair >= rows * K) return;
    const int row = pair / K;
    const int expert = selected_experts[pair];
    if (expert < 0 || expert >= num_experts) return;
    const float rw = routing_weights[pair];

    const __nv_bfloat16* token_hidden = hidden + static_cast<size_t>(row) * hidden_dim;
    // Resolve expert weights: through the per-expert indirection tables when
    // offload is active, otherwise contiguously from the base pointers.
    const __nv_bfloat16* gu;
    const __nv_bfloat16* dw;
    if (gate_up_ptrs != nullptr) {
        gu = gate_up_ptrs[expert];
        dw = down_ptrs[expert];
        if (gu == nullptr || dw == nullptr) return;
    } else {
        gu = gate_up + static_cast<size_t>(expert) * gate_up_stride;
        dw = down + static_cast<size_t>(expert) * down_stride;
    }

    __nv_bfloat16* gu_out = scratch_gate_up +
        (static_cast<size_t>(pair) * 2 * inter);
    __nv_bfloat16* act_out = scratch_activated +
        (static_cast<size_t>(pair) * inter);

    // Stage the BF16 hidden vector into shared memory as FP32 once per block.
    // All gate_up channels read from this shared tile instead of issuing
    // global loads for every output channel. Down-projection reads `act_out`
    // which is already produced cooperatively in this block, so it does not
    // need additional shared staging.  Hidden vectors up to a few thousand
    // elements fit comfortably in shared memory for the model shapes we
    // target (LFM2.5-8B-A1B: H=2048, ONE_BLOCK_H<=1024 here); larger vectors
    // tile through the buffer in chunks of ONE_BLOCK_H.
    constexpr int ONE_BLOCK_H = 1024;
    extern __shared__ float smem[];
    float* shared_hidden = smem;       // [ONE_BLOCK_H]
    // The down-projection accumulator is reused across threads via atomicAdd;
    // no extra shared memory is required for that pass.

    // gate_up = hidden @ gu^T   -> [2*inter]
    // For each output channel c, the inner product walks `hidden_dim` BF16
    // elements. We tile the hidden vector through shared memory in chunks of
    // ONE_BLOCK_H so this works for H > 1024.
    for (int c = threadIdx.x; c < 2 * inter; c += blockDim.x) {
        float acc = 0.0f;
        const __nv_bfloat16* col = gu + static_cast<size_t>(c) * hidden_dim;
        for (int h_base = 0; h_base < hidden_dim; h_base += ONE_BLOCK_H) {
            // Cooperatively load+cast the hidden tile for this chunk.
            for (int i = threadIdx.x; i < ONE_BLOCK_H && (h_base + i) < hidden_dim; i += blockDim.x) {
                shared_hidden[i] = bf16_to_f(token_hidden[h_base + i]);
            }
            __syncthreads();
            const int chunk_len = min(ONE_BLOCK_H, hidden_dim - h_base);
            for (int h = 0; h < chunk_len; ++h) {
                acc += shared_hidden[h] * bf16_to_f(col[h_base + h]);
            }
            // All threads must finish reading the shared tile before the next
            // chunk overwrites it. Avoid racing with the channel loop above.
            __syncthreads();
        }
        gu_out[c] = f_to_bf16(acc);
    }
    __syncthreads();

    // SwiGLU: activated[i] = silu(gate[i]) * up[i]
    for (int i = threadIdx.x; i < inter; i += blockDim.x) {
        const float gate = bf16_to_f(gu_out[i]);
        const float up = bf16_to_f(gu_out[inter + i]);
        const float silu = gate / (1.0f + expf(-gate));
        act_out[i] = f_to_bf16(silu * up);
    }
    __syncthreads();

    // down = activated @ dw^T   -> [hidden]
    // Each thread reduces over `inter` BF16 activations to produce one element
    // of the hidden output, scales by the routing weight, and atomically adds
    // to the FP32 accumulator for this token row.
    for (int h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        float acc = 0.0f;
        const __nv_bfloat16* col = dw + static_cast<size_t>(h) * inter;
        for (int i = 0; i < inter; ++i) acc += bf16_to_f(act_out[i]) * bf16_to_f(col[i]);
        const float contrib = acc * rw;
        // atomicAdd on FP32 is native on every supported CC; no CAS loop, no
        // per-contribution BF16 round. The caller casts to BF16 once.
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

    const int block = (device.inter <= 64 && device.hidden_dim <= 64) ? 64 : 128;
    const int pairs = rows * K;
    // Shared memory holds `ONE_BLOCK_H` floats for the hidden tile staging.
    // For H<=1024 (covers all LFM2.5 hidden sizes), this is 4 KB.
    constexpr int ONE_BLOCK_H = 1024;
    const size_t smem_bytes = ONE_BLOCK_H * sizeof(float);
    moe_ffn_kernel<<<pairs, block, smem_bytes, stream>>>(
        device.gate_up, device.down, device.gate_up_ptrs, device.down_ptrs,
        device.num_experts, device.inter,
        device.hidden_dim, device.expert_gate_up_stride,
        device.expert_down_stride, selected_experts, routing_weights, hidden,
        output_accum, rows, K, scratch_gate_up, scratch_activated);
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
