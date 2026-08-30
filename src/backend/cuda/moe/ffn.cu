#include "backend/cuda/moe.hpp"
#include "utils.cuh"
#include "celeg/checkpoint/gguf_blocks.hpp"

#include <cuda_runtime.h>

namespace celeg {

namespace {

#pragma pack(push, 1)
struct GgufQ4KBlock {
    __half d;
    __half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct GgufQ6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    __half d;
};
#pragma pack(pop)

__device__ __forceinline__ float gguf_q4k_value(const GgufQ4KBlock* block,
                                                 int col) {
    uint8_t scale = 0, minimum = 0;
    celeg::gguf_blocks::q4k_scale_min(col >> 5, block->scales, scale, minimum);
    const uint8_t packed = block->qs[(col >> 6) * 32 + (col & 31)];
    const int q = (col & 32) != 0 ? packed >> 4 : packed & 15;
    return __half2float(block->d) * static_cast<float>(scale * q) -
        __half2float(block->dmin) * static_cast<float>(minimum);
}

__device__ __forceinline__ float gguf_q6k_value(const GgufQ6KBlock* block,
                                                 int col) {
    const int half = col >> 7;
    const int index = col & 127;
    const int lane = index & 31;
    const int group = index >> 5;
    const uint8_t* ql = block->ql + half * 64;
    const uint8_t* qh = block->qh + half * 32;
    int q = 0;
    if (group == 0) q = (ql[lane] & 15) | ((qh[lane] & 3) << 4);
    else if (group == 1) q = (ql[lane + 32] & 15) | (((qh[lane] >> 2) & 3) << 4);
    else if (group == 2) q = (ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4);
    else q = (ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4);
    const int scale_index = half * 8 + group * 2 + (lane >> 4);
    return __half2float(block->d) * static_cast<float>(block->scales[scale_index]) *
        static_cast<float>(q - 32);
}

__device__ __forceinline__ float moe_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

template <typename BlockT, float (*ValueFn)(const BlockT*, int)>
__device__ __forceinline__ float gguf_expert_dot_warp(
    const uint8_t* row_blocks, const __nv_bfloat16* input, int cols,
    size_t row_bytes) {
    float sum = 0.0f;
    const int lane = threadIdx.x & 31;
    const int blocks = cols / 256;
    for (int block = 0; block < blocks; ++block) {
        const BlockT* packed = reinterpret_cast<const BlockT*>(
            row_blocks + static_cast<size_t>(block) * sizeof(BlockT));
        const int base = block * 256;
        for (int col = lane; col < 256; col += 32) {
            sum += __bfloat162float(input[base + col]) * ValueFn(packed, col);
        }
    }
    (void)row_bytes;
    return moe_warp_sum(sum);
}

template <typename BlockT, float (*ValueFn)(const BlockT*, int)>
__global__ void moe_gguf_gate_up_kernel(
    const uint8_t* gate_up, GgmlType gate_up_type, size_t expert_stride,
    size_t row_bytes, int num_experts, int inter, int hidden_dim,
    const int* selected_experts, const __nv_bfloat16* hidden,
    int rows, int K, __nv_bfloat16* activated) {
    const int pair = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int channel = blockIdx.y * (blockDim.x / 32) + warp;
    if (pair >= rows * K || channel >= inter) return;
    const int expert = selected_experts[pair];
    if (expert < 0 || expert >= num_experts) return;
    const uint8_t* expert_base = gate_up +
        static_cast<size_t>(expert) * expert_stride;
    const uint8_t* gate_row = expert_base +
        static_cast<size_t>(channel) * row_bytes;
    const uint8_t* up_row = expert_base +
        static_cast<size_t>(inter + channel) * row_bytes;
    const __nv_bfloat16* input = hidden + static_cast<size_t>(pair / K) * hidden_dim;
    const float gate = gguf_expert_dot_warp<BlockT, ValueFn>(
        gate_row, input, hidden_dim, row_bytes);
    const float up = gguf_expert_dot_warp<BlockT, ValueFn>(
        up_row, input, hidden_dim, row_bytes);
    if (lane == 0) {
        activated[static_cast<size_t>(pair) * inter + channel] =
            __float2bfloat16(moe_sigmoid(gate) * up);
    }
    (void)gate_up_type;
}

template <typename BlockT, float (*ValueFn)(const BlockT*, int)>
__global__ void moe_gguf_down_kernel(
    const uint8_t* down, size_t expert_stride, size_t row_bytes,
    int num_experts, int inter, int hidden_dim, const int* selected_experts,
    const float* routing_weights, const __nv_bfloat16* activated,
    float* output_accum, int rows, int K) {
    const int pair = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int channel = blockIdx.y * (blockDim.x / 32) + warp;
    if (pair >= rows * K || channel >= hidden_dim) return;
    const int expert = selected_experts[pair];
    if (expert < 0 || expert >= num_experts) return;
    const uint8_t* row = down + static_cast<size_t>(expert) * expert_stride +
        static_cast<size_t>(channel) * row_bytes;
    const __nv_bfloat16* input = activated + static_cast<size_t>(pair) * inter;
    const float value = gguf_expert_dot_warp<BlockT, ValueFn>(
        row, input, inter, row_bytes) * routing_weights[pair];
    if (lane == 0) {
        atomicAdd(output_accum + static_cast<size_t>(pair / K) * hidden_dim + channel,
                  value);
    }
}

void launch_moe_gguf_ffn(const MoeFfnDevice& device,
                         const int* selected_experts,
                         const float* routing_weights,
                         const __nv_bfloat16* hidden,
                         float* output_accum, int rows, int K,
                         __nv_bfloat16* scratch_activated,
                         cudaStream_t stream) {
    const int pairs = rows * K;
    constexpr int block_size = 256;
    const int warps_per_block = block_size / 32;
    const dim3 gate_grid(pairs, (device.inter + warps_per_block - 1) / warps_per_block);
    const dim3 block(block_size);
    if (device.gate_up_gguf_type == GgmlType::Q4_K) {
        moe_gguf_gate_up_kernel<GgufQ4KBlock, gguf_q4k_value>
            <<<gate_grid, block, 0, stream>>>(
                device.gate_up_gguf, device.gate_up_gguf_type,
                device.expert_gate_up_byte_stride, device.expert_gate_up_row_bytes,
                device.num_experts, device.inter, device.hidden_dim,
                selected_experts, hidden, rows, K, scratch_activated);
    } else {
        moe_gguf_gate_up_kernel<GgufQ6KBlock, gguf_q6k_value>
            <<<gate_grid, block, 0, stream>>>(
                device.gate_up_gguf, device.gate_up_gguf_type,
                device.expert_gate_up_byte_stride, device.expert_gate_up_row_bytes,
                device.num_experts, device.inter, device.hidden_dim,
                selected_experts, hidden, rows, K, scratch_activated);
    }
    CELEG_KERNEL_CHECK();

    const dim3 down_grid(pairs, (device.hidden_dim + warps_per_block - 1) / warps_per_block);
    if (device.down_gguf_type == GgmlType::Q4_K) {
        moe_gguf_down_kernel<GgufQ4KBlock, gguf_q4k_value>
            <<<down_grid, block, 0, stream>>>(
                device.down_gguf, device.expert_down_byte_stride,
                device.expert_down_row_bytes, device.num_experts, device.inter,
                device.hidden_dim, selected_experts, routing_weights,
                scratch_activated, output_accum, rows, K);
    } else {
        moe_gguf_down_kernel<GgufQ6KBlock, gguf_q6k_value>
            <<<down_grid, block, 0, stream>>>(
                device.down_gguf, device.expert_down_byte_stride,
                device.expert_down_row_bytes, device.num_experts, device.inter,
                device.hidden_dim, selected_experts, routing_weights,
                scratch_activated, output_accum, rows, K);
    }
    CELEG_KERNEL_CHECK();
}

__device__ inline float bf16_to_f(const __nv_bfloat16 v) {
    return __bfloat162float(v);
}
__device__ inline __nv_bfloat16 f_to_bf16(float v) {
    return __float2bfloat16(v);
}

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

}

void launch_moe_ffn(const MoeFfnDevice& device,
                    const int* selected_experts,
                    const float* routing_weights,
                    const __nv_bfloat16* hidden,
                    float* output_accum,
                    int rows, int K,
                    __nv_bfloat16* scratch_gate_up,
                    __nv_bfloat16* scratch_activated,
                    cudaStream_t stream) {
    const bool indirect =
        device.gate_up_ptrs != nullptr || device.down_ptrs != nullptr;
    const bool native_gguf =
        device.gate_up_gguf != nullptr || device.down_gguf != nullptr;
    if (device.num_experts <= 0 || device.inter <= 0 || device.hidden_dim <= 0) {
        throw std::invalid_argument("invalid MoE FFN device configuration");
    }
    if (indirect) {
        if (device.gate_up_ptrs == nullptr || device.down_ptrs == nullptr) {
            throw std::invalid_argument(
                "MoE FFN offload requires both gate_up_ptrs and down_ptrs");
        }
    } else if (!native_gguf && (device.expert_gate_up_stride == 0 ||
               device.expert_down_stride == 0)) {
        throw std::invalid_argument("invalid MoE FFN device configuration");
    }
    if (rows <= 0 || K <= 0) {
        throw std::invalid_argument("invalid MoE FFN dimensions");
    }
    if (selected_experts == nullptr || routing_weights == nullptr ||
        hidden == nullptr || output_accum == nullptr) {
        throw std::invalid_argument("null MoE FFN pointer");
    }

    if (device.gate_up_gguf != nullptr || device.down_gguf != nullptr) {
        if (device.gate_up_gguf == nullptr || device.down_gguf == nullptr ||
            (device.gate_up_gguf_type != GgmlType::Q4_K &&
             device.gate_up_gguf_type != GgmlType::Q6_K) ||
            (device.down_gguf_type != GgmlType::Q4_K &&
             device.down_gguf_type != GgmlType::Q6_K) ||
            device.expert_gate_up_row_bytes == 0 ||
            device.expert_down_row_bytes == 0 ||
            device.expert_gate_up_byte_stride == 0 ||
            device.expert_down_byte_stride == 0) {
            throw std::invalid_argument("invalid native GGUF MoE expert storage");
        }
        launch_moe_gguf_ffn(device, selected_experts, routing_weights, hidden,
                            output_accum, rows, K, scratch_activated, stream);
        return;
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
    CELEG_KERNEL_CHECK();

    dim3 grid_dw(pairs, (device.hidden_dim + tile_size - 1) / tile_size);
    moe_down_tiled_kernel<<<grid_dw, tile_size, 0, stream>>>(
        device.down, device.down_ptrs,
        device.num_experts, device.inter, device.hidden_dim,
        device.expert_down_stride, selected_experts, routing_weights,
        output_accum, rows, K, scratch_activated);
    CELEG_KERNEL_CHECK();
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
    CELEG_KERNEL_CHECK();
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
    CELEG_KERNEL_CHECK();
}

}
