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

// One block per (token, selected-expert) pair. Computes the full SwiGLU FFN
// for that token through the selected expert and atomically accumulates the
// routing-weighted result into the token's output vector.
__global__ void moe_ffn_kernel(const __nv_bfloat16* gate_up,
                               const __nv_bfloat16* down,
                               int num_experts, int inter, int hidden_dim,
                               size_t gate_up_stride, size_t down_stride,
                               const int* selected_experts,
                               const float* routing_weights,
                               const __nv_bfloat16* hidden,
                               __nv_bfloat16* output,
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
    const __nv_bfloat16* gu = gate_up + static_cast<size_t>(expert) * gate_up_stride;
    const __nv_bfloat16* dw = down + static_cast<size_t>(expert) * down_stride;

    __nv_bfloat16* gu_out = scratch_gate_up +
        (static_cast<size_t>(pair) * 2 * inter);
    __nv_bfloat16* act_out = scratch_activated +
        (static_cast<size_t>(pair) * inter);

    // gate_up = hidden @ gu^T   -> [2*inter]
    // Parallel over output channels; each thread reduces over hidden_dim.
    for (int c = threadIdx.x; c < 2 * inter; c += blockDim.x) {
        float acc = 0.0f;
        const __nv_bfloat16* col = gu + static_cast<size_t>(c) * hidden_dim;
        for (int h = 0; h < hidden_dim; ++h) acc += bf16_to_f(token_hidden[h]) * bf16_to_f(col[h]);
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
    for (int h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        float acc = 0.0f;
        const __nv_bfloat16* col = dw + static_cast<size_t>(h) * inter;
        for (int i = 0; i < inter; ++i) acc += bf16_to_f(act_out[i]) * bf16_to_f(col[i]);
        const float contrib = acc * rw;
        __nv_bfloat16* out_slot = const_cast<__nv_bfloat16*>(
            output + static_cast<size_t>(row) * hidden_dim + h);
        float old = bf16_to_f(*out_slot);
        // Atomic add into the token's output accumulator via a bf16 slot CAS.
        while (true) {
            const __nv_bfloat16 expected = f_to_bf16(old);
            const __nv_bfloat16 updated = f_to_bf16(old + contrib);
            const unsigned short exp_bits =
                *reinterpret_cast<const unsigned short*>(&expected);
            const unsigned short new_bits =
                *reinterpret_cast<const unsigned short*>(&updated);
            const unsigned short prev = atomicCAS(
                reinterpret_cast<unsigned short*>(out_slot), exp_bits, new_bits);
            if (prev == exp_bits) break;
            old = bf16_to_f(*out_slot);
        }
    }
}

} // namespace

void launch_moe_ffn(const MoeFfnDevice& device,
                    const int* selected_experts,
                    const float* routing_weights,
                    const __nv_bfloat16* hidden,
                    __nv_bfloat16* output,
                    int rows, int K,
                    __nv_bfloat16* scratch_gate_up,
                    __nv_bfloat16* scratch_activated,
                    cudaStream_t stream) {
    if (device.num_experts <= 0 || device.inter <= 0 || device.hidden_dim <= 0 ||
        device.expert_gate_up_stride == 0 || device.expert_down_stride == 0) {
        throw std::invalid_argument("invalid MoE FFN device configuration");
    }
    if (rows <= 0 || K <= 0) {
        throw std::invalid_argument("invalid MoE FFN dimensions");
    }
    if (selected_experts == nullptr || routing_weights == nullptr ||
        hidden == nullptr || output == nullptr) {
        throw std::invalid_argument("null MoE FFN pointer");
    }

    const int block = (device.inter <= 64 && device.hidden_dim <= 64) ? 64 : 128;
    const int pairs = rows * K;
    moe_ffn_kernel<<<pairs, block, 0, stream>>>(
        device.gate_up, device.down, device.num_experts, device.inter,
        device.hidden_dim, device.expert_gate_up_stride,
        device.expert_down_stride, selected_experts, routing_weights, hidden,
        output, rows, K, scratch_gate_up, scratch_activated);
    LFM_KERNEL_CHECK();
}

} // namespace lfm
