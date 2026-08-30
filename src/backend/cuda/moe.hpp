#pragma once

#include <cstdint>
#include <cmath>
#include <vector>

#include <celeg/checkpoint/formats/gguf.hpp>

#if defined(__CUDACC__) || defined(__CUDA_BF16_H__)
#include <cuda_bf16.h>
using cudaStream_t = struct CUstream_st*;
#else
struct __opencode_bf16 {
    uint16_t value = 0;
};
using __nv_bfloat16 = __opencode_bf16;
using cudaStream_t = struct CUstream_st*;
#endif

namespace celeg {

#if defined(__CUDACC__)
__host__ __device__
#endif
inline float moe_sigmoid(float x) {
#if defined(__CUDACC__)
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
#else
    if (x >= 0.0f) return 1.0f / (1.0f + std::exp(-x));
    const float e = std::exp(x);
    return e / (1.0f + e);
#endif
}

struct MoeRouterConfig {
    int num_experts = 0;
    int experts_per_token = 0;
    bool normalize_topk = false;
    bool softmax = false;
    bool use_expert_bias = false;
    float routed_scaling_factor = 1.0f;
    int group_count = 0;
    int experts_per_group = 0;
    int groups_per_token = 0;
    int group_score_top_k = 0;
};

struct MoeRouterDevice {
    const float* router_weight = nullptr;
    const float* expert_bias = nullptr;
    const float* hidden_data = nullptr;
    int* selected_experts = nullptr;
    float* routing_weights = nullptr;
    int rows = 0;
    int hidden_dim = 0;
};

void launch_moe_router(const MoeRouterDevice& device,
                       const MoeRouterConfig& cfg,
                       float* scratch_logits,
                       cudaStream_t stream);

void compute_moe_router(const std::vector<float>& hidden,
                        const std::vector<float>& router_weight,
                        const std::vector<float>* expert_bias,
                        int rows, int hidden_dim,
                        const MoeRouterConfig& config,
                        std::vector<int>& selected_experts,
                        std::vector<float>& routing_weights);

struct MoeFfnDevice {
    const __nv_bfloat16* gate_up = nullptr;
    const __nv_bfloat16* down = nullptr;
    int num_experts = 0;
    int inter = 0;
    int hidden_dim = 0;
    size_t expert_gate_up_stride = 0;
    size_t expert_down_stride = 0;

    const uint8_t* gate_up_gguf = nullptr;
    const uint8_t* down_gguf = nullptr;
    GgmlType gate_up_gguf_type = GgmlType::Unknown;
    GgmlType down_gguf_type = GgmlType::Unknown;
    size_t expert_gate_up_row_bytes = 0;
    size_t expert_down_row_bytes = 0;
    size_t expert_gate_up_byte_stride = 0;
    size_t expert_down_byte_stride = 0;

    const __nv_bfloat16* const* gate_up_ptrs = nullptr;
    const __nv_bfloat16* const* down_ptrs = nullptr;
};

void launch_moe_ffn(const MoeFfnDevice& device,
                    const int* selected_experts,
                    const float* routing_weights,
                    const __nv_bfloat16* hidden,
                    float* output_accum,
                    int rows, int K,
                    __nv_bfloat16* scratch_gate_up,
                    __nv_bfloat16* scratch_activated,
                    cudaStream_t stream);

void launch_finalize_moe_output(const float* accum,
                                __nv_bfloat16* output,
                                int count,
                                cudaStream_t stream);

void compute_moe_ffn(const std::vector<float>& hidden,
                     const std::vector<float>& gate_up,
                     const std::vector<float>& down,
                     const std::vector<int>& selected_experts,
                     const std::vector<float>& routing_weights,
                     int rows, int hidden_dim, int inter, int num_experts,
                     std::vector<float>& output);

void launch_cast_bf16_to_float(const __nv_bfloat16* input,
                               float* output,
                               int count,
                               cudaStream_t stream);

}
