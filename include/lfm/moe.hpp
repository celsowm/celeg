#pragma once

#include <cstdint>
#include <vector>

// Make __nv_bfloat16 available to this header in both CUDA and host-only
// translation units. Under CUDA it comes from cuda_bf16.h; in host-only TUs
// (e.g. moe_router_test) we define a minimal placeholder since the type is
// only ever used opaquely as a pointer there.
#ifdef __CUDACC__
#include <cuda_bf16.h>
using cudaStream_t = struct CUstream_st*;
#else
struct __opencode_bf16 {
    uint16_t value = 0;
};
using __nv_bfloat16 = __opencode_bf16;
using cudaStream_t = struct CUstream_st*;
#endif

namespace lfm {

// Router configuration for the LFM2 MoE feed-forward block.
struct MoeRouterConfig {
    int num_experts = 0;
    int experts_per_token = 0;
    bool normalize_topk = false;   // norm_topk_prob
    bool use_expert_bias = false;
    float routed_scaling_factor = 1.0f;
};

// Device pointer trio consumed by the CUDA routing kernel. All pointers are
// device-resident; the caller owns memory. `router_weight` is [E * hidden],
// `expert_bias` is [E] (may be nullptr when !use_expert_bias), `hidden_data`
// is [rows * hidden_dim], `selected_experts` and `routing_weights` are both
// [rows * experts_per_token].
struct MoeRouterDevice {
    const float* router_weight = nullptr;  // [E * hidden_dim]
    const float* expert_bias = nullptr;    // [E] or nullptr
    const float* hidden_data = nullptr;     // [rows * hidden_dim]
    int* selected_experts = nullptr;        // [rows * K]
    float* routing_weights = nullptr;       // [rows * K]
    int rows = 0;
    int hidden_dim = 0;
};

// CUDA device implementation of compute_moe_router semantics. Each thread block
// processes one row of the hidden state (one token), computing the top-K
// experts by sigmoid probability with expert-bias-aware selection.
//
// `cfg` mirrors MoeRouterConfig, passed by value so the kernel is self
// contained. The scratch buffer must hold at least E floats per row
// (i.e. rows * E floats) and is used for transient logits/probabilities.
void launch_moe_router(const MoeRouterDevice& device,
                       const MoeRouterConfig& cfg,
                       float* scratch_logits,  // [rows * E]
                       cudaStream_t stream);

// Reference (host) implementation of the LFM2 MoE router.
//
// Semantics (matching the official Transformers Lfm2Moe implementation):
//   1. router_logits = hidden @ router_weight^T   (router_weight: [E, H])
//   2. routing_probabilities = sigmoid(router_logits)   (NOT softmax)
//   3. when use_expert_bias: selection_scores = routing_probabilities + bias,
//      used ONLY to choose the top-k expert IDs.
//   4. select the top `experts_per_token` experts by selection score.
//   5. gather routing weights from the ORIGINAL sigmoid probabilities (not the
//      bias-adjusted scores).
//   6. when normalize_topk: weight_i /= (sum(selected weights) + 1e-6).
//   7. weight_i *= routed_scaling_factor.
//
// Tie-break rule: when two experts have equal selection scores, the smaller
// expert index is preferred (deterministic).
//
// `hidden` is [rows * hidden]; `router_weight` is [num_experts * hidden];
// `expert_bias` is [num_experts] or nullptr when !use_expert_bias.
// Outputs `selected_experts` and `routing_weights` are both
// [rows * experts_per_token].
void compute_moe_router(const std::vector<float>& hidden,
                        const std::vector<float>& router_weight,
                        const std::vector<float>* expert_bias,
                        int rows, int hidden_dim,
                        const MoeRouterConfig& config,
                        std::vector<int>& selected_experts,
                        std::vector<float>& routing_weights);

// BF16 expert weight tensors for the MoE feed-forward block. `gate_up` packs
// each expert's fused gate+up projection as [2 * moe_inter, hidden] stored
// contiguously across experts: gate_up[expert] shape is [2*inter, hidden].
// `down` packs each expert's down projection [hidden, moe_inter]:
// down[expert] shape is [hidden, moe_inter]. `inter` is moe_intermediate_size.
struct MoeFfnDevice {
    const __nv_bfloat16* gate_up = nullptr;  // [E * 2*inter * hidden]
    const __nv_bfloat16* down = nullptr;      // [E * hidden * inter]
    int num_experts = 0;
    int inter = 0;        // moe_intermediate_size
    int hidden_dim = 0;
    size_t expert_gate_up_stride = 0;  // elements per expert in gate_up
    size_t expert_down_stride = 0;      // elements per expert in down
};

// CUDA device implementation of the LFM2 MoE feed-forward block. For every
// token, routes to the K selected experts (selection provided by the caller,
// e.g. via launch_moe_router) and for each selected expert computes
//   activated = silu(hidden @ gate_up[e]^T) * (hidden @ up[e]^T)
//   out += routing_weight * (activated @ down[e]^T)
// accumulating the K expert contributions into `output` [rows * hidden].
//
// `selected_experts` is [rows * K] (int), `routing_weights` is [rows * K]
// (float). The scratch buffers `scratch_gate_up` ([rows * K * 2*inter]) and
// `scratch_activated` ([rows * K * inter]) are caller-owned transient storage.
void launch_moe_ffn(const MoeFfnDevice& device,
                    const int* selected_experts,   // [rows * K]
                    const float* routing_weights,  // [rows * K]
                    const __nv_bfloat16* hidden,   // [rows * hidden]
                    __nv_bfloat16* output,         // [rows * hidden]
                    int rows, int K,
                    __nv_bfloat16* scratch_gate_up,  // [rows * K * 2*inter]
                    __nv_bfloat16* scratch_activated, // [rows * K * inter]
                    cudaStream_t stream);

// Reference (host) implementation of the LFM2 MoE feed-forward block. Operates
// in float for an exact-parity baseline against the BF16 CUDA kernel (tolerance
// applied at the comparison site). `gate_up` and `down` are row-major expert
// tensors laid out exactly as MoeFfnDevice (expert-major, [2*inter, hidden] and
// [hidden, inter] per expert). `selected_experts`/`routing_weights` are
// [rows * K].
void compute_moe_ffn(const std::vector<float>& hidden,
                     const std::vector<float>& gate_up,   // [E*2*inter*hidden]
                     const std::vector<float>& down,      // [E*hidden*inter]
                     const std::vector<int>& selected_experts,
                     const std::vector<float>& routing_weights,
                     int rows, int hidden_dim, int inter, int num_experts,
                     std::vector<float>& output);  // [rows * hidden]

// Element-wise cast __nv_bfloat16 -> float. Used to feed BF16 activations
// into the float router kernels without changing the rest of the pipeline.
void launch_cast_bf16_to_float(const __nv_bfloat16* input,
                               float* output,
                               int count,
                               cudaStream_t stream);

} // namespace lfm
