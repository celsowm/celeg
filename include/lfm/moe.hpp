#pragma once

#include <cstdint>
#include <vector>

// Forward declaration so this header can be consumed by host-only targets
// (e.g. moe_router_test) without pulling in the CUDA runtime headers.
using cudaStream_t = struct CUstream_st*;

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

} // namespace lfm
