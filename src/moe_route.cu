#include "lfm/moe.hpp"
#include "lfm/cuda_utils.cuh"

#include <cuda_runtime.h>

namespace lfm {

namespace {

__device__ inline float moe_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

// One block per row (token). Computes router logits/probs/scores in shared
// memory, performs a small-K top-K selection (K <= 64), and writes the
// selected expert ids and (normalized, scaled) sigmoid routing weights.
//
// Top-K uses a shared K-wide buffer guarded by atomicCAS so the globally
// top-K (by selection score, ties broken by smaller expert id) survives the
// parallel per-expert competition deterministically.
__global__ void moe_router_kernel(const float* router_weight,
                                  const float* expert_bias,
                                  const float* hidden_data,
                                  int* selected_experts,
                                  float* routing_weights,
                                  int rows, int hidden_dim, int E, int K,
                                  bool use_expert_bias,
                                  bool normalize_topk,
                                  float routed_scaling_factor,
                                  float* scratch) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared[];
    float* logits = &scratch[static_cast<size_t>(row) * E];
    float* probs = shared;
    float* scores = shared + E;

    const float* row_hidden = hidden_data + static_cast<size_t>(row) * hidden_dim;
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        const float* w = router_weight + static_cast<size_t>(e) * hidden_dim;
        float logit = 0.0f;
        for (int h = 0; h < hidden_dim; ++h) logit += row_hidden[h] * w[h];
        logits[e] = logit;
        const float prob = moe_sigmoid(logit);
        probs[e] = prob;
        scores[e] = use_expert_bias ? prob + expert_bias[e] : prob;
    }
    __syncthreads();

    // Shared K-wide top buffer initialized to worst-possible.
    __shared__ int top_expert[64];
    __shared__ float top_score[64];
    if (threadIdx.x < K) {
        top_expert[threadIdx.x] = -1;
        top_score[threadIdx.x] = -1e30f;
    }
    __syncthreads();

    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        const float s = scores[e];
        for (int k = 0; k < K; ++k) {
            const int occ = top_expert[k];
            const float occ_s = top_score[k];
            bool better = false;
            if (s != occ_s) better = s > occ_s;
            else better = e < occ;
            if (!better) continue;

            // Claim slot k via CAS on the expert id; if we win, the displaced
            // occupant is pushed down to compete for the following slots.
            int expected = occ;
            if (atomicCAS(&top_expert[k], expected, e) == expected) {
                int bump_e = occ;
                float bump_s = occ_s;
                for (int kk = k + 1; kk < K; ++kk) {
                    const int occ2 = top_expert[kk];
                    const float occ2_s = top_score[kk];
                    bool bump_better = false;
                    if (bump_s != occ2_s) bump_better = bump_s > occ2_s;
                    else bump_better = bump_e < occ2;
                    if (!bump_better) break;
                    int exp2 = occ2;
                    if (atomicCAS(&top_expert[kk], exp2, bump_e) == exp2) {
                        const float old_s = bump_s;
                        bump_e = exp2;
                        bump_s = old_s;  // displaced entry becomes new candidate
                    }
                }
            }
        }
    }
    __syncthreads();

    // Gather original sigmoid probabilities, normalize, and scale.
    if (threadIdx.x < K) {
        const int expert = top_expert[threadIdx.x];
        if (expert >= 0) {
            float w = probs[expert];
            if (normalize_topk) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    const int ex = top_expert[k];
                    if (ex >= 0) sum += probs[ex];
                }
                w /= (sum + 1e-6f);
            }
            w *= routed_scaling_factor;
            const size_t out = static_cast<size_t>(row) * K + threadIdx.x;
            selected_experts[out] = expert;
            routing_weights[out] = w;
        }
    }
}

} // namespace

void launch_moe_router(const MoeRouterDevice& device,
                       const MoeRouterConfig& cfg,
                       float* scratch_logits,
                       cudaStream_t stream) {
    if (cfg.num_experts <= 0 || cfg.experts_per_token <= 0 ||
        cfg.experts_per_token > cfg.num_experts) {
        throw std::invalid_argument("invalid MoE router configuration");
    }
    if (cfg.use_expert_bias && device.expert_bias == nullptr) {
        throw std::invalid_argument("expert bias enabled but not provided");
    }
    if (device.rows <= 0 || device.hidden_dim <= 0) {
        throw std::invalid_argument("invalid MoE router dimensions");
    }

    const int E = cfg.num_experts;
    const int block = (E <= 64) ? 64 : 128;
    const size_t shared_bytes = static_cast<size_t>(2) * E * sizeof(float);
    moe_router_kernel<<<device.rows, block, shared_bytes, stream>>>(
        device.router_weight, device.expert_bias, device.hidden_data,
        device.selected_experts, device.routing_weights,
        device.rows, device.hidden_dim, E, cfg.experts_per_token,
        cfg.use_expert_bias, cfg.normalize_topk, cfg.routed_scaling_factor,
        scratch_logits);
    LFM_KERNEL_CHECK();
}

} // namespace lfm
