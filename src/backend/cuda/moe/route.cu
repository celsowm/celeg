#include "lfm/runtime/moe.hpp"
#include "lfm/backend/cuda/utils.cuh"

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
// The selection is done deterministically: every thread loads the full
// scores[0..E-1] into shared memory, then the first K threads each pick the
// best expert not yet taken, scanning in order. This is O(E*K) per block and
// avoids the race-prone atomicCAS cascade.
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

    // Deterministic top-K over the shared scores array. Slot k (0..K-1) is
    // filled in order: thread k picks the highest-scoring expert not yet taken
    // (ties broken by smaller expert id) and publishes it; a sync after each
    // slot guarantees later slots observe earlier selections. -1 marks an
    // unfilled slot (must not collide with a valid expert id, which start at 0).
    __shared__ int taken[64];
    if (threadIdx.x < K) taken[threadIdx.x] = -1;
    __syncthreads();
    for (int k = 0; k < K; ++k) {
        int best = -1;
        float best_s = -1e30f;
        if (threadIdx.x == k) {
            for (int e = 0; e < E; ++e) {
                bool is_taken = false;
                for (int t = 0; t < K; ++t) {
                    if (taken[t] == e) { is_taken = true; break; }
                }
                if (is_taken) continue;
                const float s = scores[e];
                bool better = false;
                if (s != best_s) better = s > best_s;
                else better = e < best;
                if (better) { best = e; best_s = s; }
            }
            taken[k] = best;
        }
        __syncthreads();
    }

    // Gather original sigmoid probabilities, normalize, and scale.
    if (threadIdx.x < K) {
        const int expert = taken[threadIdx.x];
        if (expert >= 0) {
            float w = probs[expert];
            if (normalize_topk) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    const int ex = taken[k];
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
