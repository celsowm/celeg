#include "celeg/backend/cuda/moe.hpp"
#include "celeg/backend/cuda/utils.cuh"

#include <cuda_runtime.h>

namespace celeg {

namespace {

using celeg::moe_sigmoid;

// One block per row (token). Consumes GEMM-produced router logits, computes
// probabilities/scores in shared memory, performs a small-K top-K selection
// (K <= 64), and writes the
// selected expert ids and (normalized, scaled) sigmoid routing weights.
//
// The selection is done deterministically: every thread loads the full
// scores[0..E-1] into shared memory, then the first K threads each pick the
// best expert not yet taken, scanning in order. This is O(E*K) per block and
// avoids the race-prone atomicCAS cascade.
__global__ void moe_router_kernel(const float* expert_bias,
                                  int* selected_experts,
                                  float* routing_weights,
                                  int rows, int E, int K,
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

    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        const float logit = logits[e];
        const float prob = moe_sigmoid(logit);
        probs[e] = prob;
        scores[e] = use_expert_bias ? prob + expert_bias[e] : prob;
    }
    __syncthreads();

    // Deterministic top-K over the shared scores array. Slot k (0..K-1) is
    // filled in order: thread k picks the highest-scoring expert not yet taken
    // (ties broken by smaller expert id) and publishes it; a sync after each
    // slot guarantees later slots observe earlier selections. A bitmask
    // replaces the O(K)-per-check scan of a taken[] array, making selection
    // O(E*K) instead of O(E*K^2).
    __shared__ int taken[64];
    __shared__ unsigned taken_mask;
    if (threadIdx.x == 0) taken_mask = 0u;
    if (threadIdx.x < K) taken[threadIdx.x] = -1;
    __syncthreads();
    for (int k = 0; k < K; ++k) {
        int best = -1;
        float best_s = -1e30f;
        if (threadIdx.x == k) {
            const unsigned mask = taken_mask;
            for (int e = 0; e < E; ++e) {
                if (mask & (1u << e)) continue;
                const float s = scores[e];
                bool better = false;
                if (s != best_s) better = s > best_s;
                else better = e < best;
                if (better) { best = e; best_s = s; }
            }
            taken[k] = best;
            taken_mask = mask | (1u << best);
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
    CublasHandle cublas(stream);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // C = W * hidden^T, with both inputs supplied in row-major storage. In
    // cuBLAS' column-major view this is W^T(op=T) times hidden(op=N).
    CELEG_CUBLAS(cublasSgemm(
        cublas.get(), CUBLAS_OP_T, CUBLAS_OP_N,
        E, device.rows, device.hidden_dim,
        &alpha, device.router_weight, device.hidden_dim,
        device.hidden_data, device.hidden_dim,
        &beta, scratch_logits, E));
    const int block = (E <= 64) ? 64 : 128;
    const size_t shared_bytes = static_cast<size_t>(2) * E * sizeof(float);
    moe_router_kernel<<<device.rows, block, shared_bytes, stream>>>(
        device.expert_bias,
        device.selected_experts, device.routing_weights,
        device.rows, E, cfg.experts_per_token,
        cfg.use_expert_bias, cfg.normalize_topk, cfg.routed_scaling_factor,
        scratch_logits);
    CELEG_KERNEL_CHECK();
}

} // namespace celeg
