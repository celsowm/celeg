#include "celeg/backend/cuda/moe.hpp"
#include "celeg/backend/cuda/utils.cuh"

#include <cuda_runtime.h>

namespace celeg {

namespace {

using celeg::moe_sigmoid;

__global__ void moe_router_kernel(const float* expert_bias,
                                  int* selected_experts,
                                  float* routing_weights,
                                  int rows, int E, int K,
                                  bool use_expert_bias,
                                  bool softmax,
                                  bool normalize_topk,
                                  float routed_scaling_factor,
                                  int group_count,
                                  int experts_per_group,
                                  int groups_per_token,
                                  int group_score_top_k,
                                  float* scratch) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared[];
    float* logits = &scratch[static_cast<size_t>(row) * E];
    float* probs = shared;
    float* scores = shared + E;
    __shared__ float group_scores[128];
    __shared__ int selected_groups[128];

    if (softmax) {
        if (threadIdx.x == 0) {
            float maximum = -1e30f;
            for (int e = 0; e < E; ++e) maximum = fmaxf(maximum, logits[e]);
            float denominator = 0.0f;
            for (int e = 0; e < E; ++e) {
                probs[e] = expf(logits[e] - maximum);
                denominator += probs[e];
            }
            for (int e = 0; e < E; ++e) probs[e] /= denominator;
        }
    } else {
        for (int e = threadIdx.x; e < E; e += blockDim.x) {
            probs[e] = moe_sigmoid(logits[e]);
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        scores[e] = use_expert_bias ? probs[e] + expert_bias[e] : probs[e];
    }
    __syncthreads();

    if (group_count > 0) {
        if (group_count > 128 || experts_per_group <= 0 ||
            groups_per_token <= 0 || groups_per_token > group_count ||
            group_score_top_k <= 0 || group_score_top_k > experts_per_group) {
            return;
        }
        if (threadIdx.x < group_count) {
            const int group = threadIdx.x;
            float best[128];
            for (int i = 0; i < group_score_top_k; ++i) best[i] = -1e30f;
            for (int offset = 0; offset < experts_per_group; ++offset) {
                const float value = probs[group * experts_per_group + offset];
                for (int slot = 0; slot < group_score_top_k; ++slot) {
                    if (value > best[slot]) {
                        for (int move = group_score_top_k - 1;
                             move > slot; --move) best[move] = best[move - 1];
                        best[slot] = value;
                        break;
                    }
                }
            }
            float total = 0.0f;
            for (int slot = 0; slot < group_score_top_k; ++slot) total += best[slot];
            group_scores[group] = total;
            selected_groups[group] = 0;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int selected = 0; selected < groups_per_token; ++selected) {
                int best_group = -1;
                float best_score = -1e30f;
                for (int group = 0; group < group_count; ++group) {
                    if (selected_groups[group]) continue;
                    if (group_scores[group] > best_score ||
                        (group_scores[group] == best_score && group < best_group)) {
                        best_group = group;
                        best_score = group_scores[group];
                    }
                }
                if (best_group >= 0) selected_groups[best_group] = 1;
            }
        }
        __syncthreads();
    }

    __shared__ int taken[64];
    if (threadIdx.x < K) taken[threadIdx.x] = -1;
    __syncthreads();
    for (int k = 0; k < K; ++k) {
        int best = -1;
        float best_s = -1e30f;
        if (threadIdx.x == k) {
            for (int e = 0; e < E; ++e) {
                bool already_taken = false;
                for (int previous = 0; previous < k; ++previous) {
                    if (taken[previous] == e) already_taken = true;
                }
                if (already_taken) continue;
                if (group_count > 0 &&
                    !selected_groups[e / experts_per_group]) continue;
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

}

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
        cfg.use_expert_bias, cfg.softmax, cfg.normalize_topk, cfg.routed_scaling_factor,
        cfg.group_count, cfg.experts_per_group, cfg.groups_per_token,
        cfg.group_score_top_k,
        scratch_logits);
    CELEG_KERNEL_CHECK();
}

}
