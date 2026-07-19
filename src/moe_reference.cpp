#include "lfm/moe.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lfm {

namespace {

inline float sigmoid(float x) {
    // Numerically stable sigmoid.
    if (x >= 0.0f) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    const float e = std::exp(x);
    return e / (1.0f + e);
}

} // namespace

void compute_moe_router(const std::vector<float>& hidden,
                        const std::vector<float>& router_weight,
                        const std::vector<float>* expert_bias,
                        int rows, int hidden_dim,
                        const MoeRouterConfig& config,
                        std::vector<int>& selected_experts,
                        std::vector<float>& routing_weights) {
    if (config.num_experts <= 0 || config.experts_per_token <= 0 ||
        config.experts_per_token > config.num_experts) {
        throw std::invalid_argument("invalid MoE router configuration");
    }
    if (config.use_expert_bias && expert_bias == nullptr) {
        throw std::invalid_argument("expert bias enabled but not provided");
    }

    selected_experts.assign(static_cast<size_t>(rows) * config.experts_per_token, 0);
    routing_weights.assign(static_cast<size_t>(rows) * config.experts_per_token, 0.0f);

    const int E = config.num_experts;
    const int K = config.experts_per_token;
    std::vector<float> logits(static_cast<size_t>(E));
    std::vector<float> probs(static_cast<size_t>(E));
    std::vector<float> scores(static_cast<size_t>(E));
    // Best (score, expert) pairs; smaller expert index wins ties.
    std::vector<std::pair<float, int>> best;

    for (int r = 0; r < rows; ++r) {
        const float* row = hidden.data() + static_cast<size_t>(r) * hidden_dim;
        for (int e = 0; e < E; ++e) {
            const float* w = router_weight.data() + static_cast<size_t>(e) * hidden_dim;
            float logit = 0.0f;
            for (int h = 0; h < hidden_dim; ++h) logit += row[h] * w[h];
            logits[e] = logit;
            const float prob = sigmoid(logit);
            probs[e] = prob;
            scores[e] = config.use_expert_bias ? prob + (*expert_bias)[e] : prob;
        }

        // Select top-K by score (descending); tie-break by smaller expert id.
        best.clear();
        best.reserve(static_cast<size_t>(E));
        for (int e = 0; e < E; ++e) best.emplace_back(scores[e], e);
        std::partial_sort(
            best.begin(), best.begin() + K, best.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                if (a.first != b.first) return a.first > b.first;
                return a.second < b.second;
            });

        float weight_sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            const int expert = best[k].second;
            selected_experts[static_cast<size_t>(r) * K + k] = expert;
            const float w = probs[expert];  // original sigmoid, not score
            routing_weights[static_cast<size_t>(r) * K + k] = w;
            weight_sum += w;
        }

        if (config.normalize_topk) {
            const float inv = 1.0f / (weight_sum + 1e-6f);
            for (int k = 0; k < K; ++k) {
                routing_weights[static_cast<size_t>(r) * K + k] *= inv;
            }
        }
        for (int k = 0; k < K; ++k) {
            routing_weights[static_cast<size_t>(r) * K + k] *= config.routed_scaling_factor;
        }
    }
}

} // namespace lfm
