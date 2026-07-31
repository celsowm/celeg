#include "celeg/runtime/moe.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace celeg {

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
            const float prob = moe_sigmoid(logit);
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

void compute_moe_ffn(const std::vector<float>& hidden,
                     const std::vector<float>& gate_up,
                     const std::vector<float>& down,
                     const std::vector<int>& selected_experts,
                     const std::vector<float>& routing_weights,
                     int rows, int hidden_dim, int inter, int num_experts,
                     std::vector<float>& output) {
    if (num_experts <= 0 || inter <= 0 || hidden_dim <= 0) {
        throw std::invalid_argument("invalid MoE FFN configuration");
    }
    const int K = static_cast<int>(routing_weights.size()) / (rows == 0 ? 1 : rows);
    if (selected_experts.size() != routing_weights.size()) {
        throw std::invalid_argument("MoE FFN expert/weight count mismatch");
    }
    output.assign(static_cast<size_t>(rows) * hidden_dim, 0.0f);

    const size_t gu_stride = static_cast<size_t>(2) * inter * hidden_dim;
    const size_t down_stride = static_cast<size_t>(hidden_dim) * inter;

    std::vector<float> guo(2 * inter);
    std::vector<float> activated(inter);

    for (int r = 0; r < rows; ++r) {
        const float* h = hidden.data() + static_cast<size_t>(r) * hidden_dim;
        for (int k = 0; k < K; ++k) {
            const int expert = selected_experts[static_cast<size_t>(r) * K + k];
            if (expert < 0 || expert >= num_experts) continue;
            const float rw = routing_weights[static_cast<size_t>(r) * K + k];
            const float* gu = gate_up.data() + static_cast<size_t>(expert) * gu_stride;
            const float* dw = down.data() + static_cast<size_t>(expert) * down_stride;

            for (int c = 0; c < 2 * inter; ++c) {
                const float* col = gu + static_cast<size_t>(c) * hidden_dim;
                float acc = 0.0f;
                for (int i = 0; i < hidden_dim; ++i) acc += h[i] * col[i];
                guo[c] = acc;
            }
            for (int i = 0; i < inter; ++i) {
                activated[i] = (guo[i] / (1.0f + std::exp(-guo[i]))) * guo[inter + i];
            }
            for (int j = 0; j < hidden_dim; ++j) {
                const float* col = dw + static_cast<size_t>(j) * inter;
                float acc = 0.0f;
                for (int i = 0; i < inter; ++i) acc += activated[i] * col[i];
                output[static_cast<size_t>(r) * hidden_dim + j] += acc * rw;
            }
        }
    }
}

} // namespace celeg
