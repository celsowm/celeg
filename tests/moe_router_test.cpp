#include "lfm/moe.hpp"

#include <cmath>
#include <cassert>
#include <iostream>
#include <vector>

namespace {

float sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + std::exp(-x));
    const float e = std::exp(x);
    return e / (1.0f + e);
}

// Builds a router weight [E, H] that is the identity-ish projection so each
// hidden coordinate maps to one expert's logit, plus a per-expert bias.
void build_problem(int rows, int hidden, int experts,
                   std::vector<float>& hidden_vec,
                   std::vector<float>& router_w,
                   std::vector<float>& bias) {
    hidden_vec.assign(static_cast<size_t>(rows) * hidden, 0.0f);
    router_w.assign(static_cast<size_t>(experts) * hidden, 0.0f);
    bias.assign(static_cast<size_t>(experts), 0.0f);
    // router_w[e][h] = 1 when h == e % hidden, else small.
    for (int e = 0; e < experts; ++e) {
        for (int h = 0; h < hidden; ++h) {
            router_w[static_cast<size_t>(e) * hidden + h] =
                (h == (e % hidden)) ? 2.0f : 0.1f;
        }
    }
    // A simple deterministic hidden state per row.
    for (int r = 0; r < rows; ++r) {
        for (int h = 0; h < hidden; ++h) {
            hidden_vec[static_cast<size_t>(r) * hidden + h] =
                0.5f * (static_cast<float>((r + 1) * (h + 1)));
        }
    }
}

} // namespace

int main() {
    const int rows = 3, hidden = 4, experts = 6, K = 4;
    std::vector<float> hv, rw, bias;
    build_problem(rows, hidden, experts, hv, rw, bias);

    lfm::MoeRouterConfig cfg;
    cfg.num_experts = experts;
    cfg.experts_per_token = K;
    cfg.normalize_topk = false;
    cfg.use_expert_bias = false;
    cfg.routed_scaling_factor = 1.0f;

    std::vector<int> sel;
    std::vector<float> weights;
    lfm::compute_moe_router(hv, rw, nullptr, rows, hidden, cfg, sel, weights);

    // Manually verify row 0 logits and that selected experts are top-K by
    // sigmoid probability (no bias).
    {
        std::vector<float> logits(experts), probs(experts);
        for (int e = 0; e < experts; ++e) {
            float logit = 0.0f;
            for (int hh = 0; hh < hidden; ++hh) {
                logit += hv[hh] * rw[static_cast<size_t>(e) * hidden + hh];
            }
            logits[e] = logit;
            probs[e] = sigmoid(logit);
            // Large positive logits -> sigmoid near 1.
            if (logit > 5.0f) assert(probs[e] > 0.99f);
        }
        // Selected weights equal the original sigmoid of the selected experts.
        for (int k = 0; k < K; ++k) {
            const int ex = sel[static_cast<size_t>(k)];
            assert(std::abs(weights[static_cast<size_t>(k)] - probs[ex]) < 1e-5f);
        }
        // Top-K by prob: every unselected expert has prob <= the smallest
        // selected prob.
        float min_sel = 1e9f;
        for (int k = 0; k < K; ++k) min_sel = std::min(min_sel, probs[sel[static_cast<size_t>(k)]]);
        for (int e = 0; e < experts; ++e) {
            bool selected = false;
            for (int k = 0; k < K; ++k) if (sel[static_cast<size_t>(k)] == e) selected = true;
            if (!selected) assert(probs[e] <= min_sel + 1e-6f);
        }
    }

    // Expert bias must change selected IDs but NOT the gathered weight values.
    {
        std::vector<float> biased_bias(experts, 0.0f);
        // Strongly favor expert 5 (otherwise low prob).
        biased_bias[5] = 100.0f;
        std::vector<int> sel_b;
        std::vector<float> w_b;
        lfm::MoeRouterConfig cfg_b = cfg;
        cfg_b.use_expert_bias = true;
        lfm::compute_moe_router(hv, rw, &biased_bias, rows, hidden, cfg_b, sel_b, w_b);
        bool expert5_selected = false;
        for (int k = 0; k < K; ++k) if (sel_b[static_cast<size_t>(k)] == 5) expert5_selected = true;
        assert(expert5_selected);

        // The gathered weight for expert 5 equals its sigmoid prob, NOT
        // (prob + bias).
        std::vector<float> probs(experts);
        for (int e = 0; e < experts; ++e) {
            float logit = 0.0f;
            for (int hh = 0; hh < hidden; ++hh)
                logit += hv[hh] * rw[static_cast<size_t>(e) * hidden + hh];
            probs[e] = sigmoid(logit);
        }
        for (int k = 0; k < K; ++k) {
            const int ex = sel_b[static_cast<size_t>(k)];
            // The gathered weight is the original sigmoid prob only; the expert
            // bias influences selection but never leaks into the weight value.
            assert(std::abs(w_b[static_cast<size_t>(k)] - probs[ex]) < 1e-5f);
            assert(std::abs(w_b[static_cast<size_t>(k)] - (probs[ex] + biased_bias[ex])) >=
                   (biased_bias[ex] > 0.0f ? 1.0f : 0.0f));
        }
    }

    // Normalization: sum of weights (before scaling) must be ~1.0.
    {
        std::vector<int> sel_n;
        std::vector<float> w_n;
        lfm::MoeRouterConfig cfg_n = cfg;
        cfg_n.normalize_topk = true;
        lfm::compute_moe_router(hv, rw, nullptr, rows, hidden, cfg_n, sel_n, w_n);
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += w_n[static_cast<size_t>(k)];
        assert(std::abs(sum - 1.0f) < 1e-4f);
    }

    // Routed scaling multiplies weights.
    {
        std::vector<int> sel_s;
        std::vector<float> w_s;
        lfm::MoeRouterConfig cfg_s = cfg;
        cfg_s.normalize_topk = true;
        cfg_s.routed_scaling_factor = 2.0f;
        lfm::compute_moe_router(hv, rw, nullptr, rows, hidden, cfg_s, sel_s, w_s);
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += w_s[static_cast<size_t>(k)];
        assert(std::abs(sum - 2.0f) < 1e-4f);
    }

    // Tie-break: equal scores -> smaller expert index selected.
    {
        // All-zero router weight and zero bias -> all logits 0 -> all probs 0.5,
        // scores equal -> top-K must be experts 0..K-1 in order.
        std::vector<float> zero_rw(static_cast<size_t>(experts) * hidden, 0.0f);
        std::vector<float> zero_b(experts, 0.0f);
        std::vector<int> sel_t;
        std::vector<float> w_t;
        lfm::MoeRouterConfig cfg_t = cfg;
        cfg_t.use_expert_bias = true;
        lfm::compute_moe_router(hv, zero_rw, &zero_b, 1, hidden, cfg_t, sel_t, w_t);
        for (int k = 0; k < K; ++k) assert(sel_t[static_cast<size_t>(k)] == k);
        // All weights equal 0.5 (sigmoid(0)), then normalized -> 1/K each.
        for (int k = 0; k < K; ++k) assert(std::abs(w_t[static_cast<size_t>(k)] - 0.5f) < 1e-5f);
    }

    // Negative logits -> sigmoid < 0.5; large negative -> ~0.
    {
        std::vector<float> neg_rw(static_cast<size_t>(experts) * hidden, -10.0f);
        std::vector<int> sel_n2;
        std::vector<float> w_n2;
        lfm::compute_moe_router(hv, neg_rw, nullptr, 1, hidden, cfg, sel_n2, w_n2);
        for (int k = 0; k < K; ++k) assert(w_n2[static_cast<size_t>(k)] < 0.5f);
    }

    std::cout << "moe_router_test: ok\n";
    return 0;
}
