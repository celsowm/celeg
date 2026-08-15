#include "celeg/backend/cuda/moe.hpp"
#include "support/assertions.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

using celeg::moe_sigmoid;

void build_problem(int rows, int hidden, int experts,
                   std::vector<float>& hidden_vec,
                   std::vector<float>& router_w,
                   std::vector<float>& bias) {
    hidden_vec.assign(static_cast<size_t>(rows) * hidden, 0.0f);
    router_w.assign(static_cast<size_t>(experts) * hidden, 0.0f);
    bias.assign(static_cast<size_t>(experts), 0.0f);
    for (int e = 0; e < experts; ++e) {
        for (int h = 0; h < hidden; ++h) {
            router_w[static_cast<size_t>(e) * hidden + h] =
                (h == (e % hidden)) ? 2.0f : 0.1f;
        }
    }
    for (int r = 0; r < rows; ++r) {
        for (int h = 0; h < hidden; ++h) {
            hidden_vec[static_cast<size_t>(r) * hidden + h] =
                0.5f * (static_cast<float>((r + 1) * (h + 1)));
        }
    }
}

}

int main() {
    const int rows = 3, hidden = 4, experts = 6, K = 4;
    std::vector<float> hv, rw, bias;
    build_problem(rows, hidden, experts, hv, rw, bias);

    celeg::MoeRouterConfig cfg;
    cfg.num_experts = experts;
    cfg.experts_per_token = K;
    cfg.normalize_topk = false;
    cfg.use_expert_bias = false;
    cfg.routed_scaling_factor = 1.0f;

    std::vector<int> sel;
    std::vector<float> weights;
    celeg::compute_moe_router(hv, rw, nullptr, rows, hidden, cfg, sel, weights);

    {
        std::vector<float> logits(experts), probs(experts);
        for (int e = 0; e < experts; ++e) {
            float logit = 0.0f;
            for (int hh = 0; hh < hidden; ++hh) {
                logit += hv[hh] * rw[static_cast<size_t>(e) * hidden + hh];
            }
            logits[e] = logit;
            probs[e] = moe_sigmoid(logit);
            if (logit > 5.0f) CELEG_TEST_CHECK(probs[e] > 0.99f);
        }
        for (int k = 0; k < K; ++k) {
            const int ex = sel[static_cast<size_t>(k)];
            CELEG_TEST_CHECK(std::abs(weights[static_cast<size_t>(k)] - probs[ex]) < 1e-5f);
        }
        float min_sel = 1e9f;
        for (int k = 0; k < K; ++k) min_sel = std::min(min_sel, probs[sel[static_cast<size_t>(k)]]);
        for (int e = 0; e < experts; ++e) {
            bool selected = false;
            for (int k = 0; k < K; ++k) if (sel[static_cast<size_t>(k)] == e) selected = true;
            if (!selected) CELEG_TEST_CHECK(probs[e] <= min_sel + 1e-6f);
        }
    }

    {
        std::vector<float> biased_bias(experts, 0.0f);
        biased_bias[5] = 100.0f;
        std::vector<int> sel_b;
        std::vector<float> w_b;
        celeg::MoeRouterConfig cfg_b = cfg;
        cfg_b.use_expert_bias = true;
        celeg::compute_moe_router(hv, rw, &biased_bias, rows, hidden, cfg_b, sel_b, w_b);
        bool expert5_selected = false;
        for (int k = 0; k < K; ++k) if (sel_b[static_cast<size_t>(k)] == 5) expert5_selected = true;
        CELEG_TEST_CHECK(expert5_selected);

        std::vector<float> probs(experts);
        for (int e = 0; e < experts; ++e) {
            float logit = 0.0f;
            for (int hh = 0; hh < hidden; ++hh)
                logit += hv[hh] * rw[static_cast<size_t>(e) * hidden + hh];
            probs[e] = moe_sigmoid(logit);
        }
        for (int k = 0; k < K; ++k) {
            const int ex = sel_b[static_cast<size_t>(k)];
            CELEG_TEST_CHECK(std::abs(w_b[static_cast<size_t>(k)] - probs[ex]) < 1e-5f);
            CELEG_TEST_CHECK(std::abs(w_b[static_cast<size_t>(k)] - (probs[ex] + biased_bias[ex])) >=
                   (biased_bias[ex] > 0.0f ? 1.0f : 0.0f));
        }
    }

    {
        std::vector<int> sel_n;
        std::vector<float> w_n;
        celeg::MoeRouterConfig cfg_n = cfg;
        cfg_n.normalize_topk = true;
        celeg::compute_moe_router(hv, rw, nullptr, rows, hidden, cfg_n, sel_n, w_n);
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += w_n[static_cast<size_t>(k)];
        CELEG_TEST_CHECK(std::abs(sum - 1.0f) < 1e-4f);
    }

    {
        std::vector<int> sel_s;
        std::vector<float> w_s;
        celeg::MoeRouterConfig cfg_s = cfg;
        cfg_s.normalize_topk = true;
        cfg_s.routed_scaling_factor = 2.0f;
        celeg::compute_moe_router(hv, rw, nullptr, rows, hidden, cfg_s, sel_s, w_s);
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += w_s[static_cast<size_t>(k)];
        CELEG_TEST_CHECK(std::abs(sum - 2.0f) < 1e-4f);
    }

    {
        std::vector<float> zero_rw(static_cast<size_t>(experts) * hidden, 0.0f);
        std::vector<float> zero_b(experts, 0.0f);
        std::vector<int> sel_t;
        std::vector<float> w_t;
        celeg::MoeRouterConfig cfg_t = cfg;
        cfg_t.use_expert_bias = true;
        celeg::compute_moe_router(hv, zero_rw, &zero_b, 1, hidden, cfg_t, sel_t, w_t);
        for (int k = 0; k < K; ++k) CELEG_TEST_CHECK(sel_t[static_cast<size_t>(k)] == k);
        for (int k = 0; k < K; ++k) CELEG_TEST_CHECK(std::abs(w_t[static_cast<size_t>(k)] - 0.5f) < 1e-5f);
    }

    {
        std::vector<float> neg_rw(static_cast<size_t>(experts) * hidden, -10.0f);
        std::vector<int> sel_n2;
        std::vector<float> w_n2;
        celeg::compute_moe_router(hv, neg_rw, nullptr, 1, hidden, cfg, sel_n2, w_n2);
        for (int k = 0; k < K; ++k) CELEG_TEST_CHECK(w_n2[static_cast<size_t>(k)] < 0.5f);
    }

    std::cout << "moe_router_test: ok\n";
    return 0;
}
