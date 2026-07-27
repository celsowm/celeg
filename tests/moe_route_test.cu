#include "lfm/runtime/moe.hpp"
#include "support/assertions.hpp"
#include "lfm/backend/cuda/utils.cuh"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

float sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + std::exp(-x));
    const float e = std::exp(x);
    return e / (1.0f + e);
}

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

} // namespace

int main() {
    try {
        const int rows = 3, hidden = 4, experts = 6, K = 4;
        std::vector<float> hv, rw, bias;
        build_problem(rows, hidden, experts, hv, rw, bias);

        lfm::CudaStream stream;

        lfm::DeviceBuffer<float> d_hidden(rows * hidden);
        lfm::DeviceBuffer<float> d_router(static_cast<size_t>(experts) * hidden);
        lfm::DeviceBuffer<float> d_bias(experts);
        lfm::DeviceBuffer<int> d_sel(static_cast<size_t>(rows) * K);
        lfm::DeviceBuffer<float> d_wts(static_cast<size_t>(rows) * K);
        lfm::DeviceBuffer<float> d_scratch(static_cast<size_t>(rows) * experts);

        LFM_CUDA(cudaMemcpy(d_hidden.data(), hv.data(), d_hidden.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(d_router.data(), rw.data(), d_router.bytes(), cudaMemcpyHostToDevice));

        // Helper to run on GPU and compare against CPU reference for a config.
        auto run_case = [&](const lfm::MoeRouterConfig& cfg, const std::vector<float>* bias_ptr) {
            if (cfg.use_expert_bias)
                LFM_CUDA(cudaMemcpy(d_bias.data(), bias.data(), d_bias.bytes(), cudaMemcpyHostToDevice));
            else
                bias_ptr = nullptr;

            lfm::MoeRouterDevice dev;
            dev.router_weight = d_router.data();
            dev.expert_bias = cfg.use_expert_bias ? d_bias.data() : nullptr;
            dev.hidden_data = d_hidden.data();
            dev.selected_experts = d_sel.data();
            dev.routing_weights = d_wts.data();
            dev.rows = rows;
            dev.hidden_dim = hidden;

            lfm::launch_moe_router(dev, cfg, d_scratch.data(), stream.get());
            LFM_CUDA(cudaStreamSynchronize(stream.get()));

            std::vector<int> sel_gpu(static_cast<size_t>(rows) * K);
            std::vector<float> wts_gpu(static_cast<size_t>(rows) * K);
            LFM_CUDA(cudaMemcpy(sel_gpu.data(), d_sel.data(), d_sel.bytes(), cudaMemcpyDeviceToHost));
            LFM_CUDA(cudaMemcpy(wts_gpu.data(), d_wts.data(), d_wts.bytes(), cudaMemcpyDeviceToHost));

            // Compare against the existing host reference implementation.
            std::vector<int> sel_cpu;
            std::vector<float> wts_cpu;
            lfm::compute_moe_router(hv, rw, bias_ptr, rows, hidden, cfg, sel_cpu, wts_cpu);

            for (size_t i = 0; i < sel_gpu.size(); ++i) {
                if (sel_gpu[i] != sel_cpu[i])
                    throw std::runtime_error("router selection mismatch at " +
                                             std::to_string(i));
                if (std::abs(wts_gpu[i] - wts_cpu[i]) >= 1e-4f)
                    throw std::runtime_error("router weight mismatch at " +
                                             std::to_string(i));
            }
            // Also verify weights equal the original sigmoid of the selected
            // expert (bias must never leak into the gathered weight).
            for (int r = 0; r < rows; ++r) {
                for (int k = 0; k < K; ++k) {
                    const int ex = sel_gpu[static_cast<size_t>(r) * K + k];
                    float logit = 0.0f;
                    for (int hh = 0; hh < hidden; ++hh)
                        logit += hv[static_cast<size_t>(r) * hidden + hh] *
                                 rw[static_cast<size_t>(ex) * hidden + hh];
                    float expected = sigmoid(logit);
                    if (cfg.normalize_topk) {
                        float sum = 0.0f;
                        for (int kk = 0; kk < K; ++kk) {
                            const int exx = sel_gpu[static_cast<size_t>(r) * K + kk];
                            float l2 = 0.0f;
                            for (int hh = 0; hh < hidden; ++hh)
                                l2 += hv[static_cast<size_t>(r) * hidden + hh] *
                                      rw[static_cast<size_t>(exx) * hidden + hh];
                            sum += sigmoid(l2);
                        }
                        expected /= (sum + 1e-6f);
                    }
                    expected *= cfg.routed_scaling_factor;
                    if (std::abs(wts_gpu[static_cast<size_t>(r) * K + k] - expected) >= 1e-4f)
                        throw std::runtime_error("router weight recompute mismatch");
                }
            }
        };

        // Case 1: no bias, no normalization.
        {
            lfm::MoeRouterConfig cfg;
            cfg.num_experts = experts;
            cfg.experts_per_token = K;
            cfg.normalize_topk = false;
            cfg.use_expert_bias = false;
            cfg.routed_scaling_factor = 1.0f;
            run_case(cfg, nullptr);
        }

        // Case 2: with bias (must change selection ids, not weights).
        {
            lfm::MoeRouterConfig cfg;
            cfg.num_experts = experts;
            cfg.experts_per_token = K;
            cfg.normalize_topk = false;
            cfg.use_expert_bias = true;
            cfg.routed_scaling_factor = 1.0f;
            run_case(cfg, &bias);
        }

        // Case 3: normalized, scaled.
        {
            lfm::MoeRouterConfig cfg;
            cfg.num_experts = experts;
            cfg.experts_per_token = K;
            cfg.normalize_topk = true;
            cfg.use_expert_bias = false;
            cfg.routed_scaling_factor = 2.0f;
            run_case(cfg, nullptr);
        }

        // Case 4: bias + normalization + scaling combined.
        {
            lfm::MoeRouterConfig cfg;
            cfg.num_experts = experts;
            cfg.experts_per_token = K;
            cfg.normalize_topk = true;
            cfg.use_expert_bias = true;
            cfg.routed_scaling_factor = 3.0f;
            run_case(cfg, &bias);
        }

        // Case 5: single-token (rows=1) decode path. The original router
        // selection was broken for rows=1 (top-K buffer never updated its
        // scores), so this guards the decode path explicitly.
        {
            const int E1 = 6, K1 = 4, H1 = 4, R1 = 1;
            std::vector<float> hv1, rw1, b1;
            build_problem(R1, H1, E1, hv1, rw1, b1);
            lfm::DeviceBuffer<float> dh1(R1 * H1), dr1(static_cast<size_t>(E1) * H1);
            lfm::DeviceBuffer<float> db1(E1);
            lfm::DeviceBuffer<int> dsel1(static_cast<size_t>(R1) * K1);
            lfm::DeviceBuffer<float> dw1(static_cast<size_t>(R1) * K1);
            lfm::DeviceBuffer<float> dscr1(static_cast<size_t>(R1) * E1);
            LFM_CUDA(cudaMemcpy(dh1.data(), hv1.data(), dh1.bytes(), cudaMemcpyHostToDevice));
            LFM_CUDA(cudaMemcpy(dr1.data(), rw1.data(), dr1.bytes(), cudaMemcpyHostToDevice));
            LFM_CUDA(cudaMemcpy(db1.data(), b1.data(), db1.bytes(), cudaMemcpyHostToDevice));

            lfm::MoeRouterConfig cfg;
            cfg.num_experts = E1;
            cfg.experts_per_token = K1;
            cfg.normalize_topk = true;
            cfg.use_expert_bias = true;
            cfg.routed_scaling_factor = 1.0f;

            lfm::MoeRouterDevice dev;
            dev.router_weight = dr1.data();
            dev.expert_bias = db1.data();
            dev.hidden_data = dh1.data();
            dev.selected_experts = dsel1.data();
            dev.routing_weights = dw1.data();
            dev.rows = R1;
            dev.hidden_dim = H1;
            lfm::launch_moe_router(dev, cfg, dscr1.data(), stream.get());
            LFM_CUDA(cudaStreamSynchronize(stream.get()));

            std::vector<int> sg(static_cast<size_t>(R1) * K1);
            std::vector<float> wg(static_cast<size_t>(R1) * K1);
            LFM_CUDA(cudaMemcpy(sg.data(), dsel1.data(),
                                dsel1.bytes(), cudaMemcpyDeviceToHost));
            LFM_CUDA(cudaMemcpy(wg.data(), dw1.data(),
                                dw1.bytes(), cudaMemcpyDeviceToHost));
            std::vector<int> sc;
            std::vector<float> wc;
            lfm::compute_moe_router(hv1, rw1, &b1, R1, H1, cfg, sc, wc);
            for (size_t i = 0; i < sg.size(); ++i) {
                if (sg[i] != sc[i])
                    throw std::runtime_error("rows=1 router selection mismatch at " +
                                             std::to_string(i));
                if (std::abs(wg[i] - wc[i]) >= 1e-4f)
                    throw std::runtime_error("rows=1 router weight mismatch at " +
                                             std::to_string(i));
            }
        }

        // Case 6: larger expert count (E=32) to exercise multi-warp blocks.
        {
            const int E2 = 32, K2 = 4, H2 = 8, R2 = 5;
            std::vector<float> hv2, rw2, b2;
            build_problem(R2, H2, E2, hv2, rw2, b2);
            lfm::DeviceBuffer<float> dh2(R2 * H2), dr2(static_cast<size_t>(E2) * H2);
            lfm::DeviceBuffer<float> db2(E2);
            lfm::DeviceBuffer<int> dsel2(static_cast<size_t>(R2) * K2);
            lfm::DeviceBuffer<float> dw2(static_cast<size_t>(R2) * K2);
            lfm::DeviceBuffer<float> dscr2(static_cast<size_t>(R2) * E2);
            LFM_CUDA(cudaMemcpy(dh2.data(), hv2.data(), dh2.bytes(), cudaMemcpyHostToDevice));
            LFM_CUDA(cudaMemcpy(dr2.data(), rw2.data(), dr2.bytes(), cudaMemcpyHostToDevice));

            lfm::MoeRouterConfig cfg;
            cfg.num_experts = E2;
            cfg.experts_per_token = K2;
            cfg.normalize_topk = true;
            cfg.use_expert_bias = true;
            cfg.routed_scaling_factor = 1.5f;
            LFM_CUDA(cudaMemcpy(db2.data(), b2.data(), db2.bytes(), cudaMemcpyHostToDevice));

            lfm::MoeRouterDevice dev;
            dev.router_weight = dr2.data();
            dev.expert_bias = db2.data();
            dev.hidden_data = dh2.data();
            dev.selected_experts = dsel2.data();
            dev.routing_weights = dw2.data();
            dev.rows = R2;
            dev.hidden_dim = H2;
            lfm::launch_moe_router(dev, cfg, dscr2.data(), stream.get());
            LFM_CUDA(cudaStreamSynchronize(stream.get()));

            std::vector<int> sg(static_cast<size_t>(R2) * K2);
            std::vector<float> wg(static_cast<size_t>(R2) * K2);
            LFM_CUDA(cudaMemcpy(sg.data(), dsel2.data(), dsel2.bytes(), cudaMemcpyDeviceToHost));
            LFM_CUDA(cudaMemcpy(wg.data(), dw2.data(), dw2.bytes(), cudaMemcpyDeviceToHost));

            std::vector<int> sc;
            std::vector<float> wc;
            lfm::compute_moe_router(hv2, rw2, &b2, R2, H2, cfg, sc, wc);
            for (size_t i = 0; i < sg.size(); ++i) {
                LFM_TEST_CHECK(sg[i] == sc[i]);
                LFM_TEST_CHECK(std::abs(wg[i] - wc[i]) < 1e-4f);
            }
        }

        std::cout << "moe_route_test: ok\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "moe_route_test FAILED: " << e.what() << "\n";
        return 1;
    }
}
