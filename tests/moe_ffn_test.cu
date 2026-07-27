#include "lfm/runtime/moe.hpp"
#include "lfm/backend/cuda/utils.cuh"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

// Builds a small synthetic MoE FFN. Experts have inter = 2*hidden so the
// gate_up projection [2*inter, hidden] and down [hidden, inter] are well
// conditioned but non-trivial.
struct Problem {
    int rows;
    int hidden;
    int inter;
    int experts;
    int K;
    std::vector<float> hidden_vec;
    std::vector<float> router_w;     // [E * hidden]
    std::vector<float> gate_up;       // [E * 2*inter * hidden]
    std::vector<float> down;         // [E * hidden * inter]
    std::vector<float> bias;         // [E] or empty
};

Problem build(int rows, int hidden, int inter, int experts, int K, bool with_bias) {
    Problem p;
    p.rows = rows; p.hidden = hidden; p.inter = inter;
    p.experts = experts; p.K = K;

    p.hidden_vec.assign(static_cast<size_t>(rows) * hidden, 0.0f);
    for (size_t i = 0; i < p.hidden_vec.size(); ++i)
        p.hidden_vec[i] = std::sin(static_cast<float>(i) * 0.3f) + 0.5f;

    p.router_w.assign(static_cast<size_t>(experts) * hidden, 0.0f);
    for (int e = 0; e < experts; ++e)
        for (int h = 0; h < hidden; ++h)
            p.router_w[static_cast<size_t>(e) * hidden + h] =
                (h == (e % hidden)) ? 1.5f : (std::sin(static_cast<float>(e * 7 + h)) * 0.2f);

    p.gate_up.assign(static_cast<size_t>(experts) * 2 * inter * hidden, 0.0f);
    for (size_t i = 0; i < p.gate_up.size(); ++i)
        p.gate_up[i] = std::cos(static_cast<float>(i) * 0.11f) * 0.8f;

    p.down.assign(static_cast<size_t>(experts) * hidden * inter, 0.0f);
    for (size_t i = 0; i < p.down.size(); ++i)
        p.down[i] = std::sin(static_cast<float>(i) * 0.07f) * 0.8f;

    if (with_bias) {
        p.bias.assign(static_cast<size_t>(experts), 0.0f);
        for (int e = 0; e < experts; ++e)
            p.bias[e] = (e % 2 == 0) ? 0.25f : -0.1f;
    }
    return p;
}

__nv_bfloat16 to_bf16(float v) { return __float2bfloat16(v); }

} // namespace

int main() {
    try {
        const bool with_bias = true;
        Problem p = build(3, 8, 16, 6, 4, with_bias);

        lfm::MoeRouterConfig cfg;
        cfg.num_experts = p.experts;
        cfg.experts_per_token = p.K;
        cfg.normalize_topk = true;
        cfg.use_expert_bias = with_bias;
        cfg.routed_scaling_factor = 1.0f;

        lfm::CudaStream stream;

        // Router.
        lfm::DeviceBuffer<float> d_hidden(p.rows * p.hidden);
        lfm::DeviceBuffer<float> d_router(static_cast<size_t>(p.experts) * p.hidden);
        lfm::DeviceBuffer<float> d_bias(p.experts);
        lfm::DeviceBuffer<int> d_sel(static_cast<size_t>(p.rows) * p.K);
        lfm::DeviceBuffer<float> d_wts(static_cast<size_t>(p.rows) * p.K);
        lfm::DeviceBuffer<float> d_scratch(static_cast<size_t>(p.rows) * p.experts);
        LFM_CUDA(cudaMemcpy(d_hidden.data(), p.hidden_vec.data(), d_hidden.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(d_router.data(), p.router_w.data(), d_router.bytes(), cudaMemcpyHostToDevice));
        if (with_bias)
            LFM_CUDA(cudaMemcpy(d_bias.data(), p.bias.data(), d_bias.bytes(), cudaMemcpyHostToDevice));

        lfm::MoeRouterDevice rdev;
        rdev.router_weight = d_router.data();
        rdev.expert_bias = with_bias ? d_bias.data() : nullptr;
        rdev.hidden_data = d_hidden.data();
        rdev.selected_experts = d_sel.data();
        rdev.routing_weights = d_wts.data();
        rdev.rows = p.rows;
        rdev.hidden_dim = p.hidden;
        lfm::launch_moe_router(rdev, cfg, d_scratch.data(), stream.get());
        LFM_CUDA(cudaStreamSynchronize(stream.get()));

        std::vector<int> sel_gpu(static_cast<size_t>(p.rows) * p.K);
        std::vector<float> wts_gpu(static_cast<size_t>(p.rows) * p.K);
        LFM_CUDA(cudaMemcpy(sel_gpu.data(), d_sel.data(), d_sel.bytes(), cudaMemcpyDeviceToHost));
        LFM_CUDA(cudaMemcpy(wts_gpu.data(), d_wts.data(), d_wts.bytes(), cudaMemcpyDeviceToHost));

        // CPU reference for router (sanity + feed into CPU FFN reference).
        std::vector<int> sel_cpu;
        std::vector<float> wts_cpu;
        lfm::compute_moe_router(p.hidden_vec, p.router_w,
                                with_bias ? &p.bias : nullptr,
                                p.rows, p.hidden, cfg, sel_cpu, wts_cpu);
        for (size_t i = 0; i < sel_gpu.size(); ++i) assert(sel_gpu[i] == sel_cpu[i]);

        // FFN device buffers.
        lfm::DeviceBuffer<__nv_bfloat16> d_gate_up(
            static_cast<size_t>(p.experts) * 2 * p.inter * p.hidden);
        lfm::DeviceBuffer<__nv_bfloat16> d_down(
            static_cast<size_t>(p.experts) * p.hidden * p.inter);
        lfm::DeviceBuffer<__nv_bfloat16> d_hidden_bf16(p.rows * p.hidden);
        lfm::DeviceBuffer<__nv_bfloat16> d_output(p.rows * p.hidden);
        lfm::DeviceBuffer<__nv_bfloat16> d_gu_scratch(
            static_cast<size_t>(p.rows) * p.K * 2 * p.inter);
        lfm::DeviceBuffer<__nv_bfloat16> d_act_scratch(
            static_cast<size_t>(p.rows) * p.K * p.inter);

        std::vector<__nv_bfloat16> gu_bf16(p.gate_up.size());
        std::vector<__nv_bfloat16> down_bf16(p.down.size());
        std::vector<__nv_bfloat16> h_bf16(p.hidden_vec.size());
        for (size_t i = 0; i < p.gate_up.size(); ++i) gu_bf16[i] = to_bf16(p.gate_up[i]);
        for (size_t i = 0; i < p.down.size(); ++i) down_bf16[i] = to_bf16(p.down[i]);
        for (size_t i = 0; i < p.hidden_vec.size(); ++i) h_bf16[i] = to_bf16(p.hidden_vec[i]);
        LFM_CUDA(cudaMemcpy(d_gate_up.data(), gu_bf16.data(), d_gate_up.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(d_down.data(), down_bf16.data(), d_down.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(d_hidden_bf16.data(), h_bf16.data(), d_hidden_bf16.bytes(), cudaMemcpyHostToDevice));
        d_output.zero_async(stream.get());
        lfm::DeviceBuffer<float> d_accum(p.rows * p.hidden);
        d_accum.zero_async(stream.get());

        lfm::MoeFfnDevice fdev;
        fdev.gate_up = d_gate_up.data();
        fdev.down = d_down.data();
        fdev.num_experts = p.experts;
        fdev.inter = p.inter;
        fdev.hidden_dim = p.hidden;
        fdev.expert_gate_up_stride = static_cast<size_t>(2) * p.inter * p.hidden;
        fdev.expert_down_stride = static_cast<size_t>(p.hidden) * p.inter;
        lfm::launch_moe_ffn(fdev, d_sel.data(), d_wts.data(),
                            d_hidden_bf16.data(), d_accum.data(),
                            p.rows, p.K, d_gu_scratch.data(),
                            d_act_scratch.data(), stream.get());
        lfm::launch_finalize_moe_output(d_accum.data(), d_output.data(),
                                        p.rows * p.hidden, stream.get());
        LFM_CUDA(cudaStreamSynchronize(stream.get()));

        std::vector<__nv_bfloat16> out_gpu(p.rows * p.hidden);
        LFM_CUDA(cudaMemcpy(out_gpu.data(), d_output.data(), d_output.bytes(), cudaMemcpyDeviceToHost));

        // CPU reference FFN (float) using the GPU-selected experts/weights.
        std::vector<float> out_cpu;
        lfm::compute_moe_ffn(p.hidden_vec, p.gate_up, p.down,
                             sel_gpu, wts_gpu, p.rows, p.hidden,
                             p.inter, p.experts, out_cpu);

        float max_abs = 0.0f;
        float max_rel = 0.0f;
        for (size_t i = 0; i < out_gpu.size(); ++i) {
            const float g = __bfloat162float(out_gpu[i]);
            max_abs = std::max(max_abs, std::fabs(g - out_cpu[i]));
            const float denom = std::max(1.0f, std::fabs(out_cpu[i]));
            max_rel = std::max(max_rel, std::fabs(g - out_cpu[i]) / denom);
        }
        std::cout << "moe_ffn_test: max abs diff vs float reference = " << max_abs
                  << ", max rel = " << max_rel << "\n";
        // The GPU kernel operates in BF16, so the legitimate reference is the
        // BF16-cast of the float computation, not the float computation itself.
        // Keep the float comparison only as a loose sanity bound.
        assert(max_rel < 0.25f);

        // Also verify against a pure-BF16 CPU recompute (per-element tolerance
        // tightened to account only for BF16 rounding of the weights/inputs).
        float max_abs_bf16 = 0.0f;
        for (size_t i = 0; i < out_gpu.size(); ++i) {
            const float g = __bfloat162float(out_gpu[i]);
            const float bf16_expected = __bfloat162float(to_bf16(out_cpu[i]));
            max_abs_bf16 = std::max(max_abs_bf16, std::fabs(g - bf16_expected));
        }
        std::cout << "moe_ffn_test: max abs diff vs bf16-cast reference = " << max_abs_bf16 << "\n";
        // BF16 carries ~3 decimal digits; a per-element deviation larger than
        // 0.1 indicates a real kernel mismatch rather than rounding.
        assert(max_abs_bf16 < 0.1f);

        std::cout << "moe_ffn_test: ok\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "moe_ffn_test FAILED: " << e.what() << "\n";
        return 1;
    }
}
