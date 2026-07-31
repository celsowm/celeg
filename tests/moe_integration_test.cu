// Integration test for the MoE decode forward sequence wired into Model
// (run_mlp_moe_decode). It exercises the exact kernel pipeline the model uses:
//   1. RMSNorm the hidden state (BF16)
//   2. cast the normed BF16 activations to float (router input path)
//   3. launch_moe_router (sigmoid top-K + bias + normalize + scaling)
//   4. launch_moe_ffn (per-(token,expert) SwiGLU + atomic accumulate)
//   5. residual-add the routed expert output back into the hidden state
// The result is checked against a CPU reference that runs the same sequence in
// float, so the test covers the BF16->float cast path and the residual-add
// that live only in the integrated model forward, not in the isolated kernels.
//
// NOTE: the router/FFN run in BF16, so a float CPU reference is expected to
// differ by a few percent (BF16 rounding). Tolerances below reflect that.

#include "lfm/runtime/moe.hpp"
#include "lfm/backend/cuda/utils.cuh"
#include "lfm/backend/cuda/kernels/norm_conv.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

__nv_bfloat16 to_bf16(float v) { return __float2bfloat16(v); }
float from_bf16(__nv_bfloat16 v) { return __bfloat162float(v); }

struct Problem {
    int hidden;
    int inter;
    int experts;
    int K;
    float norm_eps;
    std::vector<float> hidden_vec;   // [hidden]
    std::vector<float> ffn_norm;     // [hidden]
    std::vector<float> router_w;     // [E * hidden]
    std::vector<float> gate_up;      // [E * 2*inter * hidden]
    std::vector<float> down;         // [E * hidden * inter]
    std::vector<float> bias;         // [E]
};

Problem build(int hidden, int inter, int experts, int K) {
    Problem p;
    p.hidden = hidden; p.inter = inter;
    p.experts = experts; p.K = K;
    p.norm_eps = 1.0e-5f;

    p.hidden_vec.assign(hidden, 0.0f);
    for (int i = 0; i < hidden; ++i)
        p.hidden_vec[i] = std::sin(static_cast<float>(i) * 0.3f) + 0.5f;

    p.ffn_norm.assign(hidden, 0.0f);
    for (int i = 0; i < hidden; ++i)
        p.ffn_norm[i] = 1.0f + 0.1f * std::cos(static_cast<float>(i) * 0.5f);

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

    p.bias.assign(experts, 0.0f);
    for (int e = 0; e < experts; ++e)
        p.bias[e] = (e % 2 == 0) ? 0.25f : -0.1f;
    return p;
}

// CPU RMSNorm matching launch_rmsnorm (compute in float, output BF16).
std::vector<__nv_bfloat16> cpu_rmsnorm(const std::vector<float>& x,
                                       const std::vector<float>& w,
                                       float eps) {
    const int n = static_cast<int>(x.size());
    float ss = 0.0f;
    for (float v : x) ss += v * v;
    const float rms = std::sqrt(ss / n + eps);
    const float inv = 1.0f / rms;
    std::vector<__nv_bfloat16> out(n);
    for (int i = 0; i < n; ++i) out[i] = to_bf16(x[i] * inv * w[i]);
    return out;
}

} // namespace

int main() {
    try {
        Problem p = build(8, 16, 6, 4);

        lfm::MoeRouterConfig cfg;
        cfg.num_experts = p.experts;
        cfg.experts_per_token = p.K;
        cfg.normalize_topk = true;
        cfg.use_expert_bias = true;
        cfg.routed_scaling_factor = 1.0f;

        lfm::CudaStream stream;

        // Hidden state (BF16) and its float copy for the reference.
        lfm::DeviceBuffer<__nv_bfloat16> d_hidden(p.hidden);
        std::vector<__nv_bfloat16> h_bf16(p.hidden);
        for (int i = 0; i < p.hidden; ++i) h_bf16[i] = to_bf16(p.hidden_vec[i]);
        LFM_CUDA(cudaMemcpy(d_hidden.data(), h_bf16.data(),
                            d_hidden.bytes(), cudaMemcpyHostToDevice));

        // ffn_norm (BF16) on device.
        lfm::DeviceBuffer<__nv_bfloat16> d_ffn_norm(p.hidden);
        std::vector<__nv_bfloat16> w_bf16(p.hidden);
        for (int i = 0; i < p.hidden; ++i) w_bf16[i] = to_bf16(p.ffn_norm[i]);
        LFM_CUDA(cudaMemcpy(d_ffn_norm.data(), w_bf16.data(),
                            d_ffn_norm.bytes(), cudaMemcpyHostToDevice));

        // Step 1: RMSNorm -> normed BF16.
        lfm::DeviceBuffer<__nv_bfloat16> d_normed(p.hidden);
        lfm::launch_rmsnorm(d_hidden.data(), d_ffn_norm.data(), d_normed.data(),
                            1, p.hidden, p.norm_eps, stream.get());

        // Step 2: cast normed BF16 -> float (the router input path).
        lfm::DeviceBuffer<float> d_hidden_float(p.hidden);
        lfm::launch_cast_bf16_to_float(d_normed.data(), d_hidden_float.data(),
                                       p.hidden, stream.get());

        // Step 3: router.
        lfm::DeviceBuffer<float> d_router(static_cast<size_t>(p.experts) * p.hidden);
        lfm::DeviceBuffer<float> d_bias(p.experts);
        lfm::DeviceBuffer<int> d_sel(p.K);
        lfm::DeviceBuffer<float> d_wts(p.K);
        lfm::DeviceBuffer<float> d_scratch(p.experts);
        LFM_CUDA(cudaMemcpy(d_router.data(), p.router_w.data(),
                            d_router.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(d_bias.data(), p.bias.data(),
                            d_bias.bytes(), cudaMemcpyHostToDevice));
        lfm::MoeRouterDevice rdev;
        rdev.router_weight = d_router.data();
        rdev.expert_bias = d_bias.data();
        rdev.hidden_data = d_hidden_float.data();
        rdev.selected_experts = d_sel.data();
        rdev.routing_weights = d_wts.data();
        rdev.rows = 1;
        rdev.hidden_dim = p.hidden;
        lfm::launch_moe_router(rdev, cfg, d_scratch.data(), stream.get());

        // Step 4: expert FFN accumulate.
        lfm::DeviceBuffer<__nv_bfloat16> d_gate_up(
            static_cast<size_t>(p.experts) * 2 * p.inter * p.hidden);
        lfm::DeviceBuffer<__nv_bfloat16> d_down(
            static_cast<size_t>(p.experts) * p.hidden * p.inter);
        lfm::DeviceBuffer<__nv_bfloat16> d_output(p.hidden);
        lfm::DeviceBuffer<float> d_accum(p.hidden);
        lfm::DeviceBuffer<__nv_bfloat16> d_gu_scratch(
            static_cast<size_t>(p.K) * 2 * p.inter);
        lfm::DeviceBuffer<__nv_bfloat16> d_act_scratch(
            static_cast<size_t>(p.K) * p.inter);
        std::vector<__nv_bfloat16> gu_bf16(p.gate_up.size());
        std::vector<__nv_bfloat16> down_bf16(p.down.size());
        for (size_t i = 0; i < p.gate_up.size(); ++i) gu_bf16[i] = to_bf16(p.gate_up[i]);
        for (size_t i = 0; i < p.down.size(); ++i) down_bf16[i] = to_bf16(p.down[i]);
        LFM_CUDA(cudaMemcpy(d_gate_up.data(), gu_bf16.data(),
                            d_gate_up.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(d_down.data(), down_bf16.data(),
                            d_down.bytes(), cudaMemcpyHostToDevice));
        d_output.zero_async(stream.get());
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
                            d_normed.data(), d_accum.data(),
                            1, p.K, d_gu_scratch.data(),
                            d_act_scratch.data(), stream.get());
        lfm::launch_finalize_moe_output(d_accum.data(), d_output.data(),
                                        p.hidden, stream.get());

        // CPU reference of the full pipeline (rmsnorm -> router -> ffn -> residual).
        const std::vector<__nv_bfloat16> normed_cpu_ref = cpu_rmsnorm(
            p.hidden_vec, p.ffn_norm, p.norm_eps);
        std::vector<float> normed_float(p.hidden);
        for (int i = 0; i < p.hidden; ++i) normed_float[i] = from_bf16(normed_cpu_ref[i]);

        std::vector<int> sel_cpu;
        std::vector<float> wts_cpu;
        lfm::compute_moe_router(normed_float, p.router_w, &p.bias, 1, p.hidden,
                                cfg, sel_cpu, wts_cpu);

        std::vector<float> ffn_cpu;
        lfm::compute_moe_ffn(normed_float, p.gate_up, p.down, sel_cpu, wts_cpu,
                             1, p.hidden, p.inter, p.experts, ffn_cpu);

        // Step 5: residual add into hidden.
        lfm::launch_residual_add(d_hidden.data(), d_output.data(),
                                 p.hidden, stream.get());
        LFM_CUDA(cudaStreamSynchronize(stream.get()));

        std::vector<__nv_bfloat16> hidden_gpu(p.hidden);
        LFM_CUDA(cudaMemcpy(hidden_gpu.data(), d_hidden.data(),
                            d_hidden.bytes(), cudaMemcpyDeviceToHost));

        // (a) The router selection/weights must match the reference exactly.
        std::vector<int> sel_gpu(p.K);
        std::vector<float> wts_gpu(p.K);
        LFM_CUDA(cudaMemcpy(sel_gpu.data(), d_sel.data(),
                            d_sel.bytes(), cudaMemcpyDeviceToHost));
        LFM_CUDA(cudaMemcpy(wts_gpu.data(), d_wts.data(),
                            d_wts.bytes(), cudaMemcpyDeviceToHost));
        for (int i = 0; i < p.K; ++i) {
            if (sel_gpu[i] != sel_cpu[i])
                throw std::runtime_error("MoE router selection mismatch at slot " +
                                         std::to_string(i));
            if (std::fabs(wts_gpu[i] - wts_cpu[i]) > 5.0e-4f) {
                std::cout << "  slot " << i << " gpu=" << wts_gpu[i]
                          << " cpu=" << wts_cpu[i]
                          << " diff=" << std::fabs(wts_gpu[i] - wts_cpu[i]) << "\n";
                throw std::runtime_error("MoE router weight mismatch at slot " +
                                         std::to_string(i));
            }
        }

        // (b) The full pipeline (rmsnorm + ffn + residual) must track the float
        // reference within BF16 tolerance.
        float max_rel = 0.0f;
        for (int i = 0; i < p.hidden; ++i) {
            const float got = from_bf16(hidden_gpu[i]);
            const float expected = p.hidden_vec[i] + ffn_cpu[i];
            const float denom = std::max(1.0f, std::fabs(expected));
            max_rel = std::max(max_rel, std::fabs(got - expected) / denom);
        }
        std::cout << "moe_integration_test: max rel diff vs full-pipeline reference = "
                  << max_rel << "\n";
        if (max_rel >= 0.08f) {
            throw std::runtime_error("moe_integration_test: rel diff too large: " +
                                     std::to_string(max_rel));
        }

        std::cout << "moe_integration_test: ok\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "moe_integration_test FAILED: " << e.what() << "\n";
        return 1;
    }
}
