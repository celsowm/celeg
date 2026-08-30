#include "celeg/backend/cuda/moe.hpp"
#include "support/assertions.hpp"
#include "celeg/backend/cuda/utils.cuh"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

struct Problem {
    int rows;
    int hidden;
    int inter;
    int experts;
    int K;
    std::vector<float> hidden_vec;
    std::vector<float> router_w;
    std::vector<float> gate_up;
    std::vector<float> down;
    std::vector<float> bias;
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

void run_native_quantized_case(celeg::GgmlType gate_type,
                               celeg::GgmlType down_type) {
    constexpr int rows = 2;
    constexpr int hidden = 256;
    constexpr int inter = 256;
    constexpr int experts = 2;
    constexpr int K = 2;
    constexpr size_t q4_row_bytes = 144;
    constexpr size_t q6_row_bytes = 210;

    const size_t gate_row_bytes = gate_type == celeg::GgmlType::Q4_K
        ? q4_row_bytes : q6_row_bytes;
    const size_t down_row_bytes = down_type == celeg::GgmlType::Q4_K
        ? q4_row_bytes : q6_row_bytes;
    const size_t gate_stride = static_cast<size_t>(2 * inter) * gate_row_bytes;
    const size_t down_stride = static_cast<size_t>(hidden) * down_row_bytes;

    auto make_blocks = [](celeg::GgmlType type, size_t count) {
        const size_t row_bytes = type == celeg::GgmlType::Q4_K
            ? q4_row_bytes : q6_row_bytes;
        std::vector<uint8_t> bytes(count * row_bytes, 0);
        for (size_t i = 0; i < count; ++i) {
            uint8_t* block = bytes.data() + i * row_bytes;
            block[type == celeg::GgmlType::Q4_K ? 0 : 208] = 0x00;
            block[type == celeg::GgmlType::Q4_K ? 1 : 209] = 0x3c;
            if (type == celeg::GgmlType::Q4_K) {
                for (int s = 4; s < 12; ++s) block[s] = 1;
                for (int q = 16; q < 144; ++q) block[q] = 0x22;
            } else {
                for (int q = 0; q < 128; ++q) block[q] = 0x11;
                for (int q = 128; q < 192; ++q) block[q] = 0xaa;
                for (int q = 192; q < 208; ++q) block[q] = 1;
            }
        }
        return bytes;
    };

    const std::vector<uint8_t> gate_host = make_blocks(
        gate_type, static_cast<size_t>(experts) * 2 * inter);
    const std::vector<uint8_t> down_host = make_blocks(
        down_type, static_cast<size_t>(experts) * hidden);
    celeg::DeviceBuffer<uint8_t> gate_device(gate_host.size());
    celeg::DeviceBuffer<uint8_t> down_device(down_host.size());
    CELEG_CUDA(cudaMemcpy(gate_device.data(), gate_host.data(), gate_host.size(),
                        cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(down_device.data(), down_host.data(), down_host.size(),
                        cudaMemcpyHostToDevice));

    std::vector<float> hidden_host(static_cast<size_t>(rows) * hidden);
    for (size_t i = 0; i < hidden_host.size(); ++i)
        hidden_host[i] = 0.01f + 0.0001f * static_cast<float>(i % 17);
    std::vector<__nv_bfloat16> hidden_bf16(hidden_host.size());
    for (size_t i = 0; i < hidden_host.size(); ++i)
        hidden_bf16[i] = to_bf16(hidden_host[i]);
    celeg::DeviceBuffer<__nv_bfloat16> hidden_device(hidden_bf16.size());
    CELEG_CUDA(cudaMemcpy(hidden_device.data(), hidden_bf16.data(),
                        hidden_device.bytes(), cudaMemcpyHostToDevice));

    const int selected_host[] = {0, 1, 1, 0};
    const float routing_host[] = {0.25f, 0.75f, 0.6f, 0.4f};
    celeg::DeviceBuffer<int> selected_device(rows * K);
    celeg::DeviceBuffer<float> routing_device(rows * K);
    CELEG_CUDA(cudaMemcpy(selected_device.data(), selected_host,
                        selected_device.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(routing_device.data(), routing_host,
                        routing_device.bytes(), cudaMemcpyHostToDevice));

    celeg::DeviceBuffer<float> accum(rows * hidden);
    celeg::DeviceBuffer<__nv_bfloat16> activated(rows * K * inter);
    celeg::CudaStream stream;
    accum.zero_async(stream.get());
    celeg::MoeFfnDevice device;
    device.gate_up_gguf = gate_device.data();
    device.down_gguf = down_device.data();
    device.gate_up_gguf_type = gate_type;
    device.down_gguf_type = down_type;
    device.expert_gate_up_row_bytes = gate_row_bytes;
    device.expert_down_row_bytes = down_row_bytes;
    device.expert_gate_up_byte_stride = gate_stride;
    device.expert_down_byte_stride = down_stride;
    device.num_experts = experts;
    device.inter = inter;
    device.hidden_dim = hidden;

    celeg::launch_moe_ffn(device, selected_device.data(), routing_device.data(),
                        hidden_device.data(), accum.data(), rows, K, nullptr,
                        activated.data(), stream.get());
    celeg::DeviceBuffer<__nv_bfloat16> output(rows * hidden);
    celeg::launch_finalize_moe_output(accum.data(), output.data(), rows * hidden,
                                    stream.get());
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));

    std::vector<float> actual(rows * hidden);
    std::vector<__nv_bfloat16> output_host(static_cast<size_t>(rows) * hidden);
    CELEG_CUDA(cudaMemcpy(output_host.data(), output.data(), output.bytes(),
                        cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < actual.size(); ++i)
        actual[i] = __bfloat162float(output_host[i]);
    for (int row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (int h = 0; h < hidden; ++h)
            sum += __bfloat162float(hidden_bf16[static_cast<size_t>(row) * hidden + h]);
        const float activated_value = (1.0f / (1.0f + std::exp(-sum))) * sum;
        const float expected = (routing_host[row * K] + routing_host[row * K + 1]) *
            static_cast<float>(inter) * activated_value;
        for (int h = 0; h < hidden; ++h) {
            const float denom = std::max(1.0f, std::fabs(expected));
            CELEG_TEST_CHECK(std::fabs(actual[static_cast<size_t>(row) * hidden + h] - expected) /
                           denom < 0.02f);
        }
    }
}

}

int main() {
    try {
        const bool with_bias = true;
        Problem p = build(3, 8, 16, 6, 4, with_bias);

        run_native_quantized_case(celeg::GgmlType::Q4_K, celeg::GgmlType::Q6_K);
        run_native_quantized_case(celeg::GgmlType::Q6_K, celeg::GgmlType::Q4_K);

        celeg::MoeRouterConfig cfg;
        cfg.num_experts = p.experts;
        cfg.experts_per_token = p.K;
        cfg.normalize_topk = true;
        cfg.use_expert_bias = with_bias;
        cfg.routed_scaling_factor = 1.0f;

        celeg::CudaStream stream;

        celeg::DeviceBuffer<float> d_hidden(p.rows * p.hidden);
        celeg::DeviceBuffer<float> d_router(static_cast<size_t>(p.experts) * p.hidden);
        celeg::DeviceBuffer<float> d_bias(p.experts);
        celeg::DeviceBuffer<int> d_sel(static_cast<size_t>(p.rows) * p.K);
        celeg::DeviceBuffer<float> d_wts(static_cast<size_t>(p.rows) * p.K);
        celeg::DeviceBuffer<float> d_scratch(static_cast<size_t>(p.rows) * p.experts);
        CELEG_CUDA(cudaMemcpy(d_hidden.data(), p.hidden_vec.data(), d_hidden.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(d_router.data(), p.router_w.data(), d_router.bytes(), cudaMemcpyHostToDevice));
        if (with_bias)
            CELEG_CUDA(cudaMemcpy(d_bias.data(), p.bias.data(), d_bias.bytes(), cudaMemcpyHostToDevice));

        celeg::MoeRouterDevice rdev;
        rdev.router_weight = d_router.data();
        rdev.expert_bias = with_bias ? d_bias.data() : nullptr;
        rdev.hidden_data = d_hidden.data();
        rdev.selected_experts = d_sel.data();
        rdev.routing_weights = d_wts.data();
        rdev.rows = p.rows;
        rdev.hidden_dim = p.hidden;
        celeg::launch_moe_router(rdev, cfg, d_scratch.data(), stream.get());
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));

        std::vector<int> sel_gpu(static_cast<size_t>(p.rows) * p.K);
        std::vector<float> wts_gpu(static_cast<size_t>(p.rows) * p.K);
        CELEG_CUDA(cudaMemcpy(sel_gpu.data(), d_sel.data(), d_sel.bytes(), cudaMemcpyDeviceToHost));
        CELEG_CUDA(cudaMemcpy(wts_gpu.data(), d_wts.data(), d_wts.bytes(), cudaMemcpyDeviceToHost));

        std::vector<int> sel_cpu;
        std::vector<float> wts_cpu;
        celeg::compute_moe_router(p.hidden_vec, p.router_w,
                                with_bias ? &p.bias : nullptr,
                                p.rows, p.hidden, cfg, sel_cpu, wts_cpu);
        for (size_t i = 0; i < sel_gpu.size(); ++i) CELEG_TEST_CHECK(sel_gpu[i] == sel_cpu[i]);

        celeg::DeviceBuffer<__nv_bfloat16> d_gate_up(
            static_cast<size_t>(p.experts) * 2 * p.inter * p.hidden);
        celeg::DeviceBuffer<__nv_bfloat16> d_down(
            static_cast<size_t>(p.experts) * p.hidden * p.inter);
        celeg::DeviceBuffer<__nv_bfloat16> d_hidden_bf16(p.rows * p.hidden);
        celeg::DeviceBuffer<__nv_bfloat16> d_output(p.rows * p.hidden);
        celeg::DeviceBuffer<__nv_bfloat16> d_gu_scratch(
            static_cast<size_t>(p.rows) * p.K * 2 * p.inter);
        celeg::DeviceBuffer<__nv_bfloat16> d_act_scratch(
            static_cast<size_t>(p.rows) * p.K * p.inter);

        std::vector<__nv_bfloat16> gu_bf16(p.gate_up.size());
        std::vector<__nv_bfloat16> down_bf16(p.down.size());
        std::vector<__nv_bfloat16> h_bf16(p.hidden_vec.size());
        for (size_t i = 0; i < p.gate_up.size(); ++i) gu_bf16[i] = to_bf16(p.gate_up[i]);
        for (size_t i = 0; i < p.down.size(); ++i) down_bf16[i] = to_bf16(p.down[i]);
        for (size_t i = 0; i < p.hidden_vec.size(); ++i) h_bf16[i] = to_bf16(p.hidden_vec[i]);
        CELEG_CUDA(cudaMemcpy(d_gate_up.data(), gu_bf16.data(), d_gate_up.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(d_down.data(), down_bf16.data(), d_down.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(d_hidden_bf16.data(), h_bf16.data(), d_hidden_bf16.bytes(), cudaMemcpyHostToDevice));
        d_output.zero_async(stream.get());
        celeg::DeviceBuffer<float> d_accum(p.rows * p.hidden);
        d_accum.zero_async(stream.get());

        celeg::MoeFfnDevice fdev;
        fdev.gate_up = d_gate_up.data();
        fdev.down = d_down.data();
        fdev.num_experts = p.experts;
        fdev.inter = p.inter;
        fdev.hidden_dim = p.hidden;
        fdev.expert_gate_up_stride = static_cast<size_t>(2) * p.inter * p.hidden;
        fdev.expert_down_stride = static_cast<size_t>(p.hidden) * p.inter;
        celeg::launch_moe_ffn(fdev, d_sel.data(), d_wts.data(),
                            d_hidden_bf16.data(), d_accum.data(),
                            p.rows, p.K, d_gu_scratch.data(),
                            d_act_scratch.data(), stream.get());
        celeg::launch_finalize_moe_output(d_accum.data(), d_output.data(),
                                        p.rows * p.hidden, stream.get());
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));

        std::vector<__nv_bfloat16> out_gpu(p.rows * p.hidden);
        CELEG_CUDA(cudaMemcpy(out_gpu.data(), d_output.data(), d_output.bytes(), cudaMemcpyDeviceToHost));

        std::vector<float> out_cpu;
        celeg::compute_moe_ffn(p.hidden_vec, p.gate_up, p.down,
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
        CELEG_TEST_CHECK(max_rel < 0.25f);

        float max_abs_bf16 = 0.0f;
        for (size_t i = 0; i < out_gpu.size(); ++i) {
            const float g = __bfloat162float(out_gpu[i]);
            const float bf16_expected = __bfloat162float(to_bf16(out_cpu[i]));
            max_abs_bf16 = std::max(max_abs_bf16, std::fabs(g - bf16_expected));
        }
        std::cout << "moe_ffn_test: max abs diff vs bf16-cast reference = " << max_abs_bf16 << "\n";
        CELEG_TEST_CHECK(max_abs_bf16 < 0.1f);

        std::cout << "moe_ffn_test: ok\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "moe_ffn_test FAILED: " << e.what() << "\n";
        return 1;
    }
}
