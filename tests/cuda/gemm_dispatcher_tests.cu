#include "gemm_dispatcher_tests.hpp"

#include "utils.cuh"
#include "../support/assertions.hpp"
#include "../support/cuda_kernel_assertions.cuh"
#include "kernels/kernels.cuh"
#include "backend/cuda/gemm_dispatcher.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <tuple>
#include <vector>

namespace celeg::cuda_test {

void run_gemm_dispatcher_tests(celeg::CudaStream& stream) {
/// FP8 W8A8: exercises GemmDispatcher's LinearKernelKind::Fp8W8A8 path
/// end to end (dynamic per-token activation quant -> raw cuBLASLt fp8
/// matmul -> manual outer-product scale-apply epilogue). The reference is
/// built from the kernel's own quantized values (round-tripped through
/// the same launch_quantize_e4m3_per_row kernel under test) rather than
/// an independently reimplemented e4m3 rounding rule, so this isolates
/// "does the matmul+scale-apply reproduce the quantized dot product"
/// from "is the quantization rounding bit-for-bit what a host
/// reimplementation would produce" -- the latter isn't this kernel's
/// contract (it only needs to round *some* IEEE-754-correct e4m3 way).
/// Run the FP8 W8A8 check for two shapes: one aligned enough for the
/// cuBLASLt fp8 heuristic to find an algorithm, and one (n=3) that isn't
/// -- exercising GemmDispatcher's naive-kernel fallback for shapes with
/// no available fp8 cuBLASLt algorithm (see
/// docs/QWEN3_5_NVFP4_FP8_SUPPORT_PLAN.md Phase 3).
for (const auto& [m, n, k] : {std::tuple{4, 8, 32}, std::tuple{2, 3, 32}}) {
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<__nv_bfloat16> x(m * k), w(n * k);
    for (auto& v : x) v = to_bf16(dist(rng));
    for (auto& v : w) v = to_bf16(dist(rng));

    celeg::DeviceBuffer<__nv_bfloat16> dx(x.size()), dw_bf16(w.size());
    CELEG_CUDA(cudaMemcpy(dx.data(), x.data(), dx.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dw_bf16.data(), w.data(), dw_bf16.bytes(), cudaMemcpyHostToDevice));

    /// Quantize both operands with the kernel under test so the scalar
    /// reference matches exactly what the dispatcher will consume/produce.
    celeg::DeviceBuffer<__nv_fp8_e4m3> dx_q(x.size()), dw_q(w.size());
    celeg::DeviceBuffer<float> dx_scales(m), dw_scales(n);
    celeg::launch_quantize_e4m3_per_row(dx.data(), dx_q.data(), dx_scales.data(), m, k, stream.get());
    celeg::launch_quantize_e4m3_per_row(dw_bf16.data(), dw_q.data(), dw_scales.data(), n, k, stream.get());
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));

    std::vector<__nv_fp8_e4m3> x_q(x.size()), w_q(w.size());
    std::vector<float> x_scales(m), w_scales(n);
    CELEG_CUDA(cudaMemcpy(x_q.data(), dx_q.data(), dx_q.bytes(), cudaMemcpyDeviceToHost));
    CELEG_CUDA(cudaMemcpy(w_q.data(), dw_q.data(), dw_q.bytes(), cudaMemcpyDeviceToHost));
    CELEG_CUDA(cudaMemcpy(x_scales.data(), dx_scales.data(), dx_scales.bytes(), cudaMemcpyDeviceToHost));
    CELEG_CUDA(cudaMemcpy(w_scales.data(), dw_scales.data(), dw_scales.bytes(), cudaMemcpyDeviceToHost));

    std::vector<float> reference(m * n, 0.0f);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (int i = 0; i < k; ++i) {
                acc += float(x_q[row * k + i]) * float(w_q[col * k + i]);
            }
            reference[row * n + col] = acc * x_scales[row] * w_scales[col];
        }
    }

    celeg::CudaModelOptions dispatcher_options;
    celeg::GemmDispatcher dispatcher(stream.get(), dispatcher_options);
    celeg::LinearWeight weight;
    weight.rows = n;
    weight.cols = k;
    weight.kernel = celeg::LinearKernelKind::Fp8W8A8;
    weight.storage = celeg::Fp8LinearStorage{dw_q.data(), dw_scales.data()};
    const celeg::CudaExecutionPlan plan =
        celeg::CudaExecutionPlan::compile(dispatcher_options, 1024);

    celeg::DeviceBuffer<__nv_bfloat16> dy(m * n);
    dispatcher.linear(dx.data(), weight, dy.data(), m, n, k, 0.0f, plan);
    std::vector<__nv_bfloat16> y(m * n);
    CELEG_CUDA(cudaMemcpyAsync(y.data(), dy.data(), dy.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    for (int i = 0; i < m * n; ++i) {
        expect_near(to_float(y[i]), reference[i],
                   0.02f * std::max(1.0f, std::abs(reference[i])));
    }
}

/// NVFP4 W4A4: exercises GemmDispatcher's LinearKernelKind::Nvfp4W4A4
/// path end to end -- dynamic per-16-block e2m1 activation quantization,
/// both operands' UE4M3 scale tensors rearranged into cuBLASLt's 128x4
/// tiled layout, a native block-scaled fp4 matmul, then the per-tensor
/// global-scale post-multiply. See docs/QWEN3_5_NVFP4_FP8_SUPPORT_PLAN.md
/// Phase 4 for how this layout was found and verified (bit-exact against
/// a scalar reference at several shapes in the standalone spike). Run
/// for two shapes: one row-tile-aligned (n=128) and one not (n=6, which
/// only differs in output width -- the scale/data padding logic handles
/// both) to cover the padding path.
static const float kE2m1Lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
auto decode_e2m1_ref = [](uint8_t nibble) -> float {
    const bool sign = nibble & 0x8;
    const float mag = kE2m1Lut[nibble & 0x7];
    return sign ? -mag : mag;
};
for (const auto& [m, n, k] : {std::tuple{4, 8, 32}, std::tuple{8, 128, 256}}) {
    const int block = celeg::kNvfp4BlockSize;
    const int blocks_per_row = k / block;
    constexpr float weight_global_scale = 1.5f;
    /// Matches GemmDispatcher's current default.
    constexpr float act_global_scale = 1.0f;

    std::mt19937 rng(13);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<__nv_bfloat16> x(m * k), w(n * k);
    for (auto& v : x) v = to_bf16(dist(rng));
    for (auto& v : w) v = to_bf16(dist(rng));

    celeg::DeviceBuffer<__nv_bfloat16> dx(x.size()), dw_bf16(w.size());
    CELEG_CUDA(cudaMemcpy(dx.data(), x.data(), dx.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dw_bf16.data(), w.data(), dw_bf16.bytes(), cudaMemcpyHostToDevice));

    /// Quantize the weight with the kernel under test (same one
    /// GemmDispatcher uses for activations) so the scalar reference
    /// matches exactly what the dispatcher will consume/produce.
    celeg::DeviceBuffer<uint8_t> dw_packed(static_cast<size_t>(n) * k / 2);
    celeg::DeviceBuffer<__nv_fp8_e4m3> dw_scales(static_cast<size_t>(n) * blocks_per_row);
    celeg::launch_quantize_e2m1_per_block(dw_bf16.data(), dw_packed.data(), dw_scales.data(),
                                          n, k, block, weight_global_scale, stream.get());
    /// Reference-only: quantize activations too (GemmDispatcher does
    /// this internally with act_global_scale=1.0) so the reference dot
    /// product uses the exact same quantized values on both sides.
    celeg::DeviceBuffer<uint8_t> dx_packed(static_cast<size_t>(m) * k / 2);
    celeg::DeviceBuffer<__nv_fp8_e4m3> dx_scales(static_cast<size_t>(m) * blocks_per_row);
    celeg::launch_quantize_e2m1_per_block(dx.data(), dx_packed.data(), dx_scales.data(),
                                          m, k, block, act_global_scale, stream.get());
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));

    std::vector<uint8_t> x_packed(dx_packed.size()), w_packed(dw_packed.size());
    std::vector<__nv_fp8_e4m3> x_scales(dx_scales.size()), w_scales(dw_scales.size());
    CELEG_CUDA(cudaMemcpy(x_packed.data(), dx_packed.data(), dx_packed.bytes(), cudaMemcpyDeviceToHost));
    CELEG_CUDA(cudaMemcpy(w_packed.data(), dw_packed.data(), dw_packed.bytes(), cudaMemcpyDeviceToHost));
    CELEG_CUDA(cudaMemcpy(x_scales.data(), dx_scales.data(), dx_scales.bytes(), cudaMemcpyDeviceToHost));
    CELEG_CUDA(cudaMemcpy(w_scales.data(), dw_scales.data(), dw_scales.bytes(), cudaMemcpyDeviceToHost));

    auto dequant_row = [&](const std::vector<uint8_t>& packed, const std::vector<__nv_fp8_e4m3>& scales,
                          int row, float global_scale) {
        std::vector<float> out(k);
        for (int b = 0; b < blocks_per_row; ++b) {
            const float scale = float(scales[row * blocks_per_row + b]) / global_scale;
            for (int i = 0; i < block; i += 2) {
                const int idx = b * block + i;
                const uint8_t byte = packed[(row * k + idx) / 2];
                out[idx] = decode_e2m1_ref(byte & 0xF) * scale;
                out[idx + 1] = decode_e2m1_ref((byte >> 4) & 0xF) * scale;
            }
        }
        return out;
    };
    std::vector<float> reference(m * n, 0.0f);
    for (int row = 0; row < m; ++row) {
        auto xr = dequant_row(x_packed, x_scales, row, act_global_scale);
        for (int col = 0; col < n; ++col) {
            auto wr = dequant_row(w_packed, w_scales, col, weight_global_scale);
            float acc = 0.0f;
            for (int i = 0; i < k; ++i) acc += xr[i] * wr[i];
            reference[row * n + col] = acc;
        }
    }

    celeg::CudaModelOptions dispatcher_options;
    celeg::GemmDispatcher dispatcher(stream.get(), dispatcher_options);
    celeg::LinearWeight weight;
    weight.rows = n;
    weight.cols = k;
    weight.kernel = celeg::LinearKernelKind::Nvfp4W4A4;
    weight.storage = celeg::Nvfp4LinearStorage{dw_packed.data(), dw_scales.data(), weight_global_scale};
    const celeg::CudaExecutionPlan plan =
        celeg::CudaExecutionPlan::compile(dispatcher_options, 1024);

    celeg::DeviceBuffer<__nv_bfloat16> dy(m * n);
    dispatcher.linear(dx.data(), weight, dy.data(), m, n, k, 0.0f, plan);
    std::vector<__nv_bfloat16> y(m * n);
    CELEG_CUDA(cudaMemcpyAsync(y.data(), dy.data(), dy.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    for (int i = 0; i < m * n; ++i) {
        expect_near(to_float(y[i]), reference[i],
                   0.02f * std::max(1.0f, std::abs(reference[i])));
    }
}
}

}
