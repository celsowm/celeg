#pragma once

#include "utils.cuh"
#include "backend/cuda/execution_plan.hpp"
#include "runtime_types.hpp"
#include "model/detail/gemm_plan.hpp"
#include "model/detail/linear_weights.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

namespace celeg {

struct CompiledLinearBinding {
    const LinearWeight* weight = nullptr;
    LinearKernelKind kernel = LinearKernelKind::Bf16Cublas;
    int rows = 0;
    int cols = 0;
    uint64_t plan_fingerprint = 0;
};

class GemmDispatcher {
public:
    class NativeFanoutScope {
    public:
        NativeFanoutScope(GemmDispatcher* dispatcher,
                          const __nv_bfloat16* x, int m, int k);
        ~NativeFanoutScope();

        NativeFanoutScope(const NativeFanoutScope&) = delete;
        NativeFanoutScope& operator=(const NativeFanoutScope&) = delete;
        NativeFanoutScope(NativeFanoutScope&& other) noexcept;
        NativeFanoutScope& operator=(NativeFanoutScope&& other) noexcept;

    private:
        GemmDispatcher* dispatcher_ = nullptr;
    };

    GemmDispatcher(cudaStream_t stream,
                   const CudaModelOptions& options);
    ~GemmDispatcher();

    GemmDispatcher(const GemmDispatcher&) = delete;
    GemmDispatcher& operator=(const GemmDispatcher&) = delete;

    void linear(const __nv_bfloat16* x,
                const LinearWeight& weight,
                __nv_bfloat16* y,
                int m, int n, int k,
                float beta,
                const CudaExecutionPlan& plan);

    const CompiledLinearBinding& compile_linear_binding(
        const LinearWeight& weight, const CudaExecutionPlan& plan);

    void linear_cublas(const __nv_bfloat16* x,
                       const __nv_bfloat16* weight,
                       __nv_bfloat16* y,
                       int m, int n, int k,
                       float beta);

    void linear_cublaslt(const __nv_bfloat16* x,
                         const __nv_bfloat16* weight,
                         __nv_bfloat16* y,
                         int m, int n, int k,
                         float beta);

    LtPlan& get_or_create_lt_plan(const __nv_bfloat16* x,
                                  const __nv_bfloat16* weight,
                                  int m, int n, int k);

    // W8A8: dynamic per-token E4M3 activation quantization, a raw (unscaled)
    // FP8xFP8->FP32 cuBLASLt matmul, then a manual outer-product dequant
    // scale applied as a separate kernel epilogue -- see linear.cuh and
    // docs/QWEN3_5_NVFP4_FP8_SUPPORT_PLAN.md Phase 3 for why the scale isn't
    // applied via cuBLASLt's native scale-vector attribute.
    void linear_fp8_w8a8(const __nv_bfloat16* x,
                         const Fp8LinearStorage& weight,
                         __nv_bfloat16* y,
                         int m, int n, int k,
                         float beta);

    LtPlan& get_or_create_fp8_lt_plan(int m, int n, int k);

    // W4A4 (NVFP4): dynamic per-16-block e2m1 activation quantization, both
    // operands' UE4M3 block-scale tensors rearranged into cuBLASLt's
    // documented 128x4 tiled layout, a native block-scaled fp4 matmul, then
    // a post-multiply by the two per-tensor global scales. Falls back to
    // dequantizing the weight to bf16 (linear.cuh's launch_dequant_nvfp4)
    // if no cuBLASLt algorithm is found for the shape. See linear.cuh and
    // docs/QWEN3_5_NVFP4_FP8_SUPPORT_PLAN.md Phase 4.
    void linear_nvfp4_w4a4(const __nv_bfloat16* x,
                           const Nvfp4LinearStorage& weight,
                           __nv_bfloat16* y,
                           int m, int n, int k,
                           float beta);

    LtPlan& get_or_create_nvfp4_lt_plan(int m, int n, int k);

    void begin_native_fanout(const __nv_bfloat16* x, int m, int k);
    void end_native_fanout();

    cudaStream_t stream() const { return stream_; }
    CublasHandle& cublas() { return handles_.cublas; }
    CublasLtHandle& cublas_lt() { return handles_.cublas_lt; }
    size_t workspace_bytes() const { return lt_workspace_.bytes(); }

private:
    struct Handles {
        explicit Handles(cudaStream_t stream) : cublas(stream), cublas_lt() {}
        CublasHandle cublas;
        CublasLtHandle cublas_lt;
    };

    struct LtPlanCache {
        std::unordered_map<MatmulKey, std::unique_ptr<LtPlan>, MatmulKeyHash> plans;
    };

    struct LtAutotuner {
        int choose(bool enabled, const std::vector<int>& candidates,
                   int fallback,
                   const std::function<float(int)>& benchmark) const {
            if (!enabled || candidates.size() <= 1) return fallback;
            int selected = fallback;
            float best = std::numeric_limits<float>::infinity();
            for (const int candidate : candidates) {
                const float elapsed = benchmark(candidate);
                if (elapsed < best) {
                    best = elapsed;
                    selected = candidate;
                }
            }
            return selected;
        }
    };

    struct MmqWorkspace {
        DeviceBuffer<int8_t> q8;
        DeviceBuffer<float> scales;
        DeviceBuffer<float> sums;
        int capacity_m = 0;
        int capacity_k = 0;
        const __nv_bfloat16* fanout_input = nullptr;
        int fanout_m = 0;
        int fanout_k = 0;
    };

    struct Fp8Workspace {
        DeviceBuffer<__nv_fp8_e4m3> act_q;
        DeviceBuffer<float> act_scales;
        DeviceBuffer<float> raw;
        int capacity_m = 0;
        int capacity_n = 0;
        int capacity_k = 0;
    };

    struct Nvfp4Workspace {
        DeviceBuffer<uint8_t> act_packed;
        DeviceBuffer<__nv_fp8_e4m3> act_scale_raw;
        DeviceBuffer<__nv_fp8_e4m3> act_scale_swizzled;
        DeviceBuffer<__nv_fp8_e4m3> weight_scale_swizzled;
        DeviceBuffer<float> raw;
        int capacity_m = 0;
        int capacity_n = 0;
        int capacity_k = 0;
    };

    cudaStream_t stream_;
    int device_ordinal_ = -1;
    const CudaModelOptions& options_;
    Handles handles_;
    DeviceBuffer<std::byte> lt_workspace_;
    LtPlanCache lt_cache_;
    LtPlanCache fp8_lt_cache_;
    LtPlanCache nvfp4_lt_cache_;
    LtAutotuner lt_autotuner_;
    MmqWorkspace mmq_workspace_;
    Fp8Workspace fp8_workspace_;
    Nvfp4Workspace nvfp4_workspace_;
    std::unordered_map<const LinearWeight*, CompiledLinearBinding>
        linear_bindings_;

    void ensure_mmq_capacity(int m, int k);
    void ensure_fp8_capacity(int m, int n, int k);
    void ensure_nvfp4_capacity(int m, int n, int k);
    bool has_native_fanout(const __nv_bfloat16* x, int m, int k) const;
};

}
