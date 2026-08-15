#pragma once

#include "celeg/backend/cuda/utils.cuh"
#include "celeg/backend/cuda/execution_plan.hpp"
#include "celeg/backend/cuda/runtime_types.hpp"
#include "celeg/detail/model/gemm_plan.hpp"
#include "celeg/detail/model/linear_weights.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
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

    cudaStream_t stream_;
    int device_ordinal_ = -1;
    const CudaModelOptions& options_;
    Handles handles_;
    DeviceBuffer<std::byte> lt_workspace_;
    LtPlanCache lt_cache_;
    LtAutotuner lt_autotuner_;
    MmqWorkspace mmq_workspace_;
    std::unordered_map<const LinearWeight*, CompiledLinearBinding>
        linear_bindings_;

    void ensure_mmq_capacity(int m, int k);
    bool has_native_fanout(const __nv_bfloat16* x, int m, int k) const;
};

}
