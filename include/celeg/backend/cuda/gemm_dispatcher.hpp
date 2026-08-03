#pragma once

#include "celeg/backend/cuda/utils.cuh"
#include "celeg/backend/cuda/execution_plan.hpp"
#include "celeg/backend/cuda/runtime_types.hpp"
#include "celeg/detail/model/types.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

namespace celeg {

// Dispatches linear (GEMM) operations across cuBLAS, cuBLASLt, and
// specialized INT4/INT8 kernels based on the active CudaExecutionPlan and the
// weight storage kind. Extracted from the compiled model for Single
// Responsibility: this class owns the cuBLAS handles, the cuBLASLt plan
// cache, and the autotuning workspace; the compiled model retains the forward pass,
// session state, and graph capture. New GEMM backends (e.g. CUTLASS) are
// added by extending the dispatch switch (Open/Closed Principle).
class GemmDispatcher {
public:
    // Constructs a dispatcher bound to a stream + CudaModelOptions. Allocates
    // the cuBLAS / cuBLASLt handles and the cuBLASLt workspace.
    GemmDispatcher(cudaStream_t stream,
                   const CudaModelOptions& options);
    ~GemmDispatcher();

    GemmDispatcher(const GemmDispatcher&) = delete;
    GemmDispatcher& operator=(const GemmDispatcher&) = delete;

    // Main dispatch entry point. Selects the GEMM backend from `plan` and
    // the weight storage kind. `weight` must outlive the call.
    void linear(const __nv_bfloat16* x,
                const LinearWeight& weight,
                __nv_bfloat16* y,
                int m, int n, int k,
                float beta,
                const CudaExecutionPlan& plan);

    // Direct cuBLAS GEMM (BF16, transposed weight).
    void linear_cublas(const __nv_bfloat16* x,
                       const __nv_bfloat16* weight,
                       __nv_bfloat16* y,
                       int m, int n, int k,
                       float beta);

    // Direct cuBLASLt GEMM (BF16, transposed weight). Falls back to
    // linear_cublas if no plan is available.
    void linear_cublaslt(const __nv_bfloat16* x,
                         const __nv_bfloat16* weight,
                         __nv_bfloat16* y,
                         int m, int n, int k,
                         float beta);

    // Looks up or creates a cuBLASLt plan for (m, n, k). Auto-tunes for
    // decode shapes (m == 1) when lt_autotune is enabled.
    LtPlan& get_or_create_lt_plan(const __nv_bfloat16* x,
                                  const __nv_bfloat16* weight,
                                  int m, int n, int k);

    // Begins an explicit fan-out scope for native GGUF projections sharing
    // exactly one input matrix. The activation is quantized once and reused
    // by subsequent linear() calls in the scope. Call end_native_fanout()
    // before the source buffer is written again.
    void begin_native_fanout(const __nv_bfloat16* x, int m, int k);
    void end_native_fanout();

    cudaStream_t stream() const { return stream_; }
    CublasHandle& cublas() { return cublas_; }
    CublasLtHandle& cublas_lt() { return cublas_lt_; }
    size_t workspace_bytes() const { return lt_workspace_.bytes(); }

private:
    cudaStream_t stream_;
    const CudaModelOptions& options_;
    CublasHandle cublas_;
    CublasLtHandle cublas_lt_;
    DeviceBuffer<std::byte> lt_workspace_;
    std::unordered_map<MatmulKey, std::unique_ptr<LtPlan>, MatmulKeyHash> lt_plans_;

    // Scratch buffers for the MMQ (native-quant) path. Lazily resized to
    // the largest (m, k) seen across linear() calls. `mmq_q8_` holds the
    // Q8_1-quantized activations, `mmq_scales_` and `mmq_sums_` hold the
    // per-32-element block scales and integer sums.
    DeviceBuffer<int8_t> mmq_q8_;
    DeviceBuffer<float> mmq_scales_;
    DeviceBuffer<float> mmq_sums_;
    int mmq_capacity_m_ = 0;
    int mmq_capacity_k_ = 0;
    const __nv_bfloat16* native_fanout_input_ = nullptr;
    int native_fanout_m_ = 0;
    int native_fanout_k_ = 0;

    void ensure_mmq_capacity(int m, int k);
    bool has_native_fanout(const __nv_bfloat16* x, int m, int k) const;
};

} // namespace celeg
