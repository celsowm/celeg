#pragma once

#include "lfm/cuda_utils.cuh"
#include "lfm/execution_plan.hpp"
#include "lfm/runtime_types.hpp"
#include "lfm/detail/model_types.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

namespace lfm {

// Dispatches linear (GEMM) operations across cuBLAS, cuBLASLt, and
// specialized INT4/INT8 kernels based on the active ExecutionPlan and the
// weight storage kind. Extracted from LfmModel::Impl for Single
// Responsibility: this class owns the cuBLAS handles, the cuBLASLt plan
// cache, and the autotuning workspace; the Impl retains the forward pass,
// session state, and graph capture. New GEMM backends (e.g. CUTLASS) are
// added by extending the dispatch switch (Open/Closed Principle).
class GemmDispatcher {
public:
    // Constructs a dispatcher bound to a stream + ModelOptions. Allocates
    // the cuBLAS / cuBLASLt handles and the cuBLASLt workspace.
    GemmDispatcher(cudaStream_t stream,
                   const ModelOptions& options);
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
                const ExecutionPlan& plan);

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

    cudaStream_t stream() const { return stream_; }
    CublasHandle& cublas() { return cublas_; }
    CublasLtHandle& cublas_lt() { return cublas_lt_; }
    size_t workspace_bytes() const { return lt_workspace_.bytes(); }

private:
    cudaStream_t stream_;
    const ModelOptions& options_;
    CublasHandle cublas_;
    CublasLtHandle cublas_lt_;
    DeviceBuffer<std::byte> lt_workspace_;
    std::unordered_map<MatmulKey, std::unique_ptr<LtPlan>, MatmulKeyHash> lt_plans_;
};

} // namespace lfm
