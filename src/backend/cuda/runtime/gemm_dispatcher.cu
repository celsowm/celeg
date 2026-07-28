#include "lfm/backend/cuda/gemm_dispatcher.hpp"
#include "lfm/backend/cuda/kernels/gguf.cuh"
#include "lfm/backend/cuda/kernels/embedding.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace lfm {

GemmDispatcher::GemmDispatcher(cudaStream_t stream,
                               const ModelOptions& options)
    : stream_(stream),
      options_(options),
      cublas_(stream),
      cublas_lt_() {
    if (options.gemm_backend == GemmBackend::CublasLt &&
        options.lt_workspace_bytes > 0) {
        lt_workspace_.reset(options.lt_workspace_bytes);
    }
}

GemmDispatcher::~GemmDispatcher() = default;

void GemmDispatcher::linear_cublas(const __nv_bfloat16* x,
                                   const __nv_bfloat16* weight,
                                   __nv_bfloat16* y,
                                   int m, int n, int k,
                                   float beta) {
    const float alpha = 1.0f;
    LFM_CUBLAS(cublasGemmEx(
        cublas_.get(), CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k, &alpha,
        weight, CUDA_R_16BF, k,
        x, CUDA_R_16BF, k,
        &beta,
        y, CUDA_R_16BF, n,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

LtPlan& GemmDispatcher::get_or_create_lt_plan(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    int m, int n, int k) {
    const MatmulKey key{m, n, k};
    if (auto it = lt_plans_.find(key); it != lt_plans_.end()) {
        return *it->second;
    }

    auto plan = std::make_unique<LtPlan>();
    LFM_CUBLAS(cublasLtMatmulDescCreate(
        &plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t transa = CUBLAS_OP_T;
    const cublasOperation_t transb = CUBLAS_OP_N;
    LFM_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_TRANSA,
        &transa, sizeof(transa)));
    LFM_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_TRANSB,
        &transb, sizeof(transb)));

    // Buffers are row-major in the runtime. Interpreting them as column-major
    // transposes the logical matrices: W[n,k] -> A[k,n], X[m,k] -> B[k,m],
    // and Y[m,n] -> D[n,m]. TRANSA restores W to [n,k].
    LFM_CUBLAS(cublasLtMatrixLayoutCreate(
        &plan->a, CUDA_R_16BF, k, n, k));
    LFM_CUBLAS(cublasLtMatrixLayoutCreate(
        &plan->b, CUDA_R_16BF, k, m, k));
    LFM_CUBLAS(cublasLtMatrixLayoutCreate(
        &plan->c, CUDA_R_16BF, n, m, n));
    LFM_CUBLAS(cublasLtMatrixLayoutCreate(
        &plan->d, CUDA_R_16BF, n, m, n));

    cublasLtMatmulPreference_t preference = nullptr;
    LFM_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    try {
        const uint64_t workspace_limit =
            static_cast<uint64_t>(options_.lt_workspace_bytes);
        LFM_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_limit, sizeof(workspace_limit)));

        std::vector<cublasLtMatmulHeuristicResult_t> results(
            static_cast<size_t>(options_.lt_heuristics));
        int returned = 0;
        LFM_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
            cublas_lt_.get(), plan->operation,
            plan->a, plan->b, plan->c, plan->d,
            preference, options_.lt_heuristics,
            results.data(), &returned));

        std::vector<int> valid;
        for (int i = 0; i < returned; ++i) {
            if (results[static_cast<size_t>(i)].state == CUBLAS_STATUS_SUCCESS &&
                results[static_cast<size_t>(i)].workspaceSize <=
                    options_.lt_workspace_bytes) {
                valid.push_back(i);
            }
        }

        if (!valid.empty()) {
            int selected = valid.front();
            // Autotuning is intentionally restricted to decode shapes. Large
            // prefill outputs would require an excessive temporary buffer.
            if (options_.lt_autotune && m == 1 && valid.size() > 1) {
                DeviceBuffer<__nv_bfloat16> scratch(static_cast<size_t>(n));
                const float alpha = 1.0f;
                const float beta = 0.0f;
                float best_ms = std::numeric_limits<float>::infinity();
                for (const int candidate : valid) {
                    const auto& result = results[static_cast<size_t>(candidate)];
                    LFM_CUBLAS(cublasLtMatmul(
                        cublas_lt_.get(), plan->operation,
                        &alpha, weight, plan->a, x, plan->b,
                        &beta, scratch.data(), plan->c,
                        scratch.data(), plan->d, &result.algo,
                        lt_workspace_.data(), result.workspaceSize,
                        stream_));
                    CudaEvent begin;
                    CudaEvent end;
                    begin.record(stream_);
                    constexpr int iterations = 3;
                    for (int iteration = 0; iteration < iterations; ++iteration) {
                        LFM_CUBLAS(cublasLtMatmul(
                            cublas_lt_.get(), plan->operation,
                            &alpha, weight, plan->a, x, plan->b,
                            &beta, scratch.data(), plan->c,
                            scratch.data(), plan->d, &result.algo,
                            lt_workspace_.data(), result.workspaceSize,
                            stream_));
                    }
                    end.record(stream_);
                    end.synchronize();
                    const float elapsed = CudaEvent::elapsed_ms(begin, end);
                    if (elapsed < best_ms) {
                        best_ms = elapsed;
                        selected = candidate;
                    }
                }
            }
            const auto& chosen = results[static_cast<size_t>(selected)];
            plan->algorithm = chosen.algo;
            plan->workspace_size = chosen.workspaceSize;
            plan->available = true;
        }
    } catch (...) {
        cublasLtMatmulPreferenceDestroy(preference);
        throw;
    }
    LFM_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));

    auto [it, inserted] = lt_plans_.emplace(key, std::move(plan));
    if (!inserted) throw std::runtime_error("duplicate cuBLASLt plan");
    return *it->second;
}

void GemmDispatcher::linear_cublaslt(const __nv_bfloat16* x,
                                     const __nv_bfloat16* weight,
                                     __nv_bfloat16* y,
                                     int m, int n, int k,
                                     float beta) {
    LtPlan& plan = get_or_create_lt_plan(x, weight, m, n, k);
    if (!plan.available) {
        linear_cublas(x, weight, y, m, n, k, beta);
        return;
    }
    const float alpha = 1.0f;
    LFM_CUBLAS(cublasLtMatmul(
        cublas_lt_.get(), plan.operation,
        &alpha, weight, plan.a, x, plan.b,
        &beta, y, plan.c, y, plan.d,
        &plan.algorithm, lt_workspace_.data(), plan.workspace_size,
        stream_));
}

void GemmDispatcher::linear(const __nv_bfloat16* x,
                            const LinearWeight& weight,
                            __nv_bfloat16* y,
                            int m, int n, int k,
                            float beta,
                            const ExecutionPlan& plan) {
    if (weight.rows != n || weight.cols != k) {
        throw std::runtime_error("linear weight shape does not match the requested GEMM");
    }
    weight.validate_storage();
    if (weight.gguf_quantized()) {
        bool first = true;
        for (const GgufLinearSegment& segment : weight.gguf_segments) {
            if (segment.cols != k) {
                throw std::runtime_error("GGUF segment width does not match GEMM");
            }
            launch_gguf_linear_segment(
                x, segment.blocks, segment.type,
                y + static_cast<size_t>(segment.row_offset), m, segment.rows, k,
                segment.row_bytes, n, first ? beta : 1.0f, stream_);
            first = false;
        }
        return;
    }
    switch (plan.linear_kernel()) {
        case LinearKernelKind::W4A16:
            if (!weight.int4_quantized()) {
                throw std::runtime_error("execution plan requires INT4 weights");
            }
            launch_w4a16_linear(x, weight.int4, weight.scales, y,
                                m, n, k, beta, stream_);
            return;
        case LinearKernelKind::W8A16:
            if (!weight.int8_quantized()) {
                throw std::runtime_error("execution plan requires INT8 weights");
            }
            launch_w8a16_linear(x, weight.int8, weight.scales, y,
                                m, n, k, beta, stream_);
            return;
        case LinearKernelKind::Bf16CublasLt:
            if (weight.kind != LinearStorageKind::Bf16) {
                throw std::runtime_error("execution plan requires BF16 weights");
            }
            linear_cublaslt(x, weight.bf16, y, m, n, k, beta);
            return;
        case LinearKernelKind::Bf16Cublas:
            if (weight.kind != LinearStorageKind::Bf16) {
                throw std::runtime_error("execution plan requires BF16 weights");
            }
            linear_cublas(x, weight.bf16, y, m, n, k, beta);
            return;
    }
    throw std::runtime_error("unknown linear execution plan");
}

} // namespace lfm
