#include "celeg/backend/cuda/gemm_dispatcher.hpp"
#include "celeg/backend/cuda/kernels/gguf.cuh"
#include "celeg/backend/cuda/kernels/mmq.hpp"
#include "celeg/backend/cuda/kernels/embedding.hpp"
#include "celeg/backend/cuda/kernels/gemv_kernels.cuh"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace celeg {
namespace {

void launch_bf16_gemv(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                      __nv_bfloat16* y, int n, int k, float beta,
                      cudaStream_t stream) {
    constexpr int warps_per_block = 8;
    const dim3 grid(static_cast<unsigned>((n + warps_per_block - 1) / warps_per_block));
    bf16_gemv_kernel<<<grid, warps_per_block * 32, 0, stream>>>(x, weight, y, n, k, beta);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

} // namespace

GemmDispatcher::GemmDispatcher(cudaStream_t stream,
                               const CudaModelOptions& options)
    : stream_(stream),
      options_(options),
      cublas_(stream),
      cublas_lt_() {
    CELEG_CUDA(cudaGetDevice(&device_ordinal_));
    if (options.gemm_backend == GemmBackend::CublasLt &&
        options.lt_workspace_bytes > 0) {
        lt_workspace_.reset(options.lt_workspace_bytes);
    }
}

GemmDispatcher::~GemmDispatcher() = default;

GemmDispatcher::NativeFanoutScope::NativeFanoutScope(
    GemmDispatcher* dispatcher, const __nv_bfloat16* x, int m, int k)
    : dispatcher_(dispatcher) {
    if (dispatcher_ != nullptr) dispatcher_->begin_native_fanout(x, m, k);
}

GemmDispatcher::NativeFanoutScope::~NativeFanoutScope() {
    if (dispatcher_ != nullptr) dispatcher_->end_native_fanout();
}

GemmDispatcher::NativeFanoutScope::NativeFanoutScope(
    NativeFanoutScope&& other) noexcept
    : dispatcher_(std::exchange(other.dispatcher_, nullptr)) {}

GemmDispatcher::NativeFanoutScope&
GemmDispatcher::NativeFanoutScope::operator=(NativeFanoutScope&& other) noexcept {
    if (this == &other) return *this;
    if (dispatcher_ != nullptr) dispatcher_->end_native_fanout();
    dispatcher_ = std::exchange(other.dispatcher_, nullptr);
    return *this;
}

void GemmDispatcher::linear_cublas(const __nv_bfloat16* x,
                                   const __nv_bfloat16* weight,
                                   __nv_bfloat16* y,
                                   int m, int n, int k,
                                   float beta) {
    const float alpha = 1.0f;
    CELEG_CUBLAS(cublasGemmEx(
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
    CELEG_CUBLAS(cublasLtMatmulDescCreate(
        &plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t transa = CUBLAS_OP_T;
    const cublasOperation_t transb = CUBLAS_OP_N;
    CELEG_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_TRANSA,
        &transa, sizeof(transa)));
    CELEG_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_TRANSB,
        &transb, sizeof(transb)));

    // Buffers are row-major in the runtime. Interpreting them as column-major
    // transposes the logical matrices: W[n,k] -> A[k,n], X[m,k] -> B[k,m],
    // and Y[m,n] -> D[n,m]. TRANSA restores W to [n,k].
    CELEG_CUBLAS(cublasLtMatrixLayoutCreate(
        &plan->a, CUDA_R_16BF, k, n, k));
    CELEG_CUBLAS(cublasLtMatrixLayoutCreate(
        &plan->b, CUDA_R_16BF, k, m, k));
    CELEG_CUBLAS(cublasLtMatrixLayoutCreate(
        &plan->c, CUDA_R_16BF, n, m, n));
    CELEG_CUBLAS(cublasLtMatrixLayoutCreate(
        &plan->d, CUDA_R_16BF, n, m, n));

    cublasLtMatmulPreference_t preference = nullptr;
    CELEG_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    try {
        const uint64_t workspace_limit =
            static_cast<uint64_t>(options_.lt_workspace_bytes);
        CELEG_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_limit, sizeof(workspace_limit)));

        std::vector<cublasLtMatmulHeuristicResult_t> results(
            static_cast<size_t>(options_.lt_heuristics));
        int returned = 0;
        CELEG_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
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
            // Autotune: try each candidate algorithm and pick the fastest.
            // For decode (m=1) the scratch is just n elements; for prefill
            // (m>1) we need m*n for the full output.
            if (options_.lt_autotune && valid.size() > 1) {
                DeviceBuffer<__nv_bfloat16> scratch(static_cast<size_t>(m) * n);
                const float alpha = 1.0f;
                const float beta = 0.0f;
                float best_ms = std::numeric_limits<float>::infinity();
                for (const int candidate : valid) {
                    const auto& result = results[static_cast<size_t>(candidate)];
                    CELEG_CUBLAS(cublasLtMatmul(
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
                        CELEG_CUBLAS(cublasLtMatmul(
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
    CELEG_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));

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
    CELEG_CUBLAS(cublasLtMatmul(
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
                            const CudaExecutionPlan& plan) {
    if (plan.device().device_ordinal >= 0 &&
        plan.device().device_ordinal != device_ordinal_) {
        throw std::invalid_argument(
            "execution plan device does not match GEMM dispatcher device");
    }
    if (weight.rows != n || weight.cols != k) {
        throw std::runtime_error("linear weight shape does not match the requested GEMM: weight=" +
            std::to_string(weight.rows) + "x" + std::to_string(weight.cols) +
            " requested=" + std::to_string(n) + "x" + std::to_string(k));
    }
    weight.validate_storage();
    if (weight.gguf_quantized()) {
        if (!has_native_fanout(x, m, k)) {
            ensure_mmq_capacity(m, k);
            launch_quantize_q8_1(x, mmq_q8_.data(), mmq_scales_.data(),
                                 mmq_sums_.data(), m, k, stream_);
        }
        for (const GgufLinearSegment& segment : weight.gguf_segments) {
            if (segment.cols != k) {
                throw std::runtime_error("GGUF segment width does not match GEMM");
            }
            __nv_bfloat16* seg_y = y + static_cast<size_t>(segment.row_offset);
            if (segment.type == GgmlType::Q4_K) {
                launch_q4k_mmq_with_policy(mmq_q8_.data(), mmq_scales_.data(),
                               mmq_sums_.data(), segment.blocks,
                               seg_y, m, segment.rows, k,
                               segment.row_bytes, n, beta,
                               plan.mmq_tensor_cores_enabled(), stream_);
            } else {
                launch_q6k_mmq_with_policy(mmq_q8_.data(), mmq_scales_.data(),
                               mmq_sums_.data(), segment.blocks,
                               seg_y, m, segment.rows, k,
                               segment.row_bytes, n, beta,
                               plan.mmq_tensor_cores_enabled(), stream_);
            }
        }
        return;
    }
    switch (plan.linear_kernel()) {
        case LinearKernelKind::W4A16:
            if (!weight.int4_quantized()) {
                throw std::runtime_error("execution plan requires INT4 weights");
            }
            if (m > 1 && weight.bf16) {
                if (options_.gemm_backend == GemmBackend::CublasLt) {
                    linear_cublaslt(x, weight.bf16, y, m, n, k, beta);
                } else {
                    linear_cublas(x, weight.bf16, y, m, n, k, beta);
                }
                return;
            }
            launch_w4a16_linear(x, weight.int4, weight.scales, y,
                                m, n, k, beta, stream_);
            return;
        case LinearKernelKind::W8A16:
            if (!weight.int8_quantized()) {
                throw std::runtime_error("execution plan requires INT8 weights");
            }
            if (m > 1 && weight.bf16) {
                if (options_.gemm_backend == GemmBackend::CublasLt) {
                    linear_cublaslt(x, weight.bf16, y, m, n, k, beta);
                } else {
                    linear_cublas(x, weight.bf16, y, m, n, k, beta);
                }
                return;
            }
            launch_w8a16_linear(x, weight.int8, weight.scales, y,
                                m, n, k, beta, stream_);
            return;
        case LinearKernelKind::Bf16CublasLt:
            if (weight.kind != LinearStorageKind::Bf16) {
                throw std::runtime_error("execution plan requires BF16 weights");
            }
            if (m == 1) {
                launch_bf16_gemv(x, weight.bf16, y, n, k, beta, stream_);
                return;
            }
            linear_cublaslt(x, weight.bf16, y, m, n, k, beta);
            return;
        case LinearKernelKind::Bf16Cublas:
            if (weight.kind != LinearStorageKind::Bf16) {
                throw std::runtime_error("execution plan requires BF16 weights");
            }
            if (m == 1) {
                launch_bf16_gemv(x, weight.bf16, y, n, k, beta, stream_);
                return;
            }
            linear_cublas(x, weight.bf16, y, m, n, k, beta);
            return;
        case LinearKernelKind::Q4kMmq:
        case LinearKernelKind::Q6kMmq:
            throw std::runtime_error("MMQ kernels are dispatched via gguf_quantized(), not the plan switch");
    }
    throw std::runtime_error("unknown linear execution plan");
}

void GemmDispatcher::begin_native_fanout(const __nv_bfloat16* x, int m, int k) {
    if (x == nullptr || m <= 0 || k <= 0 || k % kMmqQ8_1BlockSize != 0) {
        throw std::invalid_argument("native GGUF fan-out requires a non-empty Q8_1-aligned input");
    }
    if (native_fanout_input_ != nullptr) {
        throw std::logic_error("nested native GGUF fan-out is not supported");
    }
    ensure_mmq_capacity(m, k);
    launch_quantize_q8_1(x, mmq_q8_.data(), mmq_scales_.data(), mmq_sums_.data(),
                         m, k, stream_);
    native_fanout_input_ = x;
    native_fanout_m_ = m;
    native_fanout_k_ = k;
}

void GemmDispatcher::end_native_fanout() {
    native_fanout_input_ = nullptr;
    native_fanout_m_ = 0;
    native_fanout_k_ = 0;
}

void GemmDispatcher::ensure_mmq_capacity(int m, int k) {
    if (m <= mmq_capacity_m_ && k <= mmq_capacity_k_) return;
    mmq_capacity_m_ = std::max(mmq_capacity_m_, m);
    mmq_capacity_k_ = std::max(mmq_capacity_k_, k);
    mmq_q8_.reset(static_cast<size_t>(mmq_capacity_m_) * mmq_capacity_k_);
    const size_t blocks = static_cast<size_t>(mmq_capacity_m_) *
                          (mmq_capacity_k_ / kMmqQ8_1BlockSize);
    mmq_scales_.reset(blocks);
    mmq_sums_.reset(blocks);
}

bool GemmDispatcher::has_native_fanout(const __nv_bfloat16* x, int m, int k) const {
    return native_fanout_input_ == x && native_fanout_m_ == m && native_fanout_k_ == k;
}

} // namespace celeg
