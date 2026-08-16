#include "celeg/backend/cuda/gemm_dispatcher.hpp"
#include "celeg/backend/cuda/kernels/gguf.cuh"
#include "celeg/backend/cuda/kernels/mmq.hpp"
#include "celeg/backend/cuda/kernels/embedding.hpp"
#include "../kernels/gemv_launch.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace celeg {
GemmDispatcher::GemmDispatcher(cudaStream_t stream,
                               const CudaModelOptions& options)
    : stream_(stream),
      options_(options),
      handles_(stream) {
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
        handles_.cublas.get(), CUBLAS_OP_T, CUBLAS_OP_N,
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
    if (auto it = lt_cache_.plans.find(key); it != lt_cache_.plans.end()) {
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
            handles_.cublas_lt.get(), plan->operation,
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
            const int selected = lt_autotuner_.choose(
                options_.lt_autotune, valid, valid.front(),
                [&](int candidate) {
                    DeviceBuffer<__nv_bfloat16> scratch(static_cast<size_t>(m) * n);
                    const auto& result = results[static_cast<size_t>(candidate)];
                    const float alpha = 1.0f;
                    const float beta = 0.0f;
                    CELEG_CUBLAS(cublasLtMatmul(
                        handles_.cublas_lt.get(), plan->operation,
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
                            handles_.cublas_lt.get(), plan->operation,
                            &alpha, weight, plan->a, x, plan->b,
                            &beta, scratch.data(), plan->c,
                            scratch.data(), plan->d, &result.algo,
                            lt_workspace_.data(), result.workspaceSize,
                            stream_));
                    }
                    end.record(stream_);
                    end.synchronize();
                    return CudaEvent::elapsed_ms(begin, end);
                });
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

    auto [it, inserted] = lt_cache_.plans.emplace(key, std::move(plan));
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
        handles_.cublas_lt.get(), plan.operation,
        &alpha, weight, plan.a, x, plan.b,
        &beta, y, plan.c, y, plan.d,
        &plan.algorithm, lt_workspace_.data(), plan.workspace_size,
        stream_));
}

const CompiledLinearBinding& GemmDispatcher::compile_linear_binding(
    const LinearWeight& weight, const CudaExecutionPlan& plan) {
    auto it = linear_bindings_.find(&weight);
    if (it != linear_bindings_.end() &&
        it->second.plan_fingerprint == plan.fingerprint()) {
        return it->second;
    }
    if (weight.rows <= 0 || weight.cols <= 0) {
        throw std::runtime_error("linear weight dimensions must be positive");
    }
    CompiledLinearBinding binding;
    binding.weight = &weight;
    binding.kernel = plan.linear_kernel();
    binding.rows = weight.rows;
    binding.cols = weight.cols;
    binding.plan_fingerprint = plan.fingerprint();
    auto [inserted, ignored] = linear_bindings_.insert_or_assign(&weight, binding);
    (void)ignored;
    return inserted->second;
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
    const CompiledLinearBinding& binding = compile_linear_binding(weight, plan);
    if (binding.rows != n || binding.cols != k) {
        throw std::runtime_error("linear weight shape does not match the requested GEMM: weight=" +
            std::to_string(weight.rows) + "x" + std::to_string(weight.cols) +
            " requested=" + std::to_string(n) + "x" + std::to_string(k));
    }
    const LinearWeight& bound_weight = *binding.weight;
    if (const auto* gguf = std::get_if<GgufLinearStorage>(&bound_weight.storage)) {
        if (!has_native_fanout(x, m, k)) {
            ensure_mmq_capacity(m, k);
            launch_quantize_q8_1(x, mmq_workspace_.q8.data(), mmq_workspace_.scales.data(),
                                 mmq_workspace_.sums.data(), m, k, stream_);
        }
        for (const GgufLinearSegment& segment : gguf->segments) {
            if (segment.cols != k) {
                throw std::runtime_error("GGUF segment width does not match GEMM");
            }
            __nv_bfloat16* seg_y = y + static_cast<size_t>(segment.row_offset);
            if (segment.type == GgmlType::Q4_K) {
                launch_q4k_mmq_with_policy(mmq_workspace_.q8.data(), mmq_workspace_.scales.data(),
                               mmq_workspace_.sums.data(), segment.blocks,
                               seg_y, m, segment.rows, k,
                               segment.row_bytes, n, beta,
                               plan.mmq_tensor_cores_enabled(), stream_);
            } else {
                launch_q6k_mmq_with_policy(mmq_workspace_.q8.data(), mmq_workspace_.scales.data(),
                               mmq_workspace_.sums.data(), segment.blocks,
                               seg_y, m, segment.rows, k,
                               segment.row_bytes, n, beta,
                               plan.mmq_tensor_cores_enabled(), stream_);
            }
        }
        return;
    }
    switch (binding.kernel) {
        case LinearKernelKind::W4A16: {
            const auto* int4 = std::get_if<Int4LinearStorage>(&weight.storage);
            if (!int4) {
                throw std::runtime_error("execution plan requires INT4 weights");
            }
            if (m > 1 && int4->bf16_fallback) {
                if (options_.gemm_backend == GemmBackend::CublasLt) {
                    linear_cublaslt(x, int4->bf16_fallback, y, m, n, k, beta);
                } else {
                    linear_cublas(x, int4->bf16_fallback, y, m, n, k, beta);
                }
                return;
            }
            launch_w4a16_linear(x, int4->data, int4->scales, y,
                                m, n, k, beta, stream_);
            return;
        }
        case LinearKernelKind::W8A16: {
            const auto* int8 = std::get_if<Int8LinearStorage>(&weight.storage);
            if (!int8) {
                throw std::runtime_error("execution plan requires INT8 weights");
            }
            if (m > 1 && int8->bf16_fallback) {
                if (options_.gemm_backend == GemmBackend::CublasLt) {
                    linear_cublaslt(x, int8->bf16_fallback, y, m, n, k, beta);
                } else {
                    linear_cublas(x, int8->bf16_fallback, y, m, n, k, beta);
                }
                return;
            }
            launch_w8a16_linear(x, int8->data, int8->scales, y,
                                m, n, k, beta, stream_);
            return;
        }
        case LinearKernelKind::Bf16CublasLt: {
            const auto* bf16 = std::get_if<Bf16LinearStorage>(&weight.storage);
            if (!bf16) {
                throw std::runtime_error("execution plan requires BF16 weights");
            }
            if (m == 1) {
                launch_bf16_gemv(x, bf16->data, y, n, k, beta, stream_);
                return;
            }
            linear_cublaslt(x, bf16->data, y, m, n, k, beta);
            return;
        }
        case LinearKernelKind::Bf16Cublas: {
            const auto* bf16 = std::get_if<Bf16LinearStorage>(&weight.storage);
            if (!bf16) {
                throw std::runtime_error("execution plan requires BF16 weights");
            }
            if (m == 1) {
                launch_bf16_gemv(x, bf16->data, y, n, k, beta, stream_);
                return;
            }
            linear_cublas(x, bf16->data, y, m, n, k, beta);
            return;
        }
        case LinearKernelKind::Q4kMmq:
        case LinearKernelKind::Q6kMmq:
            throw std::runtime_error("MMQ kernels are dispatched via the GGUF storage branch, not the plan switch");
        case LinearKernelKind::MixedBf16AndGgufMmq: {
            const auto* bf16 = std::get_if<Bf16LinearStorage>(&bound_weight.storage);
            if (!bf16 || !bf16->data) {
                throw std::runtime_error("mixed native GGUF plan has no executable linear storage");
            }
            if (m == 1) {
                launch_bf16_gemv(x, bf16->data, y, n, k, beta, stream_);
            } else if (options_.gemm_backend == GemmBackend::CublasLt) {
                linear_cublaslt(x, bf16->data, y, m, n, k, beta);
            } else {
                linear_cublas(x, bf16->data, y, m, n, k, beta);
            }
            return;
        }
    }
    throw std::runtime_error("unknown linear execution plan");
}

void GemmDispatcher::begin_native_fanout(const __nv_bfloat16* x, int m, int k) {
    if (x == nullptr || m <= 0 || k <= 0 || k % kMmqQ8_1BlockSize != 0) {
        throw std::invalid_argument("native GGUF fan-out requires a non-empty Q8_1-aligned input");
    }
    if (mmq_workspace_.fanout_input != nullptr) {
        throw std::logic_error("nested native GGUF fan-out is not supported");
    }
    ensure_mmq_capacity(m, k);
    launch_quantize_q8_1(x, mmq_workspace_.q8.data(), mmq_workspace_.scales.data(), mmq_workspace_.sums.data(),
                         m, k, stream_);
    mmq_workspace_.fanout_input = x;
    mmq_workspace_.fanout_m = m;
    mmq_workspace_.fanout_k = k;
}

void GemmDispatcher::end_native_fanout() {
    mmq_workspace_.fanout_input = nullptr;
    mmq_workspace_.fanout_m = 0;
    mmq_workspace_.fanout_k = 0;
}

void GemmDispatcher::ensure_mmq_capacity(int m, int k) {
    if (m <= mmq_workspace_.capacity_m && k <= mmq_workspace_.capacity_k) return;
    mmq_workspace_.capacity_m = std::max(mmq_workspace_.capacity_m, m);
    mmq_workspace_.capacity_k = std::max(mmq_workspace_.capacity_k, k);
    mmq_workspace_.q8.reset(static_cast<size_t>(mmq_workspace_.capacity_m) * mmq_workspace_.capacity_k);
    const size_t blocks = static_cast<size_t>(mmq_workspace_.capacity_m) *
                          (mmq_workspace_.capacity_k / kMmqQ8_1BlockSize);
    mmq_workspace_.scales.reset(blocks);
    mmq_workspace_.sums.reset(blocks);
}

bool GemmDispatcher::has_native_fanout(const __nv_bfloat16* x, int m, int k) const {
    return mmq_workspace_.fanout_input == x && mmq_workspace_.fanout_m == m && mmq_workspace_.fanout_k == k;
}

}
