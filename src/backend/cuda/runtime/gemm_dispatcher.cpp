#include "backend/cuda/gemm_dispatcher.hpp"
#include "kernels/gguf.cuh"
#include "kernels/mmq.hpp"
#include "kernels/embedding.hpp"
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
    binding.kernel = weight.kernel.value_or(plan.linear_kernel());
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
            } else if (segment.type == GgmlType::Q6_K) {
                launch_q6k_mmq_with_policy(mmq_workspace_.q8.data(), mmq_workspace_.scales.data(),
                               mmq_workspace_.sums.data(), segment.blocks,
                               seg_y, m, segment.rows, k,
                               segment.row_bytes, n, beta,
                               plan.mmq_tensor_cores_enabled(), stream_);
            } else {
                // The loader must never bind a segment whose type has no MMQ
                // kernel; treating an unmatched type as Q6_K would decode
                // foreign blocks and silently corrupt the output.
                throw std::runtime_error(std::string("no MMQ kernel for GGUF segment type ") +
                                         ggml_type_name(segment.type));
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
        case LinearKernelKind::Fp8W8A8: {
            const auto* fp8 = std::get_if<Fp8LinearStorage>(&weight.storage);
            if (!fp8) {
                throw std::runtime_error("execution plan requires FP8 weights");
            }
            linear_fp8_w8a8(x, *fp8, y, m, n, k, beta);
            return;
        }
        case LinearKernelKind::Nvfp4W4A4: {
            const auto* nvfp4 = std::get_if<Nvfp4LinearStorage>(&weight.storage);
            if (!nvfp4) {
                throw std::runtime_error("execution plan requires NVFP4 weights");
            }
            linear_nvfp4_w4a4(x, *nvfp4, y, m, n, k, beta);
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

void GemmDispatcher::ensure_fp8_capacity(int m, int n, int k) {
    if (m <= fp8_workspace_.capacity_m && n <= fp8_workspace_.capacity_n &&
        k <= fp8_workspace_.capacity_k) {
        return;
    }
    fp8_workspace_.capacity_m = std::max(fp8_workspace_.capacity_m, m);
    fp8_workspace_.capacity_n = std::max(fp8_workspace_.capacity_n, n);
    fp8_workspace_.capacity_k = std::max(fp8_workspace_.capacity_k, k);
    fp8_workspace_.act_q.reset(
        static_cast<size_t>(fp8_workspace_.capacity_m) * fp8_workspace_.capacity_k);
    fp8_workspace_.act_scales.reset(static_cast<size_t>(fp8_workspace_.capacity_m));
    fp8_workspace_.raw.reset(
        static_cast<size_t>(fp8_workspace_.capacity_m) * fp8_workspace_.capacity_n);
}

// Raw (unscaled) FP8xFP8->FP32 matmul plan. Mirrors get_or_create_lt_plan's
// TRANSA=T/TRANSB=N layout convention (A=weight [k,n,k], B=activation
// [k,m,k], D=[n,m,n]) so the raw accumulation buffer has the same physical
// row-major-[m,n] layout the rest of the codebase's y buffers use. No scale
// pointers are set here -- see linear_fp8_w8a8 and
// docs/QWEN3_5_NVFP4_FP8_SUPPORT_PLAN.md Phase 3 for why the dequant scale
// is applied as a separate kernel instead of via cuBLASLt's scale-vector
// attribute (empirically unsupported on this hardware/toolkit combination).
LtPlan& GemmDispatcher::get_or_create_fp8_lt_plan(int m, int n, int k) {
    const MatmulKey key{m, n, k};
    if (auto it = fp8_lt_cache_.plans.find(key); it != fp8_lt_cache_.plans.end()) {
        return *it->second;
    }

    auto plan = std::make_unique<LtPlan>();
    CELEG_CUBLAS(cublasLtMatmulDescCreate(
        &plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t transa = CUBLAS_OP_T;
    const cublasOperation_t transb = CUBLAS_OP_N;
    CELEG_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CELEG_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    CELEG_CUBLAS(cublasLtMatrixLayoutCreate(&plan->a, CUDA_R_8F_E4M3, k, n, k));
    CELEG_CUBLAS(cublasLtMatrixLayoutCreate(&plan->b, CUDA_R_8F_E4M3, k, m, k));
    CELEG_CUBLAS(cublasLtMatrixLayoutCreate(&plan->c, CUDA_R_32F, n, m, n));
    CELEG_CUBLAS(cublasLtMatrixLayoutCreate(&plan->d, CUDA_R_32F, n, m, n));

    cublasLtMatmulPreference_t preference = nullptr;
    CELEG_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    try {
        const uint64_t workspace_limit = static_cast<uint64_t>(options_.lt_workspace_bytes);
        CELEG_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_limit, sizeof(workspace_limit)));

        cublasLtMatmulHeuristicResult_t result{};
        int returned = 0;
        const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
            handles_.cublas_lt.get(), plan->operation,
            plan->a, plan->b, plan->c, plan->d,
            preference, 1, &result, &returned);
        if (status == CUBLAS_STATUS_SUCCESS && returned > 0 &&
            result.workspaceSize <= options_.lt_workspace_bytes) {
            plan->algorithm = result.algo;
            plan->workspace_size = result.workspaceSize;
            plan->available = true;
        }
    } catch (...) {
        cublasLtMatmulPreferenceDestroy(preference);
        throw;
    }
    CELEG_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));

    auto [it, inserted] = fp8_lt_cache_.plans.emplace(key, std::move(plan));
    if (!inserted) throw std::runtime_error("duplicate fp8 cuBLASLt plan");
    return *it->second;
}

void GemmDispatcher::linear_fp8_w8a8(const __nv_bfloat16* x,
                                     const Fp8LinearStorage& weight,
                                     __nv_bfloat16* y,
                                     int m, int n, int k,
                                     float beta) {
    if (!weight.data || !weight.scales) {
        throw std::runtime_error("FP8 linear weight is missing data or scales");
    }
    ensure_fp8_capacity(m, n, k);
    launch_quantize_e4m3_per_row(x, fp8_workspace_.act_q.data(),
                                 fp8_workspace_.act_scales.data(), m, k, stream_);

    LtPlan& plan = get_or_create_fp8_lt_plan(m, n, k);
    if (!plan.available) {
        // No cuBLASLt fp8 algorithm exists for this shape (observed for
        // small/unaligned n on RTX 5090/CUDA 13.2 -- see
        // docs/QWEN3_5_NVFP4_FP8_SUPPORT_PLAN.md Phase 3). Fall back to a
        // naive kernel that is correct for every shape instead of failing.
        launch_fp8_w8a8_naive(fp8_workspace_.act_q.data(), fp8_workspace_.act_scales.data(),
                              weight.data, weight.scales, y, m, n, k, beta, stream_);
        return;
    }
    const float alpha = 1.0f;
    const float raw_beta = 0.0f;
    CELEG_CUBLAS(cublasLtMatmul(
        handles_.cublas_lt.get(), plan.operation,
        &alpha, weight.data, plan.a, fp8_workspace_.act_q.data(), plan.b,
        &raw_beta, fp8_workspace_.raw.data(), plan.c,
        fp8_workspace_.raw.data(), plan.d,
        &plan.algorithm, lt_workspace_.data(), plan.workspace_size, stream_));

    launch_fp8_scale_apply(fp8_workspace_.raw.data(), fp8_workspace_.act_scales.data(),
                           weight.scales, y, m, n, beta, stream_);
}

namespace {
// Byte size of the 128x4-tiled VEC16_UE4M3 scale buffer for a `rows` x
// `k_scale` (row-major) logical scale tensor -- see linear.cuh's
// swizzle_nvfp4_scale_kernel comment for the layout this mirrors.
size_t nvfp4_scale_buffer_bytes(int rows, int k_scale) {
    const int tiles_m = (rows + 127) / 128;
    const int tiles_n = (k_scale + 3) / 4;
    return static_cast<size_t>(tiles_m) * tiles_n * 512;
}

bool supports_native_nvfp4(int device_ordinal) {
    int major = 0;
    return cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                  device_ordinal) == cudaSuccess && major >= 10;
}
}

void GemmDispatcher::ensure_nvfp4_capacity(int m, int n, int k) {
    if (m <= nvfp4_workspace_.capacity_m && n <= nvfp4_workspace_.capacity_n &&
        k <= nvfp4_workspace_.capacity_k) {
        return;
    }
    nvfp4_workspace_.capacity_m = std::max(nvfp4_workspace_.capacity_m, m);
    nvfp4_workspace_.capacity_n = std::max(nvfp4_workspace_.capacity_n, n);
    nvfp4_workspace_.capacity_k = std::max(nvfp4_workspace_.capacity_k, k);
    const int cm = nvfp4_workspace_.capacity_m;
    const int cn = nvfp4_workspace_.capacity_n;
    const int ck = nvfp4_workspace_.capacity_k;
    const int k_scale = ck / kNvfp4BlockSize;

    nvfp4_workspace_.act_packed.reset(static_cast<size_t>(cm) * ck / 2);
    nvfp4_workspace_.act_scale_raw.reset(static_cast<size_t>(cm) * k_scale);
    nvfp4_workspace_.act_scale_swizzled.reset(nvfp4_scale_buffer_bytes(cm, k_scale));
    nvfp4_workspace_.weight_scale_swizzled.reset(nvfp4_scale_buffer_bytes(cn, k_scale));
    nvfp4_workspace_.raw.reset(static_cast<size_t>(cm) * cn);
}

// Native NVFP4 block-scaled fp4 matmul plan. Mirrors get_or_create_fp8_lt_plan's
// TRANSA=T/TRANSB=N layout convention. A_SCALE_MODE/B_SCALE_MODE are fixed
// per plan (VEC16_UE4M3); the actual scale POINTERs are set per-call in
// linear_nvfp4_w4a4 since they point at per-call-refreshed workspace
// buffers. See docs/QWEN3_5_NVFP4_FP8_SUPPORT_PLAN.md Phase 4 for how the
// 128x4 tiled scale layout this depends on was found and verified.
LtPlan& GemmDispatcher::get_or_create_nvfp4_lt_plan(int m, int n, int k) {
    const MatmulKey key{m, n, k};
    if (auto it = nvfp4_lt_cache_.plans.find(key); it != nvfp4_lt_cache_.plans.end()) {
        return *it->second;
    }

    auto plan = std::make_unique<LtPlan>();
    CELEG_CUBLAS(cublasLtMatmulDescCreate(
        &plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t transa = CUBLAS_OP_T;
    const cublasOperation_t transb = CUBLAS_OP_N;
    CELEG_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CELEG_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CELEG_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    CELEG_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    // cublasLtMatmulAlgoGetHeuristic requires the scale POINTERS to already
    // be set (not just the scale mode) or it returns INVALID_VALUE with
    // zero results -- found empirically, undocumented. ensure_nvfp4_capacity
    // was already called for this exact (m,n,k) by the caller, so these
    // addresses are valid now; linear_nvfp4_w4a4 re-sets them on every call
    // anyway (workspace buffers can grow/realloc for a larger shape later,
    // which would otherwise leave this cached plan's pointers dangling).
    __nv_fp8_e4m3* a_scale_ptr = nvfp4_workspace_.weight_scale_swizzled.data();
    __nv_fp8_e4m3* b_scale_ptr = nvfp4_workspace_.act_scale_swizzled.data();
    CELEG_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_ptr, sizeof(a_scale_ptr)));
    CELEG_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_ptr, sizeof(b_scale_ptr)));

    CELEG_CUBLAS(cublasLtMatrixLayoutCreate(&plan->a, CUDA_R_4F_E2M1, k, n, k));
    CELEG_CUBLAS(cublasLtMatrixLayoutCreate(&plan->b, CUDA_R_4F_E2M1, k, m, k));
    CELEG_CUBLAS(cublasLtMatrixLayoutCreate(&plan->c, CUDA_R_32F, n, m, n));
    CELEG_CUBLAS(cublasLtMatrixLayoutCreate(&plan->d, CUDA_R_32F, n, m, n));

    cublasLtMatmulPreference_t preference = nullptr;
    CELEG_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    try {
        const uint64_t workspace_limit = static_cast<uint64_t>(options_.lt_workspace_bytes);
        CELEG_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_limit, sizeof(workspace_limit)));

        cublasLtMatmulHeuristicResult_t result{};
        int returned = 0;
        const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
            handles_.cublas_lt.get(), plan->operation,
            plan->a, plan->b, plan->c, plan->d,
            preference, 1, &result, &returned);
        if (status == CUBLAS_STATUS_SUCCESS && returned > 0 &&
            result.workspaceSize <= options_.lt_workspace_bytes) {
            plan->algorithm = result.algo;
            plan->workspace_size = result.workspaceSize;
            plan->available = true;
        }
    } catch (...) {
        cublasLtMatmulPreferenceDestroy(preference);
        throw;
    }
    CELEG_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));

    auto [it, inserted] = nvfp4_lt_cache_.plans.emplace(key, std::move(plan));
    if (!inserted) throw std::runtime_error("duplicate nvfp4 cuBLASLt plan");
    return *it->second;
}

void GemmDispatcher::linear_nvfp4_w4a4(const __nv_bfloat16* x,
                                       const Nvfp4LinearStorage& weight,
                                       __nv_bfloat16* y,
                                       int m, int n, int k,
                                       float beta) {
    if (!weight.data || !weight.block_scales) {
        throw std::runtime_error("NVFP4 linear weight is missing data or block scales");
    }
    if (k % kNvfp4BlockSize != 0) {
        throw std::runtime_error("NVFP4 linear requires k to be a multiple of the block size");
    }
    ensure_nvfp4_capacity(m, n, k);

    const bool native_supported = supports_native_nvfp4(device_ordinal_);
    LtPlan* plan = native_supported ? &get_or_create_nvfp4_lt_plan(m, n, k) : nullptr;
    if (!plan || !plan->available) {
        launch_quantize_e2m1_per_block(
            x, nvfp4_workspace_.act_packed.data(),
            nvfp4_workspace_.act_scale_raw.data(), m, k, kNvfp4BlockSize,
            weight.input_global_scale, stream_);
        launch_nvfp4_w4a4_fallback(
            nvfp4_workspace_.act_packed.data(),
            nvfp4_workspace_.act_scale_raw.data(), weight.input_global_scale,
            weight.data, weight.block_scales, weight.global_scale,
            y, m, n, k, beta, stream_);
        return;
    }

    const int k_scale = k / kNvfp4BlockSize;
    launch_quantize_e2m1_per_block(x, nvfp4_workspace_.act_packed.data(),
                                   nvfp4_workspace_.act_scale_raw.data(),
                                   m, k, kNvfp4BlockSize, weight.input_global_scale, stream_);

    nvfp4_workspace_.act_scale_swizzled.zero_async(stream_);
    nvfp4_workspace_.weight_scale_swizzled.zero_async(stream_);
    launch_swizzle_nvfp4_scale(nvfp4_workspace_.act_scale_raw.data(),
                               nvfp4_workspace_.act_scale_swizzled.data(), m, k_scale, stream_);
    launch_swizzle_nvfp4_scale(weight.block_scales,
                               nvfp4_workspace_.weight_scale_swizzled.data(), n, k_scale, stream_);

    __nv_fp8_e4m3* a_scale_ptr = nvfp4_workspace_.weight_scale_swizzled.data();
    __nv_fp8_e4m3* b_scale_ptr = nvfp4_workspace_.act_scale_swizzled.data();
    CELEG_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_ptr, sizeof(a_scale_ptr)));
    CELEG_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_ptr, sizeof(b_scale_ptr)));

    const float alpha = 1.0f;
    const float raw_beta = 0.0f;
    CELEG_CUBLAS(cublasLtMatmul(
        handles_.cublas_lt.get(), plan->operation,
        &alpha, weight.data, plan->a, nvfp4_workspace_.act_packed.data(), plan->b,
        &raw_beta, nvfp4_workspace_.raw.data(), plan->c,
        nvfp4_workspace_.raw.data(), plan->d,
        &plan->algorithm, lt_workspace_.data(), plan->workspace_size, stream_));

    const float total_scale = 1.0f / (weight.global_scale * weight.input_global_scale);
    launch_nvfp4_global_scale_apply(nvfp4_workspace_.raw.data(), total_scale, y, m, n, beta, stream_);
}

}
