#include "quant_registry.hpp"

#include "detail.hpp"

#include <algorithm>
#include <cstdlib>
#include <string>

namespace celeg {

namespace {

/// @brief Whether the device is Apple M5, resolved once per process.
bool is_apple_m5(id<MTLDevice> device) noexcept {
    if (device == nil) return false;
    NSString* name = device.name;
    if (name == nil) return false;
    std::string value = metal_model_detail::ns_string(name);
    return value.find("Apple M5") != std::string::npos;
}

bool dense_rows_enabled(const MetalModelOptions& options, id<MTLDevice> device) noexcept {
    if (options.numerical_policy != MetalNumericalPolicy::Fast) return false;
    if (!is_apple_m5(device)) return false;
    const char* env = std::getenv("CELEG_METAL_DENSE_MATVEC_ROWS");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

constexpr NSUInteger kTensorTileRows = 64;
constexpr NSUInteger kTensorTileTokens = 128;
constexpr NSUInteger kTensorTileK = 64;
constexpr NSUInteger kQ4KStrictStageK = 128;
constexpr NSUInteger kTensorTileBytes = kTensorTileRows * kTensorTileK * sizeof(uint16_t);
constexpr NSUInteger kQ4KStrictStageBytes = kTensorTileRows * kQ4KStrictStageK * sizeof(uint16_t);

constexpr std::string_view kQ4KStrictKernel = "celeg_matmul_tensor_q4k_static_stage128";
constexpr std::string_view kF16RelaxedKernel = "celeg_matmul_tensor_f16_fast";
constexpr std::string_view kBF16RelaxedKernel = "celeg_matmul_tensor_bf16_fast";
constexpr std::string_view kQ40RelaxedKernel = "celeg_matmul_tensor_q4_0_relaxed";
constexpr std::string_view kQ4KRelaxedKernel = "celeg_matmul_tensor_q4k_relaxed";
constexpr std::string_view kQ5KRelaxedKernel = "celeg_matmul_tensor_q5k_relaxed";
constexpr std::string_view kQ6KRelaxedKernel = "celeg_matmul_tensor_q6k_fast";
constexpr std::string_view kQ80RelaxedKernel = "celeg_matmul_tensor_q8_0_relaxed";
constexpr std::string_view kF16RelaxedN32Kernel = "celeg_matmul_tensor_f16_fast_n32";
constexpr std::string_view kBF16RelaxedN32Kernel = "celeg_matmul_tensor_bf16_fast_n32";
constexpr std::string_view kQ40RelaxedN32Kernel = "celeg_matmul_tensor_q4_0_relaxed_n32";
constexpr std::string_view kQ4KRelaxedN32Kernel = "celeg_matmul_tensor_q4k_relaxed_n32";
constexpr std::string_view kQ5KRelaxedN32Kernel = "celeg_matmul_tensor_q5k_relaxed_n32";
constexpr std::string_view kQ6KRelaxedN32Kernel = "celeg_matmul_tensor_q6k_fast_n32";
constexpr std::string_view kQ6KStrictFastKernel = "celeg_matmul_tensor_q6k_fast_strict";
constexpr std::string_view kQ6KStrictFastN32Kernel = "celeg_matmul_tensor_q6k_fast_strict_n32";
constexpr std::string_view kQ80RelaxedN32Kernel = "celeg_matmul_tensor_q8_0_relaxed_n32";

}

std::optional<std::string_view> quant_linear_kernel(
    metal_model_detail::MetalLinearStorage storage,
    LinearOperationKind operation) noexcept {
    static constexpr const char* kGeneric[8][8] = {
        {nullptr, nullptr, "celeg_matmul", nullptr, nullptr, "celeg_embedding", "celeg_embedding_batch", nullptr},
        {nullptr, nullptr, "celeg_matmul_f16", nullptr, "celeg_matmul_tensor_f16", "celeg_embedding_f16", "celeg_embedding_f16_batch", nullptr},
        {nullptr, nullptr, "celeg_matmul_bf16", nullptr, "celeg_matmul_tensor_bf16", "celeg_embedding_bf16", "celeg_embedding_bf16_batch", nullptr},
        {nullptr, nullptr, "celeg_matmul_q4_0", nullptr, "celeg_matmul_tensor_q4_0", "celeg_embedding_q4_0", "celeg_embedding_q4_0_batch", nullptr},
        {nullptr, nullptr, "celeg_matmul_q4k", nullptr, "celeg_matmul_tensor_q4k", "celeg_embedding_q4k", "celeg_embedding_q4k_batch", nullptr},
        {nullptr, nullptr, "celeg_matmul_q5k", nullptr, "celeg_matmul_tensor_q5k", "celeg_embedding_q5k", "celeg_embedding_q5k_batch", nullptr},
        {nullptr, nullptr, "celeg_matmul_q6k", nullptr, "celeg_matmul_tensor_q6k", "celeg_embedding_q6k", "celeg_embedding_q6k_batch", nullptr},
        {nullptr, nullptr, "celeg_matmul_q8_0", nullptr, "celeg_matmul_tensor_q8_0", "celeg_embedding_q8_0", "celeg_embedding_q8_0_batch", nullptr},
    };
    const auto si = static_cast<std::size_t>(storage);
    const auto oi = static_cast<std::size_t>(operation);
    if (si >= 8 || oi >= 8) return std::nullopt;
    const char* name = kGeneric[si][oi];
    if (name == nullptr) return std::nullopt;
    return std::string_view{name};
}

MetalMatvecKernel quant_matvec_kernel(
    metal_model_detail::MetalLinearStorage storage,
    uint32_t rows,
    uint32_t cols,
    const MetalModelOptions& options,
    id<MTLDevice> device) noexcept {
    const bool ffn_expansion = rows >= 4096 && rows < 32768 && cols <= 2048;
    const bool ffn_contraction = rows <= 2048 && cols >= 4096 && cols < 32768;
    const bool m5_fast = options.numerical_policy == MetalNumericalPolicy::Fast && is_apple_m5(device);
    const bool dense_rows = dense_rows_enabled(options, device);
    using S = metal_model_detail::MetalLinearStorage;
    switch (storage) {
        case S::Float32: return {"celeg_matvec", 8, 256, 0};
        case S::Float16:
            if (dense_rows && rows == 1024 && cols == 1024) return {"celeg_matvec_f16_rows8", 8, 128, 32};
            if (dense_rows && cols == 1024 && (rows == 3072 || rows == 4608)) return {"celeg_matvec_f16_rows4", 4, 128, 16};
            return {"celeg_matvec_f16", 2, 128, 8};
        case S::BFloat16:
            if (dense_rows && rows == 1024 && cols == 4608) return {"celeg_matvec_bf16_rows8", 8, 128, 32};
            return {"celeg_matvec_bf16", 2, 128, 8};
        case S::Q4_0: return {"celeg_matvec_q4_0", 16, 128, 0};
        case S::Q4K: return {"celeg_matvec_q4k", 4, 64, 0};
        case S::Q5K:
            // The rows8 specialization remains compiled for differential work, but it is
            // quarantined from production selection until it has independent Q5_K parity
            // coverage. The default kernel already has a host-reference integration test.
            return {"celeg_matvec_q5k", 16, 128, 0};
        case S::Q6K: return MetalMatvecKernel{"celeg_matvec_q6k_llama", 4, 64, 0};
        case S::Q8_0: return m5_fast ? MetalMatvecKernel{"celeg_matvec_q8_0_m5", 2, 128, 8} : ffn_expansion || ffn_contraction ? MetalMatvecKernel{"celeg_matvec_q8_0_rows8", 32, 128, 0} : MetalMatvecKernel{"celeg_matvec_q8_0", 16, 128, 0};
    }
    return {};
}

MetalMatvecKernel quant_swiglu_matvec_kernel(
    metal_model_detail::MetalLinearStorage storage,
    uint32_t rows,
    uint32_t cols,
    const MetalModelOptions& options,
    id<MTLDevice> device) noexcept {
    const bool m5_fast = options.numerical_policy == MetalNumericalPolicy::Fast && is_apple_m5(device);
    using S = metal_model_detail::MetalLinearStorage;
    switch (storage) {
        case S::Q4_0: return {"celeg_swiglu_matvec_q4_0", 16, 128, 0};
        case S::Q4K: return {"celeg_swiglu_matvec_q4k", 4, 64, 0};
        case S::Q5K: return {"celeg_swiglu_matvec_q5k", 16, 128, 0};
        case S::Q6K: return MetalMatvecKernel{"celeg_swiglu_matvec_q6k_llama", 4, 64, 0};
        case S::Q8_0: return rows <= 2048 && cols >= 4096 && cols < 32768 ? m5_fast ? MetalMatvecKernel{"celeg_swiglu_matvec_q8_0_m5", 2, 128, 8} : MetalMatvecKernel{"celeg_swiglu_matvec_q8_0_rows8", 32, 128, 0} : MetalMatvecKernel{};
        default: return {};
    }
}

bool quant_tensor_matmul_available(
    metal_model_detail::MetalLinearStorage storage,
    const MetalPipelineCache& cache) noexcept {
    if (false) return false;
    using S = metal_model_detail::MetalLinearStorage;
    switch (storage) {
        case S::Float16: return cache.tensor_matmul_f16;
        case S::BFloat16: return cache.tensor_matmul_bf16;
        case S::Q4_0: return cache.tensor_matmul_q4_0;
        case S::Q4K: return cache.tensor_matmul_q4k;
        case S::Q5K: return cache.tensor_matmul_q5k;
        case S::Q6K: return cache.tensor_matmul_q6k;
        case S::Q8_0: return cache.tensor_matmul_q8_0;
        case S::Float32: return false;
    }
    return false;
}

bool quant_fast_tensor_matmul_available(
    metal_model_detail::MetalLinearStorage storage,
    const MetalPipelineCache& cache) noexcept {
    using S = metal_model_detail::MetalLinearStorage;
    switch (storage) {
        case S::Float16: return cache.tensor_fast_f16;
        case S::BFloat16: return cache.tensor_fast_bf16;
        case S::Q4_0: return cache.tensor_fast_q4_0;
        case S::Q4K: return cache.tensor_fast_q4k;
        case S::Q5K:
            // Keep the relaxed Q5_K TensorOps library built, but do not advertise it
            // to production dispatch until the specialized path has its own numeric
            // differential fixture. Strict Q5_K remains available through tensor_matmul.
            return false;
        case S::Q6K: return cache.tensor_fast_q6k;
        case S::Q8_0: return cache.tensor_fast_q8_0;
        case S::Float32: return false;
    }
    return false;
}

QuantTensorKernel quant_select_tensor_kernel(
    metal_model_detail::MetalLinearStorage storage,
    uint32_t rows,
    uint32_t weight_rows,
    uint32_t weight_cols,
    TensorRole role,
    const MetalPipelineCache& cache,
    const MetalModelOptions& options,
    id<MTLDevice> device) noexcept {
    QuantTensorKernel out;
    const bool relaxed = options.numerical_policy == MetalNumericalPolicy::Fast &&
        quant_fast_tensor_matmul_available(storage, cache);
    const bool m5_q6 = is_apple_m5(device) && role == TensorRole::FfnDown;
    using S = metal_model_detail::MetalLinearStorage;
    if (relaxed && storage == S::Float16) {
        out.name = rows <= 32 ? kF16RelaxedN32Kernel : kF16RelaxedKernel;
        out.custom = true;
    } else if (relaxed && storage == S::BFloat16) {
        out.name = rows <= 32 ? kBF16RelaxedN32Kernel : kBF16RelaxedKernel;
        out.custom = true;
    } else if (relaxed && storage == S::Q4_0) {
        out.name = rows <= 32 ? kQ40RelaxedN32Kernel : kQ40RelaxedKernel;
        out.custom = true;
    } else if (relaxed && storage == S::Q4K) {
        out.name = rows <= 32 ? kQ4KRelaxedN32Kernel : kQ4KRelaxedKernel;
        out.custom = true;
    } else if (relaxed && storage == S::Q5K) {
        out.name = rows <= 32 ? kQ5KRelaxedN32Kernel : kQ5KRelaxedKernel;
        out.custom = true;
    } else if (relaxed && storage == S::Q6K) {
        if (m5_q6) out.name = rows <= 32 ? kQ6KRelaxedN32Kernel : kQ6KRelaxedKernel;
        else out.name = rows <= 32 ? kQ6KStrictFastN32Kernel : kQ6KStrictFastKernel;
        out.custom = true;
    } else if (relaxed && storage == S::Q8_0) {
        out.name = rows <= 32 ? kQ80RelaxedN32Kernel : kQ80RelaxedKernel;
        out.custom = true;
    } else if (storage == S::Q4K && rows % kTensorTileTokens == 0 &&
               weight_cols % kQ4KStrictStageK == 0 && weight_rows % kTensorTileRows == 0 &&
               static_cast<NSUInteger>(device.maxThreadgroupMemoryLength) >= kQ4KStrictStageBytes) {
        out.name = kQ4KStrictKernel;
        out.shared_bytes = kQ4KStrictStageBytes;
        out.exact_groups = true;
        out.custom = true;
        out.tile_tokens = kTensorTileTokens;
        return out;
    } else {
        auto fallback = quant_linear_kernel(storage, LinearOperationKind::MatMulTensor);
        out.name = fallback ? *fallback : std::string_view{};
        out.shared_bytes = kTensorTileBytes;
        out.tile_tokens = kTensorTileTokens;
        out.custom = false;
        return out;
    }
    out.shared_bytes = kTensorTileBytes;
    out.tile_tokens = relaxed && rows <= 32 ? 32 : kTensorTileTokens;
    return out;
}

id<MTLLibrary> quant_tensor_library_for(
    std::string_view kernel_name,
    const MetalPipelineCache& cache) noexcept {
    if (kernel_name.find("_relaxed") != std::string_view::npos || kernel_name.find("_fast") != std::string_view::npos) {
        if (kernel_name.find("_f16_") != std::string_view::npos || kernel_name.find("_bf16_") != std::string_view::npos) return cache.tensor_fast_dense_library;
        if (kernel_name.find("_q4_0_") != std::string_view::npos) return cache.tensor_fast_q4_0_library;
        if (kernel_name.find("_q4k_") != std::string_view::npos) return cache.tensor_fast_q4k_library;
        if (kernel_name.find("_q5k_") != std::string_view::npos) return cache.tensor_fast_q5k_library;
        if (kernel_name.find("_q6k_") != std::string_view::npos) return cache.tensor_fast_q6k_library;
        if (kernel_name.find("_q8_0_") != std::string_view::npos) return cache.tensor_fast_q8_0_library;
        return nil;
    }
    return cache.tensor_library;
}

}