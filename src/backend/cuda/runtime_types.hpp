#pragma once

#include "celeg/model/runtime_types.hpp"
#include "celeg/runtime/concurrency/policy_types.hpp"
#include "backend/cuda/moe/offload.hpp"

#include <cstddef>
#include <cstdlib>
#include <optional>
#include <string_view>

namespace celeg {

struct CudaDeviceCapabilities {
    int device_ordinal = -1;
    int compute_major = 0;
    int compute_minor = 0;
    bool mmq_tensor_core_supported = false;
    bool mmq_tensor_core_enabled = false;

    friend bool operator==(const CudaDeviceCapabilities&, const CudaDeviceCapabilities&) = default;
};

enum class GemmBackend {
    Cublas,
    CublasLt,
};

enum class WeightMode {
    Bf16,
    Int8,
    Int4,
    NativeGguf,
};

inline constexpr bool is_rowwise_quantized_weight_mode(WeightMode mode) {
    return mode == WeightMode::Int8 || mode == WeightMode::Int4;
}

inline constexpr std::string_view weight_mode_name(WeightMode mode) {
    switch (mode) {
    case WeightMode::Bf16: return "bf16";
    case WeightMode::Int8: return "int8";
    case WeightMode::Int4: return "int4";
    case WeightMode::NativeGguf: return "native-gguf";
    }
    return "unknown";
}

enum class KvCacheMode {
    Bf16,
    Int8,
};

struct CudaModelOptions {
    bool fused_residuals = false;
    bool fast_attention = false;
    bool fused_projections = false;
    bool cuda_graph = true;
    GemmBackend gemm_backend = GemmBackend::Cublas;
    size_t lt_workspace_bytes = 64ULL * 1024ULL * 1024ULL;
    int lt_heuristics = 8;
    bool lt_autotune = false;
    WeightMode weight_mode = WeightMode::Bf16;
    KvCacheMode kv_cache_mode = KvCacheMode::Bf16;
    AttentionMode attention_mode = AttentionMode::Single;
    int attention_chunk_tokens = 32;
    int attention_auto_threshold = 1;
    bool allocate_local_kv_cache = true;
    bool enable_mtp = false;
    int mtp_speculative_tokens = 1;
    ExpertOffloadOptions expert_offload;
    bool flash_attn = false;
    std::optional<bool> mmq_tensor_cores;
    bool managed_weights = false;
};

inline CudaModelOptions cuda_model_options_from_environment(CudaModelOptions options = {}) {
    if (const char* value = std::getenv("CELEG_FLASH_ATTN");
        value != nullptr && value[0] != '\0') {
        options.flash_attn = value[0] != '0';
    }
    if (const char* value = std::getenv("CELEG_MMQ_TENSOR_CORES"); value != nullptr) {
        options.mmq_tensor_cores = std::string_view(value) == "1";
    }
    if (const char* value = std::getenv("CELEG_CUDA_MANAGED_WEIGHTS"); value != nullptr) {
        options.managed_weights = std::string_view(value) == "1";
    }
    return options;
}

}
