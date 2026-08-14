#pragma once

#include "celeg/model/runtime_types.hpp"
#include "celeg/runtime/concurrency/policy_types.hpp"
#include "celeg/backend/cuda/moe/offload.hpp"

#include <cstddef>
#include <cstdlib>
#include <optional>
#include <string_view>

namespace celeg {

namespace detail {

// Resolves a boolean environment variable exactly once, at the call site of
// a configuration struct's default member initializer (i.e. at
// construction/bootstrap time, once per model configuration object).
// Execution/kernel-launch code must never call std::getenv() itself; it
// must read the already-resolved option field instead.
//
// "Nonzero flag" semantics: unset or empty => fallback; otherwise true
// unless the first character is '0'. Matches historical CELEG_FLASH_ATTN
// behavior.
inline bool env_nonzero_flag(const char* name, bool fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    return value[0] != '0';
}

// "Exact match to '1'" semantics, returning nullopt when unset so callers
// can distinguish "no override" from "explicitly disabled". Matches
// historical CELEG_MMQ_TENSOR_CORES / CELEG_CUDA_MANAGED_WEIGHTS behavior.
inline std::optional<bool> env_exact_one_flag(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) return std::nullopt;
    return std::string_view(value) == "1";
}

} // namespace detail

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
    // Enable the native auxiliary predictor when the checkpoint provides an
    // official MTP head.  The path is intentionally explicit because it
    // changes loading, prefill scheduling, and the decode state machine.
    bool enable_mtp = false;
    int mtp_speculative_tokens = 1;
    ExpertOffloadOptions expert_offload;

    // Resolved-once execution policy knobs. These default to the
    // corresponding environment variable at construction time (a
    // configuration/bootstrap boundary) so backend execution/kernel-launch
    // code never needs to call getenv() itself. See
    // docs/SOLID_REVIEW_BACKENDS.md §6.1/§6.3.
    //
    // CELEG_FLASH_ATTN: opt in to the flash-attention prefill kernel outside
    // its always-preferred head-dim range (see CudaExecutionPlan/prefill
    // policy for how this combines with kernel support).
    bool flash_attn = detail::env_nonzero_flag("CELEG_FLASH_ATTN", false);
    // CELEG_MMQ_TENSOR_CORES: explicit override for whether native-GGUF MMQ
    // GEMMs use tensor-core kernels. nullopt means "auto": follow device
    // support as resolved by CudaExecutionPlan::compile().
    std::optional<bool> mmq_tensor_cores = detail::env_exact_one_flag("CELEG_MMQ_TENSOR_CORES");
    // CELEG_CUDA_MANAGED_WEIGHTS: allocate checkpoint weights via CUDA
    // managed memory instead of device memory.
    bool managed_weights = detail::env_exact_one_flag("CELEG_CUDA_MANAGED_WEIGHTS").value_or(false);
};

} // namespace celeg
