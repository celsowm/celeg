#pragma once

#include "celeg/model/runtime_types.hpp"
#include "celeg/runtime/concurrency/policy_types.hpp"
#include "celeg/backend/cuda/moe/offload.hpp"

#include <cstddef>

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
};

} // namespace celeg
