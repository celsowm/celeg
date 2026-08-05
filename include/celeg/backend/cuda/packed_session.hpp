#pragma once

#include "celeg/backend/cuda/utils.cuh"
#include "celeg/backend/cuda/runtime_types.hpp"
#include "celeg/model/resolved.hpp"
#include "celeg/backend/cuda/weight_layout.hpp"
#include "celeg/detail/model/types.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace celeg {

// Immutable identity of every execution-relevant choice shared by a packed
// batch. This is produced when a compiled model binds its resources; packed
// validation only compares values and never reconstructs compatibility from
// mutable option objects.
struct PackedCompatibilityKey {
    const void* weights_identity = nullptr;
    uint64_t execution_plan_fingerprint = 0;
    uint64_t compiled_program_id = 0;
    uint64_t expert_residency_fingerprint = 0;
    int device_ordinal = -1;
    int max_context = 0;
    uint8_t weight_mode = 0;
    uint8_t kv_cache_mode = 0;
    uint8_t gemm_backend = 0;
    uint8_t attention_mode = 0;
    bool fast_attention = false;
    bool fused_projections = false;
    bool fused_residuals = false;
    bool cuda_graph = false;
    bool lt_autotune = false;
    bool allocate_local_kv_cache = false;
    size_t lt_workspace_bytes = 0;
    int lt_heuristics = 0;
    int attention_chunk_tokens = 0;
    int attention_auto_threshold = 0;

    friend bool operator==(const PackedCompatibilityKey&,
                           const PackedCompatibilityKey&) = default;
};

// Operation-specific, non-owning context for one lane in a packed decode or
// ragged-prefill batch.  It contains only the resources that the packed
// executor is allowed to touch; model loading, graph capture, and unrelated
// session operations are deliberately absent.  The context is copied into a
// batch before execution, while its pointed-to state remains owned by CudaModel.
//
// Callback fields are construction-time bindings to the concrete model
// program. They avoid a virtual getter interface and keep architecture and
// dispatch decisions out of the packed executor's layer loop.
struct PackedSessionContext {
    using SegmentedAttentionFn = bool (*)(const void*, int);
    using ExpertResidencyFn = void (*)(void*, int, const int*, int,
                                       cudaStream_t, const float*);

    void* owner = nullptr;
    // Monotonic generation for the session-owned cache/state allocations.
    // Pointer identity alone is insufficient because a reset may replace a
    // buffer while retaining the same model owner.
    uint64_t storage_generation_value = 0;
    PackedCompatibilityKey compatibility_key;
    SessionPhase* phase_state = nullptr;
    int* position_state = nullptr;
    int max_context_value = 0;
    const bool* local_kv_cache_available_state = nullptr;
    bool* active_segmented_attention_state = nullptr;
    const CudaModelOptions* options_state = nullptr;
    const GenerationConfig* generation_state = nullptr;
    const RuntimeTopology* shape_state = nullptr;
    std::shared_ptr<SharedModelWeights> weights_state;

    DeviceBuffer<__nv_bfloat16>* logits_state = nullptr;
    DeviceBuffer<uint8_t>* seen_tokens_state = nullptr;
    DeviceBuffer<uint64_t>* rng_state_buffer = nullptr;
    DeviceBuffer<int32_t>* sampled_device_state = nullptr;
    DeviceBuffer<int32_t>* position_device_state = nullptr;
    PinnedBuffer<int32_t>* sampled_host_state = nullptr;
    std::vector<Layer>* layers_state = nullptr;
    RuntimeMetrics* metrics_state = nullptr;
    IWeightLayout* weight_layout_state = nullptr;
    const LinearWeight* embedding_weight = nullptr;
    const LinearWeight* logits_weight_value = nullptr;
    const __nv_bfloat16* final_norm_value = nullptr;

    SegmentedAttentionFn segmented_attention = nullptr;
    ExpertResidencyFn ensure_expert_residency = nullptr;

    SessionPhase phase() const { return *phase_state; }
    void set_phase(SessionPhase value) const { *phase_state = value; }
    int position() const { return *position_state; }
    void set_position(int value) const { *position_state = value; }
    int max_context() const { return max_context_value; }
    bool local_kv_cache_available() const {
        return *local_kv_cache_available_state;
    }
    bool active_segmented_attention() const {
        return *active_segmented_attention_state;
    }
    void set_active_segmented_attention(bool value) const {
        *active_segmented_attention_state = value;
    }
    const CudaModelOptions& options() const { return *options_state; }
    const GenerationConfig& generation() const { return *generation_state; }
    const RuntimeTopology& shape() const { return *shape_state; }
    const std::shared_ptr<SharedModelWeights>& weights() const {
        return weights_state;
    }
    DeviceBuffer<__nv_bfloat16>& logits() const { return *logits_state; }
    DeviceBuffer<uint8_t>& seen_tokens() const { return *seen_tokens_state; }
    DeviceBuffer<uint64_t>& rng_state() const { return *rng_state_buffer; }
    DeviceBuffer<int32_t>& sampled_device() const { return *sampled_device_state; }
    DeviceBuffer<int32_t>& position_device() const { return *position_device_state; }
    PinnedBuffer<int32_t>& sampled_host() const { return *sampled_host_state; }
    std::vector<Layer>& layers() const { return *layers_state; }
    RuntimeMetrics& metrics() const { return *metrics_state; }
    int32_t sampled_host_value() const { return sampled_host().data()[0]; }
    void set_sampled_host_value(int32_t value) const {
        sampled_host().data()[0] = value;
    }
    bool use_segmented_attention(int host_position) const {
        return segmented_attention(owner, host_position);
    }
    IWeightLayout& weight_layout() const { return *weight_layout_state; }
    const LinearWeight* embedding() const { return embedding_weight; }
    const LinearWeight* logits_weight() const { return logits_weight_value; }
    const __nv_bfloat16* final_norm() const { return final_norm_value; }
};

} // namespace celeg
