#pragma once

#include "celeg/backend/cuda/execution_plan.hpp"
#include "celeg/backend/cuda/utils.cuh"
#include "celeg/backend/cuda/runtime_types.hpp"
#include "celeg/model/resolved.hpp"
#include "celeg/model/session.hpp"
#include "celeg/backend/cuda/weight_layout.hpp"
#include "celeg/detail/model/layer_state.hpp"
#include "celeg/detail/model/linear_weights.hpp"
#include "celeg/detail/model/shared_weights.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace celeg {

// Independent identities only. CudaExecutionPlan::fingerprint() already mixes
// every CudaModelOptions field (weight/kv-cache/gemm/attention modes, fusion
// and autotune flags, MTP settings, workspace sizing, expert-offload config,
// device capabilities, device_ordinal, max_context) -- duplicating those
// fields here would let two independently-maintained representations of the
// same plan drift apart. device_ordinal is kept because it is read directly
// (not just compared) by executor placement checks.
struct PackedCompatibilityKey {
    const SharedModelWeights* weights_identity = nullptr;
    uint64_t execution_plan_fingerprint = 0;
    uint64_t compiled_program_id = 0;
    int device_ordinal = -1;

    friend bool operator==(const PackedCompatibilityKey&,
                           const PackedCompatibilityKey&) = default;
};

struct PackedExecutionServices {
    const SessionState* session_identity = nullptr;
    const CudaExecutionPlan* execution_plan = nullptr;
    const CompiledModelProgram* program = nullptr;
    SharedModelWeights* weights = nullptr;
    ExpertResidencyWorkspace* residency_workspace = nullptr;

    // A packed session is only usable once every required dependency is
    // bound. session_identity alone was previously trusted as a proxy for
    // overall validity, which let a partially-populated instance (e.g.
    // constructed by hand rather than through packed_session_context())
    // pass validity checks while other pointers stayed null.
    explicit operator bool() const noexcept {
        return session_identity != nullptr && execution_plan != nullptr &&
               program != nullptr && weights != nullptr &&
               residency_workspace != nullptr;
    }

    friend bool operator==(const PackedExecutionServices& services,
                           std::nullptr_t) noexcept {
        return !services;
    }
    friend bool operator==(std::nullptr_t,
                           const PackedExecutionServices& services) noexcept {
        return services == nullptr;
    }
    friend bool operator==(const PackedExecutionServices& lhs,
                           const PackedExecutionServices& rhs) noexcept {
        return lhs.session_identity == rhs.session_identity;
    }
};

inline bool packed_segmented_attention(
    const PackedExecutionServices& services, int host_position) {
    return services.execution_plan &&
           services.execution_plan->segmented_attention(host_position);
}

inline void packed_ensure_expert_residency(
    const PackedExecutionServices& services, int layer, const int* selected_device,
    int rows, cudaStream_t stream, const float* route_scores_device) {
    if (!services.weights || !services.weights->residency_coordinator) return;
    if (!services.program || !services.residency_workspace) {
        throw std::invalid_argument("packed expert residency service is incomplete");
    }
    const CompiledLayerProgram& layer_program = services.program->layers.at(
        static_cast<size_t>(layer));
    const auto* moe = std::get_if<MoeLayerProgram>(&layer_program.feed_forward);
    if (!moe) {
        throw std::invalid_argument("packed residency requested for a non-MoE layer");
    }
    services.weights->residency_coordinator->ensure(ExpertResidencyRequest{
        layer,
        selected_device,
        rows,
        moe->router.experts_per_token,
        moe->router.expert_count,
        stream,
        route_scores_device,
        services.residency_workspace});
}

struct PackedImmutableView {
    int max_context_value = 0;
    const CudaModelOptions* options_state = nullptr;
    const ExecutionTopology* shape_state = nullptr;
    const CompiledModelProgram* program_state = nullptr;
    std::shared_ptr<SharedModelWeights> weights_state;
    IWeightLayout* weight_layout_state = nullptr;
    const LinearWeight* embedding_weight = nullptr;
    const LinearWeight* logits_weight_value = nullptr;
    const __nv_bfloat16* final_norm_value = nullptr;
};

struct PackedSessionStateView {
    SessionPhase* phase_state = nullptr;
    int* position_state = nullptr;
    const bool* local_kv_cache_available_state = nullptr;
    bool* active_segmented_attention_state = nullptr;
    const GenerationConfig* generation_state = nullptr;
    DeviceBuffer<__nv_bfloat16>* logits_state = nullptr;
    DeviceBuffer<uint8_t>* seen_tokens_state = nullptr;
    DeviceBuffer<uint64_t>* rng_state_buffer = nullptr;
    DeviceBuffer<int32_t>* sampled_device_state = nullptr;
    DeviceBuffer<int32_t>* position_device_state = nullptr;
    PinnedBuffer<int32_t>* sampled_host_state = nullptr;
    std::vector<Layer>* layers_state = nullptr;
    RuntimeMetrics* metrics_state = nullptr;
};

struct PackedSessionContext {
    using SegmentedAttentionFn = bool (*)(const PackedExecutionServices&, int);
    using ExpertResidencyFn = void (*)(const PackedExecutionServices&, int,
                                       const int*, int, cudaStream_t,
                                       const float*);

    PackedExecutionServices owner;
    uint64_t storage_generation_value = 0;
    PackedCompatibilityKey compatibility_key;
    PackedImmutableView immutable;
    PackedSessionStateView session;

    SegmentedAttentionFn segmented_attention = &packed_segmented_attention;
    ExpertResidencyFn ensure_expert_residency = &packed_ensure_expert_residency;

    SessionPhase phase() const { return *session.phase_state; }
    void set_phase(SessionPhase value) const { *session.phase_state = value; }
    int position() const { return *session.position_state; }
    void set_position(int value) const { *session.position_state = value; }
    int max_context() const { return immutable.max_context_value; }
    bool local_kv_cache_available() const {
        return *session.local_kv_cache_available_state;
    }
    bool active_segmented_attention() const {
        return *session.active_segmented_attention_state;
    }
    void set_active_segmented_attention(bool value) const {
        *session.active_segmented_attention_state = value;
    }
    const CudaModelOptions& options() const { return *immutable.options_state; }
    const GenerationConfig& generation() const { return *session.generation_state; }
    const ExecutionTopology& shape() const { return *immutable.shape_state; }
    const CompiledModelProgram& program() const { return *immutable.program_state; }
    const std::shared_ptr<SharedModelWeights>& weights() const {
        return immutable.weights_state;
    }
    DeviceBuffer<__nv_bfloat16>& logits() const { return *session.logits_state; }
    DeviceBuffer<uint8_t>& seen_tokens() const { return *session.seen_tokens_state; }
    DeviceBuffer<uint64_t>& rng_state() const { return *session.rng_state_buffer; }
    DeviceBuffer<int32_t>& sampled_device() const {
        return *session.sampled_device_state;
    }
    DeviceBuffer<int32_t>& position_device() const {
        return *session.position_device_state;
    }
    PinnedBuffer<int32_t>& sampled_host() const { return *session.sampled_host_state; }
    std::vector<Layer>& layers() const { return *session.layers_state; }
    RuntimeMetrics& metrics() const { return *session.metrics_state; }
    int32_t sampled_host_value() const { return sampled_host().data()[0]; }
    void set_sampled_host_value(int32_t value) const {
        sampled_host().data()[0] = value;
    }
    bool use_segmented_attention(int host_position) const {
        return segmented_attention(owner, host_position);
    }
    IWeightLayout& weight_layout() const { return *immutable.weight_layout_state; }
    const LinearWeight* embedding() const { return immutable.embedding_weight; }
    const LinearWeight* logits_weight() const {
        return immutable.logits_weight_value;
    }
    const __nv_bfloat16* final_norm() const { return immutable.final_norm_value; }
};

}
