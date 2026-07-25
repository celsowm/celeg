#pragma once

#include "lfm/cuda_utils.cuh"
#include "lfm/runtime_types.hpp"
#include "lfm/model_shape.hpp"
#include "lfm/weight_layout.hpp"
#include "lfm/detail/model_types.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace lfm {

// Narrow interface that PackedDecodeExecutor depends on for multi-session
// packed decode. Extracted from the concrete LfmModel::Impl to remove the
// `friend struct PackedDecodeExecutorImpl` breach (Interface Segregation
// Principle): the packed executor now depends only on the accessors it
// actually calls, and new code can implement IPackedSession for testing
// without constructing a full LfmModel. LfmModel::Impl publicly implements
// this interface.
//
// The interface exposes the session-scoped state the packed executor reads
// or writes during a single decode step: position, phase, generation
// config, the sampling device buffers, and the layer vector. Immutable
// checkpoint state (weights, embedding) is shared across sessions and
// accessed through the concrete LfmModel.
class IPackedSession {
public:
    virtual ~IPackedSession() = default;

    // Session state.
    virtual SessionPhase phase() const = 0;
    virtual void set_phase(SessionPhase phase) = 0;
    virtual int position() const = 0;
    virtual void set_position(int position) = 0;
    virtual int max_context() const = 0;
    virtual bool local_kv_cache_available() const = 0;
    virtual bool active_segmented_attention() const = 0;
    virtual void set_active_segmented_attention(bool value) = 0;

    // Configuration.
    virtual const ModelOptions& options() const = 0;
    virtual const GenerationConfig& generation() const = 0;
    virtual const ModelShape& shape() const = 0;
    // Shared checkpoint owner. Two sessions with the same owner pointer
    // share device weights and can be packed together.
    virtual const std::shared_ptr<SharedModelWeights>& weights() const = 0;

    // Sampling device buffers (writable).
    virtual DeviceBuffer<__nv_bfloat16>& logits() = 0;
    virtual DeviceBuffer<uint8_t>& seen_tokens() = 0;
    virtual DeviceBuffer<uint64_t>& rng_state() = 0;
    virtual DeviceBuffer<int32_t>& sampled_device() = 0;
    virtual DeviceBuffer<int32_t>& position_device() = 0;
    virtual PinnedBuffer<int32_t>& sampled_host() = 0;

    // Per-layer K/V and conv state.
    virtual std::vector<Layer>& layers() = 0;
    virtual const std::vector<Layer>& layers() const = 0;

    // Helpers.
    virtual bool use_segmented_attention(int host_position) const = 0;
    virtual RuntimeMetrics& metrics() = 0;
    virtual int32_t sampled_host_value() const = 0;
    virtual void set_sampled_host_value(int32_t value) = 0;

    // Expert offload: ensure selected experts are GPU-resident before the
    // MoE FFN reads them.  Called by the packed executor after the router.
    // `sel_dev` points to this session's selected-expert indices (K ints),
    // `rows` is 1 for decode, and `route_scores_dev` may be null.
    virtual void ensure_moe_experts_resident_packed(
        int layer, const int* sel_dev, int rows, cudaStream_t stream,
        const float* route_scores_dev) = 0;

    // Shared checkpoint state. The packed executor uses the reference
    // session's weight layout for embedding lookup, the final norm, and the
    // RoPE tables (these are identical across all sessions sharing the
    // checkpoint).
    virtual IWeightLayout& weight_layout() = 0;
    virtual const LinearWeight* embedding() const = 0;
    virtual const LinearWeight* logits_weight() const = 0;
    virtual const __nv_bfloat16* final_norm() const = 0;
    virtual const __nv_bfloat16* rope_cos() const = 0;
    virtual const __nv_bfloat16* rope_sin() const = 0;
    virtual void linear(const __nv_bfloat16* x,
                        const LinearWeight& weight,
                        __nv_bfloat16* y,
                        int m, int n, int k,
                        float beta = 0.0f) = 0;
};

} // namespace lfm
