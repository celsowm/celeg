#pragma once

#include "lfm/cuda_utils.cuh"
#include "lfm/safetensors.hpp"
#include "lfm/safetensors_repo.hpp"
#include "lfm/runtime_types.hpp"
#include "lfm/detail/model_types.hpp"

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfm {

// Loads and quantizes SafeTensor weights into device memory, with a
// process-wide shared-weight cache keyed by (device, checkpoint path,
// weight mode). Multiple LfmModel sessions on the same device + checkpoint
// + weight_mode share one SharedModelWeights instance to avoid duplicate
// GPU allocations.
//
// Extracted from LfmModel::Impl for Single Responsibility: this class does
// only I/O + quantization + caching; the Impl retains the forward pass,
// session state, and graph capture. New weight formats are added by
// extending the quantization branches here without touching the inference
// path (Open/Closed Principle).
class WeightLoader {
public:
    // Looks up or creates the SharedModelWeights for the given
    // (device, safetensors_path, weight_mode) tuple. The returned shared
    // pointer keeps the arena alive; the caller should hold it for the
    // lifetime of its session.
    static std::shared_ptr<SharedModelWeights> acquire(
        const std::string& safetensors_path,
        WeightMode weight_mode);

    // Constructs a loader bound to a specific shared arena and weight mode.
    // The caller must hold `weights` alive for the lifetime of this loader.
    WeightLoader(std::shared_ptr<SharedModelWeights> weights,
                 WeightMode weight_mode);

    // Loads a BF16 weight tensor by name. Returns a device pointer into
    // the shared arena. Caches by name so repeated calls are cheap.
    const __nv_bfloat16* load_weight(
        const IWeightRepository& repo,
        const std::string& name,
        std::vector<int64_t> expected = {});

    // Loads a 2D linear weight, quantizing to Int8/Int4 if the weight mode
    // demands it. Returns a LinearWeight view into the shared arena.
    const LinearWeight* load_linear_weight(
        const IWeightRepository& repo,
        const std::string& name,
        std::vector<int64_t> expected);

    // Loads a synthetic concatenated linear weight from multiple source
    // tensors (e.g. w1 + w3 -> w13). The result is cached under
    // `synthetic_name`.
    const LinearWeight* load_concat_linear_weight(
        const IWeightRepository& repo,
        const std::string& synthetic_name,
        const std::vector<std::pair<std::string, std::vector<int64_t>>>& parts);

    // Loads a single packed 3D BF16 expert tensor of total shape
    // [experts * rows_per_expert, cols] (or [experts, rows_per_expert, cols])
    // into an ExpertLinearWeight. Used for synthetic/explicitly-packed
    // checkpoints and tests.
    const ExpertLinearWeight* load_expert_linear_weight(
        const IWeightRepository& repo,
        const std::string& name,
        int experts, int rows_per_expert, int cols);

    // Packs per-expert gate (w1) and up (w3) tensors of a MoE layer into a
    // single [experts, 2 * moe_intermediate, hidden] ExpertLinearWeight.
    // Official tensor names:
    //   model.layers.{layer}.feed_forward.experts.{j}.w1.weight  [moe_inter, hidden]
    //   model.layers.{layer}.feed_forward.experts.{j}.w3.weight  [moe_inter, hidden]
    const ExpertLinearWeight* load_moe_gate_up(
        const IWeightRepository& repo, int layer,
        int num_experts, int moe_intermediate, int hidden);

    // Packs per-expert down (w2) tensors of a MoE layer into a
    // single [experts, hidden, moe_intermediate] ExpertLinearWeight.
    // Official tensor name:
    //   model.layers.{layer}.feed_forward.experts.{j}.w2.weight  [hidden, moe_inter]
    const ExpertLinearWeight* load_moe_down(
        const IWeightRepository& repo, int layer,
        int num_experts, int moe_intermediate, int hidden);

    // Loads an F32 tensor (e.g. expert_bias) and returns a device pointer.
    const float* load_f32_weight(
        const IWeightRepository& repo,
        const std::string& name,
        std::vector<int64_t> expected);

    // Loads the router/gate weight [num_experts, hidden] as a LinearWeight.
    // Official tensor name:
    //   model.layers.{layer}.feed_forward.gate.weight
    const LinearWeight* load_router(
        const IWeightRepository& repo, int layer,
        int num_experts, int hidden);

    // Result of loading a MoE layer's experts into host-backed storage for
    // offload. `gate_up_host_dev` / `down_host_dev` are [num_experts]
    // device-visible pointers (CUDA UVA) into `store`, laid out per expert as
    // gate_up [2*moe_inter, hidden] and down [hidden, moe_inter].
    struct HostExpertLayer {
        std::vector<const __nv_bfloat16*> gate_up_host_dev;
        std::vector<const __nv_bfloat16*> down_host_dev;
        size_t gate_up_bytes = 0;  // bytes per expert
        size_t down_bytes = 0;     // bytes per expert
    };

    // Loads a MoE layer's per-expert gate/up/down tensors into the supplied
    // HostExpertStore instead of eagerly uploading them to the GPU. Returns
    // device-visible host pointers per expert. Used by the expert-offload
    // residency path; the caller drives promotion into a GPU ExpertLayerCache.
    // Does NOT touch the shared device weight arena. `host_mode` selects whether
    // the host tier is a pinned copy or a mapped arena.
    HostExpertLayer load_moe_experts_host(
        const IWeightRepository& repo, int layer,
        int num_experts, int moe_intermediate, int hidden,
        class HostExpertStore& store, ExpertHostMode host_mode);

    WeightMode weight_mode() const { return weight_mode_; }

private:
    std::shared_ptr<SharedModelWeights> weights_;
    WeightMode weight_mode_;
    // Keeps ExpertLinearWeight views (pointing into arena buffers) alive for
    // the lifetime of this loader.
    std::unordered_map<std::string, ExpertLinearWeight> expert_cache_;
};

} // namespace lfm
