#pragma once

#include "lfm/cuda_utils.cuh"
#include "lfm/safetensors.hpp"
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
        const SafeTensorFile& file,
        const std::string& name,
        std::vector<int64_t> expected = {});

    // Loads a 2D linear weight, quantizing to Int8/Int4 if the weight mode
    // demands it. Returns a LinearWeight view into the shared arena.
    const LinearWeight* load_linear_weight(
        const SafeTensorFile& file,
        const std::string& name,
        std::vector<int64_t> expected);

    // Loads a synthetic concatenated linear weight from multiple source
    // tensors (e.g. w1 + w3 -> w13). The result is cached under
    // `synthetic_name`.
    const LinearWeight* load_concat_linear_weight(
        const SafeTensorFile& file,
        const std::string& synthetic_name,
        const std::vector<std::pair<std::string, std::vector<int64_t>>>& parts);

    WeightMode weight_mode() const { return weight_mode_; }

private:
    std::shared_ptr<SharedModelWeights> weights_;
    WeightMode weight_mode_;
};

} // namespace lfm
