#pragma once

#include "celeg/backend/cuda/utils.cuh"
#include "celeg/model/runtime_types.hpp"

#include <cstdint>

namespace celeg {

// Owns the buffers and execution policy that form one sampling session. The
// compiled model supplies logits/configuration; this collaborator owns how a
// token is selected and how its sampling state advances.
class CudaSamplingState final {
public:
    CudaSamplingState()
        : sampled_device(1), sampled_host(1), topk_values(static_cast<size_t>(kMaxTopK)),
          topk_indices(static_cast<size_t>(kMaxTopK)),
          partial_values(static_cast<size_t>(kSamplingPartialBlocks) * kMaxTopK),
          partial_indices(static_cast<size_t>(kSamplingPartialBlocks) * kMaxTopK),
          rng_state(1) {}

    void reset_vocabulary(size_t vocabulary_size) {
        seen_tokens.reset(vocabulary_size);
        sampling_scores.reset(vocabulary_size);
    }

    void enqueue(const DeviceBuffer<__nv_bfloat16>& logits,
                 int vocab_size,
                 const GenerationConfig& generation,
                 cudaStream_t stream);

    size_t bytes() const {
        return seen_tokens.bytes() + sampling_scores.bytes() + topk_values.bytes() +
               topk_indices.bytes() + partial_values.bytes() + partial_indices.bytes() +
               rng_state.bytes() + sampled_device.bytes() +
               sampled_host.size() * sizeof(int32_t);
    }

    DeviceBuffer<uint8_t> seen_tokens;
    DeviceBuffer<float> sampling_scores;
    DeviceBuffer<float> topk_values;
    DeviceBuffer<int32_t> topk_indices;
    // Scratch for the grid-parallel top-k merge in launch_fused_sample_topk.
    DeviceBuffer<float> partial_values;
    DeviceBuffer<int32_t> partial_indices;
    DeviceBuffer<uint64_t> rng_state;
    DeviceBuffer<int32_t> sampled_device;
    PinnedBuffer<int32_t> sampled_host;
};

} // namespace celeg
