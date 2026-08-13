#pragma once

#include "celeg/backend/cuda/utils.cuh"
#include "celeg/model/resolved.hpp"
#include "celeg/model/runtime_types.hpp"

#include <cstdint>

namespace celeg {

class CudaSampler final {
public:
    static void enqueue(const DeviceBuffer<__nv_bfloat16>& logits,
                        DeviceBuffer<std::uint8_t>& seen_tokens,
                        DeviceBuffer<float>& sampling_scores,
                        DeviceBuffer<float>& topk_values,
                        DeviceBuffer<std::int32_t>& topk_indices,
                        DeviceBuffer<float>& partial_values,
                        DeviceBuffer<std::int32_t>& partial_indices,
                        const ExecutionTopology& shape,
                        int vocab_size,
                        const GenerationConfig& generation,
                        DeviceBuffer<std::uint64_t>& rng_state,
                        DeviceBuffer<std::int32_t>& sampled_device,
                        cudaStream_t stream);
};

} // namespace celeg
