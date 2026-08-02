#include "celeg/backend/cuda/sampler.hpp"

#include "celeg/backend/cuda/kernels/kernels.cuh"

#include <algorithm>

namespace celeg {

void CudaSampler::enqueue(const DeviceBuffer<__nv_bfloat16>& logits,
                          DeviceBuffer<std::uint8_t>& seen_tokens,
                          DeviceBuffer<float>& sampling_scores,
                          DeviceBuffer<float>& topk_values,
                          DeviceBuffer<std::int32_t>& topk_indices,
                          const RuntimeTopology& shape,
                          const GenerationConfig& generation,
                          DeviceBuffer<std::uint64_t>& rng_state,
                          DeviceBuffer<std::int32_t>& sampled_device,
                          cudaStream_t stream) {
    const int effective_top_k = generation.greedy() ? 1 : generation.top_k;
    const float effective_temperature = generation.temperature > 0.0f
        ? generation.temperature : 1.0f;
    launch_fused_sample_topk(
        logits.data(), seen_tokens.data(), sampling_scores.data(),
        topk_values.data(), topk_indices.data(), shape.vocab_size,
        effective_temperature, generation.repetition_penalty,
        effective_top_k, generation.greedy() ? 1.0f : generation.top_p,
        rng_state.data(), sampled_device.data(), stream);
}

} // namespace celeg
