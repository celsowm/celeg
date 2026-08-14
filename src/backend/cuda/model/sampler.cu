#include "celeg/backend/cuda/sampler.hpp"
#include "celeg/backend/cuda/sampling_state.hpp"

#include "celeg/backend/cuda/kernels/kernels.cuh"

namespace celeg {

void CudaSampler::enqueue(const DeviceBuffer<__nv_bfloat16>& logits,
                          DeviceBuffer<std::uint8_t>& seen_tokens,
                          DeviceBuffer<float>& sampling_scores,
                          DeviceBuffer<float>& topk_values,
                          DeviceBuffer<std::int32_t>& topk_indices,
                          DeviceBuffer<float>& partial_values,
                          DeviceBuffer<std::int32_t>& partial_indices,
                          int vocab_size,
                          const GenerationConfig& generation,
                          DeviceBuffer<std::uint64_t>& rng_state,
                          DeviceBuffer<std::int32_t>& sampled_device,
                          cudaStream_t stream) {
    if (generation.greedy()) {
        launch_argmax_bf16(logits.data(), seen_tokens.data(), vocab_size,
                           generation.repetition_penalty, sampled_device.data(),
                           stream);
        launch_mark_seen(sampled_device.data(), seen_tokens.data(),
                         vocab_size, stream);
        return;
    }
    const int effective_top_k = generation.top_k;
    const float effective_temperature = generation.temperature > 0.0f
        ? generation.temperature : 1.0f;
    launch_fused_sample_topk(
        logits.data(), seen_tokens.data(), sampling_scores.data(),
        topk_values.data(), topk_indices.data(),
        partial_values.data(), partial_indices.data(), vocab_size,
        effective_temperature, generation.repetition_penalty,
        effective_top_k, generation.top_p,
        rng_state.data(), sampled_device.data(), stream);
}

void CudaSamplingState::enqueue(const DeviceBuffer<__nv_bfloat16>& logits,
                                int vocab_size,
                                const GenerationConfig& generation,
                                cudaStream_t stream) {
    if (generation.forced_prefix &&
        generation.forced_prefix->position < generation.forced_prefix->tokens.size()) {
        const int32_t* token = generation.forced_prefix->tokens.data() +
            generation.forced_prefix->position;
        CELEG_CUDA(cudaMemcpyAsync(
            sampled_device.data(), token, sizeof(*token),
            cudaMemcpyHostToDevice, stream));
        launch_mark_seen(sampled_device.data(), seen_tokens.data(),
                         vocab_size, stream);
        return;
    }
    CudaSampler::enqueue(
        logits, seen_tokens, sampling_scores, topk_values, topk_indices,
        partial_values, partial_indices, vocab_size, generation, rng_state,
        sampled_device, stream);
}

} // namespace celeg
