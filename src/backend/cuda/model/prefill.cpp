#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/paged_kv.hpp"
#include "celeg/backend/cuda/phase_profile.hpp"
#include "celeg/backend/cuda/weight_layout.hpp"
#include "celeg/backend/cuda/moe.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace celeg {

void CudaCompiledModel::validate_token_ids(
    const std::vector<int32_t>& tokens) const {
    for (const int32_t token : tokens) {
        if (token < 0 || token >= resources_.shape_.vocab_size) {
            throw std::invalid_argument("CUDA token id is outside the vocabulary");
        }
    }
}

void CudaCompiledModel::prefill(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) {
        throw std::invalid_argument("prefill needs at least one token");
    }
    if (tokens.size() > static_cast<size_t>(max_context_)) {
        throw std::invalid_argument("prefill exceeds max_context");
    }
    validate_token_ids(tokens);
    if (resources_.mtp_.available()) {
        prefill_chunk(tokens, true, true);
        return;
    }
    // Disk-backed MoE residency needs the active routed set to fit in the
    // per-layer GPU cache.  The packed prefill path can route an entire prompt
    // at once (K * rows unique experts), so use the tokenwise scheduler here;
    // it preserves recurrent GDN state and lets the residency coordinator
    // promote exactly one routed set at a time.
    if (resources_.options_.expert_offload.enabled() &&
        resources_.options_.expert_offload.backing == ExpertBackingMode::DiskCached) {
        prefill_chunk(tokens, true, true);
        return;
    }
    if (resources_.shape_.mamba2_layer_count > 0) {
        prefill_chunk(tokens, true, true);
        return;
    }
    const auto begin = std::chrono::steady_clock::now();
    prefill_batched(tokens);
    const auto end = std::chrono::steady_clock::now();
    session_.metrics_.last_prefill_ms =
        std::chrono::duration<double, std::milli>(end - begin).count();
    session_.metrics_.prefill_tokens = tokens.size();
    session_.metrics_.cumulative_decode_ms = 0.0;
    session_.metrics_.decoded_tokens = 0;
}

void CudaCompiledModel::prefill(const std::vector<int32_t>& tokens,
                                const PromptEmbedding& embeddings) {
    if (tokens.empty()) throw std::invalid_argument("prefill needs at least one token");
    if (tokens.size() > static_cast<size_t>(max_context_)) {
        throw std::invalid_argument("prefill exceeds max_context");
    }
    if (!embeddings.empty() &&
        (embeddings.width <= 0 ||
         embeddings.values.size() != embeddings.positions.size() *
             static_cast<size_t>(embeddings.width))) {
        throw std::invalid_argument("invalid CUDA prompt embedding layout");
    }
    if (embeddings.has_rope_positions && embeddings.rope_positions.size() != tokens.size()) {
        throw std::invalid_argument("CUDA prompt M-RoPE positions must cover every token");
    }
    validate_token_ids(tokens);
    reset();
    session_.phase_ = SessionPhase::Prefilling;
    session_.metrics_ = {};
    const auto begin = std::chrono::steady_clock::now();
    for (size_t index = 0; index < tokens.size(); ++index) {
        const float* raw = embeddings.at_position(index);
        forward_token_host(tokens[index], index + 1 == tokens.size(), raw,
                           embeddings.rope_at_position(index));
    }
    if (embeddings.has_rope_positions) {
        session_.next_rope_position_ = embeddings.next_rope_position;
        CELEG_CUDA(cudaMemcpyAsync(mrope_position_device_.data(),
                                   session_.next_rope_position_.data(),
                                   sizeof(session_.next_rope_position_),
                                   cudaMemcpyHostToDevice, stream_.get()));
    }
    CELEG_CUDA(cudaMemcpyAsync(position_device_.data(), &session_.position_,
                               sizeof(session_.position_), cudaMemcpyHostToDevice,
                               stream_.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));
    const auto end = std::chrono::steady_clock::now();
    session_.metrics_.last_prefill_ms =
        std::chrono::duration<double, std::milli>(end - begin).count();
    session_.metrics_.prefill_tokens = tokens.size();
    session_.phase_ = SessionPhase::Ready;
    session_.active_segmented_attention_ = use_segmented_attention(session_.position_);
}

void CudaCompiledModel::prefill_chunk(const std::vector<int32_t>& tokens,
                              bool begin, bool finalize) {
    if (tokens.empty()) {
        throw std::invalid_argument("prefill_chunk needs at least one token");
    }
    validate_token_ids(tokens);
    if (begin) {
        reset();
        session_.metrics_ = {};
    } else if (session_.position_ == 0) {
        throw std::runtime_error(
            "prefill_chunk continuation requires an initial chunk");
    }
    if (session_.phase_ == SessionPhase::Ready) {
        throw std::runtime_error("cannot append prefill after finalization");
    }
    if (session_.position_ + static_cast<int>(tokens.size()) > max_context_) {
        throw std::invalid_argument("prefill chunks exceed max_context");
    }
    session_.phase_ = SessionPhase::Prefilling;

    DeviceBuffer<int32_t> input(tokens.size());
    CELEG_CUDA(cudaMemcpyAsync(input.data(), tokens.data(),
                             tokens.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    launch_mark_seen_batch(input.data(), static_cast<int>(tokens.size()),
                           sampling_.seen_tokens.data(), resources_.shape_.vocab_size,
                           stream_.get());
    const auto started = std::chrono::steady_clock::now();
    for (size_t i = 0; i < tokens.size(); ++i) {
        forward_token_host(tokens[i], finalize && i + 1 == tokens.size());
    }
    CELEG_CUDA(cudaMemcpyAsync(position_device_.data(), &session_.position_,
                             sizeof(session_.position_), cudaMemcpyHostToDevice,
                             stream_.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));
    const auto ended = std::chrono::steady_clock::now();
    session_.metrics_.last_prefill_ms +=
        std::chrono::duration<double, std::milli>(ended - started).count();
    session_.metrics_.prefill_tokens += tokens.size();
    if (finalize) {
        session_.phase_ = SessionPhase::Ready;
        session_.active_segmented_attention_ = use_segmented_attention(session_.position_);
    }
}

} // namespace celeg
