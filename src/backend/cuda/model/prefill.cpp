#include "lfm/detail/model/impl.hpp"
#include "lfm/backend/cuda/kernels/kernels.cuh"
#include "lfm/backend/cuda/paged_kv.hpp"
#include "lfm/backend/cuda/phase_profile.hpp"
#include "lfm/model/weights/layout.hpp"
#include "lfm/runtime/moe.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lfm {

void Model::Impl::prefill_legacy(const std::vector<int32_t>& tokens) {
    reset();
    DeviceBuffer<int32_t> input(tokens.size());
    LFM_CUDA(cudaMemcpyAsync(input.data(), tokens.data(),
                             tokens.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    launch_mark_seen_batch(input.data(), static_cast<int>(tokens.size()),
                           seen_tokens_.data(), shape_.vocab_size,
                           stream_.get());
    for (size_t i = 0; i < tokens.size(); ++i) {
        forward_token_host(tokens[i], i + 1 == tokens.size());
    }
    LFM_CUDA(cudaMemcpyAsync(position_device_.data(), &position_,
                             sizeof(position_), cudaMemcpyHostToDevice,
                             stream_.get()));
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    phase_ = SessionPhase::Ready;
}

void Model::Impl::prefill(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) {
        throw std::invalid_argument("prefill needs at least one token");
    }
    if (tokens.size() > static_cast<size_t>(max_context_)) {
        throw std::invalid_argument("prefill exceeds max_context");
    }
    const auto begin = std::chrono::steady_clock::now();
    if (options_.legacy_prefill) {
        prefill_legacy(tokens);
    } else {
        prefill_batched(tokens);
    }
    const auto end = std::chrono::steady_clock::now();
    metrics_.last_prefill_ms =
        std::chrono::duration<double, std::milli>(end - begin).count();
    metrics_.prefill_tokens = tokens.size();
    metrics_.cumulative_decode_ms = 0.0;
    metrics_.decoded_tokens = 0;
}

void Model::Impl::prefill_chunk(const std::vector<int32_t>& tokens,
                              bool begin, bool finalize) {
    if (tokens.empty()) {
        throw std::invalid_argument("prefill_chunk needs at least one token");
    }
    if (begin) {
        reset();
        metrics_ = {};
    } else if (position_ == 0) {
        throw std::runtime_error(
            "prefill_chunk continuation requires an initial chunk");
    }
    if (phase_ == SessionPhase::Ready) {
        throw std::runtime_error("cannot append prefill after finalization");
    }
    if (position_ + static_cast<int>(tokens.size()) > max_context_) {
        throw std::invalid_argument("prefill chunks exceed max_context");
    }
    phase_ = SessionPhase::Prefilling;

    DeviceBuffer<int32_t> input(tokens.size());
    LFM_CUDA(cudaMemcpyAsync(input.data(), tokens.data(),
                             tokens.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    launch_mark_seen_batch(input.data(), static_cast<int>(tokens.size()),
                           seen_tokens_.data(), shape_.vocab_size,
                           stream_.get());
    const auto started = std::chrono::steady_clock::now();
    for (size_t i = 0; i < tokens.size(); ++i) {
        forward_token_host(tokens[i], finalize && i + 1 == tokens.size());
    }
    LFM_CUDA(cudaMemcpyAsync(position_device_.data(), &position_,
                             sizeof(position_), cudaMemcpyHostToDevice,
                             stream_.get()));
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    const auto ended = std::chrono::steady_clock::now();
    metrics_.last_prefill_ms +=
        std::chrono::duration<double, std::milli>(ended - started).count();
    metrics_.prefill_tokens += tokens.size();
    if (finalize) {
        phase_ = SessionPhase::Ready;
        active_segmented_attention_ = use_segmented_attention(position_);
    }
}

} // namespace lfm
