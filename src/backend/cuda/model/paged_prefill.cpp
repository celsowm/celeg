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

void CudaCompiledModel::prefill_chunk_paged(
    const std::vector<int32_t>& tokens, bool begin, bool finalize,
    PhysicalPagedKvCache& paged_kv,
    const std::vector<uint32_t>& page_table) {
    if (tokens.empty()) {
        throw std::invalid_argument("prefill_chunk_paged needs at least one token");
    }
    if (paged_kv.mode() != resources_.options_.kv_cache_mode) {
        throw std::invalid_argument("model and physical paged KV modes differ");
    }
    if (begin) {
        release_local_kv_cache();
        reset(false);
        session_.metrics_ = {};
    } else if (session_.position_ == 0) {
        throw std::runtime_error(
            "paged prefill continuation requires an initial chunk or prefix state");
    }
    if (session_.position_ + static_cast<int>(tokens.size()) > max_context_) {
        throw std::invalid_argument("paged prefill chunks exceed max_context");
    }
    const size_t final_position =
        static_cast<size_t>(session_.position_) + tokens.size();
    const size_t pages_needed =
        (final_position + static_cast<size_t>(paged_kv.page_tokens()) - 1) /
        static_cast<size_t>(paged_kv.page_tokens());
    if (page_table.size() < pages_needed ||
        page_table.size() > static_cast<size_t>(paged_kv.max_pages_per_request())) {
        throw std::invalid_argument("paged prefill page table has invalid length");
    }
    if (workspace_.paged_page_table_.size() < page_table.size()) {
        workspace_.paged_page_table_.reset(page_table.size());
    }
    CELEG_CUDA(cudaMemcpyAsync(workspace_.paged_page_table_.data(), page_table.data(),
                             page_table.size() * sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    if (workspace_.paged_prefill_tokens_.size() < tokens.size()) {
        workspace_.paged_prefill_tokens_.reset(tokens.size());
    }
    CELEG_CUDA(cudaMemcpyAsync(workspace_.paged_prefill_tokens_.data(), tokens.data(),
                             tokens.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    launch_mark_seen_batch(workspace_.paged_prefill_tokens_.data(),
                           static_cast<int>(tokens.size()), sampling_.seen_tokens.data(),
                           resources_.shape_.vocab_size, stream_.get());
    session_.phase_ = SessionPhase::Prefilling;
    const auto started = std::chrono::steady_clock::now();
    for (size_t i = 0; i < tokens.size(); ++i) {
        forward_token_paged_host(tokens[i], finalize && i + 1 == tokens.size(),
                                 paged_kv, workspace_.paged_page_table_.data(),
                                 static_cast<int>(page_table.size()));
    }
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

