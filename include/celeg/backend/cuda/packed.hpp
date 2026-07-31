#pragma once

#include "celeg/model/resolved.hpp"
#include "celeg/backend/cuda/packed_session.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace celeg {

class PhysicalPagedKvCache;
class Model;
struct PackedDecodeExecutorImpl;

// Backend-internal factory. Keeping this operation-specific context factory
// out of the public Model surface prevents CUDA packed-execution types from
// becoming part of the generic model API.
PackedSessionContext packed_session_context(Model& model);

struct PackedDecodeMetrics {
    uint64_t steps = 0;
    uint64_t tokens = 0;
    uint64_t fallback_batches = 0;
    uint64_t segmented_paged_steps = 0;
    uint64_t segmented_paged_tokens = 0;
    uint64_t ragged_prefill_steps = 0;
    uint64_t ragged_prefill_tokens = 0;
    // Number of full transformer passes issued for ragged prefill.  A
    // flattened ragged batch must contribute exactly one.
    uint64_t ragged_prefill_transformer_passes = 0;
    size_t maximum_batch = 0;
    size_t maximum_prefill_batch = 0;
    double cumulative_ms = 0.0;
    double cumulative_prefill_ms = 0.0;

    double tokens_per_second() const {
        return cumulative_ms > 0.0
            ? static_cast<double>(tokens) * 1000.0 / cumulative_ms
            : 0.0;
    }
};

struct PackedPrefillRow {
    size_t token_offset = 0;
    size_t token_count = 0;
    uint8_t finalize = 0;
};

// Executes one decode token for several independent sessions in one packed
// model pass. The sessions retain separate KV/ShortConv/RNG state, while all
// linear layers use M=batch_size against one shared checkpoint allocation.
class PackedDecodeExecutor {
public:
    PackedDecodeExecutor(size_t maximum_sessions,
                         size_t maximum_prefill_tokens,
                         PhysicalPagedKvCache* paged_kv,
                         const RuntimeTopology& shape);
    ~PackedDecodeExecutor();

    PackedDecodeExecutor(const PackedDecodeExecutor&) = delete;
    PackedDecodeExecutor& operator=(const PackedDecodeExecutor&) = delete;

    bool eligible(const PackedSessionContext& session,
                  std::string* reason = nullptr) const;
    std::vector<int32_t> decode(
        const std::vector<PackedSessionContext>& sessions);
    std::vector<int32_t> decode(
        const std::vector<PackedSessionContext>& sessions,
        const std::vector<std::vector<uint32_t>>& page_tables);
    // Advances a flattened ragged prompt batch. Each row consumes
    // `rows[i].token_count` tokens from `tokens` beginning at
    // `rows[i].token_offset`.
    void prefill(const std::vector<PackedSessionContext>& sessions,
                 const std::vector<std::vector<uint32_t>>& page_tables,
                 const std::vector<int32_t>& tokens,
                 const std::vector<PackedPrefillRow>& rows);
    size_t maximum_batch() const;
    size_t maximum_prefill_tokens() const;
    PackedDecodeMetrics metrics() const;

private:
    std::unique_ptr<PackedDecodeExecutorImpl> impl_;
};

} // namespace celeg
