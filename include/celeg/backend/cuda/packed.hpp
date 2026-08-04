#pragma once

#include "celeg/model/resolved.hpp"
#include "celeg/backend/cuda/packed_session.hpp"
#include "celeg/backend/cuda/execution_plan.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace celeg {

class PhysicalPagedKvCache;
class CudaModel;
struct PackedDecodeExecutorImpl;

// Immutable sizing contract for the packed executor. Every reusable host and
// device allocation is derived from this value so a layer-specific shape
// cannot silently exceed a buffer sized from the first layer.
struct PackedWorkspaceRequirements {
    size_t maximum_batch = 0;
    size_t maximum_prefill_tokens = 0;
    size_t maximum_projection_width = 0;
    size_t maximum_ffn_intermediate = 0;
    size_t moe_intermediate = 0;
    size_t layer_slots = 0;
    size_t page_table_entries = 0;

    static PackedWorkspaceRequirements derive(
        size_t maximum_batch,
        size_t maximum_prefill_tokens,
        size_t page_table_stride,
        const RuntimeTopology& shape);
};

enum class PackedOperation { Decode, Prefill };

struct PackedEligibility {
    bool accepted = false;
    std::string reason;

    explicit operator bool() const { return accepted; }
};

struct PackedExecutorCapabilities {
    bool physical_paged_kv = false;
};

class PackedBatchValidator {
public:
    PackedEligibility validate_session(
        const PackedSessionContext& session,
        PackedOperation operation,
        PackedExecutorCapabilities capabilities) const;
};

// Backend-internal factory. Keeping this operation-specific context factory
// out of the public CudaModel surface prevents CUDA packed-execution types from
// becoming part of the generic model API.
PackedSessionContext packed_session_context(CudaModel& model);

struct PackedDecodeMetrics {
    struct Timing {
        double host_prepare_ms = 0.0;
        double gpu_execute_ms = 0.0;
        double host_commit_ms = 0.0;
        double end_to_end_ms = 0.0;
    };

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
    Timing last_decode_timing;
    Timing last_prefill_timing;

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
                         const RuntimeTopology& shape,
                         CudaExecutionPlan plan);
    ~PackedDecodeExecutor();

    PackedDecodeExecutor(const PackedDecodeExecutor&) = delete;
    PackedDecodeExecutor& operator=(const PackedDecodeExecutor&) = delete;

    PackedEligibility validate_session(const PackedSessionContext& session,
                                      PackedOperation operation) const;
    std::vector<int32_t> decode(
        const std::vector<PackedSessionContext>& sessions);
    std::vector<int32_t> decode(
        const std::vector<PackedSessionContext>& sessions,
        const std::vector<std::vector<uint32_t>>& page_tables);
    // Allocation-free form for schedulers that retain their result buffer.
    void decode_into(const std::vector<PackedSessionContext>& sessions,
                     std::span<int32_t> output);
    void decode_into(const std::vector<PackedSessionContext>& sessions,
                     const std::vector<std::vector<uint32_t>>& page_tables,
                     std::span<int32_t> output);
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
    std::unique_ptr<PackedDecodeExecutorImpl> state_;
};

} // namespace celeg
