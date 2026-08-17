#pragma once

#include "celeg/model/resolved.hpp"
#include "celeg/model/program.hpp"
#include "celeg/backend/cuda/packed/session.hpp"
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

struct PackedWorkspaceRequirements {
    size_t maximum_batch = 0;
    size_t maximum_prefill_tokens = 0;
    size_t maximum_projection_width = 0;
    size_t maximum_mamba_projection_width = 0;
    size_t maximum_mamba_intermediate = 0;
    size_t maximum_ffn_intermediate = 0;
    size_t max_gated_delta_qkv = 0;
    size_t max_gated_delta_output = 0;
    size_t max_gated_delta_gate = 0;
    int attention_query_heads = 0;
    int attention_head_dim = 0;
    size_t moe_intermediate = 0;
    size_t layer_slots = 0;
    size_t page_table_entries = 0;

    static PackedWorkspaceRequirements derive(
        size_t maximum_batch,
        size_t maximum_prefill_tokens,
        size_t page_table_stride,
        const ExecutionTopology& shape,
        const CompiledModelProgram& program);
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

PackedEligibility validate_packed_batch_common(
    const std::vector<PackedSessionContext>& sessions,
    size_t maximum_batch, PackedOperation operation);

class PackedBatchValidator {
public:
    PackedEligibility validate_session(
        const PackedSessionContext& session,
        PackedOperation operation,
        PackedExecutorCapabilities capabilities) const;
};

class PackedCompatibilityPolicy {
public:
    PackedCompatibilityPolicy(uint64_t execution_plan_fingerprint,
                              int device_ordinal)
        : execution_plan_fingerprint_(execution_plan_fingerprint),
          device_ordinal_(device_ordinal) {}

    PackedEligibility validate_executor(
        const PackedSessionContext& session) const;
    PackedEligibility validate_batch(
        const PackedSessionContext& reference,
        const std::vector<PackedSessionContext>& sessions) const;

private:
    uint64_t execution_plan_fingerprint_ = 0;
    int device_ordinal_ = -1;
};

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

class PackedDecodeExecutor {
public:
    PackedDecodeExecutor(size_t maximum_sessions,
                         size_t maximum_prefill_tokens,
                         PhysicalPagedKvCache* paged_kv,
                         const ExecutionTopology& shape,
                         const CompiledModelProgram& program,
                         int vocab_size,
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
    void decode_into(const std::vector<PackedSessionContext>& sessions,
                     std::span<int32_t> output);
    void decode_into(const std::vector<PackedSessionContext>& sessions,
                     const std::vector<std::vector<uint32_t>>& page_tables,
                     std::span<int32_t> output);
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

}
