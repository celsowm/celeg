#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lfm {

class LfmModel;
class PhysicalPagedKvCache;
struct PackedDecodeExecutorImpl;

struct PackedDecodeMetrics {
    uint64_t steps = 0;
    uint64_t tokens = 0;
    uint64_t fallback_batches = 0;
    uint64_t segmented_paged_steps = 0;
    uint64_t segmented_paged_tokens = 0;
    uint64_t ragged_prefill_steps = 0;
    uint64_t ragged_prefill_tokens = 0;
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

// Executes one decode token for several independent sessions in one packed
// model pass. The sessions retain separate KV/ShortConv/RNG state, while all
// linear layers use M=batch_size against one shared checkpoint allocation.
class PackedDecodeExecutor {
public:
    explicit PackedDecodeExecutor(size_t maximum_batch,
                                  PhysicalPagedKvCache* paged_kv = nullptr);
    ~PackedDecodeExecutor();

    PackedDecodeExecutor(const PackedDecodeExecutor&) = delete;
    PackedDecodeExecutor& operator=(const PackedDecodeExecutor&) = delete;

    bool eligible(const LfmModel& model, std::string* reason = nullptr) const;
    std::vector<int32_t> decode(const std::vector<LfmModel*>& models);
    std::vector<int32_t> decode(
        const std::vector<LfmModel*>& models,
        const std::vector<std::vector<uint32_t>>& page_tables);
    // Advances one explicit prompt token for each row. Rows may have
    // different positions and page tables, which enables wavefront ragged
    // prefill while keeping GEMM M equal to the active prefill batch.
    void prefill_step(const std::vector<LfmModel*>& models,
                      const std::vector<std::vector<uint32_t>>& page_tables,
                      const std::vector<int32_t>& tokens,
                      const std::vector<uint8_t>& finalize_rows);
    size_t maximum_batch() const;
    PackedDecodeMetrics metrics() const;

private:
    std::unique_ptr<PackedDecodeExecutorImpl> impl_;
};

} // namespace lfm
