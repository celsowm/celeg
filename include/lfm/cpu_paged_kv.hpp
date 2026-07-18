#pragma once

#include "lfm/cpu_runtime_types.hpp"
#include "lfm/cpu_thread_pool.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace lfm {

using CpuKvPageId = uint32_t;
inline constexpr CpuKvPageId kInvalidCpuKvPage = UINT32_MAX;

struct CpuKvPageStats {
    size_t total_pages = 0;
    size_t used_pages = 0;
    size_t retained_references = 0;
    size_t bytes_reserved = 0;
    size_t numa_bound_pages = 0;
    size_t numa_binding_failures = 0;
};

// A physically paged, reference-counted K/V arena for one attention layer.
// Pages are stable in memory and may later be shared by prefix-cache entries.
class CpuKvPagePool {
public:
    CpuKvPagePool(CpuKvCacheMode mode, size_t page_tokens, size_t kv_width);
    ~CpuKvPagePool();

    CpuKvPagePool(const CpuKvPagePool&) = delete;
    CpuKvPagePool& operator=(const CpuKvPagePool&) = delete;

    CpuKvPageId allocate(int numa_node = -1);
    void retain(CpuKvPageId page);
    void release(CpuKvPageId page);
    size_t reference_count(CpuKvPageId page) const;
    int numa_node(CpuKvPageId page) const;
    CpuKvPageId clone_prefix(CpuKvPageId source, size_t used_tokens,
                             int numa_node = -1);

    void write(CpuKvPageId page, size_t token_offset,
               const float* key, const float* value);

    const float* key_fp32(CpuKvPageId page, size_t token_offset) const;
    const float* value_fp32(CpuKvPageId page, size_t token_offset) const;
    const uint16_t* key_bf16(CpuKvPageId page, size_t token_offset) const;
    const uint16_t* value_bf16(CpuKvPageId page, size_t token_offset) const;

    CpuKvCacheMode mode() const { return mode_; }
    size_t page_tokens() const { return page_tokens_; }
    size_t kv_width() const { return kv_width_; }
    size_t page_bytes() const { return page_bytes_; }
    CpuKvPageStats stats() const;

private:
    struct Page;
    const Page& checked_page(CpuKvPageId page) const;
    Page& checked_page(CpuKvPageId page);

    CpuKvCacheMode mode_;
    size_t page_tokens_;
    size_t kv_width_;
    size_t page_bytes_;
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<Page>> pages_;
    std::vector<CpuKvPageId> free_pages_;
};

struct CpuPagedAttentionOptions {
    size_t parallel_threshold = 256;
    size_t page_tile = 4;
};

struct CpuPagedAttentionStats {
    size_t tasks = 0;
    size_t page_tiles = 0;
    bool parallel = false;
};

void cpu_gqa_decode_paged(
    const float* q,
    const CpuKvPagePool& pool,
    std::span<const CpuKvPageId> pages,
    float* output,
    int sequence_length,
    int q_heads,
    int kv_heads,
    int head_dim);

void cpu_gqa_decode_paged_parallel(
    const float* q,
    const CpuKvPagePool& pool,
    std::span<const CpuKvPageId> pages,
    float* output,
    int sequence_length,
    int q_heads,
    int kv_heads,
    int head_dim,
    CpuThreadPool& thread_pool,
    CpuPagedAttentionOptions options = {},
    CpuPagedAttentionStats* stats = nullptr);

} // namespace lfm
