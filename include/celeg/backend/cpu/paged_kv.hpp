#pragma once

#include "celeg/backend/cpu/runtime_types.hpp"
#include "celeg/backend/cpu/thread_pool.hpp"
#include "celeg/model/graph.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <variant>
#include <vector>

namespace celeg {

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

struct CpuStatePageLayout {
    size_t key_width = 0;
    size_t value_width = 0;
    size_t latent_width = 0;
    size_t rotary_width = 0;

    size_t token_elements() const {
        return key_width + value_width + latent_width + rotary_width;
    }
    void validate() const;
};

class CpuKvPagePool {
public:
    CpuKvPagePool(CpuKvCacheMode mode, size_t page_tokens,
                  CpuStatePageLayout layout);
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
    void write_latent(CpuKvPageId page, size_t token_offset,
                      const float* key, const float* value,
                      const float* rotary);

    const float* key_fp32(CpuKvPageId page, size_t token_offset) const;
    const float* value_fp32(CpuKvPageId page, size_t token_offset) const;
    const uint16_t* key_bf16(CpuKvPageId page, size_t token_offset) const;
    const uint16_t* value_bf16(CpuKvPageId page, size_t token_offset) const;
    const float* latent_key_fp32(CpuKvPageId page, size_t token_offset) const;
    const float* latent_value_fp32(CpuKvPageId page, size_t token_offset) const;
    const float* rotary_fp32(CpuKvPageId page, size_t token_offset) const;
    const uint16_t* latent_key_bf16(CpuKvPageId page, size_t token_offset) const;
    const uint16_t* latent_value_bf16(CpuKvPageId page, size_t token_offset) const;
    const uint16_t* rotary_bf16(CpuKvPageId page, size_t token_offset) const;

    CpuKvCacheMode mode() const { return mode_; }
    size_t page_tokens() const { return page_tokens_; }
    const CpuStatePageLayout& layout() const { return layout_; }
    size_t key_width() const { return layout_.key_width; }
    size_t value_width() const { return layout_.value_width; }
    size_t page_bytes() const { return page_bytes_; }
    CpuKvPageStats stats() const;

private:
    struct Page;
    const Page& checked_page(CpuKvPageId page) const;
    Page& checked_page(CpuKvPageId page);

    CpuKvCacheMode mode_;
    size_t page_tokens_;
    CpuStatePageLayout layout_;
    size_t page_bytes_;
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<Page>> pages_;
    std::vector<CpuKvPageId> free_pages_;
};

struct CpuPagedAttentionOptions {
    size_t parallel_threshold = 256;
    size_t page_tile = 4;
};

class CpuExternalAttentionMemory {
public:
    CpuExternalAttentionMemory(CpuKvCacheMode mode, size_t page_tokens,
                               size_t key_width, size_t value_width);
    ~CpuExternalAttentionMemory() { clear(); }

    CpuExternalAttentionMemory(const CpuExternalAttentionMemory&) = delete;
    CpuExternalAttentionMemory& operator=(const CpuExternalAttentionMemory&) = delete;

    void append(const float* key, const float* value);
    void clear() noexcept;
    size_t token_count() const { return token_count_; }
    const CpuKvPagePool& pool() const { return pool_; }
    std::span<const CpuKvPageId> pages() const { return pages_; }

private:
    CpuKvPagePool pool_;
    std::vector<CpuKvPageId> pages_;
    size_t token_count_ = 0;
};

struct CpuPagedAttentionStats {
    size_t tasks = 0;
    size_t page_tiles = 0;
    bool parallel = false;
};

// The CPU backend needs no data beyond what the semantic pattern already
// carries, so it reuses AttentionPatternSpec directly instead of lowering
// into a parallel kind + flattened-payload representation.
struct CpuAttentionPattern {
    AttentionPatternSpec storage = FullCausalPattern{};

    static CpuAttentionPattern lower(const AttentionPatternSpec& pattern) {
        return {pattern};
    }
    bool allows(int query_position, int key_position) const;
    bool may_read_future(int query_position, int sequence_length) const;
    int first_candidate(int query_position) const;
    bool parallel_safe() const {
        return std::holds_alternative<FullCausalPattern>(storage);
    }
};

struct CpuNoAttentionBiasView {};

struct CpuAlibiBiasView {
    const float* slopes = nullptr;
    size_t slope_count = 0;
};

struct CpuRelativeBiasView {
    const float* values = nullptr;
    int bucket_count = 0;
    int max_distance = 0;
    bool bidirectional = false;
};

using CpuAttentionBiasStorage = std::variant<CpuNoAttentionBiasView, CpuAlibiBiasView,
                                             CpuRelativeBiasView>;

struct CpuAttentionBias {
    CpuAttentionBiasStorage storage = CpuNoAttentionBiasView{};

    static CpuAttentionBias lower(const AttentionBiasSpec& bias,
                                  std::span<const float> relative_values = {},
                                  int query_heads = 0);
    bool empty() const {
        return std::holds_alternative<CpuNoAttentionBiasView>(storage);
    }
    float score(int query_head, int query_position, int key_position) const;
};

void cpu_gqa_decode_paged(
    const float* q,
    const CpuKvPagePool& pool,
    std::span<const CpuKvPageId> pages,
    float* output,
    int sequence_length,
    int q_heads,
    int kv_heads,
    int head_dim,
    CpuAttentionPattern pattern = {}, CpuAttentionBias bias = {},
    int query_position = -1);

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
    CpuAttentionPattern pattern = {},
    CpuAttentionBias bias = {},
    CpuPagedAttentionOptions options = {},
    CpuPagedAttentionStats* stats = nullptr);

void cpu_latent_attention_decode_paged(
    const float* query_content, const float* query_rope,
    const CpuKvPagePool& pool, std::span<const CpuKvPageId> pages,
    float* output, int sequence_length, int query_heads, int latent_rank,
    int rope_head_dim, float scale, CpuAttentionPattern pattern = {},
    CpuAttentionBias bias = {}, int query_position = -1);

void cpu_gqa_prefill_paged(
    const float* queries,
    size_t query_rows,
    size_t query_stride,
    const CpuKvPagePool& pool,
    std::span<const CpuKvPageId> pages,
    float* output,
    int base_sequence_length,
    int q_heads,
    int kv_heads,
    int head_dim,
    CpuThreadPool& thread_pool,
    CpuAttentionPattern pattern = {}, CpuAttentionBias bias = {});

}
