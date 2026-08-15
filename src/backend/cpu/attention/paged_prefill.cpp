#include "celeg/backend/cpu/paged_kv.hpp"

#include <stdexcept>

namespace celeg {

void cpu_gqa_prefill_paged(
    const float* queries, size_t query_rows, size_t query_stride,
    const CpuKvPagePool& pool,
    std::span<const CpuKvPageId> pages, float* output,
    int base_sequence_length, int q_heads, int kv_heads, int head_dim,
    CpuThreadPool& thread_pool, CpuAttentionPattern pattern,
    CpuAttentionBias bias) {
    if (!queries || !output || query_rows == 0 || base_sequence_length < 0 ||
        q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0 || head_dim <= 0) {
        throw std::invalid_argument("invalid CPU paged prefill GQA arguments");
    }
    const size_t query_width = static_cast<size_t>(q_heads) * head_dim;
    if (query_stride < query_width) {
        throw std::invalid_argument("CPU paged prefill GQA query stride is too small");
    }
    const int committed_sequence_length = base_sequence_length +
        static_cast<int>(query_rows);
    thread_pool.parallel_for(0, query_rows, 1, [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) {
            const int query_position = base_sequence_length + static_cast<int>(row);
            const int sequence_length = pattern.may_read_future(
                query_position, committed_sequence_length)
                ? committed_sequence_length : query_position + 1;
            cpu_gqa_decode_paged(queries + row * query_stride, pool, pages,
                                 output + row * query_width, sequence_length,
                                 q_heads, kv_heads, head_dim, pattern,
                                 bias, query_position);
        }
    });
}

}
