/**
 * @brief Flash-style decode attention with 8-wide blocks, QK scalar + PV 8x8.
 *
 * One TG per head, 8 SGs, 8 keys/SG/iter (64 keys/iter). QK scalar, PV 8x8.
 */

/// Reuses kCelegAttentionSlots, CelegAttentionNoBias, CelegAttentionSpan and
/// celeg_attention_decode_span from common.metal.

template <typename Bias>
void celeg_attention_decode_fa_span(device const float* query,
                                    device const float* key_cache,
                                    device const float* value_cache,
                                    device float* output,
                                    CelegAttentionSpan span,
                                    Bias bias,
                                    threadgroup float* shared,
                                    uint lane,
                                    uint simd,
                                    uint simd_count) {
    const uint head_dim = span.head_dim;
    const size_t key_width = static_cast<size_t>(span.key_heads) * head_dim;
    const size_t key_offset = static_cast<size_t>(span.key_head) * head_dim;

    float query_values[kCelegAttentionSlots];
    float accumulator[kCelegAttentionSlots];
    for (uint slot = 0; slot < kCelegAttentionSlots; ++slot) {
        const uint dimension = lane + slot * 32u;
        query_values[slot] = dimension < head_dim ? query[span.query_base + dimension] : 0.0f;
        accumulator[slot] = 0.0f;
    }

    float maximum = -INFINITY;
    float denominator = 0.0f;

    threadgroup float* q_block = shared;
    threadgroup float* prod_store = shared + 64;
    threadgroup float* p_block = shared + 128;
    threadgroup float* acc_block = shared + 192;
    threadgroup float* shared_maximum = shared + 256;
    threadgroup float* shared_denominator = shared + 264;
    threadgroup float* shared_accumulator = shared + 272;

    for (uint base = span.start + simd * 8u; base < span.sequence_length;
         base += simd_count * 8u) {
        float scores[8];
        for (uint k = 0; k < 8u; ++k) {
            const uint position = base + k;
            if (position >= span.sequence_length) {
                scores[k] = -INFINITY;
                continue;
            }
            const size_t key_base = static_cast<size_t>(position) * key_width + key_offset;
            float partial = 0.0f;
            for (uint slot = 0; slot < kCelegAttentionSlots; ++slot) {
                const uint dimension = lane + slot * 32u;
                if (dimension < head_dim) partial += query_values[slot] * key_cache[key_base + dimension];
            }
            const float score = simd_sum(partial) * span.scale + bias.value(span.head, span.query_position, position);
            scores[k] = score;
        }
        float block_maximum = scores[0];
        for (uint c = 1; c < 8u; ++c) block_maximum = max(block_maximum, scores[c]);
        const float updated = max(maximum, block_maximum);
        const float rescale = maximum == -INFINITY ? 0.0f : exp(maximum - updated);
        float block_sum = 0.0f;
        for (uint c = 0; c < 8u; ++c) {
            const uint pos = base + c;
            float p = 0.0f;
            if (pos < span.sequence_length) {
                p = exp(scores[c] - updated);
                block_sum += p;
            }
            scores[c] = p;
        }
        denominator = denominator * rescale + block_sum;
        for (uint slot = 0; slot < kCelegAttentionSlots; ++slot) {
            const uint d = lane + slot * 32u;
            if (d < head_dim) accumulator[slot] *= rescale;
        }
        for (uint c = 0; c < 8u; ++c) {
            const uint pos = base + c;
            if (pos >= span.sequence_length) continue;
            scores[c] = exp(scores[c] - updated);
        }
        // PV via scalar still (8×64) — 8x8 MMA would need p 8→64 replication
        for (uint c = 0; c < 8u; ++c) {
            const uint pos = base + c;
            if (pos >= span.sequence_length) continue;
            const float w = scores[c];
            if (w == 0.0f) continue;
            const size_t vb = static_cast<size_t>(pos) * key_width + key_offset;
            for (uint slot = 0; slot < kCelegAttentionSlots; ++slot) {
                const uint d = lane + slot * 32u;
                if (d < head_dim) accumulator[slot] += w * value_cache[vb + d];
            }
        }
        maximum = updated;
    }

    if (lane == 0) {
        shared_maximum[simd] = maximum;
        shared_denominator[simd] = denominator;
    }
    for (uint slot = 0; slot < kCelegAttentionSlots; ++slot) {
        const uint d = lane + slot * 32u;
        if (d < head_dim) shared_accumulator[simd * head_dim + d] = accumulator[slot];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd != 0) return;

    float global_maximum = -INFINITY;
    for (uint g = 0; g < simd_count; ++g) global_maximum = max(global_maximum, shared_maximum[g]);
    float total = 0.0f;
    for (uint g = 0; g < simd_count; ++g) total += shared_denominator[g] * exp(shared_maximum[g] - global_maximum);
    for (uint slot = 0; slot < kCelegAttentionSlots; ++slot) {
        const uint d = lane + slot * 32u;
        if (d >= head_dim) continue;
        float v = 0.0f;
        for (uint g = 0; g < simd_count; ++g) v += shared_accumulator[g * head_dim + d] * exp(shared_maximum[g] - global_maximum);
        output[span.output_base + d] = v / total;
    }
}

kernel void celeg_attention_decode_fa(
    device const float* query [[buffer(0)]],
    device const float* key_cache [[buffer(1)]],
    device const float* value_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& sequence_length [[buffer(4)]],
    constant uint& query_heads [[buffer(5)]],
    constant uint& key_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant uint& page_tokens [[buffer(9)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simd_count [[simdgroups_per_threadgroup]],
    uint2 grid [[threadgroup_position_in_grid]]) {
    if (grid.x >= query_heads) return;
    celeg_attention_decode_fa_span(query, key_cache, value_cache, output,
        celeg_attention_decode_span(grid.x, query_heads, key_heads, head_dim, sequence_length, 0u, page_tokens, scale),
        CelegAttentionNoBias{}, shared, lane, simd, simd_count);
}

kernel void celeg_attention_decode_fa_sliding(
    device const float* query [[buffer(0)]],
    device const float* key_cache [[buffer(1)]],
    device const float* value_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& sequence_length [[buffer(4)]],
    constant uint& query_heads [[buffer(5)]],
    constant uint& key_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant uint& page_tokens [[buffer(9)]],
    constant uint& window_size [[buffer(10)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simd_count [[simdgroups_per_threadgroup]],
    uint2 grid [[threadgroup_position_in_grid]]) {
    if (grid.x >= query_heads) return;
    const uint start = sequence_length > window_size ? sequence_length - window_size : 0u;
    celeg_attention_decode_fa_span(query, key_cache, value_cache, output,
        celeg_attention_decode_span(grid.x, query_heads, key_heads, head_dim, sequence_length, start, page_tokens, scale),
        CelegAttentionNoBias{}, shared, lane, simd, simd_count);
}
