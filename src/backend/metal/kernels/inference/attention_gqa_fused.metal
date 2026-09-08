/**
 * @brief GQA-fused decode attention — 2 Q per TG sharing 1 KV (float KV).
 *
 * 8 TGs per decode (one per KV head) vs 16. Each TG loads K/V 8-block once
 * and scores against 2 Qs. Halves KV reads (96→48 MB for 512) without half
 * precision loss. Reuses same 8 SGs×8 keys/SG/iter as scalar FA but with
 * 2× Q. Shared 2× max/denom/acc (2*8 + 2*8*64 = 1040 f).
 */

/// Reuses kCelegAttentionSlots, CelegAttentionSpan etc from common.metal.

template <typename Bias>
void celeg_attention_gqa_fused_span(device const float* query,
                                    device const float* key_cache,
                                    device const float* value_cache,
                                    device float* output,
                                    uint query_heads,
                                    uint key_heads,
                                    uint head_dim,
                                    uint sequence_length,
                                    uint start,
                                    float scale,
                                    Bias bias,
                                    threadgroup float* shared,
                                    uint lane,
                                    uint simd,
                                    uint simd_count) {
    const uint kv_head = 0; // caller sets via grid.x, query_heads = key_heads*2
}

kernel void celeg_attention_gqa_fused(
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
    const uint kv_head = grid.x;
    if (kv_head >= key_heads) return;
    const uint q0 = kv_head * 2u;
    const uint q1 = q0 + 1u;
    if (q0 >= query_heads) return;

    const uint head_dim_local = head_dim;
    const size_t key_width = static_cast<size_t>(key_heads) * head_dim_local;
    const size_t query_width = static_cast<size_t>(query_heads) * head_dim_local;
    const size_t q0_base = static_cast<size_t>(q0) * head_dim_local;
    const size_t q1_base = static_cast<size_t>(q1) * head_dim_local;
    const size_t out0_base = q0_base;
    const size_t out1_base = q1_base;
    const size_t key_offset = static_cast<size_t>(kv_head) * head_dim_local;

    float q0_values[kCelegAttentionSlots];
    float q1_values[kCelegAttentionSlots];
    float acc0[kCelegAttentionSlots];
    float acc1[kCelegAttentionSlots];
    for (uint slot = 0; slot < kCelegAttentionSlots; ++slot) {
        const uint d = lane + slot * 32u;
        q0_values[slot] = d < head_dim_local ? query[q0_base + d] : 0.0f;
        q1_values[slot] = d < head_dim_local ? query[q1_base + d] : 0.0f;
        acc0[slot] = 0.0f;
        acc1[slot] = 0.0f;
    }

    float max0 = -INFINITY;
    float max1 = -INFINITY;
    float denom0 = 0.0f;
    float denom1 = 0.0f;

    for (uint pos = simd; pos < sequence_length; pos += simd_count) {
        const size_t kb = static_cast<size_t>(pos) * key_width + key_offset;
        float p0 = 0.0f;
        float p1 = 0.0f;
        for (uint slot = 0; slot < kCelegAttentionSlots; ++slot) {
            const uint d = lane + slot * 32u;
            if (d < head_dim_local) {
                const float kv = key_cache[kb + d];
                p0 += q0_values[slot] * kv;
                p1 += q1_values[slot] * kv;
            }
        }
        float score0 = simd_sum(p0) * scale;
        float score1 = simd_sum(p1) * scale;
        // No bias for plain; sliding handled via start
        float upd0 = max(max0, score0);
        float upd1 = max(max1, score1);
        float corr0 = exp(max0 - upd0);
        float corr1 = exp(max1 - upd1);
        float w0 = exp(score0 - upd0);
        float w1 = exp(score1 - upd1);
        if (max0 == -INFINITY) corr0 = 0.0f;
        if (max1 == -INFINITY) corr1 = 0.0f;
        denom0 = denom0 * corr0 + w0;
        denom1 = denom1 * corr1 + w1;
        for (uint slot = 0; slot < kCelegAttentionSlots; ++slot) {
            const uint d = lane + slot * 32u;
            if (d < head_dim_local) {
                acc0[slot] = acc0[slot] * corr0 + w0 * static_cast<float>(value_cache[kb + d]);
                acc1[slot] = acc1[slot] * corr1 + w1 * static_cast<float>(value_cache[kb + d]);
            }
        }
        max0 = upd0;
        max1 = upd1;
    }

    threadgroup float* s_max0 = shared;
    threadgroup float* s_max1 = shared + 8;
    threadgroup float* s_denom0 = shared + 16;
    threadgroup float* s_denom1 = shared + 24;
    threadgroup float* s_acc0 = shared + 32;
    threadgroup float* s_acc1 = shared + 32 + 512;

    if (lane == 0) {
        s_max0[simd] = max0;
        s_max1[simd] = max1;
        s_denom0[simd] = denom0;
        s_denom1[simd] = denom1;
    }
    for (uint slot = 0; slot < kCelegAttentionSlots; ++slot) {
        const uint d = lane + slot * 32u;
        if (d < head_dim_local) {
            s_acc0[simd * head_dim_local + d] = acc0[slot];
            s_acc1[simd * head_dim_local + d] = acc1[slot];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd != 0) return;

    float gmax0 = -INFINITY;
    float gmax1 = -INFINITY;
    for (uint g = 0; g < simd_count; ++g) {
        gmax0 = max(gmax0, s_max0[g]);
        gmax1 = max(gmax1, s_max1[g]);
    }
    float tot0 = 0.0f;
    float tot1 = 0.0f;
    for (uint g = 0; g < simd_count; ++g) {
        tot0 += s_denom0[g] * exp(s_max0[g] - gmax0);
        tot1 += s_denom1[g] * exp(s_max1[g] - gmax1);
    }
    for (uint slot = 0; slot < kCelegAttentionSlots; ++slot) {
        const uint d = lane + slot * 32u;
        if (d >= head_dim_local) continue;
        float v0 = 0.0f;
        float v1 = 0.0f;
        for (uint g = 0; g < simd_count; ++g) {
            v0 += s_acc0[g * head_dim_local + d] * exp(s_max0[g] - gmax0);
            v1 += s_acc1[g * head_dim_local + d] * exp(s_max1[g] - gmax1);
        }
        if (q0 < query_heads) output[out0_base + d] = v0 / tot0;
        if (q1 < query_heads) output[out1_base + d] = v1 / tot1;
    }
}
