// Benchmark-only bit-exact candidate for Celeg's flash-style attention core.
// It preserves the online-softmax recurrence but exploits the invariant that
// one of correction/weight is always exp(0) == 1, reducing two exponentials
// per key position to one.

template <typename Bias>
void celeg_attention_span_one_exp(device const float* query,
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
        query_values[slot] = dimension < head_dim
            ? query[span.query_base + dimension] : 0.0f;
        accumulator[slot] = 0.0f;
    }

    float maximum = -INFINITY;
    float denominator = 0.0f;
    for (uint position = span.start + simd; position < span.sequence_length;
         position += simd_count) {
        const size_t key_base =
            static_cast<size_t>(position) * key_width + key_offset;
        float partial = 0.0f;
        for (uint slot = 0; slot < kCelegAttentionSlots; ++slot) {
            const uint dimension = lane + slot * 32u;
            if (dimension < head_dim) {
                partial += query_values[slot] * key_cache[key_base + dimension];
            }
        }
        const float score = simd_sum(partial) * span.scale +
            bias.value(span.head, span.query_position, position);

        float updated;
        float correction;
        float weight;
        if (score > maximum) {
            updated = score;
            correction = exp(maximum - score);
            weight = 1.0f;
        } else {
            updated = maximum;
            correction = 1.0f;
            weight = exp(score - maximum);
        }
        denominator = denominator * correction + weight;
        for (uint slot = 0; slot < kCelegAttentionSlots; ++slot) {
            const uint dimension = lane + slot * 32u;
            if (dimension < head_dim) {
                accumulator[slot] = accumulator[slot] * correction +
                    weight * value_cache[key_base + dimension];
            }
        }
        maximum = updated;
    }

    threadgroup float* shared_maximum = shared;
    threadgroup float* shared_denominator = shared + simd_count;
    threadgroup float* shared_accumulator = shared + 2u * simd_count;
    if (lane == 0) {
        shared_maximum[simd] = maximum;
        shared_denominator[simd] = denominator;
    }
    for (uint slot = 0; slot < kCelegAttentionSlots; ++slot) {
        const uint dimension = lane + slot * 32u;
        if (dimension < head_dim) {
            shared_accumulator[simd * head_dim + dimension] = accumulator[slot];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd != 0) return;

    float global_maximum = -INFINITY;
    for (uint group = 0; group < simd_count; ++group) {
        global_maximum = max(global_maximum, shared_maximum[group]);
    }
    float total = 0.0f;
    for (uint group = 0; group < simd_count; ++group) {
        total += shared_denominator[group] * exp(shared_maximum[group] - global_maximum);
    }
    for (uint slot = 0; slot < kCelegAttentionSlots; ++slot) {
        const uint dimension = lane + slot * 32u;
        if (dimension >= head_dim) continue;
        float value = 0.0f;
        for (uint group = 0; group < simd_count; ++group) {
            value += shared_accumulator[group * head_dim + dimension] *
                exp(shared_maximum[group] - global_maximum);
        }
        output[span.output_base + dimension] = value / total;
    }
}

kernel void celeg_attention_batch_one_exp(
    device const float* query [[buffer(0)]],
    device const float* key_cache [[buffer(1)]],
    device const float* value_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& base_position [[buffer(5)]],
    constant uint& query_heads [[buffer(6)]],
    constant uint& key_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& page_tokens [[buffer(10)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simd_count [[simdgroups_per_threadgroup]],
    uint2 grid [[threadgroup_position_in_grid]]) {
    if (grid.x >= query_heads || grid.y >= rows) return;
    celeg_attention_span_one_exp(
        query, key_cache, value_cache, output,
        celeg_attention_batch_span(grid.x, grid.y, base_position,
                                   query_heads, key_heads, head_dim,
                                   0u, page_tokens, scale),
        CelegAttentionNoBias{}, shared, lane, simd, simd_count);
}
