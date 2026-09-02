constant uint kCelegTiledQueriesPerSimdgroup = 8;
constant uint kCelegTiledSimdgroups = 4;
constant uint kCelegTiledHeadDim = 64;
constant uint kCelegTiledScoreFloats =
    kCelegTiledSimdgroups * kCelegTiledQueriesPerSimdgroup * 8;
constant uint kCelegTiledOutputFloats =
    kCelegTiledSimdgroups * kCelegTiledQueriesPerSimdgroup * kCelegTiledHeadDim;
constant uint kCelegTiledRows =
    kCelegTiledSimdgroups * kCelegTiledQueriesPerSimdgroup;
constant uint kCelegTiledSharedFloats =
    kCelegTiledScoreFloats + kCelegTiledOutputFloats + 3 * kCelegTiledRows;

kernel void celeg_attention_tiled_simdgroup(
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
    uint2 grid [[threadgroup_position_in_grid]]) {
    (void)base_position;
    (void)page_tokens;
    const uint head = grid.x;
    const uint query_block = grid.y * kCelegTiledRows;
    const uint query_row =
        query_block + simd * kCelegTiledQueriesPerSimdgroup;
    if (head >= query_heads || simd >= kCelegTiledSimdgroups ||
        head_dim != kCelegTiledHeadDim || query_row >= rows) {
        return;
    }

    const uint query_width = query_heads * head_dim;
    const uint key_width = key_heads * head_dim;
    const uint key_head = head / (query_heads / key_heads);
    threadgroup float* scores = shared +
        simd * (kCelegTiledQueriesPerSimdgroup * 8);
    threadgroup float* numerator = shared + kCelegTiledScoreFloats +
        simd * (kCelegTiledQueriesPerSimdgroup * kCelegTiledHeadDim);
    threadgroup float* maximum = shared + kCelegTiledScoreFloats +
        kCelegTiledOutputFloats + simd * kCelegTiledQueriesPerSimdgroup;
    threadgroup float* denominator = maximum + kCelegTiledRows;
    threadgroup float* correction = denominator + kCelegTiledRows;

    for (uint index = lane;
         index < kCelegTiledQueriesPerSimdgroup * kCelegTiledHeadDim;
         index += 32) {
        numerator[index] = 0.0f;
    }
    if (lane < kCelegTiledQueriesPerSimdgroup) {
        maximum[lane] = -INFINITY;
        denominator[lane] = 0.0f;
        correction[lane] = 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint last_query = min(
        rows - 1u, query_row + kCelegTiledQueriesPerSimdgroup - 1u);
    for (uint key_row = 0; key_row <= last_query; key_row += 8) {
        simdgroup_float8x8 product =
            make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint dimension = 0; dimension < head_dim; dimension += 8) {
            simdgroup_float8x8 q;
            simdgroup_float8x8 k;
            simdgroup_load(
                q,
                query + static_cast<size_t>(query_row) * query_width +
                    static_cast<size_t>(head) * head_dim + dimension,
                query_width, 0, false);
            simdgroup_load(
                k,
                key_cache + static_cast<size_t>(key_row) * key_width +
                    static_cast<size_t>(key_head) * head_dim + dimension,
                key_width, 0, true);
            simdgroup_multiply_accumulate(product, q, k, product);
        }
        simdgroup_store(product, scores, 8, 0, false);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < kCelegTiledQueriesPerSimdgroup) {
            const uint global_query = query_row + lane;
            float block_maximum = -INFINITY;
            for (uint column = 0; column < 8; ++column) {
                const uint global_key = key_row + column;
                if (global_key < rows && global_key <= global_query) {
                    block_maximum = max(
                        block_maximum, scores[lane * 8 + column] * scale);
                }
            }
            const float old_maximum = maximum[lane];
            const float updated_maximum = max(old_maximum, block_maximum);
            const float rescale = old_maximum == -INFINITY
                ? 0.0f : exp(old_maximum - updated_maximum);
            float block_sum = 0.0f;
            for (uint column = 0; column < 8; ++column) {
                const uint global_key = key_row + column;
                float probability = 0.0f;
                if (global_key < rows && global_key <= global_query) {
                    probability = exp(
                        scores[lane * 8 + column] * scale - updated_maximum);
                    block_sum += probability;
                }
                scores[lane * 8 + column] = probability;
            }
            correction[lane] = rescale;
            maximum[lane] = updated_maximum;
            denominator[lane] = denominator[lane] * rescale + block_sum;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint index = lane;
             index < kCelegTiledQueriesPerSimdgroup * kCelegTiledHeadDim;
             index += 32) {
            numerator[index] *= correction[index / head_dim];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 probabilities;
        simdgroup_load(probabilities, scores, 8, 0, false);
        for (uint dimension = 0; dimension < head_dim; dimension += 8) {
            simdgroup_float8x8 values;
            simdgroup_float8x8 accumulated;
            simdgroup_load(
                values,
                value_cache + static_cast<size_t>(key_row) * key_width +
                    static_cast<size_t>(key_head) * head_dim + dimension,
                key_width, 0, false);
            simdgroup_load(accumulated, numerator + dimension, head_dim, 0, false);
            simdgroup_multiply_accumulate(
                accumulated, probabilities, values, accumulated);
            simdgroup_store(accumulated, numerator + dimension, head_dim, 0, false);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint index = lane;
         index < kCelegTiledQueriesPerSimdgroup * kCelegTiledHeadDim;
         index += 32) {
        const uint local_query = index / head_dim;
        const uint dimension = index - local_query * head_dim;
        const uint global_query = query_row + local_query;
        if (global_query < rows) {
            output[static_cast<size_t>(global_query) * query_width +
                   static_cast<size_t>(head) * head_dim + dimension] =
                numerator[index] / denominator[local_query];
        }
    }
}
