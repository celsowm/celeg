// Benchmark-only fast-mode attention experiment for LFM2.5 geometry.
// It materializes the score/probability matrix in float and uses
// simdgroup_float8x8 for both Q*K^T and P*V. The goal is to isolate the
// throughput available from simdgroup matrix hardware before implementing a
// fully blocked flash-attention kernel. Numerical equivalence is reported but
// not required because this is a fast-mode candidate.

constant uint kCelegMaterializedQueriesPerThreadgroup = 32;
constant uint kCelegMaterializedRowsPerSimdgroup = 8;
constant uint kCelegMaterializedSimdgroups = 4;

kernel void celeg_attention_materialized_scores(
    device const float* query [[buffer(0)]],
    device const float* key_cache [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& query_heads [[buffer(4)]],
    constant uint& key_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint3 grid [[threadgroup_position_in_grid]]) {
    const uint head = grid.x;
    const uint query_row =
        grid.y * kCelegMaterializedQueriesPerThreadgroup +
        simd * kCelegMaterializedRowsPerSimdgroup;
    const uint key_row = grid.z * 8u;
    if (head >= query_heads || query_row >= rows || key_row >= rows) return;

    const uint query_width = query_heads * head_dim;
    const uint key_width = key_heads * head_dim;
    const uint key_head = head / (query_heads / key_heads);

    simdgroup_float8x8 product =
        make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint dimension = 0; dimension < head_dim; dimension += 8u) {
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

    device float* destination =
        scores + (static_cast<size_t>(head) * rows + query_row) * rows + key_row;
    simdgroup_store(product, destination, rows, 0, false);
}

kernel void celeg_attention_materialized_softmax(
    device float* scores [[buffer(0)]],
    constant uint& rows [[buffer(1)]],
    constant uint& query_heads [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint2 grid [[threadgroup_position_in_grid]]) {
    const uint head = grid.x;
    const uint row = grid.y * kCelegMaterializedSimdgroups + simd;
    if (head >= query_heads || row >= rows) return;

    device float* row_scores =
        scores + (static_cast<size_t>(head) * rows + row) * rows;

    float local_maximum = -INFINITY;
    for (uint key = lane; key <= row; key += 32u) {
        local_maximum = max(local_maximum, row_scores[key] * scale);
    }
    const float maximum = simd_max(local_maximum);

    float local_sum = 0.0f;
    for (uint key = lane; key < rows; key += 32u) {
        float probability = 0.0f;
        if (key <= row) {
            probability = exp(row_scores[key] * scale - maximum);
            local_sum += probability;
        }
        row_scores[key] = probability;
    }
    const float denominator = simd_sum(local_sum);

    for (uint key = lane; key <= row; key += 32u) {
        row_scores[key] /= denominator;
    }
}

kernel void celeg_attention_materialized_values(
    device const float* probabilities [[buffer(0)]],
    device const float* value_cache [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& query_heads [[buffer(4)]],
    constant uint& key_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint3 grid [[threadgroup_position_in_grid]]) {
    const uint head = grid.x;
    const uint query_row =
        grid.y * kCelegMaterializedQueriesPerThreadgroup +
        simd * kCelegMaterializedRowsPerSimdgroup;
    const uint output_dimension = grid.z * 8u;
    if (head >= query_heads || query_row >= rows || output_dimension >= head_dim) return;

    const uint query_width = query_heads * head_dim;
    const uint value_width = key_heads * head_dim;
    const uint key_head = head / (query_heads / key_heads);

    simdgroup_float8x8 product =
        make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint key_row = 0; key_row < rows; key_row += 8u) {
        simdgroup_float8x8 p;
        simdgroup_float8x8 v;
        simdgroup_load(
            p,
            probabilities +
                (static_cast<size_t>(head) * rows + query_row) * rows + key_row,
            rows, 0, false);
        simdgroup_load(
            v,
            value_cache + static_cast<size_t>(key_row) * value_width +
                static_cast<size_t>(key_head) * head_dim + output_dimension,
            value_width, 0, false);
        simdgroup_multiply_accumulate(product, p, v, product);
    }

    device float* destination =
        output + static_cast<size_t>(query_row) * query_width +
        static_cast<size_t>(head) * head_dim + output_dimension;
    simdgroup_store(product, destination, query_width, 0, false);
}
