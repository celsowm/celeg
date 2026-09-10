/**
 * @brief Q/K norm, split-half RoPE, and KV store with one thread per pair.
 *
 * The legacy mapping used one thread per head and serialized the norm, the
 * per-pair transcendentals, and the stores. Here one 32-thread threadgroup
 * serves one head and each lane owns a strided pair subset: phase one
 * reduces the per-head norm denominators with a single-simdgroup
 * `simd_sum` (no threadgroup exchange needed — the inverse stays in a
 * register), phase two scales, rotates, and stores its own pairs. The
 * per-element operation order matches the legacy kernel exactly, so
 * everything downstream of the inverse agrees bitwise; only the norm
 * summation order differs (tolerance-level drift). Launch with 32 threads
 * per threadgroup and `max(query_heads, key_heads)` threadgroups.
 */
kernel void celeg_qk_norm_rope_store_kv_split(
    device float* query [[buffer(0)]],
    device const float* query_weight [[buffer(1)]],
    device float* key [[buffer(2)]],
    device const float* key_weight [[buffer(3)]],
    device const float* value [[buffer(4)]],
    device float* key_cache [[buffer(5)]],
    device float* value_cache [[buffer(6)]],
    constant uint& query_heads [[buffer(7)]],
    constant uint& key_heads [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant uint& position [[buffer(10)]],
    constant float& theta [[buffer(11)]],
    constant float& query_scale [[buffer(12)]],
    constant float& query_epsilon [[buffer(13)]],
    constant float& key_epsilon [[buffer(14)]],
    constant uint& page_tokens [[buffer(15)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint pairs = head_dim / 2u;
    const uint head = group;
    const bool has_query = head < query_heads;
    const bool has_key = head < key_heads;
    if (!has_query && !has_key) return;
    float query_sum = 0.0f;
    float key_sum = 0.0f;
    if (has_query) {
        const size_t base = static_cast<size_t>(head) * head_dim;
        for (uint pair = lane; pair < pairs; pair += 32u) {
            const float x = query[base + pair];
            const float y = query[base + pairs + pair];
            query_sum += x * x + y * y;
        }
        if (2u * pairs < head_dim && lane == 0) {
            const float tail = query[base + head_dim - 1u];
            query_sum += tail * tail;
        }
    }
    if (has_key) {
        const size_t base = static_cast<size_t>(head) * head_dim;
        for (uint pair = lane; pair < pairs; pair += 32u) {
            const float x = key[base + pair];
            const float y = key[base + pairs + pair];
            key_sum += x * x + y * y;
        }
        if (2u * pairs < head_dim && lane == 0) {
            const float tail = key[base + head_dim - 1u];
            key_sum += tail * tail;
        }
    }
    const float query_inverse = has_query
        ? rsqrt(simd_sum(query_sum) / static_cast<float>(head_dim) + query_epsilon)
        : 0.0f;
    const float key_inverse = has_key
        ? rsqrt(simd_sum(key_sum) / static_cast<float>(head_dim) + key_epsilon)
        : 0.0f;
    if (has_query) {
        const size_t base = static_cast<size_t>(head) * head_dim;
        for (uint pair = lane; pair < pairs; pair += 32u) {
            const float frequency = pow(theta, -2.0f * static_cast<float>(pair) /
                                               static_cast<float>(head_dim));
            const float angle = static_cast<float>(position) * frequency;
            const float c = cos(angle);
            const float s = sin(angle);
            const size_t first = base + pair;
            const size_t second = base + pairs + pair;
            const float x = query[first] * (query_inverse * query_weight[pair]);
            const float y = query[second] * (query_inverse * query_weight[pairs + pair]);
            query[first] = (x * c - y * s) * query_scale;
            query[second] = (y * c + x * s) * query_scale;
        }
    }
    if (has_key) {
        const size_t base = static_cast<size_t>(head) * head_dim;
        for (uint pair = lane; pair < pairs; pair += 32u) {
            const float frequency = pow(theta, -2.0f * static_cast<float>(pair) /
                                               static_cast<float>(head_dim));
            const float angle = static_cast<float>(position) * frequency;
            const float c = cos(angle);
            const float s = sin(angle);
            const size_t first = base + pair;
            const size_t second = base + pairs + pair;
            const float x = key[first] * (key_inverse * key_weight[pair]);
            const float y = key[second] * (key_inverse * key_weight[pairs + pair]);
            key[first] = x * c - y * s;
            key[second] = y * c + x * s;
        }
        const size_t cache_base = static_cast<size_t>(position) *
            static_cast<size_t>(key_heads) * head_dim + base;
        for (uint pair = lane; pair < pairs; pair += 32u) {
            key_cache[cache_base + pair] = key[base + pair];
            key_cache[cache_base + pairs + pair] = key[base + pairs + pair];
            value_cache[cache_base + pair] = value[base + pair];
            value_cache[cache_base + pairs + pair] = value[base + pairs + pair];
        }
    }
}

inline void celeg_qk_norm_rope_batch_split_head(
    device float* data,
    device const float* weight,
    size_t base,
    uint head_dim,
    uint position,
    float theta,
    float epsilon,
    float output_scale) {
    const uint pairs = head_dim / 2;
    float sum = 0.0f;
    for (uint d = 0; d < head_dim; ++d) sum += data[base + d] * data[base + d];
    const float inverse = rsqrt(sum / static_cast<float>(head_dim) + epsilon);
    for (uint pair = 0; pair < pairs; ++pair) {
        const float frequency = pow(theta, -2.0f * static_cast<float>(pair) /
                                          static_cast<float>(head_dim));
        const float angle = static_cast<float>(position) * frequency;
        const float c = cos(angle);
        const float s = sin(angle);
        const size_t first = base + pair;
        const size_t second = base + pairs + pair;
        const float x = data[first] * (inverse * weight[pair]);
        const float y = data[second] * (inverse * weight[pairs + pair]);
        data[first] = (x * c - y * s) * output_scale;
        data[second] = (y * c + x * s) * output_scale;
    }
}

kernel void celeg_qk_norm_rope_batch_split(
    device float* query [[buffer(0)]],
    device const float* query_weight [[buffer(1)]],
    device float* key [[buffer(2)]],
    device const float* key_weight [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& query_heads [[buffer(5)]],
    constant uint& key_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& base_position [[buffer(8)]],
    constant float& theta [[buffer(9)]],
    constant float& query_scale [[buffer(10)]],
    constant float& query_epsilon [[buffer(11)]],
    constant float& key_epsilon [[buffer(12)]],
    uint index [[thread_position_in_grid]]) {
    const uint head_count = max(query_heads, key_heads);
    const uint token = index / head_count;
    const uint head = index % head_count;
    if (token >= rows) return;
    const uint position = base_position + token;
    if (head < query_heads) {
        const size_t base = static_cast<size_t>(token) * query_heads * head_dim +
            static_cast<size_t>(head) * head_dim;
        celeg_qk_norm_rope_batch_split_head(
            query, query_weight, base, head_dim, position, theta,
            query_epsilon, query_scale);
    }
    if (head < key_heads) {
        const size_t base = static_cast<size_t>(token) * key_heads * head_dim +
            static_cast<size_t>(head) * head_dim;
        celeg_qk_norm_rope_batch_split_head(
            key, key_weight, base, head_dim, position, theta,
            key_epsilon, 1.0f);
    }
}

kernel void celeg_qk_norm_rope_batch_split_store_kv(
    device float* query [[buffer(0)]],
    device const float* query_weight [[buffer(1)]],
    device float* key [[buffer(2)]],
    device const float* key_weight [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& query_heads [[buffer(5)]],
    constant uint& key_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& base_position [[buffer(8)]],
    constant float& theta [[buffer(9)]],
    constant float& query_scale [[buffer(10)]],
    constant float& query_epsilon [[buffer(11)]],
    constant float& key_epsilon [[buffer(12)]],
    device const float* value [[buffer(13)]],
    device float* key_cache [[buffer(14)]],
    device float* value_cache [[buffer(15)]],
    uint index [[thread_position_in_grid]]) {
    const uint head_count = max(query_heads, key_heads);
    const uint token = index / head_count;
    const uint head = index % head_count;
    if (token >= rows) return;
    const uint position = base_position + token;
    if (head < query_heads) {
        const size_t base = static_cast<size_t>(token) * query_heads * head_dim +
            static_cast<size_t>(head) * head_dim;
        celeg_qk_norm_rope_batch_split_head(
            query, query_weight, base, head_dim, position, theta,
            query_epsilon, query_scale);
    }
    if (head < key_heads) {
        const size_t source_base = static_cast<size_t>(token) * key_heads * head_dim +
            static_cast<size_t>(head) * head_dim;
        celeg_qk_norm_rope_batch_split_head(
            key, key_weight, source_base, head_dim, position, theta,
            key_epsilon, 1.0f);
        const size_t cache_base = static_cast<size_t>(position) * key_heads * head_dim +
            static_cast<size_t>(head) * head_dim;
        for (uint d = 0; d < head_dim; ++d) {
            key_cache[cache_base + d] = key[source_base + d];
            value_cache[cache_base + d] = value[source_base + d];
        }
    }
}

kernel void celeg_qk_norm_mrope_store_kv(
    device float* query [[buffer(0)]],
    device const float* query_weight [[buffer(1)]],
    device float* key [[buffer(2)]],
    device const float* key_weight [[buffer(3)]],
    device const float* value [[buffer(4)]],
    device float* key_cache [[buffer(5)]],
    device float* value_cache [[buffer(6)]],
    constant uint& query_heads [[buffer(7)]],
    constant uint& key_heads [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant uint& cache_position [[buffer(10)]],
    constant int* rope_position [[buffer(11)]],
    constant uint* sections [[buffer(12)]],
    constant float& theta [[buffer(13)]],
    constant float& query_scale [[buffer(14)]],
    constant float& query_epsilon [[buffer(15)]],
    constant float& key_epsilon [[buffer(16)]],
    constant uint& page_tokens [[buffer(17)]],
    constant uint& interleaved [[buffer(18)]],
    uint head [[thread_position_in_grid]]) {
    const uint pairs = head_dim / 2;
    if (sections[0] + sections[1] + sections[2] != pairs) return;
    if (head < query_heads) {
        const size_t base = static_cast<size_t>(head) * head_dim;
        float sum = 0.0f;
        for (uint d = 0; d < head_dim; ++d) sum += query[base + d] * query[base + d];
        const float inverse = rsqrt(sum / static_cast<float>(head_dim) + query_epsilon);
        for (uint d = 0; d < head_dim; ++d) query[base + d] *= inverse * query_weight[d];
        for (uint pair = 0; pair < pairs; ++pair) {
            const uint axis = celeg_mrope_axis_for_pair(
                pair, sections[0], sections[1], interleaved);
            const float frequency = pow(theta, -2.0f * static_cast<float>(pair) /
                                              static_cast<float>(head_dim));
            const float angle = static_cast<float>(rope_position[axis]) * frequency;
            const float c = cos(angle);
            const float s = sin(angle);
            const size_t first = base + pair;
            const size_t second = base + pairs + pair;
            const float x = query[first];
            const float y = query[second];
            query[first] = (x * c - y * s) * query_scale;
            query[second] = (y * c + x * s) * query_scale;
        }
    }
    if (head < key_heads) {
        const size_t base = static_cast<size_t>(head) * head_dim;
        float sum = 0.0f;
        for (uint d = 0; d < head_dim; ++d) sum += key[base + d] * key[base + d];
        const float inverse = rsqrt(sum / static_cast<float>(head_dim) + key_epsilon);
        for (uint d = 0; d < head_dim; ++d) key[base + d] *= inverse * key_weight[d];
        for (uint pair = 0; pair < pairs; ++pair) {
            const uint axis = celeg_mrope_axis_for_pair(
                pair, sections[0], sections[1], interleaved);
            const float frequency = pow(theta, -2.0f * static_cast<float>(pair) /
                                              static_cast<float>(head_dim));
            const float angle = static_cast<float>(rope_position[axis]) * frequency;
            const float c = cos(angle);
            const float s = sin(angle);
            const size_t first = base + pair;
            const size_t second = base + pairs + pair;
            const float x = key[first];
            const float y = key[second];
            key[first] = x * c - y * s;
            key[second] = y * c + x * s;
        }
        const size_t cache_base = static_cast<size_t>(cache_position) *
            static_cast<size_t>(key_heads) * head_dim + base;
        for (uint d = 0; d < head_dim; ++d) {
            key_cache[cache_base + d] = key[base + d];
            value_cache[cache_base + d] = value[base + d];
        }
    }
}
