/**
 * @brief Half-KV variants for Fast decode (F16 KV cache, half BW).
 *
 * Mirrors the float KV stores but writes half to halve 512-length BW
 * (96 MB → 48 MB per token). Reuses same threading (32 th per head).
 */

kernel void celeg_qk_norm_rope_store_kv_split_half(
    device float* query [[buffer(0)]],
    device const float* query_weight [[buffer(1)]],
    device float* key [[buffer(2)]],
    device const float* key_weight [[buffer(3)]],
    device const float* value [[buffer(4)]],
    device bfloat* key_cache [[buffer(5)]],
    device bfloat* value_cache [[buffer(6)]],
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
    const float query_inverse = has_query ? rsqrt(simd_sum(query_sum) / static_cast<float>(head_dim) + query_epsilon) : 0.0f;
    const float key_inverse = has_key ? rsqrt(simd_sum(key_sum) / static_cast<float>(head_dim) + key_epsilon) : 0.0f;
    if (has_query) {
        const size_t base = static_cast<size_t>(head) * head_dim;
        for (uint pair = lane; pair < pairs; pair += 32u) {
            const float frequency = pow(theta, -2.0f * static_cast<float>(pair) / static_cast<float>(head_dim));
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
            const float frequency = pow(theta, -2.0f * static_cast<float>(pair) / static_cast<float>(head_dim));
            const float angle = static_cast<float>(position) * frequency;
            const float c = cos(angle);
            const float s = sin(angle);
            const size_t first = base + pair;
            const size_t second = base + pairs + pair;
            const float x = key[first] * (key_inverse * key_weight[pair]);
            const float y = key[second] * (key_inverse * key_weight[pairs + pair]);
            const float kx = x * c - y * s;
            const float ky = y * c + x * s;
            key[first] = kx;
            key[second] = ky;
        }
        const size_t cache_base = static_cast<size_t>(position) * static_cast<size_t>(key_heads) * head_dim + base;
        for (uint d = lane; d < head_dim; d += 32u) {
            key_cache[cache_base + d] = static_cast<bfloat>(key[base + d]);
            value_cache[cache_base + d] = static_cast<bfloat>(value[base + d]);
        }
    }
}

kernel void celeg_qk_norm_rope_batch_split_store_kv_half(
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
    device bfloat* key_cache [[buffer(14)]],
    device bfloat* value_cache [[buffer(15)]],
    uint index [[thread_position_in_grid]]) {
    const uint head_count = max(query_heads, key_heads);
    const uint token = index / head_count;
    const uint head = index % head_count;
    if (token >= rows) return;
    const uint position = base_position + token;
    if (head < query_heads) {
        const size_t base = static_cast<size_t>(token) * query_heads * head_dim + static_cast<size_t>(head) * head_dim;
        const uint pairs = head_dim / 2u;
        float sum = 0.0f;
        for (uint pair = 0; pair < pairs; ++pair) {
            const float x = query[base + pair];
            const float y = query[base + pairs + pair];
            sum += x * x + y * y;
        }
        const float inv = rsqrt(sum / static_cast<float>(head_dim) + query_epsilon);
        for (uint pair = 0; pair < pairs; ++pair) {
            const float freq = pow(theta, -2.0f * static_cast<float>(pair) / static_cast<float>(head_dim));
            const float ang = static_cast<float>(position) * freq;
            const float c = cos(ang);
            const float s = sin(ang);
            const size_t first = base + pair;
            const size_t second = base + pairs + pair;
            const float x = query[first] * (inv * query_weight[pair]);
            const float y = query[second] * (inv * query_weight[pairs + pair]);
            query[first] = (x * c - y * s) * query_scale;
            query[second] = (y * c + x * s) * query_scale;
        }
    }
    if (head < key_heads) {
        const size_t source_base = static_cast<size_t>(token) * key_heads * head_dim + static_cast<size_t>(head) * head_dim;
        const uint pairs = head_dim / 2u;
        float sum = 0.0f;
        for (uint pair = 0; pair < pairs; ++pair) {
            const float x = key[source_base + pair];
            const float y = key[source_base + pairs + pair];
            sum += x * x + y * y;
        }
        const float inv = rsqrt(sum / static_cast<float>(head_dim) + key_epsilon);
        for (uint pair = 0; pair < pairs; ++pair) {
            const float freq = pow(theta, -2.0f * static_cast<float>(pair) / static_cast<float>(head_dim));
            const float ang = static_cast<float>(position) * freq;
            const float c = cos(ang);
            const float s = sin(ang);
            const size_t first = source_base + pair;
            const size_t second = source_base + pairs + pair;
            const float x = key[first] * (inv * key_weight[pair]);
            const float y = key[second] * (inv * key_weight[pairs + pair]);
            key[first] = x * c - y * s;
            key[second] = y * c + x * s;
        }
        const size_t cache_base = static_cast<size_t>(position) * key_heads * head_dim + static_cast<size_t>(head) * head_dim;
        for (uint d = 0; d < head_dim; ++d) {
            key_cache[cache_base + d] = static_cast<bfloat>(key[source_base + d]);
            value_cache[cache_base + d] = static_cast<bfloat>(value[source_base + d]);
        }
    }
}
