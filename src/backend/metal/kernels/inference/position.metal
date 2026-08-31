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
    uint head [[thread_position_in_grid]]) {
    const uint pairs = head_dim / 2;
    if (head < query_heads) {
        const size_t base = static_cast<size_t>(head) * head_dim;
        float sum = 0.0f;
        for (uint d = 0; d < head_dim; ++d) sum += query[base + d] * query[base + d];
        const float inverse = rsqrt(sum / static_cast<float>(head_dim) + query_epsilon);
        for (uint d = 0; d < head_dim; ++d) query[base + d] *= inverse * query_weight[d];
        for (uint pair = 0; pair < pairs; ++pair) {
            const float frequency = pow(theta, -2.0f * static_cast<float>(pair) /
                                              static_cast<float>(head_dim));
            const float angle = static_cast<float>(position) * frequency;
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
            const float frequency = pow(theta, -2.0f * static_cast<float>(pair) /
                                              static_cast<float>(head_dim));
            const float angle = static_cast<float>(position) * frequency;
            const float c = cos(angle);
            const float s = sin(angle);
            const size_t first = base + pair;
            const size_t second = base + pairs + pair;
            const float x = key[first];
            const float y = key[second];
            key[first] = x * c - y * s;
            key[second] = y * c + x * s;
        }
        const size_t cache_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
            position % page_tokens) * static_cast<size_t>(key_heads) * head_dim + base;
        for (uint d = 0; d < head_dim; ++d) {
            key_cache[cache_base + d] = key[base + d];
            value_cache[cache_base + d] = value[base + d];
        }
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
    const uint pairs = head_dim / 2;
    const uint position = base_position + token;
    if (head < query_heads) {
        const size_t base = static_cast<size_t>(token) * query_heads * head_dim +
            static_cast<size_t>(head) * head_dim;
        float sum = 0.0f;
        for (uint d = 0; d < head_dim; ++d) sum += query[base + d] * query[base + d];
        const float inverse = rsqrt(sum / static_cast<float>(head_dim) + query_epsilon);
        for (uint d = 0; d < head_dim; ++d) query[base + d] *= inverse * query_weight[d];
        for (uint pair = 0; pair < pairs; ++pair) {
            const float frequency = pow(theta, -2.0f * static_cast<float>(pair) /
                                              static_cast<float>(head_dim));
            const float angle = static_cast<float>(position) * frequency;
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
        const size_t base = static_cast<size_t>(token) * key_heads * head_dim +
            static_cast<size_t>(head) * head_dim;
        float sum = 0.0f;
        for (uint d = 0; d < head_dim; ++d) sum += key[base + d] * key[base + d];
        const float inverse = rsqrt(sum / static_cast<float>(head_dim) + key_epsilon);
        for (uint d = 0; d < head_dim; ++d) key[base + d] *= inverse * key_weight[d];
        for (uint pair = 0; pair < pairs; ++pair) {
            const float frequency = pow(theta, -2.0f * static_cast<float>(pair) /
                                              static_cast<float>(head_dim));
            const float angle = static_cast<float>(position) * frequency;
            const float c = cos(angle);
            const float s = sin(angle);
            const size_t first = base + pair;
            const size_t second = base + pairs + pair;
            const float x = key[first];
            const float y = key[second];
            key[first] = x * c - y * s;
            key[second] = y * c + x * s;
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
            const uint axis = pair % 3;
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
            const uint axis = pair % 3;
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
        const size_t cache_base = (static_cast<size_t>(cache_position / page_tokens) * page_tokens +
            cache_position % page_tokens) * static_cast<size_t>(key_heads) * head_dim + base;
        for (uint d = 0; d < head_dim; ++d) {
            key_cache[cache_base + d] = key[base + d];
            value_cache[cache_base + d] = value[base + d];
        }
    }
}
