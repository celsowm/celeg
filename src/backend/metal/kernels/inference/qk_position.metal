kernel void celeg_qk_position_store_kv(
    device float* query [[buffer(0)]],
    device float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* key_cache [[buffer(3)]],
    device float* value_cache [[buffer(4)]],
    constant uint& query_heads [[buffer(5)]],
    constant uint& key_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& cache_position [[buffer(8)]],
    constant uint& position_mode [[buffer(9)]],
    constant float& theta [[buffer(10)]],
    constant float& query_scale [[buffer(11)]],
    constant uint& page_tokens [[buffer(12)]],
    uint head [[thread_position_in_grid]]) {
    const uint pairs = head_dim / 2;
    if (head < query_heads) {
        const size_t base = static_cast<size_t>(head) * head_dim;
        if (position_mode == 1) {
            for (uint pair = 0; pair < pairs; ++pair) {
                const float frequency = pow(theta, -2.0f * static_cast<float>(pair) /
                                                  static_cast<float>(head_dim));
                const float angle = static_cast<float>(cache_position) * frequency;
                const float c = cos(angle);
                const float s = sin(angle);
                const size_t first = base + pair;
                const size_t second = base + pairs + pair;
                const float x = query[first];
                const float y = query[second];
                query[first] = x * c - y * s;
                query[second] = y * c + x * s;
            }
        } else if (position_mode == 2) {
            for (uint pair = 0; pair < pairs; ++pair) {
                const float frequency = pow(theta, -2.0f * static_cast<float>(pair) /
                                                  static_cast<float>(head_dim));
                const float angle = static_cast<float>(cache_position) * frequency;
                const float c = cos(angle);
                const float s = sin(angle);
                const size_t first = base + 2 * pair;
                const size_t second = first + 1;
                const float x = query[first];
                const float y = query[second];
                query[first] = x * c - y * s;
                query[second] = y * c + x * s;
            }
        }
        for (uint d = 0; d < head_dim; ++d) query[base + d] *= query_scale;
    }
    if (head < key_heads) {
        const size_t base = static_cast<size_t>(head) * head_dim;
        if (position_mode == 1) {
            for (uint pair = 0; pair < pairs; ++pair) {
                const float frequency = pow(theta, -2.0f * static_cast<float>(pair) /
                                                  static_cast<float>(head_dim));
                const float angle = static_cast<float>(cache_position) * frequency;
                const float c = cos(angle);
                const float s = sin(angle);
                const size_t first = base + pair;
                const size_t second = base + pairs + pair;
                const float x = key[first];
                const float y = key[second];
                key[first] = x * c - y * s;
                key[second] = y * c + x * s;
            }
        } else if (position_mode == 2) {
            for (uint pair = 0; pair < pairs; ++pair) {
                const float frequency = pow(theta, -2.0f * static_cast<float>(pair) /
                                                  static_cast<float>(head_dim));
                const float angle = static_cast<float>(cache_position) * frequency;
                const float c = cos(angle);
                const float s = sin(angle);
                const size_t first = base + 2 * pair;
                const size_t second = first + 1;
                const float x = key[first];
                const float y = key[second];
                key[first] = x * c - y * s;
                key[second] = y * c + x * s;
            }
        }
        const size_t cache_base = static_cast<size_t>(cache_position) *
            static_cast<size_t>(key_heads) * head_dim + base;
        for (uint d = 0; d < head_dim; ++d) {
            key_cache[cache_base + d] = key[base + d];
            value_cache[cache_base + d] = value[base + d];
        }
    }
}

kernel void celeg_qk_mrope_position_store_kv(
    device float* query [[buffer(0)]],
    device float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* key_cache [[buffer(3)]],
    device float* value_cache [[buffer(4)]],
    constant uint& query_heads [[buffer(5)]],
    constant uint& key_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& cache_position [[buffer(8)]],
    constant int* rope_position [[buffer(9)]],
    constant uint* sections [[buffer(10)]],
    constant float& theta [[buffer(11)]],
    constant float& query_scale [[buffer(12)]],
    constant uint& page_tokens [[buffer(13)]],
    uint head [[thread_position_in_grid]]) {
    const uint pairs = head_dim / 2;
    if (sections[0] + sections[1] + sections[2] != pairs) return;
    if (head < query_heads) {
        const size_t base = static_cast<size_t>(head) * head_dim;
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
        const size_t cache_base = static_cast<size_t>(cache_position) *
            static_cast<size_t>(key_heads) * head_dim + base;
        for (uint d = 0; d < head_dim; ++d) {
            key_cache[cache_base + d] = key[base + d];
            value_cache[cache_base + d] = value[base + d];
        }
    }
}

kernel void celeg_qk_position_batch(
    device float* query [[buffer(0)]],
    device float* key [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& query_heads [[buffer(3)]],
    constant uint& key_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& base_position [[buffer(6)]],
    constant uint& position_mode [[buffer(7)]],
    constant float& theta [[buffer(8)]],
    constant float& query_scale [[buffer(9)]],
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
        if (position_mode == 1) {
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
                query[first] = x * c - y * s;
                query[second] = y * c + x * s;
            }
        } else if (position_mode == 2) {
            for (uint pair = 0; pair < pairs; ++pair) {
                const float frequency = pow(theta, -2.0f * static_cast<float>(pair) /
                                                  static_cast<float>(head_dim));
                const float angle = static_cast<float>(position) * frequency;
                const float c = cos(angle);
                const float s = sin(angle);
                const size_t first = base + 2 * pair;
                const size_t second = first + 1;
                const float x = query[first];
                const float y = query[second];
                query[first] = x * c - y * s;
                query[second] = y * c + x * s;
            }
        }
        for (uint d = 0; d < head_dim; ++d) query[base + d] *= query_scale;
    }
    if (head < key_heads && position_mode != 0) {
        const size_t base = static_cast<size_t>(token) * key_heads * head_dim +
            static_cast<size_t>(head) * head_dim;
        if (position_mode == 1) {
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
        } else {
            for (uint pair = 0; pair < pairs; ++pair) {
                const float frequency = pow(theta, -2.0f * static_cast<float>(pair) /
                                                  static_cast<float>(head_dim));
                const float angle = static_cast<float>(position) * frequency;
                const float c = cos(angle);
                const float s = sin(angle);
                const size_t first = base + 2 * pair;
                const size_t second = first + 1;
                const float x = key[first];
                const float y = key[second];
                key[first] = x * c - y * s;
                key[second] = y * c + x * s;
            }
        }
    }
}
