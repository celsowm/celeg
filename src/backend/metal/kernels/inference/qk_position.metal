inline uint celeg_standard_rope_pairing_mode(uint position_mode) {
    return position_mode & 3u;
}

inline uint celeg_standard_rope_rotary_dim(uint position_mode, uint head_dim) {
    const uint encoded = position_mode >> 2u;
    if (encoded == 0u) return head_dim;
    return min(encoded - 1u, head_dim) & ~1u;
}

struct CelegRopeScalingSpec {
    uint mode;
    uint original_context;
    float rotary_fraction;
    float factor;
    float beta_fast;
    float beta_slow;
    float attention_factor;
    float low_frequency_factor;
    float high_frequency_factor;
};

inline void celeg_apply_rope_head(
    device float* values,
    size_t base,
    uint head_dim,
    uint position,
    uint position_mode,
    float theta,
    float scale) {
    const uint pairing_mode = celeg_standard_rope_pairing_mode(position_mode);
    const uint rotary_dim = celeg_standard_rope_rotary_dim(position_mode, head_dim);
    const uint pairs = rotary_dim / 2u;
    if (pairing_mode == 1u) {
        for (uint pair = 0; pair < pairs; ++pair) {
            const float frequency =
                celeg_rope_unscaled_frequency(theta, pair, rotary_dim);
            const float angle = static_cast<float>(position) * frequency;
            const float c = cos(angle);
            const float s = sin(angle);
            const size_t first = base + pair;
            const size_t second = base + pairs + pair;
            const float x = values[first];
            const float y = values[second];
            values[first] = x * c - y * s;
            values[second] = y * c + x * s;
        }
    } else if (pairing_mode == 2u) {
        for (uint pair = 0; pair < pairs; ++pair) {
            const float frequency =
                celeg_rope_unscaled_frequency(theta, pair, rotary_dim);
            const float angle = static_cast<float>(position) * frequency;
            const float c = cos(angle);
            const float s = sin(angle);
            const size_t first = base + 2u * pair;
            const size_t second = first + 1u;
            const float x = values[first];
            const float y = values[second];
            values[first] = x * c - y * s;
            values[second] = y * c + x * s;
        }
    }
    if (scale != 1.0f) {
        for (uint d = 0; d < head_dim; ++d) values[base + d] *= scale;
    }
}

inline void celeg_apply_rope_head_scaled(
    device float* values,
    size_t base,
    uint head_dim,
    uint position,
    uint position_mode,
    float theta,
    constant CelegRopeScalingSpec& scaling,
    device const float* short_factors,
    device const float* long_factors,
    float scale) {
    const uint pairing_mode = celeg_standard_rope_pairing_mode(position_mode);
    const uint rotary_dim = celeg_standard_rope_rotary_dim(position_mode, head_dim);
    const uint pairs = rotary_dim / 2u;
    for (uint pair = 0; pair < pairs; ++pair) {
        const CelegRopePairComponents components =
            celeg_rope_pair_components(pair, pairs, pairing_mode);
        const float frequency = celeg_rope_scaled_frequency(
            theta,
            scaling.rotary_fraction,
            pair,
            rotary_dim,
            position,
            scaling.mode,
            scaling.factor,
            scaling.beta_fast,
            scaling.beta_slow,
            scaling.original_context,
            scaling.low_frequency_factor,
            scaling.high_frequency_factor,
            short_factors,
            long_factors);
        const float angle = static_cast<float>(position) * frequency;
        const float c = cos(angle);
        const float s = sin(angle);
        const size_t first = base + components.first;
        const size_t second = base + components.second;
        const float x = values[first];
        const float y = values[second];
        values[first] = x * c - y * s;
        values[second] = y * c + x * s;
    }
    if (scale != 1.0f) {
        for (uint d = 0; d < head_dim; ++d) values[base + d] *= scale;
    }
}

inline void celeg_apply_mrope_head(
    device float* values,
    size_t base,
    uint head_dim,
    constant int* rope_position,
    uint section0,
    uint section1,
    uint interleaved,
    float theta,
    float scale) {
    const uint pairs = head_dim / 2;
    for (uint pair = 0; pair < pairs; ++pair) {
        const uint axis = celeg_mrope_axis_for_pair(
            pair, section0, section1, interleaved);
        const float frequency = pow(
            theta,
            -2.0f * static_cast<float>(pair) /
                static_cast<float>(head_dim));
        const float angle = static_cast<float>(rope_position[axis]) * frequency;
        const float c = cos(angle);
        const float s = sin(angle);
        const size_t first = base + pair;
        const size_t second = base + pairs + pair;
        const float x = values[first];
        const float y = values[second];
        values[first] = (x * c - y * s) * scale;
        values[second] = (y * c + x * s) * scale;
    }
}

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
    if (head < query_heads) {
        const size_t base = static_cast<size_t>(head) * head_dim;
        celeg_apply_rope_head(
            query, base, head_dim, cache_position, position_mode, theta, query_scale);
    }
    if (head < key_heads) {
        const size_t base = static_cast<size_t>(head) * head_dim;
        celeg_apply_rope_head(
            key, base, head_dim, cache_position, position_mode, theta, 1.0f);
        const size_t cache_base = static_cast<size_t>(cache_position) *
            static_cast<size_t>(key_heads) * head_dim + base;
        for (uint d = 0; d < head_dim; ++d) {
            key_cache[cache_base + d] = key[base + d];
            value_cache[cache_base + d] = value[base + d];
        }
    }
}

kernel void celeg_qk_position_store_kv_scaled(
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
    constant CelegRopeScalingSpec& scaling [[buffer(13)]],
    device const float* short_factors [[buffer(14)]],
    device const float* long_factors [[buffer(15)]],
    uint head [[thread_position_in_grid]]) {
    if (head < query_heads) {
        const size_t base = static_cast<size_t>(head) * head_dim;
        celeg_apply_rope_head_scaled(
            query, base, head_dim, cache_position, position_mode, theta,
            scaling, short_factors, long_factors, query_scale);
    }
    if (head < key_heads) {
        const size_t base = static_cast<size_t>(head) * head_dim;
        celeg_apply_rope_head_scaled(
            key, base, head_dim, cache_position, position_mode, theta,
            scaling, short_factors, long_factors, 1.0f);
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
    constant uint& interleaved [[buffer(14)]],
    uint head [[thread_position_in_grid]]) {
    const uint pairs = head_dim / 2;
    if (sections[0] + sections[1] + sections[2] != pairs) return;
    if (head < query_heads) {
        const size_t base = static_cast<size_t>(head) * head_dim;
        celeg_apply_mrope_head(
            query, base, head_dim, rope_position,
            sections[0], sections[1], interleaved, theta, query_scale);
    }
    if (head < key_heads) {
        const size_t base = static_cast<size_t>(head) * head_dim;
        celeg_apply_mrope_head(
            key, base, head_dim, rope_position,
            sections[0], sections[1], interleaved, theta, 1.0f);
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
    const uint position = base_position + token;
    if (head < query_heads) {
        const size_t base = static_cast<size_t>(token) * query_heads * head_dim +
            static_cast<size_t>(head) * head_dim;
        celeg_apply_rope_head(
            query, base, head_dim, position, position_mode, theta, query_scale);
    }
    if (head < key_heads && celeg_standard_rope_pairing_mode(position_mode) != 0u) {
        const size_t base = static_cast<size_t>(token) * key_heads * head_dim +
            static_cast<size_t>(head) * head_dim;
        celeg_apply_rope_head(
            key, base, head_dim, position, position_mode, theta, 1.0f);
    }
}

kernel void celeg_qk_position_batch_scaled(
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
    constant CelegRopeScalingSpec& scaling [[buffer(10)]],
    device const float* short_factors [[buffer(11)]],
    device const float* long_factors [[buffer(12)]],
    uint index [[thread_position_in_grid]]) {
    const uint head_count = max(query_heads, key_heads);
    const uint token = index / head_count;
    const uint head = index % head_count;
    if (token >= rows) return;
    const uint position = base_position + token;
    if (head < query_heads) {
        const size_t base = static_cast<size_t>(token) * query_heads * head_dim +
            static_cast<size_t>(head) * head_dim;
        celeg_apply_rope_head_scaled(
            query, base, head_dim, position, position_mode, theta,
            scaling, short_factors, long_factors, query_scale);
    }
    if (head < key_heads && celeg_standard_rope_pairing_mode(position_mode) != 0u) {
        const size_t base = static_cast<size_t>(token) * key_heads * head_dim +
            static_cast<size_t>(head) * head_dim;
        celeg_apply_rope_head_scaled(
            key, base, head_dim, position, position_mode, theta,
            scaling, short_factors, long_factors, 1.0f);
    }
}
