kernel void celeg_store_kv_batch_2d(
    device const float* key [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* key_cache [[buffer(2)]],
    device float* value_cache [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& base_position [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    uint lane [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    const uint column = group.x * 256u + lane;
    const uint row = group.y;
    if (row >= rows || column >= width) return;
    const uint position = base_position + row;
    const size_t source = static_cast<size_t>(row) * width + column;
    const size_t destination = static_cast<size_t>(position) * width + column;
    key_cache[destination] = key[source];
    value_cache[destination] = value[source];
}
