kernel void celeg_head_rmsnorm_inplace(
    device float* data [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    constant uint& heads [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    uint head [[thread_position_in_grid]]) {
    if (head >= heads) return;
    const size_t base = static_cast<size_t>(head) * head_dim;
    float sum = 0.0f;
    for (uint d = 0; d < head_dim; ++d) sum += data[base + d] * data[base + d];
    const float inverse = rsqrt(sum / static_cast<float>(head_dim) + epsilon);
    for (uint d = 0; d < head_dim; ++d) data[base + d] *= inverse * weight[d];
}

kernel void celeg_head_rmsnorm_batch_inplace(
    device float* data [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    const uint row = index / heads;
    const uint head = index % heads;
    if (row >= rows) return;
    const size_t base = (static_cast<size_t>(row) * heads + head) * head_dim;
    float sum = 0.0f;
    for (uint d = 0; d < head_dim; ++d) sum += data[base + d] * data[base + d];
    const float inverse = rsqrt(sum / static_cast<float>(head_dim) + epsilon);
    for (uint d = 0; d < head_dim; ++d) data[base + d] *= inverse * weight[d];
}
