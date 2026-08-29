#include <metal_stdlib>

using namespace metal;

float celeg_half_to_float(ushort bits);

kernel void celeg_embedding(device const float* table [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant uint& width [[buffer(2)]],
                            constant uint& token [[buffer(3)]],
                            uint index [[thread_position_in_grid]]) {
    if (index < width) output[index] = table[static_cast<size_t>(token) * width + index];
}

kernel void celeg_embedding_f16(device const half* table [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                constant uint& width [[buffer(2)]],
                                constant uint& token [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
    if (index < width) output[index] = static_cast<float>(
        table[static_cast<size_t>(token) * width + index]);
}

float celeg_bf16_to_float(ushort bits) {
    return as_type<float>(static_cast<uint>(bits) << 16);
}

kernel void celeg_embedding_bf16(device const ushort* table [[buffer(0)]],
                                 device float* output [[buffer(1)]],
                                 constant uint& width [[buffer(2)]],
                                 constant uint& token [[buffer(3)]],
                                 uint index [[thread_position_in_grid]]) {
    if (index < width) output[index] = celeg_bf16_to_float(
        table[static_cast<size_t>(token) * width + index]);
}

kernel void celeg_matvec(device const float* weights [[buffer(0)]],
                         device const float* input [[buffer(1)]],
                         device float* output [[buffer(2)]],
                         constant uint& rows [[buffer(3)]],
                         constant uint& cols [[buffer(4)]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]],
                         uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * 8 + simd;
    if (row >= rows) return;
    float sum = 0.0f;
    const size_t base = static_cast<size_t>(row) * cols;
    for (uint col = lane; col < cols; col += 32) sum += weights[base + col] * input[col];
    const float reduced = simd_sum(sum);
    if (lane == 0) output[row] = reduced;
}

kernel void celeg_matvec_f16(device const half* weights [[buffer(0)]],
                             device const float* input [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& cols [[buffer(4)]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]],
                             uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * 8 + simd;
    if (row >= rows) return;
    float sum = 0.0f;
    const size_t base = static_cast<size_t>(row) * cols;
    for (uint col = lane; col < cols; col += 32) {
        sum += static_cast<float>(weights[base + col]) * input[col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[row] = reduced;
}

kernel void celeg_matvec_bf16(device const ushort* weights [[buffer(0)]],
                              device const float* input [[buffer(1)]],
                              device float* output [[buffer(2)]],
                              constant uint& rows [[buffer(3)]],
                              constant uint& cols [[buffer(4)]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * 8 + simd;
    if (row >= rows) return;
    float sum = 0.0f;
    const size_t base = static_cast<size_t>(row) * cols;
    for (uint col = lane; col < cols; col += 32) {
        sum += celeg_bf16_to_float(weights[base + col]) * input[col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[row] = reduced;
}

float celeg_tuned_matvec_value(half value) {
    return static_cast<float>(value);
}

float celeg_tuned_matvec_value(ushort value) {
    return celeg_bf16_to_float(value);
}

template <typename T>
kernel void celeg_matvec_tuned(device const T* weights [[buffer(0)]],
                               device const float* input [[buffer(1)]],
                               device float* output [[buffer(2)]],
                               constant uint& rows [[buffer(3)]],
                               constant uint& cols [[buffer(4)]],
                               threadgroup float* partial [[threadgroup(0)]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint simd [[simdgroup_index_in_threadgroup]],
                               uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * 2;
    if (row >= rows) return;
    float sums[2] = {0.0f, 0.0f};
    for (uint column = simd * 32 + lane; column < cols; column += 128) {
        sums[0] += celeg_tuned_matvec_value(weights[static_cast<size_t>(row) * cols + column]) *
            input[column];
        if (row + 1 < rows) {
            sums[1] += celeg_tuned_matvec_value(
                weights[static_cast<size_t>(row + 1) * cols + column]) * input[column];
        }
    }
    const float reduced0 = simd_sum(sums[0]);
    const float reduced1 = simd_sum(sums[1]);
    if (lane == 0) {
        partial[simd * 2] = reduced0;
        partial[simd * 2 + 1] = reduced1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0 && lane < 2) {
        float total = partial[lane];
        for (uint other = 1; other < 4; ++other) total += partial[other * 2 + lane];
        if (row + lane < rows) output[row + lane] = total;
    }
}

template [[host_name("celeg_matvec_tuned_f16")]]
kernel void celeg_matvec_tuned<half>(
        device const half*, device const float*, device float*,
        constant uint&, constant uint&, threadgroup float*, uint, uint, uint);

template [[host_name("celeg_matvec_tuned_bf16")]]
kernel void celeg_matvec_tuned<ushort>(
        device const ushort*, device const float*, device float*,
        constant uint&, constant uint&, threadgroup float*, uint, uint, uint);

float celeg_q4_0_value(device const uchar* block, uint column) {
    const uchar packed = block[2 + (column & 15)];
    const uint value = (column & 16) == 0 ? packed & 0x0f : packed >> 4;
    return static_cast<float>(value) - 8.0f;
}

kernel void celeg_matvec_q4_0(device const uchar* weights [[buffer(0)]],
                              device const float* input [[buffer(1)]],
                              device float* output [[buffer(2)]],
                              constant uint& rows [[buffer(3)]],
                              constant uint& cols [[buffer(4)]],
                              constant uint& row_bytes [[buffer(5)]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * 8 + simd;
    if (row >= rows) return;
    const device uchar* row_data = weights + static_cast<size_t>(row) * row_bytes;
    float sum = 0.0f;
    for (uint column = lane; column < cols; column += 32) {
        const device uchar* block = row_data + static_cast<size_t>(column / 32) * 18;
        const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                            (static_cast<ushort>(block[1]) << 8));
        sum += d * celeg_q4_0_value(block, column) * input[column];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[row] = reduced;
}

kernel void celeg_matvec_q4_0_tuned(device const uchar* weights [[buffer(0)]],
                                    device const float* input [[buffer(1)]],
                                    device float* output [[buffer(2)]],
                                    constant uint& rows [[buffer(3)]],
                                    constant uint& cols [[buffer(4)]],
                                    constant uint& row_bytes [[buffer(5)]],
                                    threadgroup float* partial [[threadgroup(0)]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint simd [[simdgroup_index_in_threadgroup]],
                                    uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * 2;
    if (row >= rows) return;
    float sums[2] = {0.0f, 0.0f};
    for (uint column = simd * 32 + lane; column < cols; column += 128) {
        const device uchar* first = weights + static_cast<size_t>(row) * row_bytes +
            static_cast<size_t>(column / 32) * 18;
        const float first_scale = celeg_half_to_float(static_cast<ushort>(first[0]) |
                                                       (static_cast<ushort>(first[1]) << 8));
        sums[0] += first_scale * celeg_q4_0_value(first, column) * input[column];
        if (row + 1 < rows) {
            const device uchar* second = weights + static_cast<size_t>(row + 1) * row_bytes +
                static_cast<size_t>(column / 32) * 18;
            const float second_scale = celeg_half_to_float(static_cast<ushort>(second[0]) |
                                                            (static_cast<ushort>(second[1]) << 8));
            sums[1] += second_scale * celeg_q4_0_value(second, column) * input[column];
        }
    }
    const float reduced0 = simd_sum(sums[0]);
    const float reduced1 = simd_sum(sums[1]);
    if (lane == 0) {
        partial[simd * 2] = reduced0;
        partial[simd * 2 + 1] = reduced1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0 && lane < 2) {
        float total = partial[lane];
        for (uint other = 1; other < 4; ++other) total += partial[other * 2 + lane];
        if (row + lane < rows) output[row + lane] = total;
    }
}

float celeg_q5k_value(device const uchar* block, uint column) {
    const uint sub = column >> 5;
    const uint within = column & 31;
    const uchar packed = block[48 + (sub >> 1) * 32 + within];
    const uint low = (sub & 1) != 0 ? packed >> 4 : packed & 0x0f;
    return static_cast<float>(low | (((block[16 + within] >> sub) & 1) << 4));
}

void celeg_q5k_scale_min(device const uchar* scales, uint index,
                         thread uchar& scale, thread uchar& minimum) {
    if (index < 4) {
        scale = scales[index] & 63;
        minimum = scales[index + 4] & 63;
        return;
    }
    scale = (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4);
    minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
}

kernel void celeg_matvec_q5k(device const uchar* weights [[buffer(0)]],
                             device const float* input [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& cols [[buffer(4)]],
                             constant uint& row_bytes [[buffer(5)]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]],
                             uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * 8 + simd;
    if (row >= rows) return;
    const device uchar* row_data = weights + static_cast<size_t>(row) * row_bytes;
    float sum = 0.0f;
    for (uint column = lane; column < cols; column += 32) {
        const device uchar* block = row_data + static_cast<size_t>(column / 256) * 176;
        const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                            (static_cast<ushort>(block[1]) << 8));
        const float dmin = celeg_half_to_float(static_cast<ushort>(block[2]) |
                                               (static_cast<ushort>(block[3]) << 8));
        const uint within = column & 255;
        const uint sub = within >> 5;
        uchar scale = 0;
        uchar minimum = 0;
        celeg_q5k_scale_min(block + 4, sub, scale, minimum);
        sum += (d * static_cast<float>(scale) * celeg_q5k_value(block, within) -
                dmin * static_cast<float>(minimum)) * input[column];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[row] = reduced;
}

kernel void celeg_matvec_q5k_tuned(device const uchar* weights [[buffer(0)]],
                                   device const float* input [[buffer(1)]],
                                   device float* output [[buffer(2)]],
                                   constant uint& rows [[buffer(3)]],
                                   constant uint& cols [[buffer(4)]],
                                   constant uint& row_bytes [[buffer(5)]],
                                   threadgroup float* partial [[threadgroup(0)]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd [[simdgroup_index_in_threadgroup]],
                                   uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * 2;
    if (row >= rows) return;
    float sums[2] = {0.0f, 0.0f};
    for (uint column = simd * 32 + lane; column < cols; column += 128) {
        const uint within = column & 255;
        const device uchar* first = weights + static_cast<size_t>(row) * row_bytes +
            static_cast<size_t>(column / 256) * 176;
        const float first_d = celeg_half_to_float(static_cast<ushort>(first[0]) |
                                                   (static_cast<ushort>(first[1]) << 8));
        const float first_dmin = celeg_half_to_float(static_cast<ushort>(first[2]) |
                                                      (static_cast<ushort>(first[3]) << 8));
        uchar first_scale = 0;
        uchar first_minimum = 0;
        celeg_q5k_scale_min(first + 4, within >> 5, first_scale, first_minimum);
        sums[0] += (first_d * static_cast<float>(first_scale) *
                        celeg_q5k_value(first, within) -
                    first_dmin * static_cast<float>(first_minimum)) * input[column];
        if (row + 1 < rows) {
            const device uchar* second = weights + static_cast<size_t>(row + 1) * row_bytes +
                static_cast<size_t>(column / 256) * 176;
            const float second_d = celeg_half_to_float(static_cast<ushort>(second[0]) |
                                                        (static_cast<ushort>(second[1]) << 8));
            const float second_dmin = celeg_half_to_float(static_cast<ushort>(second[2]) |
                                                           (static_cast<ushort>(second[3]) << 8));
            uchar second_scale = 0;
            uchar second_minimum = 0;
            celeg_q5k_scale_min(second + 4, within >> 5, second_scale, second_minimum);
            sums[1] += (second_d * static_cast<float>(second_scale) *
                            celeg_q5k_value(second, within) -
                        second_dmin * static_cast<float>(second_minimum)) * input[column];
        }
    }
    const float reduced0 = simd_sum(sums[0]);
    const float reduced1 = simd_sum(sums[1]);
    if (lane == 0) {
        partial[simd * 2] = reduced0;
        partial[simd * 2 + 1] = reduced1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0 && lane < 2) {
        float total = partial[lane];
        for (uint other = 1; other < 4; ++other) total += partial[other * 2 + lane];
        if (row + lane < rows) output[row + lane] = total;
    }
}

kernel void celeg_matvec_q8_0(device const uchar* weights [[buffer(0)]],
                              device const float* input [[buffer(1)]],
                              device float* output [[buffer(2)]],
                              constant uint& rows [[buffer(3)]],
                              constant uint& cols [[buffer(4)]],
                              constant uint& row_bytes [[buffer(5)]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * 8 + simd;
    if (row >= rows) return;
    const device uchar* row_data = weights + static_cast<size_t>(row) * row_bytes;
    float sum = 0.0f;
    for (uint column = lane; column < cols; column += 32) {
        const device uchar* block = row_data + static_cast<size_t>(column / 32) * 34;
        const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                            (static_cast<ushort>(block[1]) << 8));
        sum += d * static_cast<float>(static_cast<char>(block[2 + (column & 31)])) *
            input[column];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[row] = reduced;
}

float celeg_half_to_float(ushort bits) {
    return static_cast<float>(as_type<half>(bits));
}

void celeg_q4k_scale_min(device const uchar* scales, uint index,
                         thread uchar& scale, thread uchar& minimum) {
    if (index < 4) {
        scale = scales[index] & 63;
        minimum = scales[index + 4] & 63;
        return;
    }
    scale = (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4);
    minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
}

uint celeg_q4k_value(device const uchar* block, uint column) {
    const uint sub = column >> 5;
    const uint within = column & 31;
    const uchar packed = block[16 + (sub >> 1) * 32 + within];
    return (sub & 1) != 0 ? packed >> 4 : packed & 0x0f;
}

kernel void celeg_matvec_q4k(device const uchar* weights [[buffer(0)]],
                             device const float* input [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& cols [[buffer(4)]],
                             constant uint& row_bytes [[buffer(5)]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]],
                             uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * 8 + simd;
    if (row >= rows) return;
    float sum = 0.0f;
    const device uchar* row_data = weights + static_cast<size_t>(row) * row_bytes;
    for (uint column = lane; column < cols; column += 32) {
        const uint block_index = column / 256;
        const device uchar* block = row_data + static_cast<size_t>(block_index) * 144;
        const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                            (static_cast<ushort>(block[1]) << 8));
        const float dmin = celeg_half_to_float(static_cast<ushort>(block[2]) |
                                               (static_cast<ushort>(block[3]) << 8));
        const uint within = column & 255;
        const uint sub = within >> 5;
        uchar scale = 0;
        uchar minimum = 0;
        celeg_q4k_scale_min(block + 4, sub, scale, minimum);
        const float value = d * static_cast<float>(scale) *
                            static_cast<float>(celeg_q4k_value(block, within)) -
                            dmin * static_cast<float>(minimum);
        sum += value * input[column];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[row] = reduced;
}

kernel void celeg_matvec_q4k_tuned(device const uchar* weights [[buffer(0)]],
                                   device const float* input [[buffer(1)]],
                                   device float* output [[buffer(2)]],
                                   constant uint& rows [[buffer(3)]],
                                   constant uint& cols [[buffer(4)]],
                                   constant uint& row_bytes [[buffer(5)]],
                                   threadgroup float* partial [[threadgroup(0)]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd [[simdgroup_index_in_threadgroup]],
                                   uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * 2;
    if (row >= rows) return;
    float sums[2] = {0.0f, 0.0f};
    for (uint column = simd * 32 + lane; column < cols; column += 128) {
        const uint within = column & 255;
        const device uchar* first = weights + static_cast<size_t>(row) * row_bytes +
            static_cast<size_t>(column / 256) * 144;
        const float first_d = celeg_half_to_float(static_cast<ushort>(first[0]) |
                                                   (static_cast<ushort>(first[1]) << 8));
        const float first_dmin = celeg_half_to_float(static_cast<ushort>(first[2]) |
                                                      (static_cast<ushort>(first[3]) << 8));
        uchar first_scale = 0;
        uchar first_minimum = 0;
        celeg_q4k_scale_min(first + 4, within >> 5, first_scale, first_minimum);
        sums[0] += (first_d * static_cast<float>(first_scale) *
                        static_cast<float>(celeg_q4k_value(first, within)) -
                    first_dmin * static_cast<float>(first_minimum)) * input[column];
        if (row + 1 < rows) {
            const device uchar* second = weights + static_cast<size_t>(row + 1) * row_bytes +
                static_cast<size_t>(column / 256) * 144;
            const float second_d = celeg_half_to_float(static_cast<ushort>(second[0]) |
                                                        (static_cast<ushort>(second[1]) << 8));
            const float second_dmin = celeg_half_to_float(static_cast<ushort>(second[2]) |
                                                           (static_cast<ushort>(second[3]) << 8));
            uchar second_scale = 0;
            uchar second_minimum = 0;
            celeg_q4k_scale_min(second + 4, within >> 5, second_scale, second_minimum);
            sums[1] += (second_d * static_cast<float>(second_scale) *
                            static_cast<float>(celeg_q4k_value(second, within)) -
                        second_dmin * static_cast<float>(second_minimum)) * input[column];
        }
    }
    const float reduced0 = simd_sum(sums[0]);
    const float reduced1 = simd_sum(sums[1]);
    if (lane == 0) {
        partial[simd * 2] = reduced0;
        partial[simd * 2 + 1] = reduced1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0 && lane < 2) {
        float total = partial[lane];
        for (uint other = 1; other < 4; ++other) total += partial[other * 2 + lane];
        if (row + lane < rows) output[row + lane] = total;
    }
}

int celeg_q6k_value(device const uchar* block, uint column) {
    const uint half_index = column >> 7;
    const uint index = column & 127;
    const uint lane = index & 31;
    const uint group = index >> 5;
    const device uchar* ql = block + half_index * 64;
    const device uchar* qh = block + 128 + half_index * 32;
    if (group == 0) return (ql[lane] & 0x0f) | (((qh[lane] >> 0) & 3) << 4);
    if (group == 1) return (ql[lane + 32] & 0x0f) | (((qh[lane] >> 2) & 3) << 4);
    if (group == 2) return (ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4);
    return (ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4);
}

kernel void celeg_matvec_q6k(device const uchar* weights [[buffer(0)]],
                             device const float* input [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& cols [[buffer(4)]],
                             constant uint& row_bytes [[buffer(5)]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]],
                             uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * 8 + simd;
    if (row >= rows) return;
    float sum = 0.0f;
    const device uchar* row_data = weights + static_cast<size_t>(row) * row_bytes;
    for (uint column = lane; column < cols; column += 32) {
        const uint block_index = column / 256;
        const device uchar* block = row_data + static_cast<size_t>(block_index) * 210;
        const float d = celeg_half_to_float(static_cast<ushort>(block[208]) |
                                            (static_cast<ushort>(block[209]) << 8));
        const uint within = column & 255;
        const float scale = d * static_cast<float>(
            static_cast<char>(block[192 + within / 16]));
        sum += scale * static_cast<float>(celeg_q6k_value(block, within) - 32) * input[column];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[row] = reduced;
}

kernel void celeg_matvec_q6k_tuned(device const uchar* weights [[buffer(0)]],
                                   device const float* input [[buffer(1)]],
                                   device float* output [[buffer(2)]],
                                   constant uint& rows [[buffer(3)]],
                                   constant uint& cols [[buffer(4)]],
                                   constant uint& row_bytes [[buffer(5)]],
                                   threadgroup float* partial [[threadgroup(0)]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd [[simdgroup_index_in_threadgroup]],
                                   uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * 2;
    if (row >= rows) return;
    float sums[2] = {0.0f, 0.0f};
    for (uint column = simd * 32 + lane; column < cols; column += 128) {
        const uint within = column & 255;
        const device uchar* first = weights + static_cast<size_t>(row) * row_bytes +
            static_cast<size_t>(column / 256) * 210;
        const float first_d = celeg_half_to_float(static_cast<ushort>(first[208]) |
                                                   (static_cast<ushort>(first[209]) << 8));
        const float first_scale = first_d * static_cast<float>(
            static_cast<char>(first[192 + within / 16]));
        sums[0] += first_scale * static_cast<float>(celeg_q6k_value(first, within) - 32) *
            input[column];
        if (row + 1 < rows) {
            const device uchar* second = weights + static_cast<size_t>(row + 1) * row_bytes +
                static_cast<size_t>(column / 256) * 210;
            const float second_d = celeg_half_to_float(static_cast<ushort>(second[208]) |
                                                        (static_cast<ushort>(second[209]) << 8));
            const float second_scale = second_d * static_cast<float>(
                static_cast<char>(second[192 + within / 16]));
            sums[1] += second_scale *
                static_cast<float>(celeg_q6k_value(second, within) - 32) * input[column];
        }
    }
    const float reduced0 = simd_sum(sums[0]);
    const float reduced1 = simd_sum(sums[1]);
    if (lane == 0) {
        partial[simd * 2] = reduced0;
        partial[simd * 2 + 1] = reduced1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0 && lane < 2) {
        float total = partial[lane];
        for (uint other = 1; other < 4; ++other) total += partial[other * 2 + lane];
        if (row + lane < rows) output[row + lane] = total;
    }
}

kernel void celeg_embedding_q4k(device const uchar* weights [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                constant uint& width [[buffer(2)]],
                                constant uint& token [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
    if (index >= width) return;
    const uint block_index = index / 256;
    const uint column = index & 255;
    const device uchar* block = weights + static_cast<size_t>(token) *
                                (width / 256) * 144 + block_index * 144;
    const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                        (static_cast<ushort>(block[1]) << 8));
    const float dmin = celeg_half_to_float(static_cast<ushort>(block[2]) |
                                           (static_cast<ushort>(block[3]) << 8));
    const uint sub = column >> 5;
    uchar scale = 0;
    uchar minimum = 0;
    celeg_q4k_scale_min(block + 4, sub, scale, minimum);
    output[index] = d * static_cast<float>(scale) *
                    static_cast<float>(celeg_q4k_value(block, column)) -
                    dmin * static_cast<float>(minimum);
}

kernel void celeg_embedding_q6k(device const uchar* weights [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                constant uint& width [[buffer(2)]],
                                constant uint& token [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
    if (index >= width) return;
    const uint block_index = index / 256;
    const uint column = index & 255;
    const device uchar* block = weights + static_cast<size_t>(token) *
                                (width / 256) * 210 + block_index * 210;
    const float d = celeg_half_to_float(static_cast<ushort>(block[208]) |
                                        (static_cast<ushort>(block[209]) << 8));
    const uint sub = column / 16;
    const float scale = d * static_cast<float>(static_cast<char>(block[192 + sub]));
    output[index] = scale * static_cast<float>(celeg_q6k_value(block, column) - 32);
}

kernel void celeg_embedding_q4_0(device const uchar* weights [[buffer(0)]],
                                 device float* output [[buffer(1)]],
                                 constant uint& width [[buffer(2)]],
                                 constant uint& token [[buffer(3)]],
                                 uint index [[thread_position_in_grid]]) {
    if (index >= width) return;
    const device uchar* block = weights + static_cast<size_t>(token) * (width / 32) * 18 +
        static_cast<size_t>(index / 32) * 18;
    const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                        (static_cast<ushort>(block[1]) << 8));
    output[index] = d * celeg_q4_0_value(block, index);
}

kernel void celeg_embedding_q5k(device const uchar* weights [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                constant uint& width [[buffer(2)]],
                                constant uint& token [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
    if (index >= width) return;
    const uint block_index = index / 256;
    const uint column = index & 255;
    const device uchar* block = weights + static_cast<size_t>(token) * (width / 256) * 176 +
        static_cast<size_t>(block_index) * 176;
    const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                        (static_cast<ushort>(block[1]) << 8));
    const float dmin = celeg_half_to_float(static_cast<ushort>(block[2]) |
                                           (static_cast<ushort>(block[3]) << 8));
    uchar scale = 0;
    uchar minimum = 0;
    celeg_q5k_scale_min(block + 4, column >> 5, scale, minimum);
    output[index] = d * static_cast<float>(scale) * celeg_q5k_value(block, column) -
        dmin * static_cast<float>(minimum);
}

kernel void celeg_embedding_q8_0(device const uchar* weights [[buffer(0)]],
                                 device float* output [[buffer(1)]],
                                 constant uint& width [[buffer(2)]],
                                 constant uint& token [[buffer(3)]],
                                 uint index [[thread_position_in_grid]]) {
    if (index >= width) return;
    const device uchar* block = weights + static_cast<size_t>(token) * (width / 32) * 34 +
        static_cast<size_t>(index / 32) * 34;
    const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                        (static_cast<ushort>(block[1]) << 8));
    output[index] = d * static_cast<float>(static_cast<char>(block[2 + (index & 31)]));
}
