#include <metal_stdlib>

using namespace metal;

float celeg_bf16_to_float(ushort bits) {
    return as_type<float>(static_cast<uint>(bits) << 16);
}

float celeg_q4_0_value(device const uchar* block, uint column) {
    const uchar packed = block[2 + (column & 15)];
    const uint value = (column & 16) == 0 ? packed & 0x0f : packed >> 4;
    return static_cast<float>(value) - 8.0f;
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

float celeg_half_to_float(ushort bits) {
    return static_cast<float>(as_type<half>(bits));
}

kernel void celeg_embedding(device const float* table [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant uint& width [[buffer(2)]],
                            constant uint& token [[buffer(3)]],
                            uint index [[thread_position_in_grid]]) {
    if (index < width) output[index] = table[static_cast<size_t>(token) * width + index];
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
