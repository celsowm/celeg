#pragma once

#include "celeg/checkpoint/gguf_blocks.hpp"
#include "celeg/quantization/ggml.hpp"
#include "celeg/checkpoint/gguf_iq.hpp"

#include <cstdint>
#include <cstring>

namespace celeg::ggml_detail {

using celeg::gguf_blocks::q4k_scale_min;

#pragma pack(push, 1)
struct BlockQ4K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct BlockQ5K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

struct BlockQ6K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
};

struct BlockQ2K {
    uint8_t scales[16];
    uint8_t qs[64];
    uint16_t d;
    uint16_t dmin;
};

struct BlockQ3K {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    uint16_t d;
};

struct BlockQ4_0 {
    uint16_t d;
    uint8_t qs[16];
};

struct BlockQ5_0 {
    uint16_t d;
    uint8_t qh[4];
    uint8_t qs[16];
};

struct BlockQ8_0 {
    uint16_t d;
    int8_t qs[32];
};
struct BlockQ4_1 {
    uint16_t d;
    uint16_t dmin;
    uint8_t qs[16];
};
#pragma pack(pop)

static_assert(sizeof(BlockQ4K) == 144);
static_assert(sizeof(BlockQ5K) == 176);
static_assert(sizeof(BlockQ2K) == 84);
static_assert(sizeof(BlockQ3K) == 110);
static_assert(sizeof(BlockQ6K) == 210);
static_assert(sizeof(BlockQ4_0) == 18);
static_assert(sizeof(BlockQ5_0) == 22);
static_assert(sizeof(BlockQ8_0) == 34);
static_assert(sizeof(BlockQ4_1) == 20);

inline int q4k_value(const BlockQ4K& block, int col) {
    const int sub = col >> 5;
    const int within = col & 31;
    const uint8_t byte = block.qs[(sub >> 1) * 32 + within];
    return (sub & 1) ? (byte >> 4) : (byte & 0x0f);
}

inline int q5k_value(const BlockQ5K& block, int col) {
    const int sub = col >> 5;
    const int within = col & 31;
    const uint8_t packed = block.qs[(sub >> 1) * 32 + within];
    const int low = (sub & 1) ? (packed >> 4) : (packed & 0x0f);
    return low | (((block.qh[within] >> sub) & 1) << 4);
}

inline int q6k_value(const BlockQ6K& block, int col) {
    const int half = col >> 7;
    const int index = col & 127;
    const int lane = index & 31;
    const int group = index >> 5;
    const uint8_t* ql = block.ql + half * 64;
    const uint8_t* qh = block.qh + half * 32;
    if (group == 0) {
        return (ql[lane] & 0x0f) | (((qh[lane] >> 0) & 3) << 4);
    }
    if (group == 1) {
        return (ql[lane + 32] & 0x0f) | (((qh[lane] >> 2) & 3) << 4);
    }
    if (group == 2) {
        return (ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4);
    }
    return (ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4);
}

inline int q2k_value(const BlockQ2K& block, int col) {
    const int half = col / 128;
    const int local = col % 128;
    const int lane = local % 32;
    const int shift = (local / 32) * 2;
    return (block.qs[half * 32 + lane] >> shift) & 3;
}

inline void q3k_scales(const BlockQ3K& block, int8_t* output) {
    uint32_t words[4]{};
    uint32_t source[3]{};
    std::memcpy(source, block.scales, sizeof(block.scales));
    constexpr uint32_t low_nibble = 0x0f0f0f0f;
    constexpr uint32_t high_bits = 0x03030303;
    const uint32_t packed = source[2];
    words[2] = ((source[0] >> 4) & low_nibble) |
               (((packed >> 4) & high_bits) << 4);
    words[3] = ((source[1] >> 4) & low_nibble) |
               (((packed >> 6) & high_bits) << 4);
    words[0] = (source[0] & low_nibble) |
               (((packed >> 0) & high_bits) << 4);
    words[1] = (source[1] & low_nibble) |
               (((packed >> 2) & high_bits) << 4);
    for (int i = 0; i < 16; ++i) {
        output[i] = static_cast<int8_t>(
            reinterpret_cast<const uint8_t*>(words)[i]) - 32;
    }
}

inline int q3k_value(const BlockQ3K& block, int col) {
    const int half = col / 128;
    const int local = col % 128;
    const int lane = local % 32;
    const int shift = (local / 32) * 2;
    const int high = (block.hmask[lane] & (1u << (local / 32))) ? 0 : 4;
    return ((block.qs[half * 32 + lane] >> shift) & 3) - high;
}

}

