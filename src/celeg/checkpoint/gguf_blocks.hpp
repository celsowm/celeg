#pragma once

#include <cstddef>
#include <cstdint>

namespace celeg::gguf_blocks {

/// Canonical serialized GGUF Q4_K super-block layout.
struct BlockQ4K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

/// Canonical serialized GGUF Q6_K super-block layout.
struct BlockQ6K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
};

static_assert(sizeof(BlockQ4K) == 144);
static_assert(offsetof(BlockQ4K, d) == 0);
static_assert(offsetof(BlockQ4K, dmin) == 2);
static_assert(offsetof(BlockQ4K, scales) == 4);
static_assert(offsetof(BlockQ4K, qs) == 16);

static_assert(sizeof(BlockQ6K) == 210);
static_assert(offsetof(BlockQ6K, ql) == 0);
static_assert(offsetof(BlockQ6K, qh) == 128);
static_assert(offsetof(BlockQ6K, scales) == 192);
static_assert(offsetof(BlockQ6K, d) == 208);

#if defined(__CUDACC__)
__host__ __device__
#endif
inline void q4k_scale_min(int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = static_cast<uint8_t>((q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4));
        m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

}