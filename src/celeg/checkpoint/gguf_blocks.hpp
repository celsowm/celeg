#pragma once


#include <cstdint>

namespace celeg::gguf_blocks {

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