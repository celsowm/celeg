#pragma once

// Shared GGUF (ggml) block-quantization layout and scalar decode helpers.
//
// The on-disk Q4_K / Q6_K super-block layout is fixed by the GGUF format and is
// identical across the CPU and CUDA backends. The pure-integer byte-unpacking
// helpers below are portable to both host and device, so the four previously
// duplicated copies (cpu/gguf.cpp, cpu/gguf_avx2.cpp, cuda/gguf.cu, cuda/mmq.cu)
// now delegate here. Backends keep their own block structs because the fp16
// scale field is stored as `__half` on CUDA and `uint16_t` on CPU, but every
// site calls `gguf_q4k_scale_min` from this header for the 6-bit scale/min
// extraction.

#include <cstdint>

namespace celeg::gguf_blocks {

// Decodes the 6-bit scale and min for sub-block j (0..7) from the packed
// 12-byte scales array of a Q4_K super-block. Mirrors ggml's
// get_scale_min_k4. Works on both host and device.
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

} // namespace celeg::gguf_blocks