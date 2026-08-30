#pragma once

#include "celeg/checkpoint/gguf_iq_tables.hpp"

#include <cstdint>
#include <cstring>

namespace celeg::gguf_iq {

/// Packed IQ block layouts and their element decoders, defined once and
/// shared by the CPU kernels and the CUDA host loader.
///
/// The older GGUF quantizations carry a separate copy of their block structs
/// per backend, which is how the CUDA and CPU type support drifted apart.
/// The IQ formats start out shared instead: one struct and one decoder per
/// type, so a fix to either lands everywhere at once.
///
/// Every decoder returns the value scaled by the block's own `d`, matching
/// ggml's `dequantize_row_iq*`. The dot kernels need the unscaled integer
/// magnitudes as well, so each type also exposes the pieces separately.

#pragma pack(push, 1)

/// 2.5 bpw. 8 sub-blocks of 32; each group of 8 weights is one grid entry
/// selected by qs plus two extra bits from qh, with signs in the bytes that
/// follow qs.
struct BlockIq2S {
    std::uint16_t d;
    std::uint8_t qs[64];
    std::uint8_t qh[8];
    std::uint8_t scales[8];
};

/// 3.0625 bpw. Scale and signs for each sub-block are packed together into
/// one uint32 that follows the 64 grid indices.
struct BlockIq3XXS {
    std::uint16_t d;
    std::uint8_t qs[96];
};

/// 3.4375 bpw. Like IQ3_XXS but with explicit sign bytes, a 9th grid-index
/// bit in qh, and 4-bit scales shared by pairs of sub-blocks.
struct BlockIq3S {
    std::uint16_t d;
    std::uint8_t qs[64];
    std::uint8_t qh[8];
    std::uint8_t signs[32];
    std::uint8_t scales[4];
};

/// Non-linear 4-bit over blocks of 32: the nibble indexes a codebook of
/// unevenly spaced levels instead of scaling a uniform integer.
struct BlockIq4NL {
    std::uint16_t d;
    std::uint8_t qs[16];
};

/// IQ4_NL's codebook applied over 256-element blocks, with a 6-bit scale per
/// sub-block split across scales_l and scales_h.
struct BlockIq4XS {
    std::uint16_t d;
    std::uint16_t scales_h;
    std::uint8_t scales_l[4];
    std::uint8_t qs[128];
};

#pragma pack(pop)

static_assert(sizeof(BlockIq2S) == 82);
static_assert(sizeof(BlockIq3XXS) == 98);
static_assert(sizeof(BlockIq3S) == 110);
static_assert(sizeof(BlockIq4NL) == 18);
static_assert(sizeof(BlockIq4XS) == 136);

/// -1 or +1 for element `j` of the group selected by sign byte `signs`.
inline int sign_of(std::uint8_t signs, int j) {
    return (signs & k_kmask_iq2xs[j]) ? -1 : 1;
}

/// Byte `j` of grid entry `entry`, read the way ggml reads it: through a
/// byte pointer into the table, so the packing matches on any endianness.
template <typename Entry>
inline int grid_byte(const Entry& entry, int j) {
    return reinterpret_cast<const std::uint8_t*>(&entry)[j];
}

// --- IQ2_S -----------------------------------------------------------------
// A 256-block holds 8 sub-blocks of 32; each sub-block splits into 4 groups
// of 8, and the two halves of a sub-block use the two nibbles of scales[].
// qs is addressed as two halves: 32 grid-index bytes followed by 32 sign
// bytes (ggml reaches the latter as `qs + QK_K/8`).

/// Signed grid magnitude of element `within` (0..255).
inline int iq2s_value(const BlockIq2S& block, int within) {
    const int ib32 = within / 32;
    const int group = (within % 32) / 8;
    const int j = within % 8;
    const int index = block.qs[ib32 * 4 + group] |
                      ((block.qh[ib32] << (8 - 2 * group)) & 0x300);
    const std::uint8_t signs = block.qs[32 + ib32 * 4 + group];
    return grid_byte(k_iq2s_grid[index], j) * sign_of(signs, j);
}

/// Multiplier applied to the magnitudes of sub-block `ib32`, half `half`
/// (0 for the first 16 elements, 1 for the second).
inline float iq2s_sub_scale(const BlockIq2S& block, int ib32, int half) {
    const int raw = half == 0 ? (block.scales[ib32] & 0x0f) : (block.scales[ib32] >> 4);
    return (0.5f + static_cast<float>(raw)) * 0.25f;
}

// --- IQ3_XXS ---------------------------------------------------------------

/// The scale/sign word for sub-block `ib32`, read from the 32 bytes that
/// follow the 64 grid indices.
inline std::uint32_t iq3xxs_aux(const BlockIq3XXS& block, int ib32) {
    std::uint32_t aux = 0;
    std::memcpy(&aux, block.qs + 64 + 4 * ib32, sizeof(aux));
    return aux;
}

inline float iq3xxs_sub_scale(std::uint32_t aux) {
    return (0.5f + static_cast<float>(aux >> 28)) * 0.5f;
}

/// Signed grid magnitude of element `within` (0..255).
inline int iq3xxs_value(const BlockIq3XXS& block, int within) {
    const int ib32 = within / 32;
    const int group = (within % 32) / 8;
    const int j = within % 8;
    // Each group of 8 is two grid entries of 4 bytes: j<4 from the first,
    // j>=4 from the second.
    const std::uint8_t index = block.qs[ib32 * 8 + 2 * group + (j >= 4 ? 1 : 0)];
    const std::uint8_t signs = k_ksigns_iq2xs[(iq3xxs_aux(block, ib32) >> (7 * group)) & 127];
    return grid_byte(k_iq3xxs_grid[index], j % 4) * sign_of(signs, j);
}

// --- IQ3_S -----------------------------------------------------------------

inline float iq3s_sub_scale(const BlockIq3S& block, int ib32) {
    const int raw = (ib32 % 2 == 0) ? (block.scales[ib32 / 2] & 0x0f)
                                    : (block.scales[ib32 / 2] >> 4);
    return 1.0f + 2.0f * static_cast<float>(raw);
}

/// Signed grid magnitude of element `within` (0..255).
inline int iq3s_value(const BlockIq3S& block, int within) {
    const int ib32 = within / 32;
    const int group = (within % 32) / 8;
    const int j = within % 8;
    const int high = j >= 4 ? 1 : 0;
    // qh contributes a 9th index bit; ggml shifts it by (8 - 2*group) for the
    // first grid entry of a group and (7 - 2*group) for the second.
    const int shift = (high ? 7 : 8) - 2 * group;
    const int index = block.qs[ib32 * 8 + 2 * group + high] |
                      ((block.qh[ib32] << shift) & 256);
    return grid_byte(k_iq3s_grid[index], j % 4) *
           sign_of(block.signs[ib32 * 4 + group], j);
}

// --- IQ4_NL / IQ4_XS -------------------------------------------------------

/// Codebook level of element `within` (0..31) of a 32-wide IQ4_NL block.
inline int iq4nl_value(const BlockIq4NL& block, int within) {
    // Split-half nibble layout: qs[j] holds element j low, element j+16 high.
    const std::uint8_t packed = block.qs[within & 15];
    return k_kvalues_iq4nl[within < 16 ? (packed & 0x0f) : (packed >> 4)];
}

/// Codebook level of element `within` (0..255) of an IQ4_XS block.
inline int iq4xs_value(const BlockIq4XS& block, int within) {
    const int ib = within / 32;
    const int local = within % 32;
    const std::uint8_t packed = block.qs[ib * 16 + (local & 15)];
    return k_kvalues_iq4nl[local < 16 ? (packed & 0x0f) : (packed >> 4)];
}

/// Signed 6-bit scale of sub-block `ib`, already biased by -32.
inline float iq4xs_sub_scale(const BlockIq4XS& block, int ib) {
    const int level = ((block.scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0f) |
                      (((block.scales_h >> (2 * ib)) & 3) << 4);
    return static_cast<float>(level - 32);
}

}
