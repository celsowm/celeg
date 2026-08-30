#pragma once

#include <cstdint>

namespace celeg::gguf_iq {

/// Codebooks and sign tables the GGUF IQ quantizations decode against,
/// transcribed from ggml-common.h (llama.cpp, MIT licence). Defined in
/// gguf_iq_tables.cpp so the constants exist once per program.
///
/// - k_iq2s_grid / k_iq3xxs_grid / k_iq3s_grid: each entry packs 8 (IQ2) or
///   4 (IQ3) unsigned magnitudes, one byte apiece.
/// - k_kvalues_iq4nl: the 16 non-linear levels IQ4_NL and IQ4_XS index.
/// - k_ksigns_iq2xs / k_kmask_iq2xs: sign-mask expansion, shared by IQ2 and
///   IQ3 decoders.
extern const std::uint64_t k_iq2s_grid[1024];
extern const std::uint32_t k_iq3xxs_grid[256];
extern const std::uint32_t k_iq3s_grid[512];
extern const std::int8_t k_kvalues_iq4nl[16];
extern const std::uint8_t k_kmask_iq2xs[8];
extern const std::uint8_t k_ksigns_iq2xs[128];

}
