#pragma once

#include <string>

namespace celeg {

// GGUF carries an unavoidable ambiguity: some architectures store
// "<arch>.rope.*" hyperparameters inherited from a related architecture
// family even though their reference graph builder never applies RoPE to
// attention (llama.cpp: llama_model_rope_type() == LLAMA_ROPE_TYPE_NONE).
// Nothing in the checkpoint's own tensor shapes, tensor names or other
// metadata can distinguish "RoPE metadata present and active" from "RoPE
// metadata present but vestigial" -- that fact is pure GGUF-format/
// architecture knowledge, so it is declared once here at the GGUF format
// boundary instead of inside generic, architecture-agnostic semantic
// inference.
bool gguf_architecture_never_applies_rope(const std::string& gguf_architecture);

/// GGUF carries a second unavoidable ambiguity, about the row order of the
/// attention query/key projections. llama.cpp's conversion permutes those rows
/// for architectures whose reference graph applies "normal" RoPE over pairs of
/// consecutive head components (llama_model_rope_type() ==
/// LLAMA_ROPE_TYPE_NORM), and leaves them in the original half-split order for
/// the NEOX-style architectures. The file records the architecture but never
/// the resulting row order, and no shape, tensor name, or weight statistic
/// recovers it, so -- exactly like the predicate above -- it is declared here
/// at the GGUF format boundary rather than guessed inside architecture-agnostic
/// inference. Returns true when the checkpoint's Q/K rows are stored in
/// llama.cpp's consecutive-pair order, i.e. RopePairingKind::AdjacentPairs.
bool gguf_architecture_uses_adjacent_rope_pairs(const std::string& gguf_architecture);

}
