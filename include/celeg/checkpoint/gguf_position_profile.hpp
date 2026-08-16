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

}
