#include "celeg/checkpoint/gguf_position_profile.hpp"

#include <unordered_set>

namespace celeg {

bool gguf_architecture_never_applies_rope(const std::string& gguf_architecture) {
    // Mirrors llama.cpp's llama_model_rope_type() -> LLAMA_ROPE_TYPE_NONE
    // architecture list. Update alongside upstream when new hybrid/
    // recurrent GGUF architectures are added.
    static const std::unordered_set<std::string> kArchitectures = {
        "clip", "gpt2", "gptj", "mpt", "refact", "bloom",
        "mamba", "mamba2", "jamba", "jina-bert-v2", "t5", "t5encoder",
        "jais", "rwkv6", "rwkv6qwen2", "rwkv7", "arwkv7",
        "wavtokenizer-dec", "nemotron_h", "nemotron_h_moe", "kimi-linear",
    };
    return kArchitectures.contains(gguf_architecture);
}

}
