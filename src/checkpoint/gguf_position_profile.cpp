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

bool gguf_architecture_uses_adjacent_rope_pairs(const std::string& gguf_architecture) {
    /// Mirrors llama.cpp's llama_model_rope_type() -> LLAMA_ROPE_TYPE_NORM
    /// architecture list. Every architecture absent from this set uses the
    /// NEOX-style half-split pairing, which is also the safe default because it
    /// matches the original HuggingFace row order. Update alongside upstream
    /// when new GGUF architectures are added.
    static const std::unordered_set<std::string> kArchitectures = {
        "arcee", "arctic", "baichuan", "bailingmoe", "bailingmoe3",
        "chameleon", "chatglm", "cohere2", "cohere2moe", "command-r",
        "deci", "deepseek", "deepseek2", "deepseek2-ocr", "deepseek32",
        "deepseek4", "eagle3", "ernie4_5", "ernie4_5-moe", "glm-dsa",
        "granite", "granitehybrid", "granitemoe", "graniteswitch",
        "internlm2", "llada", "llama", "llama-embed", "llama4",
        "maincoder", "minicpm", "mistral3", "mistral4", "muse-glimmer",
        "nanbeige", "neo-bert", "olmo", "plm", "pockettts", "smollm3",
        "starcoder", "xverse",
        /// glm4 selects NORM whenever it is not running the multimodal
        /// M-RoPE variant, which celeg does not implement.
        "glm4",
    };
    return kArchitectures.contains(gguf_architecture);
}

}
