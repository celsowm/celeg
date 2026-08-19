#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace celeg {

struct TokenizerData {
    std::vector<std::string> tokens;
    std::vector<std::string> merges;
    std::vector<int32_t> token_types;
    // GGUF `tokenizer.ggml.scores`: per-token SentencePiece priority used by
    // the "llama" vocab's own bigram-merge tokenizer when no explicit merge
    // list is present (see gguf_tokenizer_metadata.cpp / resolve_tokenizer_definition()).
    std::vector<float> scores;
    int32_t bos_id = 1;
    int32_t eos_id = 7;
    int32_t pad_id = 0;
    bool has_bos = false;
    std::string pre_tokenizer;
    // GGUF `tokenizer.ggml.model`: "llama" (SentencePiece, byte-fallback
    // vocabulary) vs. "gpt2"/others (byte-level BPE). Empty for non-GGUF
    // checkpoints, which use load_tokenizer_definition_json() instead.
    std::string model;
};

}
