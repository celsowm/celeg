#include "gguf_tokenizer_metadata.hpp"

namespace celeg {

TokenizerData read_gguf_tokenizer_data(const GgufFile& file) {
    TokenizerData result;
    const auto& tokens = file.value("tokenizer.ggml.tokens");
    if (tokens.kind == GgufValueKind::Array &&
        tokens.array_kind == GgufValueKind::String) {
        result.tokens = tokens.array_strings;
    }

    const auto& merges = file.value("tokenizer.ggml.merges");
    if (merges.kind == GgufValueKind::Array &&
        merges.array_kind == GgufValueKind::String) {
        result.merges = merges.array_strings;
    }

    if (file.has("tokenizer.ggml.token_type")) {
        const auto& types = file.value("tokenizer.ggml.token_type");
        result.token_types.reserve(types.array_integers.size());
        for (const int64_t type : types.array_integers) {
            result.token_types.push_back(static_cast<int32_t>(type));
        }
    }
    if (file.has("tokenizer.ggml.bos_token_id")) {
        result.bos_id = static_cast<int32_t>(file.i64("tokenizer.ggml.bos_token_id"));
    }
    if (file.has("tokenizer.ggml.eos_token_id")) {
        result.eos_id = static_cast<int32_t>(file.i64("tokenizer.ggml.eos_token_id"));
    }
    if (file.has("tokenizer.ggml.padding_token_id")) {
        result.pad_id = static_cast<int32_t>(file.i64("tokenizer.ggml.padding_token_id"));
    }

    const std::string source = file.str_or("tokenizer.ggml.pre", "");
    // GGUF stores a tokenizer implementation label rather than the semantic
    // behavior consumed by the tokenizer engine. Normalize that format
    // spelling at the import boundary; runtime composition never sees it.
    if (source == "lfm2" || source == "smaug-bpe") {
        result.pre_tokenizer = "numeric_triplets";
    } else if (source == "gpt2" || source == "granite") {
        result.pre_tokenizer = "numeric_runs";
    } else {
        result.pre_tokenizer = source;
    }
    return result;
}

} // namespace celeg
