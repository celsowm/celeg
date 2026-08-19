#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace celeg {

struct TokenizerData;

struct TokenizerSpecialToken {
    std::string text;
    int32_t id = -1;
    bool skip_on_decode = true;
};

enum class TokenizerNormalizationKind : uint8_t {
    None,
    SentencePieceSpace,
};

enum class TokenizerPreTokenizerKind : uint8_t {
    Default,
    NumericTriplets,
    NumericRuns,
    ByteLevelRegex,
    RawUtf8,
};

struct TokenizerPreTokenizerRule {
    std::string identifier;
    TokenizerPreTokenizerKind behavior = TokenizerPreTokenizerKind::Default;
    bool contains = false;
};

struct TokenizerDefinition {
    std::vector<std::string> tokens;
    std::vector<std::string> merges;
    /// Per-token SentencePiece priority (GGUF `tokenizer.ggml.scores`), used
    /// by the score-based bigram tokenizer when `merges` is empty (see
    /// BpeTokenizer's SentencePiece-without-merges path).
    std::vector<float> scores;
    std::vector<TokenizerSpecialToken> special_tokens;
    TokenizerPreTokenizerKind pre_tokenizer = TokenizerPreTokenizerKind::Default;
    TokenizerNormalizationKind normalization = TokenizerNormalizationKind::None;
    bool byte_fallback = false;
    int32_t bos_id = 1;
    int32_t eos_id = 7;
    int32_t pad_id = 0;
    bool has_bos = false;
    int tokenizer_vocab_size = 0;
};

TokenizerDefinition load_tokenizer_definition_json(const std::string& path);
TokenizerDefinition load_tokenizer_definition_json(const std::string& path, std::optional<int> target_vocab_size);
TokenizerDefinition resolve_tokenizer_definition(
    const TokenizerData& data,
    const std::vector<TokenizerPreTokenizerRule>& rules = {});

}
