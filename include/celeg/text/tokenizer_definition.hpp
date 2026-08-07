#pragma once

#include <cstdint>
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

// These are tokenizer behaviors, rather than model-family identities.  The
// source-format resolver maps checkpoint metadata to one of these values
// before the BPE engine is constructed.
enum class TokenizerPreTokenizerKind : uint8_t {
    Default,
    NumericTriplets,
    NumericRuns,
    RawUtf8,
};

struct TokenizerDefinition {
    std::vector<std::string> tokens;
    std::vector<std::string> merges;
    std::vector<TokenizerSpecialToken> special_tokens;
    TokenizerPreTokenizerKind pre_tokenizer = TokenizerPreTokenizerKind::Default;
    TokenizerNormalizationKind normalization = TokenizerNormalizationKind::None;
    bool byte_fallback = false;
    int32_t bos_id = 1;
    int32_t eos_id = 7;
    int32_t pad_id = 0;
};

TokenizerDefinition load_tokenizer_definition_json(const std::string& path);
TokenizerDefinition resolve_tokenizer_definition(const TokenizerData& data);

} // namespace celeg
