#pragma once

#include "celeg/text/tokenizer_definition.hpp"
#include "celeg/text/chat_template.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace celeg {

class ITokenizer {
public:
    virtual ~ITokenizer() = default;
    virtual std::vector<int32_t> encode(std::string_view text,
                                        bool add_bos = true) const = 0;
    virtual std::string decode(const std::vector<int32_t>& ids,
                               bool skip_special = true) const = 0;
    virtual std::string decode_token(int32_t id,
                                     bool skip_special = true) const = 0;
    virtual std::optional<int32_t> token_id(std::string_view text) const = 0;
    virtual int32_t bos_id() const = 0;
    virtual int32_t eos_id() const = 0;
    virtual int32_t pad_id() const = 0;
};

class BpeTokenizer final : public ITokenizer {
public:
    explicit BpeTokenizer(const TokenizerDefinition& definition);

    std::vector<int32_t> encode(std::string_view text, bool add_bos = true) const override;
    std::string decode(const std::vector<int32_t>& ids, bool skip_special = true) const override;
    std::string decode_token(int32_t id, bool skip_special = true) const override;
    int32_t bos_id() const override { return bos_id_; }
    int32_t eos_id() const override { return eos_id_; }
    int32_t pad_id() const override { return pad_id_; }
    std::optional<int32_t> token_id(std::string_view text) const override;

private:
    struct SpecialToken {
        std::string text;
        int32_t id;
    };

    void load_definition(const TokenizerDefinition& definition);
    // Initializes the GPT-2 byte<->unicode encoder tables shared by both the
    // JSON and GGUF load paths.
    void init_byte_encoder();

    std::vector<std::string> pretokenize(std::string_view text) const;
    std::vector<std::string> pretokenize_byte_level_regex(std::string_view text) const;
    std::vector<int32_t> encode_ordinary(std::string_view text) const;
    std::vector<std::string> bpe(std::string_view encoded_piece) const;
    std::vector<std::string> bpe_symbols(std::vector<std::string> symbols) const;
    std::string byte_encode(std::string_view bytes) const;
    std::string byte_decode(std::string_view encoded) const;

    std::unordered_map<std::string, int32_t> vocab_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> merge_rank_;
    std::vector<SpecialToken> specials_;
    std::unordered_map<int32_t, bool> special_ids_;
    std::array<std::string, 256> byte_encoder_{};
    std::unordered_map<uint32_t, uint8_t> byte_decoder_;
    TokenizerPreTokenizerKind pre_tokenizer_ = TokenizerPreTokenizerKind::Default;
    TokenizerNormalizationKind normalization_ = TokenizerNormalizationKind::None;
    bool byte_fallback_ = false;
    int32_t bos_id_ = 1;
    int32_t eos_id_ = 7;
    int32_t pad_id_ = 0;
};

} // namespace celeg
