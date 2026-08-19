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
    virtual int vocab_size() const = 0;
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
    int vocab_size() const override { return static_cast<int>(id_to_token_.size()); }
    std::optional<int32_t> token_id(std::string_view text) const override;

private:
    struct SpecialToken {
        std::string text;
        int32_t id;
    };

    void load_definition(const TokenizerDefinition& definition);
    void init_byte_encoder();

    std::vector<std::string> pretokenize(std::string_view text) const;
    std::vector<std::string> pretokenize_byte_level_regex(std::string_view text) const;
    std::vector<int32_t> encode_ordinary(std::string_view text) const;
    std::vector<std::string> bpe(std::string_view encoded_piece) const;
    std::vector<std::string> bpe_symbols(std::vector<std::string> symbols) const;
    std::vector<int32_t> spm_score_tokenize(std::string_view normalized) const;
    std::string byte_encode(std::string_view bytes) const;
    std::string byte_decode(std::string_view encoded) const;

    std::unordered_map<std::string, int32_t> vocab_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> merge_rank_;
    /// Per-token SentencePiece priority, indexed by id; parallels
    /// id_to_token_. Only populated (and only consulted, via
    /// spm_score_mode_) when the checkpoint ships scores but no merge list --
    /// e.g. GGUF's "llama" vocab type, which derives valid merges from vocab
    /// membership itself rather than an explicit merge-rank table.
    std::vector<float> id_score_;
    bool spm_score_mode_ = false;
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
    bool has_bos_ = false;
};

}
