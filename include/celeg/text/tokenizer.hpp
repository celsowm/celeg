#pragma once

#include "celeg/text/chat_template.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace celeg {

class BpeTokenizer {
public:
    // Loads the tokenizer with the default instruct chat profile.
    explicit BpeTokenizer(const std::string& tokenizer_json_path);
    // Loads the tokenizer with an explicit chat template chosen by the model
    // profile. Use this overload when the chat profile is known at construction.
    BpeTokenizer(const std::string& tokenizer_json_path,
                 std::unique_ptr<IChatTemplate> chat_template);

    // Builds the tokenizer from a GGUF metadata container (tokenizer.ggml.*
    // keys) instead of a tokenizer.json file. The tag disambiguates this from
    // the path-based constructor.
    struct FromGguf {};
    BpeTokenizer(FromGguf, const class GgufFile& gguf,
                 std::unique_ptr<IChatTemplate> chat_template);

    std::vector<int32_t> encode(std::string_view text, bool add_bos = true) const;
    std::string decode(const std::vector<int32_t>& ids, bool skip_special = true) const;
    std::string format_chat(std::span<const ChatMessage> messages,
                            bool add_generation_prompt = true) const;

    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    int32_t pad_id() const { return pad_id_; }

private:
    enum class BpeProfile : uint8_t {
        Generic,
        ByteLevelLfm2,
        ByteLevelGranite,
        RawUtf8Gemma,
    };

    struct SpecialToken {
        std::string text;
        int32_t id;
    };

    void load(const std::string& tokenizer_json_path);
    void load_gguf(const class GgufFile& gguf);
    // Initializes the GPT-2 byte<->unicode encoder tables shared by both the
    // JSON and GGUF load paths.
    void init_byte_encoder();

    std::vector<std::string> pretokenize(std::string_view text) const;
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
    BpeProfile profile_ = BpeProfile::Generic;
    bool gemma_normalization_ = false;
    bool byte_fallback_ = false;
    std::unique_ptr<IChatTemplate> chat_template_;
    int32_t bos_id_ = 1;
    int32_t eos_id_ = 7;
    int32_t pad_id_ = 0;
};

} // namespace celeg
