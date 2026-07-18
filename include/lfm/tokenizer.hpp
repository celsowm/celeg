#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lfm {

class BpeTokenizer {
public:
    explicit BpeTokenizer(const std::string& tokenizer_json_path);

    std::vector<int32_t> encode(std::string_view text, bool add_bos = true) const;
    std::string decode(const std::vector<int32_t>& ids, bool skip_special = true) const;
    std::string format_chat(std::string_view user_prompt, std::string_view system_prompt = {}) const;

    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }

private:
    struct SpecialToken {
        std::string text;
        int32_t id;
    };

    std::vector<std::string> pretokenize(std::string_view text) const;
    std::vector<int32_t> encode_ordinary(std::string_view text) const;
    std::vector<std::string> bpe(std::string_view encoded_piece) const;
    std::string byte_encode(std::string_view bytes) const;
    std::string byte_decode(std::string_view encoded) const;

    std::unordered_map<std::string, int32_t> vocab_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> merge_rank_;
    std::vector<SpecialToken> specials_;
    std::unordered_map<int32_t, bool> special_ids_;
    std::array<std::string, 256> byte_encoder_{};
    std::unordered_map<uint32_t, uint8_t> byte_decoder_;
    int32_t bos_id_ = 1;
    int32_t eos_id_ = 7;
};

} // namespace lfm
