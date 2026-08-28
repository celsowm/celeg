#include "celeg/text/tokenizer.hpp"
#include "detail.hpp"

#include <algorithm>
#include <string_view>

namespace celeg {
using namespace tokenizer_detail;

void BpeTokenizer::init_byte_encoder() {
    std::vector<int> bs;
    for (int b = 33; b <= 126; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n++);
        }
    }
    for (size_t i = 0; i < bs.size(); ++i) {
        std::string encoded;
        append_utf8(encoded, static_cast<uint32_t>(cs[i]));
        byte_encoder_[static_cast<size_t>(bs[i])] = encoded;
        byte_decoder_[static_cast<uint32_t>(cs[i])] = static_cast<uint8_t>(bs[i]);
    }
}

std::string BpeTokenizer::byte_encode(std::string_view bytes) const {
    std::string out;
    for (unsigned char byte : bytes) out += byte_encoder_[byte];
    return out;
}

std::string BpeTokenizer::byte_decode(std::string_view encoded) const {
    std::string out;
    size_t i = 0;
    while (i < encoded.size()) {
        const auto [cp, len] = next_cp(encoded, i);
        const auto it = byte_decoder_.find(cp);
        if (it != byte_decoder_.end()) out.push_back(static_cast<char>(it->second));
        i += len;
    }
    return out;
}

}
