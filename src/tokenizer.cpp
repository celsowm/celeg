#include "lfm/tokenizer.hpp"
#include "lfm/gguf.hpp"
#include "lfm/json.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <utility>

namespace lfm {
namespace {

void append_utf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7F) out.push_back(static_cast<char>(cp));
    else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

std::pair<uint32_t, size_t> next_cp(std::string_view text, size_t offset) {
    const uint8_t c0 = static_cast<uint8_t>(text[offset]);
    if (c0 < 0x80) return {c0, 1};
    if ((c0 >> 5) == 0x6 && offset + 1 < text.size()) {
        return {static_cast<uint32_t>(((c0 & 0x1F) << 6) | (static_cast<uint8_t>(text[offset + 1]) & 0x3F)), 2};
    }
    if ((c0 >> 4) == 0xE && offset + 2 < text.size()) {
        return {static_cast<uint32_t>(((c0 & 0x0F) << 12) | ((static_cast<uint8_t>(text[offset + 1]) & 0x3F) << 6) |
                                      (static_cast<uint8_t>(text[offset + 2]) & 0x3F)), 3};
    }
    if ((c0 >> 3) == 0x1E && offset + 3 < text.size()) {
        return {static_cast<uint32_t>(((c0 & 0x07) << 18) | ((static_cast<uint8_t>(text[offset + 1]) & 0x3F) << 12) |
                                      ((static_cast<uint8_t>(text[offset + 2]) & 0x3F) << 6) |
                                      (static_cast<uint8_t>(text[offset + 3]) & 0x3F)), 4};
    }
    return {0xFFFD, 1};
}

bool in_range(uint32_t cp, uint32_t first, uint32_t last) {
    return cp >= first && cp <= last;
}

bool is_space_cp(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
           cp == '\f' || cp == '\v' || cp == 0x0085 || cp == 0x00A0 ||
           cp == 0x1680 || in_range(cp, 0x2000, 0x200A) || cp == 0x2028 ||
           cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

bool is_number_cp(uint32_t cp) {
    static constexpr uint32_t ranges[][2] = {
        {0x0030, 0x0039}, {0x0660, 0x0669}, {0x06F0, 0x06F9},
        {0x07C0, 0x07C9}, {0x0966, 0x096F}, {0x09E6, 0x09EF},
        {0x0A66, 0x0A6F}, {0x0AE6, 0x0AEF}, {0x0B66, 0x0B6F},
        {0x0BE6, 0x0BEF}, {0x0C66, 0x0C6F}, {0x0CE6, 0x0CEF},
        {0x0D66, 0x0D6F}, {0x0DE6, 0x0DEF}, {0x0E50, 0x0E59},
        {0x0ED0, 0x0ED9}, {0x0F20, 0x0F29}, {0x1040, 0x1049},
        {0x17E0, 0x17E9}, {0x1810, 0x1819}, {0x1946, 0x194F},
        {0x19D0, 0x19D9}, {0x1A80, 0x1A89}, {0x1A90, 0x1A99},
        {0x1B50, 0x1B59}, {0x1BB0, 0x1BB9}, {0x1C40, 0x1C49},
        {0x1C50, 0x1C59}, {0xA620, 0xA629}, {0xA8D0, 0xA8D9},
        {0xA900, 0xA909}, {0xA9D0, 0xA9D9}, {0xA9F0, 0xA9F9},
        {0xAA50, 0xAA59}, {0xABF0, 0xABF9}, {0xFF10, 0xFF19},
    };
    for (const auto& range : ranges) {
        if (in_range(cp, range[0], range[1])) return true;
    }
    return false;
}

bool is_punctuation_or_symbol_cp(uint32_t cp) {
    if (cp < 128) return !std::isalnum(static_cast<unsigned char>(cp));
    if (in_range(cp, 0x2000, 0x206F) || in_range(cp, 0x20A0, 0x20CF) ||
        in_range(cp, 0x2100, 0x214F) || in_range(cp, 0x2190, 0x2BFF) ||
        in_range(cp, 0x2E00, 0x2E7F) || in_range(cp, 0x3001, 0x303F) ||
        in_range(cp, 0xFE10, 0xFE1F) || in_range(cp, 0xFE30, 0xFE6F) ||
        in_range(cp, 0xFF01, 0xFF0F) || in_range(cp, 0xFF1A, 0xFF20) ||
        in_range(cp, 0xFF3B, 0xFF40) || in_range(cp, 0xFF5B, 0xFF65) ||
        in_range(cp, 0x1F000, 0x1FAFF)) {
        return true;
    }
    switch (cp) {
        case 0x00A1: case 0x00A2: case 0x00A3: case 0x00A4: case 0x00A5:
        case 0x00A6: case 0x00A7: case 0x00A9: case 0x00AB: case 0x00AC:
        case 0x00AE: case 0x00B0: case 0x00B1: case 0x00B6: case 0x00B7:
        case 0x00BB: case 0x00BF: case 0x00D7: case 0x00F7:
        case 0x060C: case 0x061B: case 0x061F: case 0x066A: case 0x066B:
        case 0x066C: case 0x066D: case 0x06D4:
            return true;
        default:
            return false;
    }
}

int category(uint32_t cp) {
    if (is_space_cp(cp)) return 0;
    if (is_number_cp(cp)) return 2;
    if (is_punctuation_or_symbol_cp(cp)) return 3;
    // Remaining assigned Unicode code points are treated as letters/marks.
    // This captures the model's multilingual scripts without an ICU runtime.
    return 1;
}

bool ascii_case_equal(std::string_view text, size_t offset,
                      std::string_view expected) {
    if (offset + expected.size() > text.size()) return false;
    for (size_t i = 0; i < expected.size(); ++i) {
        unsigned char a = static_cast<unsigned char>(text[offset + i]);
        unsigned char b = static_cast<unsigned char>(expected[i]);
        if (a >= 'A' && a <= 'Z') a = static_cast<unsigned char>(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = static_cast<unsigned char>(b - 'A' + 'a');
        if (a != b) return false;
    }
    return true;
}

std::string pair_key(const std::string& a, const std::string& b) {
    std::string out;
    out.reserve(a.size() + b.size() + 1);
    out += a;
    out.push_back('\0');
    out += b;
    return out;
}

} // namespace

BpeTokenizer::BpeTokenizer(const std::string& tokenizer_json_path)
    : BpeTokenizer(tokenizer_json_path,
                   std::make_unique<Lfm2InstructChatTemplate>()) {}

BpeTokenizer::BpeTokenizer(const std::string& tokenizer_json_path,
                           std::unique_ptr<IChatTemplate> chat_template)
    : chat_template_(std::move(chat_template)) {
    if (!chat_template_) {
        throw std::invalid_argument("BpeTokenizer requires a chat template");
    }
    load(tokenizer_json_path);
}

BpeTokenizer::BpeTokenizer(FromGguf, const GgufFile& gguf,
                           std::unique_ptr<IChatTemplate> chat_template)
    : chat_template_(std::move(chat_template)) {
    if (!chat_template_) {
        throw std::invalid_argument("BpeTokenizer requires a chat template");
    }
    load_gguf(gguf);
}

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

void BpeTokenizer::load_gguf(const GgufFile& gguf) {
    const GgufValue& tokens = gguf.value("tokenizer.ggml.tokens");
    if (tokens.kind != GgufValueKind::Array ||
        tokens.array_kind != GgufValueKind::String) {
        throw std::runtime_error("gguf tokenizer.ggml.tokens missing or not string array");
    }
    const std::vector<std::string>& toks = tokens.array_strings;
    id_to_token_.resize(toks.size());
    for (size_t id = 0; id < toks.size(); ++id) {
        vocab_[toks[id]] = static_cast<int32_t>(id);
        id_to_token_[id] = toks[id];
    }

    const GgufValue& merges = gguf.value("tokenizer.ggml.merges");
    if (merges.kind != GgufValueKind::Array ||
        merges.array_kind != GgufValueKind::String) {
        throw std::runtime_error("gguf tokenizer.ggml.merges missing or not string array");
    }
    int32_t rank = 0;
    for (const std::string& line : merges.array_strings) {
        const size_t sep = line.find(' ');
        if (sep == std::string::npos) continue;
        merge_rank_[pair_key(line.substr(0, sep), line.substr(sep + 1))] = rank++;
    }

    // Special (CONTROL, type == 3) tokens drive verbatim matching in encode().
    if (gguf.has("tokenizer.ggml.token_type")) {
        const GgufValue& types = gguf.value("tokenizer.ggml.token_type");
        if (types.kind == GgufValueKind::Array) {
            for (size_t id = 0; id < types.array_integers.size() && id < toks.size();
                 ++id) {
                if (types.array_integers[id] == 3) {  // GGUF_TOKEN_TYPE_CONTROL
                    SpecialToken token{toks[id], static_cast<int32_t>(id)};
                    specials_.push_back(token);
                    special_ids_[token.id] = true;
                }
            }
            std::sort(specials_.begin(), specials_.end(),
                      [](const auto& a, const auto& b) {
                          return a.text.size() > b.text.size();
                      });
        }
    }

    if (gguf.has("tokenizer.ggml.bos_token_id")) {
        bos_id_ = static_cast<int32_t>(gguf.i64("tokenizer.ggml.bos_token_id"));
    }
    if (gguf.has("tokenizer.ggml.eos_token_id")) {
        eos_id_ = static_cast<int32_t>(gguf.i64("tokenizer.ggml.eos_token_id"));
    }

    init_byte_encoder();
}

void BpeTokenizer::load(const std::string& tokenizer_json_path) {
    const Json root = Json::parse_file(tokenizer_json_path);
    const Json& model = root["model"];
    if (model["type"].as_string() != "BPE") throw std::runtime_error("tokenizer model is not BPE");

    int32_t max_id = -1;
    for (const auto& [token, value] : model["vocab"].as_object()) {
        max_id = std::max(max_id, static_cast<int32_t>(value.as_i64()));
    }
    id_to_token_.resize(static_cast<size_t>(max_id + 1));
    for (const auto& [token, value] : model["vocab"].as_object()) {
        const int32_t id = static_cast<int32_t>(value.as_i64());
        vocab_[token] = id;
        id_to_token_[static_cast<size_t>(id)] = token;
    }

    int32_t rank = 0;
    for (const Json& merge : model["merges"].as_array()) {
        std::string left;
        std::string right;
        if (merge.is_string()) {
            const std::string& line = merge.as_string();
            const size_t sep = line.find(' ');
            if (sep == std::string::npos) continue;
            left = line.substr(0, sep);
            right = line.substr(sep + 1);
        } else {
            const auto& pair = merge.as_array();
            if (pair.size() != 2) continue;
            left = pair[0].as_string();
            right = pair[1].as_string();
        }
        merge_rank_[pair_key(left, right)] = rank++;
    }

    if (root.contains("added_tokens")) {
        for (const Json& item : root["added_tokens"].as_array()) {
            if (!item.contains("special") || !item["special"].as_bool()) continue;
            SpecialToken token{item["content"].as_string(), static_cast<int32_t>(item["id"].as_i64())};
            specials_.push_back(token);
            special_ids_[token.id] = true;
            if (token.text == "<|startoftext|>") bos_id_ = token.id;
            if (token.text == "<|im_end|>") eos_id_ = token.id;
        }
        std::sort(specials_.begin(), specials_.end(), [](const auto& a, const auto& b) {
            return a.text.size() > b.text.size();
        });
    }

    init_byte_encoder();
}

std::vector<std::string> BpeTokenizer::pretokenize(std::string_view text) const {
    std::vector<std::string> pieces;
    size_t i = 0;
    while (i < text.size()) {
        // GPT-2 contractions.
        if (text[i] == '\'' ) {
            static constexpr std::string_view suffixes[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
            bool matched = false;
            for (auto suffix : suffixes) {
                if (ascii_case_equal(text, i, suffix)) {
                    pieces.emplace_back(text.substr(i, suffix.size()));
                    i += suffix.size();
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
        }

        const auto [cp, len] = next_cp(text, i);
        if (is_space_cp(cp)) {
            // A single ASCII space is attached to the following non-space group, matching GPT-2's optional leading space.
            if (cp == ' ' && i + len < text.size()) {
                const auto [next, _] = next_cp(text, i + len);
                if (!is_space_cp(next)) {
                    const int cat = category(next);
                    size_t end = i + len;
                    while (end < text.size()) {
                        const auto [cur, cur_len] = next_cp(text, end);
                        if (category(cur) != cat) break;
                        end += cur_len;
                    }
                    pieces.emplace_back(text.substr(i, end - i));
                    i = end;
                    continue;
                }
            }
            size_t end = i + len;
            while (end < text.size()) {
                const auto [cur, cur_len] = next_cp(text, end);
                if (!is_space_cp(cur)) break;
                end += cur_len;
            }
            pieces.emplace_back(text.substr(i, end - i));
            i = end;
            continue;
        }

        const int cat = category(cp);
        size_t end = i + len;
        while (end < text.size()) {
            const auto [cur, cur_len] = next_cp(text, end);
            if (category(cur) != cat) break;
            end += cur_len;
        }
        pieces.emplace_back(text.substr(i, end - i));
        i = end;
    }
    return pieces;
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

std::vector<std::string> BpeTokenizer::bpe(std::string_view encoded_piece) const {
    std::vector<std::string> symbols;
    size_t i = 0;
    while (i < encoded_piece.size()) {
        const auto [_, len] = next_cp(encoded_piece, i);
        symbols.emplace_back(encoded_piece.substr(i, len));
        i += len;
    }
    if (symbols.size() < 2) return symbols;

    while (true) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        std::string best_left;
        std::string best_right;
        for (size_t j = 0; j + 1 < symbols.size(); ++j) {
            const auto it = merge_rank_.find(pair_key(symbols[j], symbols[j + 1]));
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_left = symbols[j];
                best_right = symbols[j + 1];
            }
        }
        if (best_rank == std::numeric_limits<int32_t>::max()) break;

        std::vector<std::string> merged;
        for (size_t j = 0; j < symbols.size();) {
            if (j + 1 < symbols.size() && symbols[j] == best_left && symbols[j + 1] == best_right) {
                merged.push_back(symbols[j] + symbols[j + 1]);
                j += 2;
            } else {
                merged.push_back(symbols[j]);
                ++j;
            }
        }
        symbols.swap(merged);
    }
    return symbols;
}

std::vector<int32_t> BpeTokenizer::encode_ordinary(std::string_view text) const {
    std::vector<int32_t> ids;
    for (const std::string& piece : pretokenize(text)) {
        const std::string encoded = byte_encode(piece);
        for (const std::string& token : bpe(encoded)) {
            const auto it = vocab_.find(token);
            if (it == vocab_.end()) throw std::runtime_error("BPE produced token absent from vocabulary");
            ids.push_back(it->second);
        }
    }
    return ids;
}

std::vector<int32_t> BpeTokenizer::encode(std::string_view text, bool add_bos) const {
    std::vector<int32_t> out;
    if (add_bos) out.push_back(bos_id_);
    size_t cursor = 0;
    while (cursor < text.size()) {
        const SpecialToken* best = nullptr;
        size_t best_pos = text.size();
        for (const auto& special : specials_) {
            const size_t pos = text.find(special.text, cursor);
            if (pos < best_pos) {
                best_pos = pos;
                best = &special;
            }
        }
        if (!best) {
            auto ordinary = encode_ordinary(text.substr(cursor));
            out.insert(out.end(), ordinary.begin(), ordinary.end());
            break;
        }
        if (best_pos > cursor) {
            auto ordinary = encode_ordinary(text.substr(cursor, best_pos - cursor));
            out.insert(out.end(), ordinary.begin(), ordinary.end());
        }
        out.push_back(best->id);
        cursor = best_pos + best->text.size();
    }
    return out;
}

std::string BpeTokenizer::decode(const std::vector<int32_t>& ids, bool skip_special) const {
    std::string encoded;
    for (int32_t id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) continue;
        if (skip_special && special_ids_.contains(id)) continue;
        encoded += id_to_token_[static_cast<size_t>(id)];
    }
    return byte_decode(encoded);
}

std::string BpeTokenizer::format_chat(std::string_view user_prompt, std::string_view system_prompt) const {
    return chat_template_->format(user_prompt, system_prompt);
}

} // namespace lfm
