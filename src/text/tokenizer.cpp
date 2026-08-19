#include "celeg/text/tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <queue>
#include <stdexcept>
#include <utility>

namespace celeg {

std::optional<int32_t> BpeTokenizer::token_id(std::string_view text) const {
    const auto it = vocab_.find(std::string(text));
    if (it == vocab_.end()) return std::nullopt;
    return it->second;
}
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
    const auto continuation = [&](size_t index) {
        return index < text.size() &&
            (static_cast<uint8_t>(text[index]) & 0xC0) == 0x80;
    };
    if ((c0 >> 5) == 0x6 && continuation(offset + 1)) {
        const uint32_t cp = static_cast<uint32_t>(((c0 & 0x1F) << 6) |
            (static_cast<uint8_t>(text[offset + 1]) & 0x3F));
        return cp >= 0x80 ? std::pair<uint32_t, size_t>{cp, 2} : std::pair<uint32_t, size_t>{0xFFFD, 1};
    }
    if ((c0 >> 4) == 0xE && continuation(offset + 1) && continuation(offset + 2)) {
        const uint32_t cp = static_cast<uint32_t>(((c0 & 0x0F) << 12) |
            ((static_cast<uint8_t>(text[offset + 1]) & 0x3F) << 6) |
            (static_cast<uint8_t>(text[offset + 2]) & 0x3F));
        return cp >= 0x800 && !(cp >= 0xD800 && cp <= 0xDFFF)
            ? std::pair<uint32_t, size_t>{cp, 3} : std::pair<uint32_t, size_t>{0xFFFD, 1};
    }
    if ((c0 >> 3) == 0x1E && continuation(offset + 1) && continuation(offset + 2) && continuation(offset + 3)) {
        const uint32_t cp = static_cast<uint32_t>(((c0 & 0x07) << 18) |
            ((static_cast<uint8_t>(text[offset + 1]) & 0x3F) << 12) |
            ((static_cast<uint8_t>(text[offset + 2]) & 0x3F) << 6) |
            (static_cast<uint8_t>(text[offset + 3]) & 0x3F));
        return cp >= 0x10000 && cp <= 0x10FFFF
            ? std::pair<uint32_t, size_t>{cp, 4} : std::pair<uint32_t, size_t>{0xFFFD, 1};
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

}

BpeTokenizer::BpeTokenizer(const TokenizerDefinition& definition) {
    load_definition(definition);
}

std::vector<std::string> BpeTokenizer::pretokenize_byte_level_regex(
    std::string_view text) const {
    std::vector<std::string> pieces;
    size_t i = 0;
    while (i < text.size()) {
        if (text[i] == '\'' && i + 1 < text.size()) {
            static constexpr std::string_view suffixes[] = {
                "'s", "'t", "'d", "'m", "'ll", "'ve", "'re"};
            bool matched = false;
            for (const std::string_view suffix : suffixes) {
                if (i + suffix.size() <= text.size() &&
                    ascii_case_equal(text, i, suffix)) {
                    pieces.emplace_back(text.substr(i, suffix.size()));
                    i += suffix.size();
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
        }

        const auto [cp, len] = next_cp(text, i);
        const int kind = category(cp);
        if (kind == 1) {
            size_t end = i + len;
            while (end < text.size()) {
                const auto [next, next_len] = next_cp(text, end);
                if (category(next) != 1) break;
                end += next_len;
            }
            pieces.emplace_back(text.substr(i, end - i));
            i = end;
            continue;
        }
        if (kind == 0 && cp == ' ' && i + len < text.size()) {
            const auto [next, next_len] = next_cp(text, i + len);
            if (category(next) == 1) {
                size_t end = i + len + next_len;
                while (end < text.size()) {
                    const auto [cur, cur_len] = next_cp(text, end);
                    if (category(cur) != 1) break;
                    end += cur_len;
                }
                pieces.emplace_back(text.substr(i, end - i));
                i = end;
                continue;
            }
        }
        if (kind == 3 || (kind == 0 && cp == ' ' && i + len < text.size() &&
                          category(next_cp(text, i + len).first) == 3)) {
            const size_t start = i;
            size_t punctuation = i;
            if (kind == 0) punctuation += len;
            const auto [punctuation_cp, punctuation_len] = next_cp(text, punctuation);
            if (kind == 3 && punctuation_cp == cp && punctuation_len == len &&
                punctuation + punctuation_len < text.size() &&
                category(next_cp(text, punctuation + punctuation_len).first) == 1) {
                size_t end = punctuation + punctuation_len;
                while (end < text.size()) {
                    const auto [next, next_len] = next_cp(text, end);
                    if (category(next) != 1) break;
                    end += next_len;
                }
                pieces.emplace_back(text.substr(start, end - start));
                i = end;
                continue;
            }
            size_t end = punctuation + punctuation_len;
            while (end < text.size()) {
                const auto [next, next_len] = next_cp(text, end);
                if (category(next) != 3) break;
                end += next_len;
            }
            while (end < text.size()) {
                const auto [next, next_len] = next_cp(text, end);
                if (next != '\n' && next != '\r') break;
                end += next_len;
            }
            pieces.emplace_back(text.substr(start, end - start));
            i = end;
            continue;
        }
        if (kind == 0) {
            size_t end = i + len;
            while (end < text.size()) {
                const auto [next, next_len] = next_cp(text, end);
                if (!is_space_cp(next)) break;
                end += next_len;
            }
            pieces.emplace_back(text.substr(i, end - i));
            i = end;
            continue;
        }
        if (kind == 2) {
            pieces.emplace_back(text.substr(i, len));
            i += len;
            continue;
        }
        size_t end = i + len;
        while (end < text.size()) {
            const auto [next, next_len] = next_cp(text, end);
            if (!is_space_cp(next)) end += next_len;
            else break;
        }
        pieces.emplace_back(text.substr(i, end - i));
        i = end;
    }
    return pieces;
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

void BpeTokenizer::load_definition(const TokenizerDefinition& definition) {
    if (definition.tokens.empty()) {
        throw std::runtime_error("tokenizer definition is incomplete");
    }
    const std::vector<std::string>& toks = definition.tokens;
    id_to_token_.resize(toks.size());
    for (size_t id = 0; id < toks.size(); ++id) {
        if (!toks[id].empty()) vocab_[toks[id]] = static_cast<int32_t>(id);
        id_to_token_[id] = toks[id];
    }

    int32_t rank = 0;
    for (const std::string& line : definition.merges) {
        const size_t sep = line.find(' ');
        if (sep == std::string::npos) continue;
        merge_rank_[pair_key(line.substr(0, sep), line.substr(sep + 1))] = rank++;
    }

    if (definition.merges.empty() && !definition.scores.empty()) {
        id_score_ = definition.scores;
        id_score_.resize(id_to_token_.size(), 0.0f);
    }

    for (const TokenizerSpecialToken& definition_token : definition.special_tokens) {
        SpecialToken token{definition_token.text, definition_token.id};
        specials_.push_back(token);
        if (definition_token.skip_on_decode) special_ids_[token.id] = true;
    }
    std::sort(specials_.begin(), specials_.end(),
              [](const auto& a, const auto& b) {
                  return a.text.size() > b.text.size();
              });

    bos_id_ = definition.bos_id;
    eos_id_ = definition.eos_id;
    pad_id_ = definition.pad_id;
    has_bos_ = definition.has_bos;

    pre_tokenizer_ = definition.pre_tokenizer;
    normalization_ = definition.normalization;
    byte_fallback_ = definition.byte_fallback;
    spm_score_mode_ = normalization_ == TokenizerNormalizationKind::SentencePieceSpace &&
        !id_score_.empty();
    init_byte_encoder();
}

std::vector<std::string> BpeTokenizer::pretokenize(std::string_view text) const {
    std::vector<std::string> pieces;
    if (pre_tokenizer_ == TokenizerPreTokenizerKind::RawUtf8) {
        if (!text.empty()) pieces.emplace_back(text);
        return pieces;
    }
    if (pre_tokenizer_ == TokenizerPreTokenizerKind::ByteLevelRegex) {
        return pretokenize_byte_level_regex(text);
    }
    size_t i = 0;
    while (i < text.size()) {
        if (text[i] == '\'' ) {
            static constexpr std::string_view suffixes[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
            bool matched = false;
            for (auto suffix : suffixes) {
                const bool case_insensitive =
                    pre_tokenizer_ != TokenizerPreTokenizerKind::NumericRuns;
                const bool suffix_match = case_insensitive
                    ? ascii_case_equal(text, i, suffix)
                    : text.substr(i, suffix.size()) == suffix;
                if (suffix_match) {
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
            if (cp == ' ' && i + len < text.size()) {
                    const auto [next, _] = next_cp(text, i + len);
                    if (!is_space_cp(next)) {
                        const int cat = category(next);
                        size_t end = i + len;
                        size_t category_count = 0;
                        while (end < text.size()) {
                            const auto [cur, cur_len] = next_cp(text, end);
                            if (category(cur) != cat) break;
                            end += cur_len;
                            if (cat == 2 && pre_tokenizer_ == TokenizerPreTokenizerKind::NumericTriplets &&
                                ++category_count >= 3) break;
                        }
                        if (cat == 3 && pre_tokenizer_ == TokenizerPreTokenizerKind::NumericTriplets) {
                            while (end < text.size()) {
                                const auto [cur, cur_len] = next_cp(text, end);
                                if (cur != '\r' && cur != '\n') break;
                                end += cur_len;
                            }
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
        if (pre_tokenizer_ == TokenizerPreTokenizerKind::NumericTriplets && cat == 3 &&
            i + len < text.size()) {
            const auto [next, next_len] = next_cp(text, i + len);
            if (category(next) == 1) {
                size_t end = i + len + next_len;
                while (end < text.size()) {
                    const auto [cur, cur_len] = next_cp(text, end);
                    if (category(cur) != 1) break;
                    end += cur_len;
                }
                pieces.emplace_back(text.substr(i, end - i));
                i = end;
                continue;
            }
        }
        if (pre_tokenizer_ == TokenizerPreTokenizerKind::NumericTriplets && cat == 3) {
            size_t end = i + len;
            while (end < text.size()) {
                const auto [cur, cur_len] = next_cp(text, end);
                if (category(cur) == 3) end += cur_len;
                else break;
            }
            while (end < text.size()) {
                const auto [cur, cur_len] = next_cp(text, end);
                if (cur != '\r' && cur != '\n') break;
                end += cur_len;
            }
            pieces.emplace_back(text.substr(i, end - i));
            i = end;
            continue;
        }
        if (cat == 2 && pre_tokenizer_ == TokenizerPreTokenizerKind::NumericTriplets) {
            size_t end = i;
            while (end < text.size()) {
                const auto [cur, cur_len] = next_cp(text, end);
                if (category(cur) != cat) break;
                end += cur_len;
                if ((end - i) >= 3) break;
            }
            pieces.emplace_back(text.substr(i, end - i));
            i = end;
            continue;
        }
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
    return bpe_symbols(std::move(symbols));
}

std::vector<std::string> BpeTokenizer::bpe_symbols(std::vector<std::string> symbols) const {
    if (symbols.size() < 2) return symbols;

    struct Node {
        std::string text;
        int prev = -1;
        int next = -1;
        bool alive = true;
    };
    struct Candidate {
        int32_t rank;
        int left;
        int right;
    };
    struct CandidateGreater {
        bool operator()(const Candidate& a, const Candidate& b) const {
            return a.rank > b.rank ||
                   (a.rank == b.rank && a.left > b.left);
        }
    };

    std::vector<Node> nodes;
    nodes.reserve(symbols.size());
    for (size_t i = 0; i < symbols.size(); ++i) {
        nodes.push_back(Node{std::move(symbols[i]), static_cast<int>(i) - 1,
                             i + 1 < symbols.size() ? static_cast<int>(i) + 1 : -1,
                             true});
    }

    std::priority_queue<Candidate, std::vector<Candidate>, CandidateGreater> queue;
    const auto add_candidate = [&](int left, int right,
                                   auto& candidate_queue) {
        if (left < 0 || right < 0 || !nodes[left].alive || !nodes[right].alive ||
            nodes[left].next != right) return;
        const auto it = merge_rank_.find(pair_key(nodes[left].text, nodes[right].text));
        if (it != merge_rank_.end()) candidate_queue.push(Candidate{it->second, left, right});
    };
    for (size_t i = 1; i < nodes.size(); ++i) {
        add_candidate(static_cast<int>(i) - 1, static_cast<int>(i), queue);
    }

    while (!queue.empty()) {
        const Candidate candidate = queue.top();
        queue.pop();
        if (!nodes[candidate.left].alive || !nodes[candidate.right].alive ||
            nodes[candidate.left].next != candidate.right ||
            nodes[candidate.right].prev != candidate.left) {
            continue;
        }
        const auto current = merge_rank_.find(
            pair_key(nodes[candidate.left].text, nodes[candidate.right].text));
        if (current == merge_rank_.end() || current->second != candidate.rank) continue;

        Node& left = nodes[candidate.left];
        Node& right = nodes[candidate.right];
        const int previous = left.prev;
        const int following = right.next;
        left.text += right.text;
        left.next = following;
        right.alive = false;
        if (following >= 0) nodes[following].prev = candidate.left;
        add_candidate(previous, candidate.left, queue);
        add_candidate(candidate.left, following, queue);
    }

    std::vector<std::string> result;
    for (int index = 0; index >= 0;) {
        if (nodes[index].alive) result.push_back(std::move(nodes[index].text));
        index = nodes[index].next;
    }
    return result;
}

// SentencePiece's own tokenizer (llama.cpp's "llama" GGUF vocab type) has no
// explicit merge-rank table: any substring that is itself a vocabulary entry
// is a valid merge, prioritized by that entry's own score (higher merges
// first; ties broken toward the leftmost position). This mirrors
// llm_tokenizer_spm_session::tokenize() in llama.cpp's llama-vocab.cpp.
std::vector<int32_t> BpeTokenizer::spm_score_tokenize(std::string_view normalized) const {
    struct Node {
        std::string text;
        int prev = -1;
        int next = -1;
        bool alive = true;
    };
    std::vector<Node> nodes;
    for (size_t offset = 0; offset < normalized.size();) {
        const auto [_, len] = next_cp(normalized, offset);
        const int index = static_cast<int>(nodes.size());
        nodes.push_back(Node{
            std::string(normalized.substr(offset, len)),
            index - 1,
            -1,
            true});
        if (index > 0) nodes[static_cast<size_t>(index - 1)].next = index;
        offset += len;
    }
    if (nodes.empty()) return {};

    struct Bigram {
        float score;
        int left;
        int right;
        size_t size;
    };
    struct BigramLess {
        bool operator()(const Bigram& a, const Bigram& b) const {
            return a.score < b.score || (a.score == b.score && a.left > b.left);
        }
    };
    std::priority_queue<Bigram, std::vector<Bigram>, BigramLess> queue;
    const auto try_add = [&](int left, int right) {
        if (left < 0 || right < 0 || !nodes[static_cast<size_t>(left)].alive ||
            !nodes[static_cast<size_t>(right)].alive ||
            nodes[static_cast<size_t>(left)].next != right) {
            return;
        }
        const std::string text =
            nodes[static_cast<size_t>(left)].text + nodes[static_cast<size_t>(right)].text;
        const auto it = vocab_.find(text);
        if (it == vocab_.end()) return;
        const float score = static_cast<size_t>(it->second) < id_score_.size()
            ? id_score_[static_cast<size_t>(it->second)] : 0.0f;
        queue.push(Bigram{score, left, right, text.size()});
    };
    for (size_t i = 1; i < nodes.size(); ++i) {
        try_add(static_cast<int>(i) - 1, static_cast<int>(i));
    }
    while (!queue.empty()) {
        const Bigram bigram = queue.top();
        queue.pop();
        Node& left = nodes[static_cast<size_t>(bigram.left)];
        Node& right = nodes[static_cast<size_t>(bigram.right)];
        if (!left.alive || !right.alive || left.text.size() + right.text.size() != bigram.size) {
            continue;
        }
        const int previous = left.prev;
        const int following = right.next;
        left.text += right.text;
        left.next = following;
        right.alive = false;
        if (following >= 0) nodes[static_cast<size_t>(following)].prev = bigram.left;
        try_add(previous, bigram.left);
        try_add(bigram.left, following);
    }

    std::vector<int32_t> ids;
    for (int index = 0; index >= 0;) {
        const Node& node = nodes[static_cast<size_t>(index)];
        const auto it = vocab_.find(node.text);
        if (it != vocab_.end()) {
            ids.push_back(it->second);
        } else if (byte_fallback_) {
            for (unsigned char byte : node.text) {
                char fallback[7] = {};
                std::snprintf(fallback, sizeof(fallback), "<0x%02X>", byte);
                const auto fallback_it = vocab_.find(fallback);
                if (fallback_it == vocab_.end()) {
                    throw std::runtime_error(
                        "SentencePiece byte-fallback token absent from vocabulary: " +
                        std::string(fallback));
                }
                ids.push_back(fallback_it->second);
            }
        } else {
            throw std::runtime_error(
                "SentencePiece produced a symbol absent from vocabulary: " + node.text);
        }
        index = node.next;
    }
    return ids;
}

std::vector<int32_t> BpeTokenizer::encode_ordinary(std::string_view text) const {
    std::vector<int32_t> ids;
    if (normalization_ == TokenizerNormalizationKind::SentencePieceSpace) {
        std::string normalized;
        for (size_t offset = 0; offset < text.size();) {
            const auto [cp, len] = next_cp(text, offset);
            if (cp == ' ') append_utf8(normalized, 0x2581);
            else normalized.append(text.substr(offset, len));
            offset += len;
        }
        if (spm_score_mode_) {
            return spm_score_tokenize(normalized);
        }
        size_t offset = 0;
        while (offset < normalized.size()) {
            const bool starts_marker = normalized.compare(offset, std::string("▁").size(), "▁") == 0;
            size_t end = offset;
            if (starts_marker) end += std::string("▁").size();
            while (end < normalized.size()) {
                const auto [cp, len] = next_cp(normalized, end);
                if (cp == 0x2581) break;
                end += len;
            }
            std::vector<std::string> symbols;
            for (size_t cursor = offset; cursor < end;) {
                const auto [cp, len] = next_cp(normalized, cursor);
                std::string symbol(normalized.substr(cursor, len));
                if (!vocab_.contains(symbol) && byte_fallback_) {
                    for (unsigned char byte : std::string(normalized.substr(cursor, len))) {
                        char fallback[7] = {};
                        std::snprintf(fallback, sizeof(fallback), "<0x%02X>", byte);
                        symbols.emplace_back(fallback);
                    }
                } else {
                    symbols.push_back(std::move(symbol));
                }
                cursor += len;
            }
            for (const std::string& token : bpe_symbols(std::move(symbols))) {
                const auto it = vocab_.find(token);
                if (it == vocab_.end()) {
                    throw std::runtime_error("SentencePiece BPE produced token absent from vocabulary: " + token);
                }
                ids.push_back(it->second);
            }
            offset = end;
        }
        return ids;
    }
    for (const std::string& piece : pretokenize(text)) {
        const std::string encoded = byte_encode(piece);
        if (pre_tokenizer_ == TokenizerPreTokenizerKind::NumericTriplets) {
            const auto direct = vocab_.find(encoded);
            if (direct != vocab_.end()) {
                ids.push_back(direct->second);
                continue;
            }
        }
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
    if (add_bos && has_bos_) out.push_back(bos_id_);
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
    if (normalization_ == TokenizerNormalizationKind::SentencePieceSpace) {
        std::string decoded;
        for (int32_t id : ids) {
            if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) continue;
            if (skip_special && special_ids_.contains(id)) continue;
            const std::string& token = id_to_token_[static_cast<size_t>(id)];
            if (token.size() == 6 && token.rfind("<0x", 0) == 0 && token.back() == '>') {
                const unsigned value = std::stoul(token.substr(3, 2), nullptr, 16);
                decoded.push_back(static_cast<char>(value));
            } else {
                decoded += token;
            }
        }
        std::string marker = "▁";
        for (size_t offset = 0; (offset = decoded.find(marker, offset)) != std::string::npos;) {
            decoded.replace(offset, marker.size(), " ");
            ++offset;
        }
        return decoded;
    }
    std::string encoded;
    for (int32_t id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) continue;
        if (skip_special && special_ids_.contains(id)) continue;
        encoded += id_to_token_[static_cast<size_t>(id)];
    }
    return byte_decode(encoded);
}

std::string BpeTokenizer::decode_token(int32_t id, bool skip_special) const {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) return {};
    if (skip_special && special_ids_.contains(id)) return {};
    const std::string& token = id_to_token_[static_cast<size_t>(id)];
    if (normalization_ == TokenizerNormalizationKind::SentencePieceSpace) {
        std::string decoded;
        if (token.size() == 6 && token.rfind("<0x", 0) == 0 && token.back() == '>') {
            const unsigned value = std::stoul(token.substr(3, 2), nullptr, 16);
            decoded.push_back(static_cast<char>(value));
        } else {
            decoded = token;
        }
        for (size_t offset = 0; (offset = decoded.find("▁", offset)) != std::string::npos;) {
            decoded.replace(offset, std::string("▁").size(), " ");
            ++offset;
        }
        return decoded;
    }
    return byte_decode(token);
}

}
