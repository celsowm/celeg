#include "celeg/text/tokenizer.hpp"
#include "detail.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace celeg {
using namespace tokenizer_detail;

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

}
