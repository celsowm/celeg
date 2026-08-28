#include "celeg/text/tokenizer.hpp"
#include "detail.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace celeg {
using namespace tokenizer_detail;

std::optional<int32_t> BpeTokenizer::token_id(std::string_view text) const {
    const auto it = vocab_.find(std::string(text));
    if (it == vocab_.end()) return std::nullopt;
    return it->second;
}

BpeTokenizer::BpeTokenizer(const TokenizerDefinition& definition) {
    load_definition(definition);
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
