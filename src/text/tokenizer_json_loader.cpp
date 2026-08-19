#include "celeg/text/tokenizer_definition.hpp"

#include "celeg/checkpoint/tokenizer.hpp"
#include "celeg/checkpoint/formats/json.hpp"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <stdexcept>

namespace celeg {
namespace {

const Json* find_regex_node(const Json& value) {
    if (value.is_object()) {
        if (value.contains("Regex") && value["Regex"].is_string()) {
            return &value["Regex"];
        }
        for (const auto& [_, child] : value.as_object()) {
            if (const Json* found = find_regex_node(child)) return found;
        }
    } else if (value.is_array()) {
        for (const Json& child : value.as_array()) {
            if (const Json* found = find_regex_node(child)) return found;
        }
    }
    return nullptr;
}

void add_reasoning_delimiters(TokenizerDefinition& definition) {
    for (const std::string_view text : {"<think>", "</think>"}) {
        const auto it = std::find(definition.tokens.begin(), definition.tokens.end(),
                                  std::string(text));
        if (it == definition.tokens.end()) continue;
        const int32_t id = static_cast<int32_t>(std::distance(definition.tokens.begin(), it));
        const auto existing = std::find_if(definition.special_tokens.begin(),
                                           definition.special_tokens.end(),
            [text](const TokenizerSpecialToken& token) { return token.text == text; });
        if (existing == definition.special_tokens.end()) {
            definition.special_tokens.push_back({std::string(text), id});
        }
    }
}

std::optional<std::string> configured_token_text(const Json& config,
                                                 std::string_view key) {
    if (!config.contains(key)) return std::nullopt;
    const Json& value = config[key];
    if (value.is_string()) return value.as_string();
    if (value.is_object() && value.contains("content") && value["content"].is_string()) {
        return value["content"].as_string();
    }
    return std::nullopt;
}

void apply_tokenizer_config(TokenizerDefinition& definition, const std::string& path) {
    const std::filesystem::path config_path =
        std::filesystem::path(path).parent_path() / "tokenizer_config.json";
    if (!std::filesystem::is_regular_file(config_path)) return;
    const Json config = Json::parse_file(config_path.string());
    const auto id_for = [&definition](const std::optional<std::string>& text) {
        if (!text) return std::optional<int32_t>{};
        const auto it = std::find_if(definition.special_tokens.begin(),
                                     definition.special_tokens.end(),
            [&text](const TokenizerSpecialToken& token) { return token.text == *text; });
        if (it == definition.special_tokens.end()) return std::optional<int32_t>{};
        return std::optional<int32_t>{it->id};
    };
    if (const auto bos = id_for(configured_token_text(config, "bos_token"))) {
        definition.bos_id = *bos;
        definition.has_bos = true;
    }
    if (const auto eos = id_for(configured_token_text(config, "eos_token"))) {
        definition.eos_id = *eos;
    }
    if (const auto pad = id_for(configured_token_text(config, "pad_token"))) {
        definition.pad_id = *pad;
    }
}

}

TokenizerDefinition load_tokenizer_definition_json(const std::string& path) {
    return load_tokenizer_definition_json(path, std::nullopt);
}

TokenizerDefinition load_tokenizer_definition_json(const std::string& path, std::optional<int> target_vocab_size) {
    const Json root = Json::parse_file(path);
    const Json& model = root["model"];
    if (model["type"].as_string() != "BPE") {
        throw std::runtime_error("tokenizer model is not BPE");
    }

    TokenizerDefinition definition;
    int32_t max_id = -1;
    for (const auto& [_, value] : model["vocab"].as_object()) {
        max_id = std::max(max_id, static_cast<int32_t>(value.as_i64()));
    }
    definition.tokens.resize(static_cast<size_t>(max_id + 1));
    for (const auto& [token, value] : model["vocab"].as_object()) {
        definition.tokens[static_cast<size_t>(value.as_i64())] = token;
    }
    for (const Json& merge : model["merges"].as_array()) {
        if (merge.is_string()) {
            definition.merges.push_back(merge.as_string());
        } else {
            const auto& pair = merge.as_array();
            if (pair.size() == 2) {
                definition.merges.push_back(pair[0].as_string() + " " + pair[1].as_string());
            }
        }
    }

    if (root.contains("added_tokens")) {
        for (const Json& item : root["added_tokens"].as_array()) {
            TokenizerSpecialToken token{item["content"].as_string(),
                                        static_cast<int32_t>(item["id"].as_i64())};
            if (token.id < 0) throw std::runtime_error("tokenizer added token has a negative id");
            if (static_cast<size_t>(token.id) >= definition.tokens.size()) {
                definition.tokens.resize(static_cast<size_t>(token.id) + 1);
            }
            definition.tokens[static_cast<size_t>(token.id)] = token.text;
            const bool is_special = item.contains("special") && item["special"].as_bool();
            token.skip_on_decode = is_special;
            definition.special_tokens.push_back(std::move(token));
        }
    }

    definition.byte_fallback = model.contains("byte_fallback") && model["byte_fallback"].as_bool();
    definition.normalization = definition.byte_fallback &&
        std::find(definition.tokens.begin(), definition.tokens.end(), "▁") != definition.tokens.end()
        ? TokenizerNormalizationKind::SentencePieceSpace
        : TokenizerNormalizationKind::None;
    if (definition.normalization == TokenizerNormalizationKind::SentencePieceSpace) {
        definition.pre_tokenizer = TokenizerPreTokenizerKind::RawUtf8;
    }
    if (root.contains("pre_tokenizer")) {
        if (const Json* regex = find_regex_node(root["pre_tokenizer"])) {
            const std::string pattern = regex->as_string();
            if (pattern.find("?+\\p{L}+") != std::string::npos) {
                definition.pre_tokenizer = TokenizerPreTokenizerKind::ByteLevelRegex;
            } else if (pattern.find("\\p{N}{1,3}") != std::string::npos) {
                definition.pre_tokenizer = TokenizerPreTokenizerKind::NumericTriplets;
            } else if (definition.pre_tokenizer != TokenizerPreTokenizerKind::RawUtf8) {
                definition.pre_tokenizer = TokenizerPreTokenizerKind::NumericRuns;
            }
        }
    }

    add_reasoning_delimiters(definition);
    int32_t endoftext_id = -1;
    bool explicit_bos = false;
    bool explicit_eos = false;
    for (const auto& token : definition.special_tokens) {
        if (token.text == "<|startoftext|>" || token.text == "<|start_of_text|>" ||
            token.text == "<|begin_of_text|>" || token.text == "<bos>" ||
            token.text == "<|end_of_text|>") {
            definition.bos_id = token.id;
            definition.has_bos = true;
            explicit_bos = true;
        }
        if (token.text == "<|im_end|>" || token.text == "<eos>" ||
            token.text == "<|end_of_text|>") definition.eos_id = token.id;
        if (token.text == "<|im_end|>" || token.text == "<eos>" ||
            token.text == "<|end_of_text|>") explicit_eos = true;
        if (token.text == "<|endoftext|>") endoftext_id = token.id;
        if (token.text == "<pad>" || token.text == "<|pad|>") definition.pad_id = token.id;
    }
    if (endoftext_id >= 0 && !explicit_bos && !explicit_eos) {
        definition.bos_id = endoftext_id;
        definition.eos_id = endoftext_id;
        definition.has_bos = true;
    }
    apply_tokenizer_config(definition, path);
    definition.tokenizer_vocab_size = static_cast<int>(definition.tokens.size());
    return definition;
}

TokenizerDefinition resolve_tokenizer_definition(
    const TokenizerData& data,
    const std::vector<TokenizerPreTokenizerRule>& rules) {
    if (data.tokens.empty()) {
        throw std::runtime_error("checkpoint tokenizer data is incomplete");
    }
    TokenizerDefinition definition;
    definition.tokens = data.tokens;
    definition.merges = data.merges;
    definition.bos_id = data.bos_id;
    definition.eos_id = data.eos_id;
    definition.pad_id = data.pad_id;
    definition.has_bos = data.has_bos;
    for (const TokenizerPreTokenizerRule& rule : rules) {
        const bool matches = rule.contains
            ? data.pre_tokenizer.find(rule.identifier) != std::string::npos
            : data.pre_tokenizer == rule.identifier;
        if (matches) {
            definition.pre_tokenizer = rule.behavior;
            break;
        }
    }
    if (rules.empty()) {
        if (data.pre_tokenizer == "numeric_triplets") {
            definition.pre_tokenizer = TokenizerPreTokenizerKind::NumericTriplets;
        } else if (data.pre_tokenizer == "numeric_runs") {
            definition.pre_tokenizer = TokenizerPreTokenizerKind::NumericRuns;
        } else if (data.pre_tokenizer == "raw_utf8") {
            definition.pre_tokenizer = TokenizerPreTokenizerKind::RawUtf8;
        }
    }
    for (size_t id = 0; id < data.token_types.size() && id < data.tokens.size(); ++id) {
        if (data.token_types[id] == 3) {
            definition.special_tokens.push_back({data.tokens[id], static_cast<int32_t>(id)});
        }
    }
    if (definition.pre_tokenizer == TokenizerPreTokenizerKind::NumericTriplets) {
        add_reasoning_delimiters(definition);
    }
    definition.tokenizer_vocab_size = static_cast<int>(definition.tokens.size());
    return definition;
}

}
