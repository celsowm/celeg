#include "celeg/checkpoint/catalog.hpp"

#include "celeg/checkpoint/formats/json.hpp"
#include "celeg/checkpoint/formats/gguf.hpp"
#include "celeg/checkpoint/repositories/gguf.hpp"
#include "formats/detail/gguf_tokenizer_metadata.hpp"
#include "celeg/checkpoint/repositories/safetensors.hpp"

#include <algorithm>
#include <fstream>
#include <string>
#include <stdexcept>

namespace celeg {
namespace {

std::filesystem::path config_path_for(const std::filesystem::path& path) {
    const auto root = std::filesystem::is_directory(path)
        ? path : path.parent_path();
    return root / "config.json";
}

class GgufFormat final : public ICheckpointFormat {
public:
    std::string_view id() const override { return "gguf"; }

    bool matches(const std::filesystem::path& path) const override {
        return !std::filesystem::is_directory(path) &&
               path.extension() == ".gguf";
    }

    CheckpointView open(const std::filesystem::path& path) const override {
        auto file = std::make_shared<GgufFile>(path.string());
        CheckpointView result;
        result.path = path;
        result.metadata = CheckpointMetadata::from_gguf(*file);
        auto repository = std::make_shared<GgufRepository>(file);
        result.tokenizer = std::make_shared<TokenizerData>(
            read_gguf_tokenizer_data(*file));
        result.repository = std::move(repository);
        return result;
    }
};

class SafeTensorsFormat final : public ICheckpointFormat {
public:
    std::string_view id() const override { return "safetensors"; }

    bool matches(const std::filesystem::path& path) const override {
        if (std::filesystem::is_regular_file(path)) {
            return path.extension() == ".safetensors";
        }
        if (!std::filesystem::is_directory(path)) return false;
        return std::filesystem::exists(path / "model.safetensors") ||
               std::filesystem::exists(path / "model.safetensors.index.json");
    }

    CheckpointView open(const std::filesystem::path& path) const override {
        const auto config = config_path_for(path);
        if (!std::filesystem::exists(config)) {
            throw std::runtime_error(
                "config.json not found alongside checkpoint: " + config.string());
        }
        CheckpointView result;
        result.path = path;
        result.metadata = CheckpointMetadata::from_json(Json::parse_file(config.string()));
        const auto root = std::filesystem::is_directory(path) ? path : path.parent_path();
        const auto chat_template = root / "chat_template.jinja";
        if (std::filesystem::is_regular_file(chat_template) &&
            !result.metadata.contains("chat_template")) {
            std::ifstream stream(chat_template, std::ios::binary);
            if (!stream) {
                throw std::runtime_error(
                    "cannot read chat template alongside checkpoint: " +
                    chat_template.string());
            }
            result.metadata.values["chat_template"] = std::string(
                std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
        }
        const auto tokenizer_config = root / "tokenizer_config.json";
        if (std::filesystem::is_regular_file(tokenizer_config) &&
            !result.metadata.contains("chat_template") &&
            !result.metadata.contains("tokenizer.chat_template")) {
            std::ifstream stream(tokenizer_config, std::ios::binary);
            if (stream) {
                try {
                    const auto tok_json = Json::parse_file(tokenizer_config.string());
                    if (tok_json.contains("chat_template")) {
                        result.metadata.values["tokenizer.chat_template"] = tok_json["chat_template"].as_string();
                    }
                } catch (...) {
                }
            }
        }
        // generation_config.json is HF's authoritative source for the token
        // ids actually used at generation time: bos/eos there can differ
        // from (and be more complete than -- e.g. a multi-id eos_token_id
        // list) config.json's, so when present it takes priority over
        // config.json's bos_token_id/eos_token_id at the unscoped metadata
        // key (which alias resolution checks before any text_config.*
        // fallback).
        const auto generation_config = root / "generation_config.json";
        if (std::filesystem::is_regular_file(generation_config)) {
            try {
                const auto gen_json = Json::parse_file(generation_config.string());
                const auto merge_token_field = [&](std::string_view key) {
                    if (!gen_json.contains(key)) return;
                    const Json& value = gen_json[key];
                    if (value.is_number()) {
                        result.metadata.values[std::string(key)] = value.as_i64();
                    } else if (value.is_array()) {
                        std::vector<int64_t> ids;
                        for (const Json& item : value.as_array()) {
                            if (item.is_number()) ids.push_back(item.as_i64());
                        }
                        if (!ids.empty()) result.metadata.values[std::string(key)] = std::move(ids);
                    }
                };
                merge_token_field("bos_token_id");
                merge_token_field("eos_token_id");
            } catch (...) {
            }
        }
        result.repository = std::make_shared<SafeTensorRepository>(path);
        return result;
    }
};

}

void CheckpointFormatCatalog::add(std::unique_ptr<ICheckpointFormat> format) {
    if (frozen_) throw std::logic_error("checkpoint format catalog is frozen");
    if (!format || format->id().empty()) {
        throw std::invalid_argument("checkpoint format must have a non-empty id");
    }
    const auto duplicate = std::find_if(formats_.begin(), formats_.end(),
        [&](const auto& existing) { return existing->id() == format->id(); });
    if (duplicate != formats_.end()) {
        throw std::invalid_argument("duplicate checkpoint format id: " +
                                    std::string(format->id()));
    }
    formats_.push_back(std::move(format));
}

void CheckpointFormatCatalog::freeze() { frozen_ = true; }

const ICheckpointFormat& CheckpointFormatCatalog::select(
    const std::filesystem::path& path) const {
    const ICheckpointFormat* selected = nullptr;
    for (const auto& format : formats_) {
        if (!format->matches(path)) continue;
        if (selected != nullptr) {
            throw std::runtime_error("multiple checkpoint formats match: " +
                                     path.string());
        }
        selected = format.get();
    }
    if (selected == nullptr) {
        throw std::runtime_error("no registered checkpoint format matches: " +
                                 path.string());
    }
    return *selected;
}

CheckpointView CheckpointFormatCatalog::open(
    const std::filesystem::path& path) const {
    return select(path).open(path);
}

std::vector<std::string_view> CheckpointFormatCatalog::ids() const {
    std::vector<std::string_view> result;
    result.reserve(formats_.size());
    for (const auto& format : formats_) result.push_back(format->id());
    return result;
}

std::shared_ptr<const CheckpointFormatCatalog>
create_builtin_checkpoint_format_catalog() {
    auto catalog = std::make_shared<CheckpointFormatCatalog>();
    add_builtin_checkpoint_formats(*catalog);
    catalog->freeze();
    return catalog;
}

void add_builtin_checkpoint_formats(CheckpointFormatCatalog& catalog) {
    catalog.add(std::make_unique<GgufFormat>());
    catalog.add(std::make_unique<SafeTensorsFormat>());
}

}
