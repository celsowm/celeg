#pragma once

#include "celeg/checkpoint/formats/gguf.hpp"
#include "celeg/checkpoint/formats/json.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace celeg {

enum class CheckpointSourceFormat : uint8_t {
    Safetensors,
    Gguf,
};

using MetadataValue = std::variant<
    int64_t,
    double,
    bool,
    std::string,
    std::vector<int64_t>,
    std::vector<double>,
    std::vector<std::string>>;

struct CheckpointMetadata {
    CheckpointSourceFormat source_format = CheckpointSourceFormat::Safetensors;
    std::string repository_hint;
    std::unordered_map<std::string, MetadataValue> values;

    bool contains(std::string_view key) const;
    const MetadataValue& value(std::string_view key) const;

    std::string string(std::string_view key) const;
    std::string string_or(std::string_view key, std::string fallback) const;
    int64_t integer(std::string_view key) const;
    int64_t integer_or(std::string_view key, int64_t fallback) const;
    double number(std::string_view key) const;
    double number_or(std::string_view key, double fallback) const;
    bool boolean(std::string_view key) const;
    bool boolean_or(std::string_view key, bool fallback) const;
    std::vector<int64_t> integers(std::string_view key) const;
    std::vector<double> numbers(std::string_view key) const;
    std::vector<std::string> strings(std::string_view key) const;

    bool is_gguf() const { return source_format == CheckpointSourceFormat::Gguf; }
    std::string key(std::string_view safetensors_key,
                    std::string_view gguf_key) const;
    std::string architecture_type() const;
    int64_t integer_for(std::string_view safetensors_key,
                        std::string_view gguf_key) const;
    int64_t integer_for_or(std::string_view safetensors_key,
                           std::string_view gguf_key, int64_t fallback) const;
    double number_for_or(std::string_view safetensors_key,
                         std::string_view gguf_key, double fallback) const;
    std::string string_for_or(std::string_view safetensors_key,
                              std::string_view gguf_key,
                              std::string fallback = {}) const;

    static CheckpointMetadata from_json(const Json& root);
    static CheckpointMetadata from_gguf(const GgufFile& file);
};

}
