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

// Format-neutral metadata. The checkpoint format readers only normalize value
// representation; architecture modules own the meaning of individual keys.
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
    std::vector<std::string> strings(std::string_view key) const;

    static CheckpointMetadata from_json(const Json& root);
    static CheckpointMetadata from_gguf(const GgufFile& file);
};

} // namespace celeg
