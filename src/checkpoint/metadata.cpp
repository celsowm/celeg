#include "celeg/checkpoint/metadata.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace celeg {
namespace {

bool special_float(const Json& value, double& result) {
    if (!value.is_object() || value.as_object().size() != 1 ||
        !value.contains("__float__")) {
        return false;
    }
    const Json& name = value.as_object().at("__float__");
    if (!name.is_string()) return false;
    const std::string& text = name.as_string();
    if (text == "Infinity" || text == "+Infinity") {
        result = std::numeric_limits<double>::infinity();
        return true;
    }
    if (text == "-Infinity") {
        result = -std::numeric_limits<double>::infinity();
        return true;
    }
    if (text == "NaN") {
        result = std::numeric_limits<double>::quiet_NaN();
        return true;
    }
    return false;
}

void flatten_json(const Json& value, const std::string& prefix,
                  CheckpointMetadata& metadata) {
    if (value.is_bool()) {
        metadata.values[prefix] = value.as_bool();
    } else if (value.is_number()) {
        const double number = value.as_number();
        const int64_t integer = value.as_i64();
        metadata.values[prefix] = static_cast<double>(integer) == number
            ? MetadataValue(integer) : MetadataValue(number);
    } else if (value.is_string()) {
        metadata.values[prefix] = value.as_string();
    } else if (value.is_array()) {
        bool all_strings = true;
        bool all_numbers = true;
        bool all_integers = true;
        std::vector<std::string> strings;
        std::vector<double> numbers;
        std::vector<int64_t> integers;
        for (const Json& item : value.as_array()) {
            if (!item.is_string()) all_strings = false;
            else strings.push_back(item.as_string());
            double special_number = 0.0;
            const bool is_special_number = special_float(item, special_number);
            if (!item.is_number() && !is_special_number) {
                all_numbers = false;
                all_integers = false;
            } else {
                const double number = is_special_number ? special_number : item.as_number();
                numbers.push_back(number);
                if (is_special_number || static_cast<double>(item.as_i64()) != number) {
                    all_integers = false;
                }
                integers.push_back(is_special_number ? 0 : item.as_i64());
            }
        }
        if (all_strings) metadata.values[prefix] = std::move(strings);
        else if (all_numbers && all_integers) metadata.values[prefix] = std::move(integers);
        else if (all_numbers) metadata.values[prefix] = std::move(numbers);
        else throw std::runtime_error("unsupported metadata array: " + prefix);
    } else if (value.is_object()) {
        double special_number = 0.0;
        if (special_float(value, special_number)) {
            metadata.values[prefix] = special_number;
            return;
        }
        for (const auto& [key, child] : value.as_object()) {
            flatten_json(child, prefix.empty() ? key : prefix + "." + key, metadata);
        }
    }
}

MetadataValue gguf_value(const GgufValue& value) {
    switch (value.kind) {
    case GgufValueKind::Bool: return value.integer != 0;
    case GgufValueKind::String: return value.str;
    case GgufValueKind::F32:
    case GgufValueKind::F64: return value.number;
    case GgufValueKind::U8:
    case GgufValueKind::I8:
    case GgufValueKind::U16:
    case GgufValueKind::I16:
    case GgufValueKind::U32:
    case GgufValueKind::I32:
    case GgufValueKind::U64:
    case GgufValueKind::I64: return value.integer;
    case GgufValueKind::Array:
        if (!value.array_strings.empty()) return value.array_strings;
        if (!value.array_numbers.empty()) return value.array_numbers;
        return value.array_integers;
    }
    throw std::runtime_error("unsupported GGUF metadata value");
}

template <typename T>
const T& as(const MetadataValue& value, std::string_view key) {
    if (const auto* result = std::get_if<T>(&value)) return *result;
    throw std::runtime_error("metadata key has unexpected type: " + std::string(key));
}

}

bool CheckpointMetadata::contains(std::string_view key) const {
    return values.find(std::string(key)) != values.end();
}

const MetadataValue& CheckpointMetadata::value(std::string_view key) const {
    const auto it = values.find(std::string(key));
    if (it == values.end()) throw std::runtime_error("missing checkpoint metadata: " + std::string(key));
    return it->second;
}

std::string CheckpointMetadata::string(std::string_view key) const {
    return as<std::string>(value(key), key);
}

std::string CheckpointMetadata::string_or(std::string_view key, std::string fallback) const {
    return contains(key) ? string(key) : std::move(fallback);
}

int64_t CheckpointMetadata::integer(std::string_view key) const {
    if (const auto* integer = std::get_if<int64_t>(&value(key))) return *integer;
    throw std::runtime_error("metadata key is not an integer: " + std::string(key));
}

int64_t CheckpointMetadata::integer_or(std::string_view key, int64_t fallback) const {
    return contains(key) ? integer(key) : fallback;
}

double CheckpointMetadata::number(std::string_view key) const {
    if (const auto* number = std::get_if<double>(&value(key))) return *number;
    if (const auto* integer = std::get_if<int64_t>(&value(key))) return static_cast<double>(*integer);
    throw std::runtime_error("metadata key is not numeric: " + std::string(key));
}

double CheckpointMetadata::number_or(std::string_view key, double fallback) const {
    return contains(key) ? number(key) : fallback;
}

bool CheckpointMetadata::boolean(std::string_view key) const {
    return as<bool>(value(key), key);
}

bool CheckpointMetadata::boolean_or(std::string_view key, bool fallback) const {
    return contains(key) ? boolean(key) : fallback;
}

std::vector<int64_t> CheckpointMetadata::integers(std::string_view key) const {
    return as<std::vector<int64_t>>(value(key), key);
}

std::vector<double> CheckpointMetadata::numbers(std::string_view key) const {
    return as<std::vector<double>>(value(key), key);
}

std::vector<std::string> CheckpointMetadata::strings(std::string_view key) const {
    return as<std::vector<std::string>>(value(key), key);
}

std::string CheckpointMetadata::key(std::string_view safetensors_key,
                                    std::string_view gguf_key) const {
    return std::string(is_gguf() ? gguf_key : safetensors_key);
}

std::string CheckpointMetadata::architecture_type() const {
    return string_or(is_gguf() ? "general.architecture" : "model_type", {});
}

int64_t CheckpointMetadata::integer_for(std::string_view safetensors_key,
                                        std::string_view gguf_key) const {
    return integer(key(safetensors_key, gguf_key));
}

int64_t CheckpointMetadata::integer_for_or(std::string_view safetensors_key,
                                           std::string_view gguf_key,
                                           int64_t fallback) const {
    return integer_or(key(safetensors_key, gguf_key), fallback);
}

double CheckpointMetadata::number_for_or(std::string_view safetensors_key,
                                         std::string_view gguf_key,
                                         double fallback) const {
    return number_or(key(safetensors_key, gguf_key), fallback);
}

std::string CheckpointMetadata::string_for_or(std::string_view safetensors_key,
                                              std::string_view gguf_key,
                                              std::string fallback) const {
    return string_or(key(safetensors_key, gguf_key), std::move(fallback));
}

CheckpointMetadata CheckpointMetadata::from_json(const Json& root) {
    CheckpointMetadata metadata;
    flatten_json(root, {}, metadata);
    metadata.repository_hint = metadata.string_or(
        "_name_or_path", metadata.string_or("_name", {}));
    return metadata;
}

CheckpointMetadata CheckpointMetadata::from_gguf(const GgufFile& file) {
    CheckpointMetadata metadata;
    metadata.source_format = CheckpointSourceFormat::Gguf;
    for (const auto& [key, value] : file.metadata()) metadata.values[key] = gguf_value(value);
    metadata.repository_hint = metadata.string_or("general.name",
                                                   metadata.string_or("general.basename", {}));
    return metadata;
}

}
