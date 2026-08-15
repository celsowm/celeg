#include "detail.hpp"

#include <cmath>

namespace celeg::descriptor_detail {

std::string selected_key(const CheckpointMetadata& metadata, const Field& field) {
    std::string key = metadata.is_gguf() ? field.gguf : field.json;
    const std::string needle = "{architecture}";
    size_t position = 0;
    while ((position = key.find(needle, position)) != std::string::npos) {
        const std::string architecture = metadata.architecture_type();
        key.replace(position, needle.size(), architecture);
        position += architecture.size();
    }
    return key;
}

std::vector<std::string> selected_keys(const CheckpointMetadata& metadata,
                                       const Field& field) {
    std::vector<std::string> keys;
    const std::string primary = selected_key(metadata, field);
    if (!primary.empty()) keys.push_back(primary);
    if (!metadata.is_gguf()) {
        for (const std::string& alternative : field.json_alternatives) {
            Field alias;
            alias.json = alternative;
            const std::string key = selected_key(metadata, alias);
            if (!key.empty()) keys.push_back(key);
        }
    }
    return keys;
}

int integer_value(const CheckpointMetadata& metadata, const Field& field,
                 int hidden, int query_heads) {
    const std::string key = selected_key(metadata, field);
    if (!key.empty() && metadata.contains(key)) {
        const MetadataValue& value = metadata.value(key);
        if (const auto* scalar = std::get_if<int64_t>(&value)) {
            return static_cast<int>(*scalar);
        }
        if (const auto* scalar = std::get_if<double>(&value)) {
            return static_cast<int>(*scalar);
        }
        if (std::holds_alternative<std::vector<int64_t>>(value)) {
            return static_cast<int>(field.fallback.value_or(0.0));
        }
        throw std::runtime_error("descriptor dimension metadata is not an integer: " + key);
    }
    if (field.fallback_expression == "hidden_div_query_heads") {
        if (hidden <= 0 || query_heads <= 0) throw std::invalid_argument("invalid descriptor dimension expression");
        return hidden / query_heads;
    }
    if (field.fallback_expression.empty()) return static_cast<int>(field.fallback.value_or(0.0));
    throw std::invalid_argument("unsupported descriptor default expression: " + field.fallback_expression);
}

double number_value(const CheckpointMetadata& metadata, const Field& field, int hidden) {
    const std::vector<std::string> keys = selected_keys(metadata, field);
    for (const std::string& key : keys) {
        if (metadata.contains(key)) return metadata.number(key);
    }
    const std::string key = keys.empty() ? selected_key(metadata, field) : keys.front();
    if (field.fallback_expression == "sqrt_hidden") {
        if (hidden <= 0) throw std::invalid_argument("invalid hidden size expression");
        return std::sqrt(static_cast<double>(hidden));
    }
    if (!field.fallback_expression.empty()) {
        throw std::invalid_argument("unsupported descriptor numeric expression: " +
                                    field.fallback_expression);
    }
    return field.fallback.value_or(0.0);
}

int scaling_integer_value(const CheckpointMetadata& metadata,
                          const std::optional<Field>& field, int fallback) {
    return field.has_value() ? integer_value(metadata, *field) : fallback;
}

double scaling_number_value(const CheckpointMetadata& metadata,
                            const std::optional<Field>& field, double fallback) {
    return field.has_value() ? number_value(metadata, *field) : fallback;
}

std::vector<float> scaling_factor_values(const CheckpointMetadata& metadata,
                                         const std::optional<Field>& field) {
    if (!field.has_value()) return {};
    const std::string key = selected_key(metadata, *field);
    if (key.empty() || !metadata.contains(key)) return {};
    const MetadataValue& value = metadata.value(key);
    std::vector<float> result;
    if (const auto* list = std::get_if<std::vector<double>>(&value)) {
        result.reserve(list->size());
        for (double item : *list) result.push_back(static_cast<float>(item));
        return result;
    }
    if (const auto* list = std::get_if<std::vector<int64_t>>(&value)) {
        result.reserve(list->size());
        for (int64_t item : *list) result.push_back(static_cast<float>(item));
        return result;
    }
    if (const auto* item = std::get_if<double>(&value)) return {static_cast<float>(*item)};
    if (const auto* item = std::get_if<int64_t>(&value)) return {static_cast<float>(*item)};
    throw std::runtime_error("RoPE factor metadata must be numeric: " + key);
}

std::string position_kind_value(const CheckpointMetadata& metadata,
                                const Descriptor& descriptor) {
    if (!descriptor.position_kind_field.has_value()) return descriptor.position_kind;
    const Field& field = *descriptor.position_kind_field;
    const std::string key = selected_key(metadata, field);
    if (!key.empty() && metadata.contains(key)) return metadata.string(key);
    return field.fallback_expression.empty() ? "rope" : field.fallback_expression;
}

RopeScalingSpec parse_scaling_kind(std::string_view value) {
    if (value.empty() || value == "none") return NoRopeScaling{};
    if (value == "linear") return LinearRopeScaling{};
    if (value == "dynamic_ntk") return DynamicNtkRopeScaling{};
    if (value == "yarn") return YarnRopeScaling{};
    if (value == "long" || value == "longrope") return LongRopeScaling{};
    if (value == "llama3_frequency") return Llama3FrequencyScaling{};
    throw std::invalid_argument("descriptor has unsupported RoPE scaling kind: " +
                                std::string(value));
}

StateScalarType parse_state_scalar(std::string_view value) {
    if (value == "fp32") return StateScalarType::FP32;
    if (value == "fp16") return StateScalarType::FP16;
    if (value == "bf16") return StateScalarType::BF16;
    if (value == "fp8") return StateScalarType::FP8;
    if (value == "int8") return StateScalarType::INT8;
    if (value == "int4") return StateScalarType::INT4;
    throw std::invalid_argument("descriptor has unsupported state storage: " +
                                std::string(value));
}

StateQuantizationGranularity parse_state_granularity(std::string_view value) {
    if (value == "per_tensor") return StateQuantizationGranularity::PerTensor;
    if (value == "per_head") return StateQuantizationGranularity::PerHead;
    if (value == "per_token") return StateQuantizationGranularity::PerToken;
    if (value == "per_block") return StateQuantizationGranularity::PerBlock;
    throw std::invalid_argument("descriptor has unsupported state quantization granularity: " +
                                std::string(value));
}

std::string scaling_kind_value(const CheckpointMetadata& metadata,
                               const Descriptor& descriptor) {
    if (!descriptor.rope_scaling_kind_field.has_value()) {
        return descriptor.rope_scaling_kind;
    }
    const Field& field = *descriptor.rope_scaling_kind_field;
    const std::string key = selected_key(metadata, field);
    if (!key.empty() && metadata.contains(key)) return metadata.string(key);
    return field.fallback_expression.empty() ? "none" : field.fallback_expression;
}

std::vector<int> integer_values(const CheckpointMetadata& metadata,
                               const std::string& key) {
    if (key.empty() || !metadata.contains(key)) return {};
    const MetadataValue& value = metadata.value(key);
    if (const auto* list = std::get_if<std::vector<int64_t>>(&value)) {
        std::vector<int> result;
        result.reserve(list->size());
        for (int64_t item : *list) result.push_back(static_cast<int>(item));
        return result;
    }
    return {};
}

std::vector<bool> attention_pattern_values(const CheckpointMetadata& metadata,
                                           const Descriptor& descriptor) {
    if (!descriptor.attention_pattern.has_value()) return {};
    const std::string key = selected_key(metadata, *descriptor.attention_pattern);
    if (key.empty() || !metadata.contains(key)) return {};
    const MetadataValue& value = metadata.value(key);
    const auto is_sliding = [&descriptor](std::string_view item) {
        return item == descriptor.sliding_pattern_value || item == "sliding";
    };
    if (const auto* list = std::get_if<std::vector<std::string>>(&value)) {
        std::vector<bool> result;
        result.reserve(list->size());
        for (const std::string& item : *list) result.push_back(is_sliding(item));
        return result;
    }
    if (const auto* list = std::get_if<std::vector<int64_t>>(&value)) {
        std::vector<bool> result;
        result.reserve(list->size());
        for (int64_t item : *list) result.push_back(item != 0);
        return result;
    }
    if (const auto* item = std::get_if<std::string>(&value)) return {is_sliding(*item)};
    if (const auto* item = std::get_if<int64_t>(&value)) return {*item != 0};
    throw std::runtime_error("attention pattern metadata must be strings or integers: " + key);
}

std::vector<int> field_integer_values(const CheckpointMetadata& metadata,
                                      const std::optional<Field>& field) {
    if (!field.has_value()) return {};
    const std::string key = selected_key(metadata, *field);
    if (key.empty() || !metadata.contains(key)) return {};
    const MetadataValue& value = metadata.value(key);
    if (const auto* list = std::get_if<std::vector<int64_t>>(&value)) {
        std::vector<int> result;
        result.reserve(list->size());
        for (int64_t item : *list) result.push_back(static_cast<int>(item));
        return result;
    }
    return {};
}

std::vector<bool> mixer_is_convolution(const CheckpointMetadata& metadata,
                                       const Descriptor& descriptor) {
    if (!descriptor.mixer_schedule.has_value()) return {};
    const std::string key = selected_key(metadata, *descriptor.mixer_schedule);
    if (key.empty() || !metadata.contains(key)) return {};
    const MetadataValue& value = metadata.value(key);
    const auto is_convolution = [&descriptor](std::string_view item) {
        return item == descriptor.convolution_value || item == "conv" ||
               item == "short_convolution";
    };
    if (const auto* list = std::get_if<std::vector<std::string>>(&value)) {
        std::vector<bool> result;
        result.reserve(list->size());
        for (const std::string& item : *list) result.push_back(is_convolution(item));
        return result;
    }
    if (const auto* list = std::get_if<std::vector<int64_t>>(&value)) {
        std::vector<bool> result;
        result.reserve(list->size());
        for (int64_t item : *list) result.push_back(item == 0);
        return result;
    }
    if (const auto* item = std::get_if<std::string>(&value)) return {is_convolution(*item)};
    if (const auto* item = std::get_if<int64_t>(&value)) return {*item == 0};
    throw std::runtime_error("mixer schedule metadata must be strings or integers: " + key);
}

std::vector<std::string> mixer_schedule_values(const CheckpointMetadata& metadata,
                                               const Descriptor& descriptor) {
    if (!descriptor.mixer_schedule.has_value()) return {};
    const std::string key = selected_key(metadata, *descriptor.mixer_schedule);
    if (key.empty() || !metadata.contains(key)) return {};
    const MetadataValue& value = metadata.value(key);
    if (const auto* list = std::get_if<std::vector<std::string>>(&value)) return *list;
    if (const auto* list = std::get_if<std::vector<int64_t>>(&value)) {
        std::vector<std::string> result;
        result.reserve(list->size());
        for (int64_t item : *list) result.push_back(std::to_string(item));
        return result;
    }
    if (const auto* item = std::get_if<std::string>(&value)) {
        if (item->size() > 1 && item->find_first_not_of("M-*_") == std::string::npos) {
            std::vector<std::string> result;
            result.reserve(item->size());
            for (char symbol : *item) result.emplace_back(1, symbol);
            return result;
        }
        return {*item};
    }
    if (const auto* item = std::get_if<int64_t>(&value)) return {std::to_string(*item)};
    throw std::runtime_error("mixer schedule metadata must be strings or integers: " + key);
}

bool boolean_value(const CheckpointMetadata& metadata, const std::optional<Field>& field,
                  bool fallback) {
    if (!field.has_value()) return fallback;
    const std::string key = selected_key(metadata, *field);
    if (key.empty() || !metadata.contains(key)) {
        if (field->fallback_expression == "true") return true;
        if (field->fallback_expression == "false" || !field->fallback_expression.empty()) {
            return false;
        }
        return field->fallback.has_value() && *field->fallback != 0.0 ? true : fallback;
    }
    return metadata.boolean(key);
}

int token_value(const CheckpointMetadata& metadata, const Field& field,
                std::string_view gguf_override) {
    if (metadata.is_gguf() && !gguf_override.empty() && metadata.contains(gguf_override)) {
        return static_cast<int>(metadata.integer(gguf_override));
    }
    return integer_value(metadata, field);
}

std::vector<int> eos_values(const CheckpointMetadata& metadata, const Descriptor& descriptor) {
    if (metadata.is_gguf()) {
        std::vector<int> result = integer_values(metadata, descriptor.gguf_eos);
        if (result.empty()) result.push_back(token_value(metadata, descriptor.eos, descriptor.gguf_eos));
        if (!descriptor.gguf_eot.empty()) {
            const std::vector<int> eot = integer_values(metadata, descriptor.gguf_eot);
            result.insert(result.end(), eot.begin(), eot.end());
            if (eot.empty() && metadata.contains(descriptor.gguf_eot)) {
                result.push_back(static_cast<int>(metadata.integer(descriptor.gguf_eot)));
            }
            std::sort(result.begin(), result.end());
            result.erase(std::unique(result.begin(), result.end()), result.end());
        }
        return result;
    }
    return integer_values(metadata, descriptor.eos.json).empty()
        ? std::vector<int>{token_value(metadata, descriptor.eos, {})}
        : integer_values(metadata, descriptor.eos.json);
}


}
