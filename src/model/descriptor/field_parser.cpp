#include "detail.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace celeg::descriptor_detail {

const Json& required(const Json& object, std::string_view key) {
    if (!object.is_object() || !object.contains(key)) {
        throw std::invalid_argument("descriptor is missing field: " + std::string(key));
    }
    return object.at(key);
}

std::string optional_string(const Json& object, std::string_view key,
                            std::string fallback) {
    return object.is_object() && object.contains(key) ? object.at(key).as_string()
                                                       : std::move(fallback);
}

bool optional_bool(const Json& object, std::string_view key, bool fallback) {
    return object.is_object() && object.contains(key) ? object.at(key).as_bool() : fallback;
}

Field parse_field(const Json& value) {
    Field result;
    result.json = optional_string(value, "json");
    if (value.is_object() && value.contains("json_alternatives")) {
        for (const Json& alternative : value.at("json_alternatives").as_array()) {
            result.json_alternatives.push_back(alternative.as_string());
        }
    }
    result.gguf = optional_string(value, "gguf");
    if (value.is_object() && value.contains("default")) {
        const Json& fallback = value.at("default");
        if (fallback.is_string()) result.fallback_expression = fallback.as_string();
        else result.fallback = fallback.as_number();
    }
    if (result.json.empty() && result.gguf.empty() && result.fallback_expression.empty() &&
        !(value.is_object() && value.contains("default"))) {
        throw std::invalid_argument("descriptor field has no metadata key or default");
    }
    return result;
}

std::optional<Field> optional_field(const Json& object, std::string_view key) {
    return object.is_object() && object.contains(key)
        ? std::optional<Field>(parse_field(object.at(key))) : std::nullopt;
}

AttentionVariant parse_attention_variant(const Json& value) {
    AttentionVariant result;
    result.query_heads = optional_field(value, "query_heads");
    result.key_value_heads = optional_field(value, "kv_heads");
    result.head_dim = optional_field(value, "head_dim");
    result.rope_theta = optional_field(value, "rope_theta");
    result.rotary_fraction = optional_field(value, "rotary_fraction");
    result.sliding_window = optional_field(value, "sliding_window");
    if (!result.query_heads.has_value() && !result.key_value_heads.has_value() &&
        !result.head_dim.has_value() && !result.rope_theta.has_value() &&
        !result.rotary_fraction.has_value() && !result.sliding_window.has_value()) {
        throw std::invalid_argument("descriptor attention variant is empty");
    }
    return result;
}

ActivationKind parse_activation_kind(std::string_view value) {
    if (value == "swiglu") return ActivationKind::SwiGLU;
    if (value == "gelu_tanh") return ActivationKind::GeluTanh;
    if (value == "relu2") return ActivationKind::Relu2;
    throw std::invalid_argument("descriptor has unsupported feed-forward activation: " +
                                std::string(value));
}

} // namespace celeg::descriptor_detail
