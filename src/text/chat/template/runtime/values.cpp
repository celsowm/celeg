#include "../program.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>
#include <stdexcept>

namespace celeg::chat_template_detail {

TemplateValue template_value_from_json(const Json& input) {
    if (input.is_null()) {
        return {};
    }
    if (input.is_bool()) {
        return TemplateValue{input.as_bool()};
    }
    if (input.is_number()) {
        return TemplateValue{input.as_i64()};
    }
    if (input.is_string()) {
        return TemplateValue{input.as_string()};
    }
    if (input.is_array()) {
        ValueList values;
        for (const Json& member : input.as_array()) {
            values.push_back(template_value_from_json(member));
        }
        return TemplateValue{std::move(values)};
    }

    ValueObject values;
    for (const auto& [key, member] : input.as_object()) {
        values.emplace(key, template_value_from_json(member));
    }
    return TemplateValue{std::move(values)};
}

std::string json_string(std::string_view input) {
    std::string out{"\""};
    for (const unsigned char character : input) {
        switch (character) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (character < 0x20) {
                constexpr char hex[] = "0123456789abcdef";
                out += "\\u00";
                out += hex[(character >> 4) & 0xf];
                out += hex[character & 0xf];
            } else {
                out += static_cast<char>(character);
            }
        }
    }
    return out + '"';
}

std::string stringify(const TemplateValue& value) {
    if (const auto* text = std::get_if<std::string>(&value.value)) {
        return *text;
    }
    if (const auto* raw = std::get_if<RawJson>(&value.value)) {
        return raw->value;
    }
    if (const auto* integer = std::get_if<std::int64_t>(&value.value)) {
        return std::to_string(*integer);
    }
    if (const auto* boolean = std::get_if<bool>(&value.value)) {
        return *boolean ? "true" : "false";
    }
    return {};
}

bool truthy(const TemplateValue& value) {
    if (const auto* boolean = std::get_if<bool>(&value.value)) {
        return *boolean;
    }
    if (const auto* integer = std::get_if<std::int64_t>(&value.value)) {
        return *integer != 0;
    }
    if (const auto* text = std::get_if<std::string>(&value.value)) {
        return !text->empty();
    }
    if (const auto* list = std::get_if<ValueList>(&value.value)) {
        return !list->empty();
    }
    if (const auto* object = std::get_if<ValueObject>(&value.value)) {
        return !object->empty();
    }
    return false;
}

std::string to_json(const TemplateValue& value) {
    if (std::holds_alternative<std::monostate>(value.value)) {
        return "null";
    }
    if (const auto* boolean = std::get_if<bool>(&value.value)) {
        return *boolean ? "true" : "false";
    }
    if (const auto* integer = std::get_if<std::int64_t>(&value.value)) {
        return std::to_string(*integer);
    }
    if (const auto* text = std::get_if<std::string>(&value.value)) {
        return json_string(*text);
    }
    if (const auto* raw = std::get_if<RawJson>(&value.value)) {
        return raw->value;
    }
    if (const auto* list = std::get_if<ValueList>(&value.value)) {
        std::string out = "[";
        for (std::size_t index = 0; index < list->size(); ++index) {
            if (index != 0) {
                out += ',';
            }
            out += to_json((*list)[index]);
        }
        return out + ']';
    }

    const auto& object = std::get<ValueObject>(value.value);
    std::string out = "{";
    bool first = true;
    for (const auto& [key, member] : object) {
        if (!first) {
            out += ',';
        }
        first = false;
        out += json_string(key) + ':' + to_json(member);
    }
    return out + '}';
}

std::optional<TemplateValue> member_of(
    const TemplateValue& value,
    std::string_view key) {
    if (const auto* object = std::get_if<ValueObject>(&value.value)) {
        const auto found = object->find(key);
        if (found != object->end()) {
            return found->second;
        }
    }
    if (key == "length") {
        if (const auto* list = std::get_if<ValueList>(&value.value)) {
            return TemplateValue{
                static_cast<std::int64_t>(list->size())};
        }
        if (const auto* text = std::get_if<std::string>(&value.value)) {
            return TemplateValue{
                static_cast<std::int64_t>(text->size())};
        }
    }
    return std::nullopt;
}


}
