#pragma once

#include "celeg/checkpoint/formats/json.hpp"
#include "celeg/text/chat_template.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace celeg::chat_template_detail {

struct RawJson {
    std::string value;
};

struct TemplateValue;
using ValueList = std::vector<TemplateValue>;
using ValueObject = std::map<std::string, TemplateValue, std::less<>>;

struct TemplateValue {
    using Storage = std::variant<std::monostate, bool, std::int64_t, std::string,
                                 RawJson, ValueList, ValueObject>;
    Storage value;

    TemplateValue() = default;
    TemplateValue(bool input) : value(input) {}
    TemplateValue(std::int64_t input) : value(input) {}
    TemplateValue(std::string input) : value(std::move(input)) {}
    TemplateValue(const char* input) : value(std::string(input)) {}
    TemplateValue(RawJson input) : value(std::move(input)) {}
    TemplateValue(ValueList input) : value(std::move(input)) {}
    TemplateValue(ValueObject input) : value(std::move(input)) {}
};

struct Expression {
    enum class Kind {
        Literal,
        Name,
        Unary,
        Binary,
        Conditional,
        Access,
        Index,
        Slice,
        Call,
        Filter,
    };

    Kind kind = Kind::Literal;
    std::string text;
    TemplateValue literal;
    std::vector<Expression> children;
};

struct TemplateNode {
    enum class Kind {
        Text,
        Output,
        If,
        For,
        Set,
        Macro,
        Generation,
    };

    Kind kind = Kind::Text;
    int line = 1;
    std::string text;
    Expression expression;
    std::optional<Expression> condition;
    std::vector<TemplateNode> body;
    std::vector<std::pair<Expression, std::vector<TemplateNode>>> branches;
    std::vector<TemplateNode> otherwise;
    std::vector<std::string> parameters;
};

TemplateValue template_value_from_json(const Json& input);
std::string json_string(std::string_view input);
std::string stringify(const TemplateValue& value);
bool truthy(const TemplateValue& value);
std::string to_json(const TemplateValue& value);
std::optional<TemplateValue> member_of(const TemplateValue& value, std::string_view key);
std::string trim(std::string_view input);

std::vector<TemplateNode> parse_template(std::string_view source, std::string origin);
void validate_program(const std::vector<TemplateNode>& nodes, std::string_view origin);
std::optional<ToolCallGrammar> derive_tool_grammar(std::string_view source);

}

namespace celeg {

class InteractionRenderProgram final {
public:
    explicit InteractionRenderProgram(
        std::vector<chat_template_detail::TemplateNode> nodes);

    std::string render(std::span<const ChatMessage> messages,
                       std::span<const ChatToolDefinition> tools,
                       bool generation_prompt,
                       const ChatTemplateOptions& options,
                       std::string_view origin,
                       std::string_view bos_token) const;

private:
    std::vector<chat_template_detail::TemplateNode> nodes_;
};

}
