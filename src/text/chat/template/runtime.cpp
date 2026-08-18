#include "program.hpp"

#include <algorithm>
#include <cctype>
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

namespace {

struct MacroDefinition {
    const TemplateNode* node = nullptr;
};

struct RenderState {
    std::vector<ValueObject> scopes;
    std::map<std::string, MacroDefinition, std::less<>> macros;
    std::string origin;
    int current_line = 1;
    bool generation_prompt = false;
};

std::string role_name(ChatRole role) {
    switch (role) {
    case ChatRole::System:
        return "system";
    case ChatRole::Developer:
        return "developer";
    case ChatRole::User:
        return "user";
    case ChatRole::Assistant:
        return "assistant";
    case ChatRole::Tool:
        return "tool";
    }
    throw std::invalid_argument("unknown chat role");
}

TemplateValue message_values(std::span<const ChatMessage> messages) {
    ValueList rendered;
    rendered.reserve(messages.size());
    for (const ChatMessage& message : messages) {
        ValueObject value;
        value.emplace("role", role_name(message.role));
        value.emplace("content", message.content.value_or(""));
        value.emplace("tool_call_id", message.tool_call_id.value_or(""));
        value.emplace("name", message.name.value_or(""));
        value.emplace(
            "reasoning_content",
            message.reasoning_content.value_or(""));
        value.emplace("thinking", message.reasoning_content.value_or(""));

        ValueList calls;
        for (const ToolCall& call : message.tool_calls) {
            ValueObject function;
            function.emplace("name", call.name);
            function.emplace("arguments", call.arguments);

            ValueObject call_value;
            call_value.emplace("id", call.id);
            call_value.emplace("name", call.name);
            call_value.emplace("arguments", call.arguments);
            call_value.emplace("type", "function");
            call_value.emplace(
                "function",
                TemplateValue{std::move(function)});
            try {
                call_value["arguments"] =
                    template_value_from_json(Json::parse(call.arguments));
                auto& callable =
                    std::get<ValueObject>(call_value["function"].value);
                callable["arguments"] = call_value["arguments"];
            } catch (const std::exception&) {
            }
            calls.emplace_back(std::move(call_value));
        }
        value.emplace("tool_calls", TemplateValue{std::move(calls)});
        rendered.emplace_back(std::move(value));
    }
    return TemplateValue{std::move(rendered)};
}

TemplateValue tool_values(std::span<const ChatToolDefinition> tools) {
    ValueList rendered;
    rendered.reserve(tools.size());
    for (const ChatToolDefinition& tool : tools) {
        ValueObject function;
        function.emplace("name", tool.function.name);
        function.emplace("description", tool.function.description);
        function.emplace(
            "parameters",
            RawJson{
                tool.function.parameters.serialized.empty()
                    ? "{}"
                    : tool.function.parameters.serialized});
        function.emplace("strict", tool.function.strict);

        ValueObject value;
        value.emplace("type", tool.type);
        value.emplace(
            "function",
            TemplateValue{std::move(function)});
        rendered.emplace_back(std::move(value));
    }
    return TemplateValue{std::move(rendered)};
}

TemplateValue tool_choice_value(const ToolChoice& choice) {
    ValueObject value;
    switch (choice.mode) {
    case ToolChoiceMode::None:
        value.emplace("mode", "none");
        break;
    case ToolChoiceMode::Auto:
        value.emplace("mode", "auto");
        break;
    case ToolChoiceMode::Required:
        value.emplace("mode", "required");
        break;
    case ToolChoiceMode::Specific:
        value.emplace("mode", "specific");
        break;
    }
    value.emplace("function_name", choice.function_name);
    return TemplateValue{std::move(value)};
}

class Renderer {
public:
    explicit Renderer(const std::vector<TemplateNode>& nodes)
        : nodes_(nodes) {}

    std::string render(
        std::span<const ChatMessage> messages,
        std::span<const ChatToolDefinition> tools,
        bool generation_prompt,
        const ChatTemplateOptions& options,
        std::string_view origin,
        std::string_view bos_token) const {
        RenderState state;
        state.origin = origin;
        state.generation_prompt = generation_prompt;

        ValueObject root;
        root.emplace("bos_token", std::string(bos_token));
        root.emplace("eos_token", "<|im_end|>");
        root.emplace("messages", message_values(messages));
        root.emplace("tools", tool_values(tools));
        root.emplace(
            "add_generation_prompt",
            TemplateValue{generation_prompt});
        root.emplace(
            "enable_thinking",
            TemplateValue{options.enable_thinking.value_or(false)});
        root.emplace("keep_past_thinking", TemplateValue{false});
        root.emplace(
            "tool_choice",
            tool_choice_value(options.tool_choice));
        state.scopes.push_back(std::move(root));

        std::string output;
        render_nodes(nodes_, state, output, generation_prompt);
        return output;
    }

private:
    TemplateValue eval(
        const Expression& expression,
        RenderState& state) const {
        switch (expression.kind) {
        case Expression::Kind::Literal:
            return expression.literal;

        case Expression::Kind::Name:
            return lookup(expression.text, state);

        case Expression::Kind::Unary:
            return TemplateValue{
                !truthy(eval(expression.children.at(0), state))};

        case Expression::Kind::Conditional:
            return truthy(eval(expression.children.at(0), state))
                ? eval(expression.children.at(1), state)
                : eval(expression.children.at(2), state);

        case Expression::Kind::Access: {
            const TemplateValue object =
                eval(expression.children.at(0), state);
            return member_of(object, expression.text)
                .value_or(TemplateValue{});
        }

        case Expression::Kind::Index:
            return index(expression, state);

        case Expression::Kind::Slice:
            return slice(expression, state);

        case Expression::Kind::Binary:
            return binary(expression, state);

        case Expression::Kind::Filter:
            return filter(expression, state);

        case Expression::Kind::Call:
            return call(expression, state);
        }
        throw_error(state, "invalid Jinja expression");
    }

    TemplateValue lookup(
        const std::string& name,
        const RenderState& state) const {
        for (auto scope = state.scopes.rbegin();
             scope != state.scopes.rend();
             ++scope) {
            const auto found = scope->find(name);
            if (found != scope->end()) {
                return found->second;
            }
        }
        return {};
    }

    TemplateValue index(
        const Expression& expression,
        RenderState& state) const {
        const TemplateValue object =
            eval(expression.children.at(0), state);
        const TemplateValue key =
            eval(expression.children.at(1), state);

        if (const auto* list = std::get_if<ValueList>(&object.value)) {
            const auto* integer =
                std::get_if<std::int64_t>(&key.value);
            if (!integer || *integer < 0 ||
                static_cast<std::size_t>(*integer) >= list->size()) {
                throw_error(state, "Jinja list index is out of range");
            }
            return (*list)[static_cast<std::size_t>(*integer)];
        }

        if (const auto* members =
                std::get_if<ValueObject>(&object.value)) {
            const std::string name = stringify(key);
            const auto found = members->find(name);
            if (found == members->end()) {
                throw_error(
                    state,
                    "undefined Jinja key '" + name + "'");
            }
            return found->second;
        }

        throw_error(state, "Jinja index applied to a non-container");
    }

    TemplateValue slice(
        const Expression& expression,
        RenderState& state) const {
        const TemplateValue object =
            eval(expression.children.at(0), state);
        const auto* list = std::get_if<ValueList>(&object.value);
        if (!list) {
            throw_error(state, "Jinja slice applied to a non-list");
        }

        const auto optional_integer =
            [&](std::size_t child) -> std::optional<std::int64_t> {
            if (child >= expression.children.size()) {
                return std::nullopt;
            }
            const TemplateValue value =
                eval(expression.children[child], state);
            if (std::holds_alternative<std::monostate>(value.value)) {
                return std::nullopt;
            }
            const auto* integer =
                std::get_if<std::int64_t>(&value.value);
            if (!integer) {
                throw_error(
                    state,
                    "Jinja slice bounds require integers");
            }
            return *integer;
        };

        const auto start = optional_integer(1);
        const auto end = optional_integer(2);
        const auto step = optional_integer(3);
        const std::int64_t stride = step.value_or(1);
        if (stride == 0) {
            throw_error(state, "Jinja slice step cannot be zero");
        }

        const auto bound = [&](std::int64_t raw) {
            const std::int64_t size =
                static_cast<std::int64_t>(list->size());
            const std::int64_t adjusted =
                raw < 0 ? raw + size : raw;
            return static_cast<std::size_t>(
                std::clamp(adjusted, std::int64_t{0}, size));
        };

        ValueList selected;
        if (stride > 0) {
            const std::size_t first = bound(start.value_or(0));
            const std::size_t last = bound(end.value_or(
                static_cast<std::int64_t>(list->size())));
            for (std::size_t position = first;
                 position < last;
                 position += static_cast<std::size_t>(stride)) {
                selected.push_back((*list)[position]);
            }
        } else {
            std::int64_t position = start.value_or(
                static_cast<std::int64_t>(list->size()) - 1);
            const std::int64_t last = end.value_or(-1);
            if (position < 0) {
                position += static_cast<std::int64_t>(list->size());
            }
            for (; position > last && position >= 0; position += stride) {
                selected.push_back(
                    (*list)[static_cast<std::size_t>(position)]);
            }
        }
        return TemplateValue{std::move(selected)};
    }

    TemplateValue binary(
        const Expression& expression,
        RenderState& state) const {
        const TemplateValue left =
            eval(expression.children.at(0), state);

        if (expression.text == "and") {
            return truthy(left)
                ? TemplateValue{
                      truthy(eval(expression.children.at(1), state))}
                : TemplateValue{false};
        }
        if (expression.text == "or") {
            return truthy(left)
                ? TemplateValue{true}
                : TemplateValue{
                      truthy(eval(expression.children.at(1), state))};
        }

        if (expression.text.starts_with("is ")) {
            return test(expression.text, left);
        }

        const TemplateValue right =
            eval(expression.children.at(1), state);

        if (expression.text == "+" &&
            std::holds_alternative<std::int64_t>(left.value) &&
            std::holds_alternative<std::int64_t>(right.value)) {
            return TemplateValue{
                std::get<std::int64_t>(left.value) +
                std::get<std::int64_t>(right.value)};
        }
        if (expression.text == "-") {
            const auto* l =
                std::get_if<std::int64_t>(&left.value);
            const auto* r =
                std::get_if<std::int64_t>(&right.value);
            if (!l || !r) {
                throw_error(
                    state,
                    "Jinja subtraction requires integers");
            }
            return TemplateValue{*l - *r};
        }
        if (expression.text == "~" || expression.text == "+") {
            return TemplateValue{
                stringify(left) + stringify(right)};
        }
        if (expression.text == "==") {
            return TemplateValue{
                stringify(left) == stringify(right)};
        }
        if (expression.text == "!=") {
            return TemplateValue{
                stringify(left) != stringify(right)};
        }
        if (expression.text == "in" ||
            expression.text == "not in") {
            bool found = false;
            if (const auto* list =
                    std::get_if<ValueList>(&right.value)) {
                found = std::any_of(
                    list->begin(),
                    list->end(),
                    [&](const TemplateValue& item) {
                        return stringify(item) == stringify(left);
                    });
            } else {
                found =
                    stringify(right).find(stringify(left)) !=
                    std::string::npos;
            }
            return TemplateValue{
                expression.text == "in" ? found : !found};
        }

        const std::string l = stringify(left);
        const std::string r = stringify(right);
        if (expression.text == ">") {
            return TemplateValue{l > r};
        }
        if (expression.text == "<") {
            return TemplateValue{l < r};
        }
        if (expression.text == ">=") {
            return TemplateValue{l >= r};
        }
        if (expression.text == "<=") {
            return TemplateValue{l <= r};
        }

        throw_error(
            state,
            "unsupported Jinja operator '" + expression.text + "'");
    }

    static TemplateValue test(
        std::string_view name,
        const TemplateValue& value) {
        if (name == "is defined") {
            return TemplateValue{
                !std::holds_alternative<std::monostate>(value.value)};
        }
        if (name == "is undefined") {
            return TemplateValue{
                std::holds_alternative<std::monostate>(value.value)};
        }
        if (name == "is none") {
            return TemplateValue{
                std::holds_alternative<std::monostate>(value.value)};
        }
        if (name == "is not none") {
            return TemplateValue{
                !std::holds_alternative<std::monostate>(value.value)};
        }
        if (name == "is string") {
            return TemplateValue{
                std::holds_alternative<std::string>(value.value)};
        }
        if (name == "is not string") {
            return TemplateValue{
                !std::holds_alternative<std::string>(value.value)};
        }
        if (name == "is true") {
            return TemplateValue{truthy(value)};
        }
        if (name == "is false") {
            return TemplateValue{!truthy(value)};
        }
        if (name == "is iterable") {
            return TemplateValue{
                std::holds_alternative<ValueList>(value.value) ||
                std::holds_alternative<ValueObject>(value.value) ||
                std::holds_alternative<std::string>(value.value)};
        }
        if (name == "is mapping") {
            return TemplateValue{
                std::holds_alternative<ValueObject>(value.value)};
        }
        if (name == "is not mapping") {
            return TemplateValue{
                !std::holds_alternative<ValueObject>(value.value)};
        }
        if (name == "is sequence") {
            return TemplateValue{
                std::holds_alternative<ValueList>(value.value)};
        }
        if (name == "is not sequence") {
            return TemplateValue{
                !std::holds_alternative<ValueList>(value.value)};
        }
        return TemplateValue{false};
    }

    TemplateValue filter(
        const Expression& expression,
        RenderState& state) const {
        TemplateValue input =
            eval(expression.children.at(0), state);
        const auto argument = [&](std::size_t index) {
            return eval(expression.children.at(index), state);
        };

        if (expression.text == "tojson") {
            return TemplateValue{to_json(input)};
        }
        if (expression.text == "string" ||
            expression.text == "safe") {
            return TemplateValue{stringify(input)};
        }
        if (expression.text == "trim") {
            return TemplateValue{trim(stringify(input))};
        }
        if (expression.text == "lower" ||
            expression.text == "upper") {
            std::string out = stringify(input);
            std::transform(
                out.begin(),
                out.end(),
                out.begin(),
                [&](unsigned char character) {
                    return static_cast<char>(
                        expression.text == "lower"
                            ? std::tolower(character)
                            : std::toupper(character));
                });
            return TemplateValue{std::move(out)};
        }
        if (expression.text == "length") {
            if (const auto* list =
                    std::get_if<ValueList>(&input.value)) {
                return TemplateValue{
                    static_cast<std::int64_t>(list->size())};
            }
            if (const auto* object =
                    std::get_if<ValueObject>(&input.value)) {
                return TemplateValue{
                    static_cast<std::int64_t>(object->size())};
            }
            return TemplateValue{
                static_cast<std::int64_t>(
                    stringify(input).size())};
        }
        if (expression.text == "default") {
            return truthy(input) ? input : argument(1);
        }
        if (expression.text == "replace") {
            std::string out = stringify(input);
            const std::string from = stringify(argument(1));
            const std::string to = stringify(argument(2));
            replace_all(out, from, to);
            return TemplateValue{std::move(out)};
        }
        if (expression.text == "join") {
            const std::string separator =
                expression.children.size() > 1
                    ? stringify(argument(1))
                    : "";
            std::string out;
            if (const auto* list =
                    std::get_if<ValueList>(&input.value)) {
                for (std::size_t index = 0;
                     index < list->size();
                     ++index) {
                    if (index != 0) {
                        out += separator;
                    }
                    out += stringify((*list)[index]);
                }
            }
            return TemplateValue{std::move(out)};
        }
        if (expression.text == "items") {
            const auto* object =
                std::get_if<ValueObject>(&input.value);
            if (!object) {
                throw_error(
                    state,
                    "items filter requires an object");
            }
            ValueList pairs;
            pairs.reserve(object->size());
            for (const auto& [key, value] : *object) {
                pairs.emplace_back(
                    ValueList{TemplateValue{key}, value});
            }
            return TemplateValue{std::move(pairs)};
        }
        if (expression.text == "list") {
            return input;
        }
        if (expression.text == "min" ||
            expression.text == "max") {
            const auto* list =
                std::get_if<ValueList>(&input.value);
            if (!list || list->empty()) {
                throw_error(
                    state,
                    expression.text +
                        " filter requires a non-empty list");
            }
            const auto compare = [](
                                     const TemplateValue& left,
                                     const TemplateValue& right) {
                if (std::holds_alternative<std::int64_t>(
                        left.value) &&
                    std::holds_alternative<std::int64_t>(
                        right.value)) {
                    return std::get<std::int64_t>(left.value) <
                           std::get<std::int64_t>(right.value);
                }
                return stringify(left) < stringify(right);
            };
            TemplateValue selected = list->front();
            for (const TemplateValue& candidate : *list) {
                if ((expression.text == "min" &&
                     compare(candidate, selected)) ||
                    (expression.text == "max" &&
                     compare(selected, candidate))) {
                    selected = candidate;
                }
            }
            return selected;
        }
        if (expression.text == "reverse") {
            if (const auto* list =
                    std::get_if<ValueList>(&input.value)) {
                ValueList reversed = *list;
                std::reverse(reversed.begin(), reversed.end());
                return TemplateValue{std::move(reversed)};
            }
            throw_error(state, "reverse filter requires a list");
        }

        throw_error(
            state,
            "unsupported Jinja filter '" + expression.text + "'");
    }

    TemplateValue call(
        const Expression& expression,
        RenderState& state) const {
        if (expression.text == "__list") {
            ValueList values;
            for (const Expression& child : expression.children) {
                values.push_back(eval(child, state));
            }
            return TemplateValue{std::move(values)};
        }
        if (expression.children.empty()) {
            throw_error(state, "unsupported Jinja call");
        }

        const Expression& target = expression.children.front();
        if (target.kind == Expression::Kind::Name) {
            if (target.text == "namespace") {
                ValueObject members;
                for (std::size_t index = 1;
                     index < expression.children.size();
                     ++index) {
                    const Expression& argument =
                        expression.children[index];
                    if (argument.kind != Expression::Kind::Binary ||
                        !argument.text.starts_with("__keyword:") ||
                        argument.children.size() != 1) {
                        throw_error(
                            state,
                            "namespace() requires named arguments");
                    }
                    members.emplace(
                        argument.text.substr(
                            std::string_view("__keyword:").size()),
                        eval(argument.children.front(), state));
                }
                return TemplateValue{std::move(members)};
            }

            if (target.text == "range") {
                if (expression.children.size() != 2 &&
                    expression.children.size() != 3) {
                    throw_error(
                        state,
                        "range() requires one or two arguments");
                }
                const std::int64_t start =
                    expression.children.size() == 2
                        ? 0
                        : require_integer(
                              eval(expression.children.at(1), state),
                              state,
                              "range() requires integer arguments");
                const std::int64_t end = require_integer(
                    eval(expression.children.back(), state),
                    state,
                    "range() requires integer arguments");
                ValueList values;
                for (std::int64_t value = start;
                     value < end;
                     ++value) {
                    values.emplace_back(value);
                }
                return TemplateValue{std::move(values)};
            }

            if (target.text == "raise_exception") {
                throw_error(
                    state,
                    stringify(eval(
                        expression.children.at(1),
                        state)));
            }

            const auto macro = state.macros.find(target.text);
            if (macro != state.macros.end()) {
                if (expression.children.size() - 1 >
                    macro->second.node->parameters.size()) {
                    throw_error(
                        state,
                        "wrong argument count for Jinja macro '" +
                            target.text + "'");
                }

                ValueObject arguments;
                for (std::size_t index = 0;
                     index < macro->second.node->parameters.size();
                     ++index) {
                    arguments.emplace(
                        macro->second.node->parameters[index],
                        index + 1 < expression.children.size()
                            ? eval(
                                  expression.children[index + 1],
                                  state)
                            : TemplateValue{});
                }
                state.scopes.push_back(std::move(arguments));
                std::string output;
                render_nodes(
                    macro->second.node->body,
                    state,
                    output,
                    state.generation_prompt);
                state.scopes.pop_back();
                return TemplateValue{std::move(output)};
            }
        }

        if (target.kind == Expression::Kind::Access) {
            const TemplateValue object =
                eval(target.children.at(0), state);
            const std::string member = target.text;
            const std::string text = stringify(object);

            if (member == "strip") {
                return TemplateValue{trim(text)};
            }
            if (member == "lstrip" || member == "rstrip") {
                std::string output = text;
                if (member == "lstrip") {
                    while (!output.empty() &&
                           std::isspace(static_cast<unsigned char>(
                               output.front()))) {
                        output.erase(output.begin());
                    }
                } else {
                    while (!output.empty() &&
                           std::isspace(static_cast<unsigned char>(
                               output.back()))) {
                        output.pop_back();
                    }
                }
                return TemplateValue{std::move(output)};
            }
            if (member == "lower" || member == "upper") {
                std::string output = text;
                std::transform(
                    output.begin(),
                    output.end(),
                    output.begin(),
                    [&](unsigned char character) {
                        return static_cast<char>(
                            member == "lower"
                                ? std::tolower(character)
                                : std::toupper(character));
                    });
                return TemplateValue{std::move(output)};
            }
            if (member == "items") {
                const auto* object_values =
                    std::get_if<ValueObject>(&object.value);
                if (!object_values) {
                    throw_error(
                        state,
                        "items() requires an object");
                }
                ValueList pairs;
                for (const auto& [key, value] : *object_values) {
                    pairs.emplace_back(
                        ValueList{TemplateValue{key}, value});
                }
                return TemplateValue{std::move(pairs)};
            }
            if (member == "split") {
                const std::string separator =
                    expression.children.size() > 1
                        ? stringify(eval(
                              expression.children.at(1),
                              state))
                        : " ";
                ValueList pieces;
                std::size_t offset = 0;
                while (offset <= text.size()) {
                    const std::size_t found =
                        text.find(separator, offset);
                    pieces.emplace_back(text.substr(
                        offset,
                        found == std::string::npos
                            ? std::string::npos
                            : found - offset));
                    if (found == std::string::npos ||
                        separator.empty()) {
                        break;
                    }
                    offset = found + separator.size();
                }
                return TemplateValue{std::move(pieces)};
            }
            if (member == "replace") {
                if (expression.children.size() != 3) {
                    throw_error(
                        state,
                        "replace() requires old and new strings");
                }
                std::string output = text;
                const std::string from = stringify(
                    eval(expression.children.at(1), state));
                const std::string to = stringify(
                    eval(expression.children.at(2), state));
                replace_all(output, from, to);
                return TemplateValue{std::move(output)};
            }
            if (member == "startswith" ||
                member == "endswith") {
                const std::string needle = stringify(
                    eval(expression.children.at(1), state));
                return TemplateValue{
                    member == "startswith"
                        ? text.starts_with(needle)
                        : text.ends_with(needle)};
            }

            if (member == "get") {
                const TemplateValue key = eval(
                    expression.children.at(1), state);
                const TemplateValue fallback =
                    expression.children.size() > 2
                        ? eval(expression.children.at(2), state)
                        : TemplateValue{};
                if (const auto* obj =
                        std::get_if<ValueObject>(&object.value)) {
                    const auto found = obj->find(stringify(key));
                    if (found != obj->end()) {
                        return found->second;
                    }
                }
                return fallback;
            }
        }

        throw_error(state, "unsupported Jinja call");
    }

    void render_nodes(
        const std::vector<TemplateNode>& nodes,
        RenderState& state,
        std::string& output,
        bool generation_prompt) const {
        for (const TemplateNode& node : nodes) {
            state.current_line = node.line;
            switch (node.kind) {
            case TemplateNode::Kind::Text:
                output += node.text;
                break;

            case TemplateNode::Kind::Output:
                output += stringify(eval(node.expression, state));
                break;

            case TemplateNode::Kind::Set:
                if (node.body.empty()) {
                    assign(
                        node.text,
                        eval(node.expression, state),
                        state);
                } else {
                    std::string captured;
                    render_nodes(node.body, state, captured, generation_prompt);
                    assign(
                        node.text,
                        TemplateValue{std::move(captured)},
                        state);
                }
                break;

            case TemplateNode::Kind::Macro:
                state.macros[node.text] = {&node};
                break;

            case TemplateNode::Kind::If: {
                bool rendered = false;
                for (const auto& [condition, body] : node.branches) {
                    if (truthy(eval(condition, state))) {
                        render_nodes(body, state, output, generation_prompt);
                        rendered = true;
                        break;
                    }
                }
                if (!rendered) {
                    render_nodes(node.otherwise, state, output, generation_prompt);
                }
                break;
            }

            case TemplateNode::Kind::For:
                render_loop(node, state, output, generation_prompt);
                break;

            case TemplateNode::Kind::Generation:
                if (generation_prompt) {
                    render_nodes(node.body, state, output, generation_prompt);
                }
                break;
            }
        }
    }

    void render_loop(
        const TemplateNode& node,
        RenderState& state,
        std::string& output,
        bool generation_prompt) const {
        const TemplateValue sequence = eval(node.expression, state);
        const auto* values = std::get_if<ValueList>(&sequence.value);
        if (!values) {
            throw_error(state, "Jinja for requires a list");
        }

        bool rendered_any = false;
        for (std::size_t index = 0;
             index < values->size();
             ++index) {
            ValueObject scope;
            const std::size_t comma = node.text.find(',');
            if (comma == std::string::npos) {
                scope.emplace(node.text, (*values)[index]);
            } else {
                const std::string first =
                    trim(node.text.substr(0, comma));
                const std::string second =
                    trim(node.text.substr(comma + 1));
                const auto* pair =
                    std::get_if<ValueList>(&(*values)[index].value);
                if (!pair || pair->size() != 2) {
                    throw_error(
                        state,
                        "Jinja destructuring loop requires pairs");
                }
                scope.emplace(first, (*pair)[0]);
                scope.emplace(second, (*pair)[1]);
            }

            ValueObject loop;
            loop.emplace(
                "index0",
                static_cast<std::int64_t>(index));
            loop.emplace(
                "index",
                static_cast<std::int64_t>(index + 1));
            loop.emplace("first", index == 0);
            loop.emplace(
                "last",
                index + 1 == values->size());
            loop.emplace(
                "length",
                static_cast<std::int64_t>(values->size()));
            if (index != 0) {
                loop.emplace("previtem", (*values)[index - 1]);
            }
            if (index + 1 < values->size()) {
                loop.emplace("nextitem", (*values)[index + 1]);
            }
            scope.emplace("loop", TemplateValue{std::move(loop)});

            state.scopes.push_back(std::move(scope));
            const bool selected =
                !node.condition ||
                truthy(eval(*node.condition, state));
            if (selected) {
                render_nodes(node.body, state, output, generation_prompt);
            }
            state.scopes.pop_back();
            rendered_any = rendered_any || selected;
        }

        if (!rendered_any) {
            render_nodes(node.otherwise, state, output, generation_prompt);
        }
    }

    void assign(
        const std::string& target,
        TemplateValue value,
        RenderState& state) const {
        const std::size_t dot = target.find('.');
        if (dot == std::string::npos) {
            state.scopes.back()[target] = std::move(value);
            return;
        }

        const std::string root = target.substr(0, dot);
        TemplateValue* current = nullptr;
        for (auto scope = state.scopes.rbegin();
             scope != state.scopes.rend() && !current;
             ++scope) {
            const auto found = scope->find(root);
            if (found != scope->end()) {
                current = &found->second;
            }
        }
        if (!current) {
            throw_error(
                state,
                "undefined Jinja assignment target '" + root + "'");
        }

        std::size_t begin = dot + 1;
        for (;;) {
            const std::size_t next = target.find('.', begin);
            const std::string member = target.substr(
                begin,
                next == std::string::npos
                    ? std::string::npos
                    : next - begin);
            auto* object =
                std::get_if<ValueObject>(&current->value);
            if (!object) {
                throw_error(
                    state,
                    "Jinja assignment target is not an object");
            }
            if (next == std::string::npos) {
                (*object)[member] = std::move(value);
                return;
            }
            current = &(*object)[member];
            begin = next + 1;
        }
    }

    static std::int64_t require_integer(
        const TemplateValue& value,
        const RenderState& state,
        std::string_view message) {
        const auto* integer =
            std::get_if<std::int64_t>(&value.value);
        if (!integer) {
            throw_error(state, std::string(message));
        }
        return *integer;
    }

    static void replace_all(
        std::string& output,
        const std::string& from,
        const std::string& to) {
        std::size_t offset = 0;
        while (!from.empty() &&
               (offset = output.find(from, offset)) !=
                   std::string::npos) {
            output.replace(offset, from.size(), to);
            offset += to.size();
        }
    }

    [[noreturn]] static void throw_error(
        const RenderState& state,
        const std::string& message) {
        throw std::invalid_argument(
            "UnsupportedChatTemplateConstruct at " + state.origin + ":" +
            std::to_string(state.current_line) + ": " + message);
    }

    const std::vector<TemplateNode>& nodes_;
};

}

std::string render_program(
    const std::vector<TemplateNode>& nodes,
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool generation_prompt,
    const ChatTemplateOptions& options,
    std::string_view origin,
    std::string_view bos_token) {
    return Renderer(nodes).render(
        messages,
        tools,
        generation_prompt,
        options,
        origin,
        bos_token);
}

}

namespace celeg {

InteractionRenderProgram::InteractionRenderProgram(
    std::vector<chat_template_detail::TemplateNode> nodes)
    : nodes_(std::move(nodes)) {}

std::string InteractionRenderProgram::render(
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool generation_prompt,
    const ChatTemplateOptions& options,
    std::string_view origin,
    std::string_view bos_token) const {
    return chat_template_detail::render_program(
        nodes_,
        messages,
        tools,
        generation_prompt,
        options,
        origin,
        bos_token);
}

}
