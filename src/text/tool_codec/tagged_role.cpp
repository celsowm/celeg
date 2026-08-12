#include "celeg/text/semantic_chat_templates.hpp"
#include "celeg/text/protocol_utils.hpp"

#include <cctype>
#include <string>

namespace celeg {
namespace {

std::string trim(std::string_view value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.remove_prefix(1);
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.remove_suffix(1);
    }
    return std::string(value);
}

std::string render_tagged_value(std::string_view value) {
    std::string result = trim(value);
    if (result.size() >= 2 && result.front() == '"' && result.back() == '"') {
        result.erase(result.begin());
        result.pop_back();
        std::string decoded;
        decoded.reserve(result.size());
        bool escaped = false;
        for (const char ch : result) {
            if (escaped) {
                decoded.push_back(ch);
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else {
                decoded.push_back(ch);
            }
        }
        if (escaped) decoded.push_back('\\');
        return decoded;
    }
    return result;
}

std::string json_argument_value(std::string_view value) {
    const std::string result = trim(value);
    if (result.empty()) return "\"\"";
    const char first = result.front();
    if (first == '"' || first == '{' || first == '[' ||
        first == '-' || std::isdigit(static_cast<unsigned char>(first)) ||
        result == "true" || result == "false" || result == "null") {
        return result;
    }
    return json_quote(result);
}

std::string json_with_template_separators(std::string_view value) {
    std::string out;
    out.reserve(value.size() + value.size() / 8);
    bool quoted = false;
    bool escaped = false;
    for (const char ch : value) {
        if (quoted) {
            out.push_back(ch);
            if (escaped) escaped = false;
            else if (ch == '\\') escaped = true;
            else if (ch == '"') quoted = false;
            continue;
        }
        if (ch == '"') {
            quoted = true;
            out.push_back(ch);
        } else if (std::isspace(static_cast<unsigned char>(ch))) {
            continue;
        } else if (ch == ',') {
            out += ", ";
        } else if (ch == ':') {
            out += ": ";
        } else {
            out.push_back(ch);
        }
    }
    return out;
}

class TaggedRoleToolCallCodec final : public IChatToolCallCodec {
public:
    bool supports_parallel_calls() const noexcept override { return true; }

    std::string render_tool_definitions(std::span<const ToolDefinition> tools,
                                         const ToolChoice& choice) const override {
        std::string out =
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
        for (const ToolDefinition& tool : tools) {
            out += "\n{\"type\": \"function\", \"function\": {\"name\": " +
                json_quote(tool.function.name);
            out += ", \"description\": " + json_quote(tool.function.description);
            out += ", \"parameters\": " +
                json_with_template_separators(
                    tool.function.parameters.serialized.empty() ? "{}" :
                    tool.function.parameters.serialized) + "}}";
        }
        out +=
            "\n</tools>\n\nIf none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.\n";
        if (choice.mode == ToolChoiceMode::Required) {
            out += "You must call a function for this request. Do not answer with prose.\n";
        } else if (choice.mode == ToolChoiceMode::Specific) {
            out += "You must call the function " + json_quote(choice.function_name) +
                   " for this request. Do not answer with prose.\n";
        }
        return out +
            "If you need to use a function, for each function call, output the function name and arguments within the following XML format:\n"
            "<tool_call>{function-name}\n<arg_key>{arg-key-1}</arg_key>\n<arg_value>{arg-value-1}</arg_value>\n<arg_key>{arg-key-2}</arg_key>\n<arg_value>{arg-value-2}</arg_value>\n...</tool_call>\n";
    }

    std::string render_assistant_tool_calls(std::span<const ToolCall> calls) const override {
        std::string out;
        for (const ToolCall& call : calls) {
            out += "<tool_call>" + call.name;
            std::string_view args = call.arguments;
            if (args.size() >= 2 && args.front() == '{' && args.back() == '}') {
                args = args.substr(1, args.size() - 2);
            }
            std::size_t cursor = 0;
            while (cursor < args.size()) {
                const std::size_t key = args.find('"', cursor);
                if (key == std::string_view::npos) break;
                const std::size_t key_end = args.find('"', key + 1);
                const std::size_t colon = args.find(':', key_end == std::string_view::npos ? key : key_end);
                if (key_end == std::string_view::npos || colon == std::string_view::npos) break;
                std::size_t end = colon + 1;
                int depth = 0;
                bool quoted = false;
                for (; end < args.size(); ++end) {
                    const char ch = args[end];
                    if (ch == '"' && (end == 0 || args[end - 1] != '\\')) quoted = !quoted;
                    if (quoted) continue;
                    if (ch == '{' || ch == '[') ++depth;
                    if (ch == '}' || ch == ']') --depth;
                    if (ch == ',' && depth == 0) break;
                }
                out += "\n<arg_key>" + std::string(args.substr(key + 1, key_end - key - 1)) +
                       "</arg_key>\n<arg_value>" +
                       render_tagged_value(args.substr(colon + 1, end - colon - 1)) +
                       "</arg_value>";
                cursor = end == args.size() ? end : end + 1;
            }
            out += "\n</tool_call>";
        }
        return out;
    }

    std::string forced_tool_call_prefix(std::span<const ToolDefinition> tools,
                                        const ToolChoice& choice) const override {
        if (tools.empty()) return {};
        const std::string& name = choice.mode == ToolChoiceMode::Specific &&
                !choice.function_name.empty() ? choice.function_name : tools.front().function.name;
        return "<tool_call>" + name;
    }

    std::string render_tool_result(const ChatMessage& message) const override {
        return "<tool_response>\n" + message.content.value_or("") + "\n</tool_response>";
    }

    ToolParseResult parse_generation(std::string_view text) const override {
        ToolParseResult result;
        constexpr std::string_view start = "<tool_call>";
        constexpr std::string_view end = "</tool_call>";
        std::size_t cursor = 0;
        while ((cursor = text.find(start, cursor)) != std::string_view::npos) {
            const std::size_t body = cursor + start.size();
            const std::size_t close = text.find(end, body);
            if (close == std::string_view::npos) {
                result.status = ToolParseStatus::Incomplete;
                result.assistant_text = std::string(text.substr(0, cursor));
                return result;
            }
            const std::size_t first_key = text.find("<arg_key>", body);
            const std::size_t name_end = first_key == std::string_view::npos ? close : first_key;
            std::string name(text.substr(body, name_end - body));
            std::string arguments = "{";
            std::size_t count = 0;
            std::size_t item = first_key;
            while (item != std::string_view::npos && item < close) {
                const std::size_t key_end = text.find("</arg_key>", item);
                const std::size_t value = text.find("<arg_value>", key_end == std::string_view::npos ? item : key_end);
                const std::size_t value_end = text.find("</arg_value>", value == std::string_view::npos ? item : value);
                if (key_end == std::string_view::npos || value == std::string_view::npos ||
                    value_end == std::string_view::npos || value_end > close) break;
                if (count++) arguments += ',';
                arguments += json_quote(text.substr(item + 9, key_end - item - 9));
                arguments += ":" + json_argument_value(
                    text.substr(value + 11, value_end - value - 11));
                item = text.find("<arg_key>", value_end + 12);
            }
            arguments += '}';
            result.calls.push_back({"call_" + std::to_string(result.calls.size()), name, arguments});
            cursor = close + end.size();
        }
        if (!result.calls.empty()) {
            result.status = ToolParseStatus::Complete;
            result.assistant_text = std::string(text.substr(0, text.find(start)));
            result.consumed_bytes = text.size();
        }
        return result;
    }
};

} // namespace

std::unique_ptr<IChatToolCallCodec> make_tagged_role_tool_codec() {
    return std::make_unique<TaggedRoleToolCallCodec>();
}

} // namespace celeg
