#include "celeg/text/semantic_chat_templates.hpp"
#include "celeg/text/protocol_utils.hpp"

#include <cctype>

namespace celeg {
namespace {

class TaggedRoleToolCallCodec final : public IChatToolCallCodec {
public:
    bool supports_parallel_calls() const noexcept override { return true; }

    std::string render_tool_definitions(std::span<const ToolDefinition> tools,
                                         const ToolChoice&) const override {
        std::string out = "# Tools\n\n<tools>";
        for (const ToolDefinition& tool : tools) {
            out += "\n{";
            out += "\"name\":" + json_quote(tool.function.name);
            out += ",\"description\":" + json_quote(tool.function.description);
            out += ",\"parameters\":" +
                (tool.function.parameters.serialized.empty() ? "{}" :
                 tool.function.parameters.serialized) + "}";
        }
        return out + "\n</tools>";
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
                std::string value(args.substr(colon + 1, end - colon - 1));
                while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) value.erase(value.begin());
                while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
                out += "<arg_key>" + std::string(args.substr(key + 1, key_end - key - 1)) +
                       "</arg_key><arg_value>" + value + "</arg_value>";
                cursor = end == args.size() ? end : end + 1;
            }
            out += "</tool_call>";
        }
        return out;
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
                arguments += ":" + std::string(text.substr(value + 11, value_end - value - 11));
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
