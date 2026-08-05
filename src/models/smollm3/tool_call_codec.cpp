#include "celeg/text/chat_profile.hpp"
#include "celeg/text/protocol_utils.hpp"

#include <cctype>
#include <string>
#include <string_view>

namespace celeg {
namespace {

std::string json_string(std::string_view value) {
    return json_quote(value);
}

std::size_t matching_brace(std::string_view text, std::size_t open,
                           std::size_t limit) {
    int depth = 0;
    bool quoted = false;
    bool escaped = false;
    for (std::size_t index = open; index < limit; ++index) {
        const char ch = text[index];
        if (escaped) { escaped = false; continue; }
        if (quoted && ch == '\\') { escaped = true; continue; }
        if (ch == '"') { quoted = !quoted; continue; }
        if (quoted) continue;
        if (ch == '{') ++depth;
        if (ch == '}' && --depth == 0) return index;
    }
    return std::string_view::npos;
}

class SmolLm3ToolCallCodec final : public IChatToolCallCodec {
public:
    bool supports_parallel_calls() const noexcept override { return true; }

    std::string render_tool_definitions(std::span<const ToolDefinition> tools,
                                         const ToolChoice&) const override {
        if (tools.empty()) return {};
        std::string out = "### Tools\n\nYou may call one or more functions to assist with the user query.\n"
                          "You are provided with function signatures within <tools> XML tags:\n<tools>\n";
        for (const ToolDefinition& tool : tools) {
            out += "{\"name\":" + json_string(tool.function.name);
            out += ",\"description\":" + json_string(tool.function.description);
            out += ",\"parameters\":" +
                (tool.function.parameters.serialized.empty() ? "{}" : tool.function.parameters.serialized);
            out += "}\n";
        }
        return out + "</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call> XML tags.\n";
    }

    std::string render_assistant_tool_calls(std::span<const ToolCall> calls) const override {
        std::string out;
        for (const ToolCall& call : calls) {
            out += "<tool_call>{\"name\":" + json_string(call.name) +
                   ",\"arguments\":" +
                   (call.arguments.empty() ? "{}" : call.arguments) + "}</tool_call>";
        }
        return out;
    }

    std::string render_tool_result(const ChatMessage& message) const override {
        return message.content.value_or("");
    }

    ToolParseResult parse_generation(std::string_view text) const override {
        ToolParseResult result;
        constexpr std::string_view start = "<tool_call>";
        constexpr std::string_view end = "</tool_call>";
        std::size_t cursor = 0;
        while ((cursor = text.find(start, cursor)) != std::string_view::npos) {
            const std::size_t body_begin = cursor + start.size();
            const std::size_t body_end = text.find(end, body_begin);
            if (body_end == std::string_view::npos) break;
            const std::string_view body = text.substr(body_begin, body_end - body_begin);
            const std::size_t name_key = body.find("\"name\"");
            const std::size_t name_quote = name_key == std::string_view::npos
                ? name_key : body.find('"', body.find(':', name_key) + 1);
            const std::size_t name_end = name_quote == std::string_view::npos
                ? name_quote : body.find('"', name_quote + 1);
            const std::size_t args_key = body.find("\"arguments\"");
            const std::size_t args_begin = args_key == std::string_view::npos
                ? args_key : body.find('{', body.find(':', args_key) + 1);
            const std::size_t args_end = args_begin == std::string_view::npos
                ? args_begin : matching_brace(body, args_begin, body.size());
            if (name_quote == std::string_view::npos || name_end == std::string_view::npos ||
                args_begin == std::string_view::npos || args_end == std::string_view::npos) {
                cursor = body_end + end.size();
                continue;
            }
            result.calls.push_back({"call_" + std::to_string(result.calls.size()),
                std::string(body.substr(name_quote + 1, name_end - name_quote - 1)),
                std::string(body.substr(args_begin, args_end - args_begin + 1))});
            cursor = body_end + end.size();
        }
        if (!result.calls.empty()) {
            result.status = ToolParseStatus::Complete;
            result.assistant_text = remove_calls(text);
            result.consumed_bytes = text.size();
        } else if (text.find(start) != std::string_view::npos &&
                   text.find(end) == std::string_view::npos) {
            result.status = ToolParseStatus::Incomplete;
            result.assistant_text = std::string(text.substr(0, text.find(start)));
        }
        return result;
    }

private:
    static std::string remove_calls(std::string_view text) {
        constexpr std::string_view start = "<tool_call>";
        constexpr std::string_view end = "</tool_call>";
        return remove_tagged_blocks(text, start, end);
    }
};

} // namespace

std::unique_ptr<IChatToolCallCodec> make_smollm3_tool_call_codec() {
    return std::make_unique<SmolLm3ToolCallCodec>();
}

} // namespace celeg
