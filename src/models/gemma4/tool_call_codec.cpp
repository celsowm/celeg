#include "celeg/text/chat_profile.hpp"

#include <string>

namespace celeg {
namespace {

class Gemma4ToolCallCodec final : public IChatToolCallCodec {
public:
    bool supports_parallel_calls() const noexcept override { return true; }

    std::string render_tool_definitions(std::span<const ToolDefinition> tools,
                                         const ToolChoice&) const override {
        std::string out;
        for (const auto& tool : tools) {
            out += "declaration:" + tool.function.name + "{description:<|\"|>" +
                   tool.function.description + "<|\"|>,parameters:" +
                   (tool.function.parameters.serialized.empty() ? "{}" : tool.function.parameters.serialized) + "}\n";
        }
        return out;
    }

    std::string render_assistant_tool_calls(std::span<const ToolCall> calls) const override {
        std::string out;
        for (const auto& call : calls) out += "<|tool_call>call:" + call.name + call.arguments + "<tool_call|>";
        return out;
    }

    std::string render_tool_result(const ChatMessage& message) const override {
        return "<|tool_response>response:" + message.tool_call_id.value_or("unknown") +
               "{value:" + message.content.value_or("") + "}<tool_response|>";
    }

    ToolParseResult parse_generation(std::string_view text) const override {
        ToolParseResult result;
        result.calls = decode_calls(text);
        if (!result.calls.empty()) {
            result.status = ToolParseStatus::Complete;
            result.assistant_text = remove_calls_impl(text);
            result.consumed_bytes = text.size();
        } else if (text.find("<|tool_call>call:") != std::string_view::npos &&
                   text.find("<tool_call|>") == std::string_view::npos) {
            result.status = ToolParseStatus::Incomplete;
            result.assistant_text = std::string(text.substr(0, text.find("<|tool_call>call:")));
        } else if (text.find("<|tool_call>") != std::string_view::npos) {
            result.status = ToolParseStatus::Invalid;
            result.error = "Gemma tool-call block contains no valid function call";
        }
        return result;
    }

private:
    std::vector<ToolCall> decode_calls(std::string_view text) const {
        std::vector<ToolCall> calls;
        constexpr std::string_view start = "<|tool_call>call:";
        constexpr std::string_view end = "<tool_call|>";
        for (std::size_t cursor = 0; (cursor = text.find(start, cursor)) != std::string_view::npos;) {
            const auto name_begin = cursor + start.size();
            const auto open = text.find('{', name_begin);
            const auto close_marker = text.find(end, name_begin);
            if (open == std::string_view::npos || close_marker == std::string_view::npos || open > close_marker) break;
            const auto close = matching_brace(text, open, close_marker);
            if (close == std::string_view::npos) break;
            calls.push_back({"call_" + std::to_string(calls.size()),
                             std::string(text.substr(name_begin, open - name_begin)),
                             std::string(text.substr(open, close - open + 1))});
            cursor = close_marker + end.size();
        }
        return calls;
    }

    std::string remove_calls_impl(std::string_view text) const {
        std::string result(text);
        constexpr std::string_view start = "<|tool_call>call:";
        constexpr std::string_view end = "<tool_call|>";
        for (std::size_t pos = 0; (pos = result.find(start, pos)) != std::string::npos;) {
            const auto finish = result.find(end, pos + start.size());
            if (finish == std::string::npos) {
                result.erase(pos);
                break;
            }
            result.erase(pos, finish + end.size() - pos);
        }
        return result;
    }

    static std::size_t matching_brace(std::string_view text, std::size_t open, std::size_t limit) {
        int depth = 0;
        bool quoted = false;
        bool escaped = false;
        for (std::size_t i = open; i < limit; ++i) {
            const char c = text[i];
            if (escaped) {
                escaped = false;
            } else if (c == '\\' && quoted) {
                escaped = true;
            } else if (c == '"') {
                quoted = !quoted;
            } else if (!quoted && c == '{') {
                ++depth;
            } else if (!quoted && c == '}' && --depth == 0) {
                return i;
            }
        }
        return std::string_view::npos;
    }
};

} // namespace

std::unique_ptr<IChatToolCallCodec> make_gemma4_tool_call_codec() {
    return std::make_unique<Gemma4ToolCallCodec>();
}

} // namespace celeg
