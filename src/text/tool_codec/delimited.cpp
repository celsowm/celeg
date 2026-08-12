#include "celeg/text/semantic_chat_templates.hpp"
#include "celeg/text/protocol_utils.hpp"

#include <cctype>
#include <string>

namespace celeg {
namespace {

std::string trim(std::string_view value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) value.remove_prefix(1);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.remove_suffix(1);
    return std::string(value);
}

class DelimitedToolCallCodec final : public IChatToolCallCodec {
public:
    bool supports_parallel_calls() const noexcept override { return true; }

    std::string render_tool_definitions(std::span<const ToolDefinition> tools,
                                         const ToolChoice&) const override {
        std::string out = "List of tools: <|tool_list_start|>[";
        for (std::size_t i = 0; i < tools.size(); ++i) {
            if (i) out += ", ";
            out += "{\"type\":\"function\",\"function\":{\"name\":\"" + tools[i].function.name +
                   "\",\"description\":\"" + tools[i].function.description + "\",\"parameters\":" +
                   (tools[i].function.parameters.serialized.empty() ? "{}" : tools[i].function.parameters.serialized) + "}}";
        }
        return out + "]<|tool_list_end|>";
    }

    std::string render_assistant_tool_calls(std::span<const ToolCall> calls) const override {
        std::string out = "<|tool_call_start|>[";
        for (std::size_t i = 0; i < calls.size(); ++i) {
            if (i) out += ", ";
            out += calls[i].name + "(" + calls[i].arguments + ")";
        }
        return out + "]<|tool_call_end|>";
    }

    std::string forced_tool_call_prefix(std::span<const ToolDefinition> tools,
                                        const ToolChoice& choice) const override {
        if (tools.empty()) return {};
        const std::string& name = choice.mode == ToolChoiceMode::Specific &&
                !choice.function_name.empty() ? choice.function_name : tools.front().function.name;
        return "<|tool_call_start|>[" + name + "(";
    }

    std::string render_tool_result(const ChatMessage& message) const override {
        return "<|tool_response_start|>" + message.content.value_or("") + "<|tool_response_end|>";
    }

    ToolParseResult parse_generation(std::string_view text) const override {
        ToolParseResult result;
        result.calls = decode_calls(text);
        if (!result.calls.empty()) {
            result.status = ToolParseStatus::Complete;
            result.assistant_text = remove_calls_impl(text);
            result.consumed_bytes = text.size();
        } else if (text.find("<|tool_call_start|>") != std::string_view::npos &&
                   text.find("<|tool_call_end|>") == std::string_view::npos) {
            result.status = ToolParseStatus::Incomplete;
            result.assistant_text = std::string(text.substr(0, text.find("<|tool_call_start|>")));
        } else if (text.find("<|tool_call_start|>") != std::string_view::npos) {
            result.status = ToolParseStatus::Invalid;
            result.error = "delimited tool-call block contains no valid function call";
        }
        return result;
    }

private:
    std::vector<ToolCall> decode_calls(std::string_view text) const {
        std::vector<ToolCall> calls;
        constexpr std::string_view start = "<|tool_call_start|>";
        constexpr std::string_view end = "<|tool_call_end|>";
        std::size_t cursor = 0;
        while ((cursor = text.find(start, cursor)) != std::string_view::npos) {
            const auto body_begin = cursor + start.size();
            const auto body_end = text.find(end, body_begin);
            if (body_end == std::string_view::npos) break;
            auto body = text.substr(body_begin, body_end - body_begin);
            if (!body.empty() && body.front() == '[') body.remove_prefix(1);
            if (!body.empty() && body.back() == ']') body.remove_suffix(1);
            for (std::size_t offset = 0; offset < body.size();) {
                const auto open = body.find('(', offset);
                if (open == std::string_view::npos) break;
                auto name = trim(body.substr(offset, open - offset));
                const auto close = body.find(')', open + 1);
                if (name.empty() || close == std::string_view::npos) break;
                calls.push_back({{}, std::move(name), normalize_arguments(body.substr(open + 1, close - open - 1))});
                offset = close + 1;
                while (offset < body.size() && (std::isspace(static_cast<unsigned char>(body[offset])) || body[offset] == ',')) ++offset;
            }
            cursor = body_end + end.size();
        }
        for (std::size_t i = 0; i < calls.size(); ++i) calls[i].id = "call_" + std::to_string(i);
        return calls;
    }

    std::string remove_calls_impl(std::string_view text) const {
        constexpr std::string_view start = "<|tool_call_start|>";
        constexpr std::string_view end = "<|tool_call_end|>";
        return remove_tagged_blocks(text, start, end);
    }

    static std::string normalize_arguments(std::string_view args) {
        auto value = trim(args);
        if (value.empty()) return "{}";
        if (value.front() == '{') return value;
        std::string json = "{";
        std::size_t cursor = 0;
        while (cursor < value.size()) {
            const auto equal = value.find('=', cursor);
            if (equal == std::string::npos) return value;
            const auto comma = value.find(',', equal + 1);
            auto key = trim(std::string_view(value).substr(cursor, equal - cursor));
            auto val = trim(std::string_view(value).substr(equal + 1, comma == std::string::npos ? std::string::npos : comma - equal - 1));
            if (key.empty()) return value;
            if (json.size() > 1) json += ',';
            json += '"' + key + "\":" + normalize_value(val);
            cursor = comma == std::string::npos ? value.size() : comma + 1;
        }
        json += '}';
        return json;
    }

    static std::string normalize_value(std::string value) {
        if (value.size() >= 2 && value.front() == '\'' && value.back() == '\'') {
            value.front() = '"';
            value.back() = '"';
            for (std::size_t i = 1; i + 1 < value.size(); ++i) {
                if (value[i] == '"' && value[i - 1] != '\\') value.insert(i++, "\\");
            }
        }
        return value;
    }
};

} // namespace

std::unique_ptr<IChatToolCallCodec> make_delimited_tool_codec() {
    return std::make_unique<DelimitedToolCallCodec>();
}

} // namespace celeg
