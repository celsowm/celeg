#include "celeg/text/semantic_chat_templates.hpp"

#include "celeg/text/protocol_utils.hpp"

#include <cctype>

namespace celeg {
namespace {

std::string trim(std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) text.remove_prefix(1);
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) text.remove_suffix(1);
    return std::string(text);
}

class ParameterXmlToolCallCodec final : public IChatToolCallCodec {
public:
    bool supports_parallel_calls() const noexcept override { return true; }

    std::string render_tool_definitions(std::span<const ToolDefinition> tools,
                                         const ToolChoice&) const override {
        std::string out = "# Tools\n\n<tools>\n";
        for (const ToolDefinition& tool : tools) {
            out += "<tool>\n<name>" + tool.function.name + "</name>\n<description>" +
                tool.function.description + "</description>\n<parameters>" +
                (tool.function.parameters.serialized.empty() ? "{}" :
                    tool.function.parameters.serialized) + "</parameters>\n</tool>\n";
        }
        return out + "</tools>\n\nUse <tool_call><function=name><parameter=arg>value</parameter></function></tool_call>.\n";
    }

    std::string render_assistant_tool_calls(std::span<const ToolCall> calls) const override {
        std::string out;
        for (const ToolCall& call : calls) {
            out += "<tool_call>\n<function=" + call.name + ">";
            std::string_view args = call.arguments;
            if (args.size() >= 2 && args.front() == '{' && args.back() == '}') args = args.substr(1, args.size() - 2);
            std::size_t cursor = 0;
            while (cursor < args.size()) {
                const std::size_t key_begin = args.find('"', cursor);
                if (key_begin == std::string_view::npos) break;
                const std::size_t key_end = args.find('"', key_begin + 1);
                const std::size_t value_begin = args.find(':', key_end == std::string_view::npos ? key_begin : key_end);
                if (key_end == std::string_view::npos || value_begin == std::string_view::npos) break;
                const std::size_t next = args.find(',', value_begin + 1);
                out += "\n<parameter=" + std::string(args.substr(key_begin + 1, key_end - key_begin - 1)) + ">" +
                    trim(args.substr(value_begin + 1, next == std::string_view::npos ? args.size() - value_begin - 1 : next - value_begin - 1)) +
                    "</parameter>";
                cursor = next == std::string_view::npos ? args.size() : next + 1;
            }
            out += "\n</function>\n</tool_call>";
        }
        return out;
    }

    std::string forced_tool_call_prefix(std::span<const ToolDefinition> tools,
                                        const ToolChoice& choice) const override {
        if (tools.empty()) return {};
        const std::string& name = choice.mode == ToolChoiceMode::Specific &&
                !choice.function_name.empty() ? choice.function_name : tools.front().function.name;
        return "<tool_call>\n<function=" + name + ">";
    }

    std::string render_tool_result(const ChatMessage& message) const override {
        return "<tool_response>\n" + message.content.value_or("") + "\n</tool_response>\n";
    }

    ToolParseResult parse_generation(std::string_view text) const override {
        ToolParseResult result;
        constexpr std::string_view start = "<tool_call>";
        constexpr std::string_view end = "</tool_call>";
        std::size_t cursor = 0;
        while ((cursor = text.find(start, cursor)) != std::string_view::npos) {
            const std::size_t function_begin = text.find("<function=", cursor + start.size());
            if (function_begin == std::string_view::npos) break;
            const std::size_t name_end = text.find('>', function_begin);
            const std::size_t body_end = text.find(end, name_end == std::string_view::npos ? function_begin : name_end);
            if (name_end == std::string_view::npos || body_end == std::string_view::npos) {
                result.status = ToolParseStatus::Incomplete;
                return result;
            }
            const std::string name(text.substr(function_begin + 10, name_end - function_begin - 10));
            const std::string_view body = text.substr(name_end + 1, body_end - name_end - 1);
            std::string arguments = "{";
            std::size_t count = 0;
            std::size_t parameter = 0;
            while ((parameter = body.find("<parameter=", parameter)) != std::string_view::npos) {
                const std::size_t key_end = body.find('>', parameter);
                const std::size_t value_end = body.find("</parameter>", key_end);
                if (key_end == std::string_view::npos || value_end == std::string_view::npos) break;
                const std::string key(body.substr(parameter + 11, key_end - parameter - 11));
                if (count++) arguments += ',';
                arguments += json_quote(key) + ":" + json_quote(trim(body.substr(key_end + 1, value_end - key_end - 1)));
                parameter = value_end + 12;
            }
            arguments += '}';
            result.calls.push_back({"call_" + std::to_string(result.calls.size()), name, arguments});
            cursor = body_end + end.size();
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

std::unique_ptr<IChatToolCallCodec> make_parameter_xml_tool_codec() {
    return std::make_unique<ParameterXmlToolCallCodec>();
}

} // namespace celeg
