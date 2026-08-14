#include "program.hpp"

namespace celeg::chat_template_detail {

std::optional<ToolCallGrammar> derive_tool_grammar(std::string_view source) {
    const std::string_view opening = "<|tool_call_start|>";
    const std::string_view closing = "<|tool_call_end|>";
    const std::size_t begin = source.find(opening);
    const std::size_t end = source.find(closing);

    if (begin == std::string_view::npos ||
        end == std::string_view::npos ||
        begin >= end) {
        if (source.find("<tool_call>") == std::string_view::npos ||
            source.find("<function=") == std::string_view::npos ||
            source.find("</function>") == std::string_view::npos) {
            return std::nullopt;
        }
        ToolCallGrammar grammar;
        grammar.opening = "<tool_call>";
        grammar.closing = "</tool_call>";
        grammar.call_prefix = "<function=";
        grammar.call_suffix = "</function>";
        grammar.parallel_calls =
            source.find("for tool_call in message.tool_calls") !=
            std::string_view::npos;
        grammar.xml_parameters = true;
        return grammar;
    }

    const std::string_view body = source.substr(
        begin + opening.size(),
        end - begin - opening.size());
    if (body.find("arguments") == std::string_view::npos ||
        body.find("name") == std::string_view::npos) {
        return std::nullopt;
    }

    ToolCallGrammar grammar;
    grammar.opening = std::string(opening);
    grammar.closing = std::string(closing);
    grammar.call_prefix =
        body.find('[') != std::string_view::npos ? "[" : "";
    grammar.call_suffix =
        body.find(']') != std::string_view::npos ? "]" : "";
    grammar.parallel_calls =
        source.find("tool_calls") != std::string_view::npos;
    return grammar;
}

} // namespace celeg::chat_template_detail

namespace celeg {

ToolParseResult ToolCallGrammar::parse(std::string_view generated) const {
    ToolParseResult result;
    const std::size_t first = generated.find(opening);
    if (first == std::string_view::npos) {
        result.assistant_text = std::string(generated);
        return result;
    }

    result.assistant_text = std::string(generated.substr(0, first));
    std::size_t cursor = first;
    std::size_t call_index = 0;

    while (cursor < generated.size()) {
        if (!generated.substr(cursor).starts_with(opening)) {
            result.status = ToolParseStatus::Invalid;
            result.error = "unexpected text between tool calls";
            return result;
        }
        cursor += opening.size();

        const std::size_t end = generated.find(closing, cursor);
        if (end == std::string_view::npos) {
            result.status = ToolParseStatus::Incomplete;
            result.consumed_bytes = cursor;
            return result;
        }

        std::string payload(generated.substr(cursor, end - cursor));
        if (xml_parameters) {
            if (!payload.starts_with(call_prefix)) {
                result.status = ToolParseStatus::Invalid;
                result.error = "tool call has no function tag";
                return result;
            }

            const std::size_t name_end = payload.find('>');
            const std::size_t function_end = payload.rfind(call_suffix);
            if (name_end == std::string::npos ||
                function_end == std::string::npos ||
                function_end < name_end) {
                result.status = ToolParseStatus::Incomplete;
                result.consumed_bytes = cursor;
                return result;
            }

            const std::string name = payload.substr(
                call_prefix.size(),
                name_end - call_prefix.size());
            std::string arguments = "{";
            std::size_t parameter = name_end + 1;
            bool first_parameter = true;

            while (parameter < function_end) {
                const std::size_t tag =
                    payload.find("<parameter=", parameter);
                if (tag == std::string::npos || tag >= function_end) {
                    break;
                }
                const std::size_t key_end = payload.find('>', tag);
                const std::size_t value_end =
                    payload.find("</parameter>", key_end);
                if (key_end == std::string::npos ||
                    value_end == std::string::npos ||
                    value_end > function_end) {
                    result.status = ToolParseStatus::Incomplete;
                    result.consumed_bytes = cursor + tag;
                    return result;
                }

                if (!first_parameter) {
                    arguments += ',';
                }
                first_parameter = false;
                const auto prefix =
                    std::string_view("<parameter=");
                arguments += chat_template_detail::json_string(
                    payload.substr(
                        tag + prefix.size(),
                        key_end - tag - prefix.size()));
                arguments += ':' + chat_template_detail::json_string(
                    chat_template_detail::trim(payload.substr(
                        key_end + 1,
                        value_end - key_end - 1)));
                parameter =
                    value_end + std::string_view("</parameter>").size();
            }

            arguments += '}';
            result.calls.push_back({
                "call_" + std::to_string(call_index++),
                name,
                std::move(arguments),
            });
            cursor = end + closing.size();
            if (!parallel_calls) {
                break;
            }
            continue;
        }

        if (!call_prefix.empty()) {
            if (!payload.starts_with(call_prefix) ||
                !payload.ends_with(call_suffix)) {
                result.status = ToolParseStatus::Invalid;
                result.error =
                    "tool call delimiters do not match grammar";
                return result;
            }
            payload = payload.substr(
                call_prefix.size(),
                payload.size() - call_prefix.size() -
                    call_suffix.size());
        }

        const std::size_t open = payload.find('(');
        const std::size_t close = payload.rfind(')');
        if (open == std::string::npos ||
            close != payload.size() - 1 ||
            open == 0) {
            result.status = ToolParseStatus::Invalid;
            result.error = "tool call is not name(arguments)";
            return result;
        }

        result.calls.push_back({
            "call_" + std::to_string(call_index++),
            payload.substr(0, open),
            payload.substr(open + 1, close - open - 1),
        });
        cursor = end + closing.size();
        if (!parallel_calls) {
            break;
        }
    }

    result.status = result.calls.empty()
        ? ToolParseStatus::NotToolCall
        : ToolParseStatus::Complete;
    result.consumed_bytes = cursor;
    return result;
}

} // namespace celeg
