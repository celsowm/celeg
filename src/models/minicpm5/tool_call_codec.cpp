#include "celeg/text/chat_profile.hpp"

#include <cctype>
#include <string>
#include <string_view>
#include <vector>

namespace celeg {
namespace {

std::string json_string(std::string_view value) {
    std::string out = "\"";
    for (const char ch : value) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(ch); break;
        }
    }
    out.push_back('"');
    return out;
}

std::string trim(std::string_view value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) value.remove_prefix(1);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.remove_suffix(1);
    return std::string(value);
}

struct JsonMember {
    std::string key;
    std::string value;
};

std::vector<JsonMember> object_members(std::string_view object) {
    std::vector<JsonMember> members;
    if (object.size() < 2 || object.front() != '{' || object.back() != '}') return members;
    std::size_t cursor = 1;
    while (cursor + 1 < object.size()) {
        while (cursor + 1 < object.size() &&
               (std::isspace(static_cast<unsigned char>(object[cursor])) || object[cursor] == ',')) ++cursor;
        if (cursor + 1 >= object.size() || object[cursor] != '"') break;
        const std::size_t key_begin = ++cursor;
        bool escaped = false;
        for (; cursor < object.size(); ++cursor) {
            if (escaped) { escaped = false; continue; }
            if (object[cursor] == '\\') { escaped = true; continue; }
            if (object[cursor] == '"') break;
        }
        if (cursor >= object.size()) break;
        const std::string key(object.substr(key_begin, cursor - key_begin));
        ++cursor;
        while (cursor < object.size() && std::isspace(static_cast<unsigned char>(object[cursor]))) ++cursor;
        if (cursor >= object.size() || object[cursor] != ':') break;
        ++cursor;
        while (cursor < object.size() && std::isspace(static_cast<unsigned char>(object[cursor]))) ++cursor;
        const std::size_t value_begin = cursor;
        int depth = 0;
        bool quoted = false;
        escaped = false;
        for (; cursor < object.size(); ++cursor) {
            const char ch = object[cursor];
            if (escaped) { escaped = false; continue; }
            if (quoted && ch == '\\') { escaped = true; continue; }
            if (ch == '"') { quoted = !quoted; continue; }
            if (quoted) continue;
            if (ch == '{' || ch == '[') ++depth;
            else if (ch == '}' || ch == ']') {
                if (ch == '}' && depth == 0) break;
                --depth;
            } else if (ch == ',' && depth == 0) break;
        }
        members.push_back({key, trim(object.substr(value_begin, cursor - value_begin))});
        if (cursor < object.size() && object[cursor] == ',') ++cursor;
    }
    return members;
}

class MiniCpm5ToolCallCodec final : public IChatToolCallCodec {
public:
    bool supports_parallel_calls() const noexcept override { return true; }

    std::string render_tool_definitions(std::span<const ToolDefinition> tools,
                                         const ToolChoice&) const override {
        if (tools.empty()) return {};
        std::string out = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n<tools>\n";
        for (const ToolDefinition& tool : tools) {
            out += "{\"type\":\"function\",\"function\":{\"name\":";
            out += json_string(tool.function.name);
            out += ",\"description\":";
            out += json_string(tool.function.description);
            out += ",\"parameters\":";
            out += tool.function.parameters.serialized.empty() ? "{}" : tool.function.parameters.serialized;
            out += "}}\n";
        }
        out += "</tools>\n\nFor each function call, return the function name and arguments inside <function></function> XML tags.\n";
        return out;
    }

    std::string render_assistant_tool_calls(std::span<const ToolCall> calls) const override {
        std::string out;
        for (const ToolCall& call : calls) {
            out += "<function name=" + json_string(call.name) + ">";
            for (const JsonMember& member : object_members(call.arguments)) {
                out += "<param name=" + json_string(member.key) + ">" + member.value + "</param>";
            }
            out += "</function>";
        }
        return out;
    }

    std::string render_tool_result(const ChatMessage& message) const override {
        return "<tool_response>\n" + message.content.value_or("") + "\n</tool_response>";
    }

    ToolParseResult parse_generation(std::string_view text) const override {
        ToolParseResult result;
        constexpr std::string_view start = "<function name=";
        constexpr std::string_view end = "</function>";
        std::size_t cursor = 0;
        while ((cursor = text.find(start, cursor)) != std::string_view::npos) {
            const std::size_t name_begin = cursor + start.size();
            if (name_begin >= text.size() || text[name_begin] != '"') break;
            const std::size_t name_end = text.find('"', name_begin + 1);
            const std::size_t body_begin = name_end == std::string_view::npos ? name_end : text.find('>', name_end);
            const std::size_t body_end = body_begin == std::string_view::npos ? body_begin : text.find(end, body_begin + 1);
            if (body_begin == std::string_view::npos || body_end == std::string_view::npos) break;
            const std::string name(text.substr(name_begin + 1, name_end - name_begin - 1));
            const std::string_view body = text.substr(body_begin + 1, body_end - body_begin - 1);
            std::string arguments = "{";
            std::size_t param_cursor = 0;
            std::size_t parameter_count = 0;
            constexpr std::string_view param_start = "<param name=";
            constexpr std::string_view param_end = "</param>";
            while ((param_cursor = body.find(param_start, param_cursor)) != std::string_view::npos) {
                const std::size_t param_name_begin = param_cursor + param_start.size();
                const std::size_t param_name_end = body.find('"', param_name_begin + 1);
                const std::size_t value_begin = param_name_end == std::string_view::npos ? param_name_end : body.find('>', param_name_end);
                const std::size_t value_end = value_begin == std::string_view::npos ? value_begin : body.find(param_end, value_begin + 1);
                if (param_name_begin >= body.size() || body[param_name_begin] != '"' ||
                    value_begin == std::string_view::npos || value_end == std::string_view::npos) break;
                if (parameter_count++) arguments += ",";
                arguments += json_string(body.substr(param_name_begin + 1, param_name_end - param_name_begin - 1));
                arguments += ":" + trim(body.substr(value_begin + 1, value_end - value_begin - 1));
                param_cursor = value_end + param_end.size();
            }
            arguments += "}";
            result.calls.push_back({"call_" + std::to_string(result.calls.size()), name, arguments});
            cursor = body_end + end.size();
        }
        if (!result.calls.empty()) {
            result.status = ToolParseStatus::Complete;
            result.assistant_text = remove_calls(text);
            result.consumed_bytes = text.size();
        } else if (text.find(start) != std::string_view::npos && text.find(end) == std::string_view::npos) {
            result.status = ToolParseStatus::Incomplete;
            result.assistant_text = std::string(text.substr(0, text.find(start)));
        }
        return result;
    }

private:
    static std::string remove_calls(std::string_view text) {
        std::string result(text);
        constexpr std::string_view start = "<function name=";
        constexpr std::string_view end = "</function>";
        for (std::size_t pos = 0; (pos = result.find(start, pos)) != std::string::npos;) {
            const std::size_t finish = result.find(end, pos + start.size());
            if (finish == std::string::npos) { result.erase(pos); break; }
            result.erase(pos, finish + end.size() - pos);
        }
        return result;
    }
};

} // namespace

std::unique_ptr<IChatToolCallCodec> make_minicpm5_tool_call_codec() {
    return std::make_unique<MiniCpm5ToolCallCodec>();
}

} // namespace celeg
