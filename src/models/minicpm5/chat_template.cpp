#include "celeg/models/minicpm5/chat_template.hpp"

#include "celeg/text/chat_profile.hpp"
#include "celeg/text/detail/chat_template_support.hpp"

#include <cctype>
#include <stdexcept>
#include <utility>
#include <vector>

namespace celeg {
namespace {

std::string tools_block(std::span<const ChatToolDefinition> tools) {
    if (tools.empty()) return {};
    std::string out = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n<tools>\n";
    for (const ChatToolDefinition& tool : tools) {
        out += "{\"type\":\"function\",\"function\":{\"name\":" +
            chat_template_detail::json_string(tool.function.name);
        out += ",\"description\":" + chat_template_detail::json_string(tool.function.description);
        out += ",\"parameters\":" +
            (tool.function.parameters.serialized.empty() ? "{}" : tool.function.parameters.serialized) + "}}\n";
    }
    return out + "</tools>\n\nFor each function call, return the function name and arguments inside <function></function> XML tags.\n";
}

const char* role_tag(ChatRole role) {
    switch (role) {
        case ChatRole::System:
        case ChatRole::Developer: return "system";
        case ChatRole::User:
        case ChatRole::Tool: return "user";
        case ChatRole::Assistant: return "assistant";
    }
    throw std::invalid_argument("unknown chat role");
}

std::vector<std::pair<std::string, std::string>> object_members(std::string_view object) {
    std::vector<std::pair<std::string, std::string>> members;
    if (object.size() < 2 || object.front() != '{' || object.back() != '}') return members;
    std::size_t cursor = 1;
    while (cursor + 1 < object.size()) {
        while (cursor + 1 < object.size() &&
               (std::isspace(static_cast<unsigned char>(object[cursor])) || object[cursor] == ',')) ++cursor;
        if (cursor + 1 >= object.size() || object[cursor] != '"') break;
        const std::size_t key_begin = ++cursor;
        while (cursor < object.size() && object[cursor] != '"') ++cursor;
        if (cursor >= object.size()) break;
        std::string key(object.substr(key_begin, cursor - key_begin));
        ++cursor;
        while (cursor < object.size() && (std::isspace(static_cast<unsigned char>(object[cursor])) || object[cursor] == ':')) ++cursor;
        const std::size_t value_begin = cursor;
        int depth = 0;
        bool quoted = false;
        bool escaped = false;
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
        std::string value(object.substr(value_begin, cursor - value_begin));
        while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
        members.emplace_back(std::move(key), std::move(value));
        if (cursor < object.size() && object[cursor] == ',') ++cursor;
    }
    return members;
}

} // namespace

std::string MiniCpm5InstructChatTemplate::format(
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool add_generation_prompt,
    const ChatTemplateOptions& options) const {
    chat_template_detail::reject_unhandled_options(options, "minicpm5-instruct");
    std::string out = "<bos>";
    const std::string tool_block = tools_block(tools);
    std::size_t first_message = 0;
    if (!tool_block.empty()) {
        out += "<|im_start|>system\n";
        if (!messages.empty() && messages.front().role == ChatRole::System) {
            out += messages.front().content.value_or("") + "\n\n";
            first_message = 1;
        }
        out += tool_block + "<|im_end|>\n";
    }
    for (std::size_t index = first_message; index < messages.size(); ++index) {
        const ChatMessage& message = messages[index];
        if (message.role == ChatRole::Tool) {
            out += "<|im_start|>user\n<tool_response>\n" + message.content.value_or("") +
                "\n</tool_response><|im_end|>\n";
            continue;
        }
        out += "<|im_start|>" + std::string(role_tag(message.role) ) + "\n" + message.content.value_or("");
        if (message.role == ChatRole::Assistant) {
            for (const ToolCall& call : message.tool_calls) {
                out += "<function name=\"" + call.name + "\">";
                for (const auto& pair : object_members(call.arguments)) {
                    out += "<param name=\"" + pair.first + "\">" + pair.second + "</param>";
                }
                out += "</function>";
            }
        }
        out += "<|im_end|>\n";
    }
    if (add_generation_prompt) out += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    return out;
}

void add_minicpm5_chat_profile(ChatProfileCatalog& catalog) {
    ChatCapabilities capabilities;
    capabilities.assistant_tool_calls = true;
    capabilities.parallel_tool_calls = true;
    capabilities.native_tool_call_codec = true;
    capabilities.roles.developer = true;
    capabilities.roles.tool = true;
    catalog.add("minicpm5-instruct", std::make_unique<MiniCpm5InstructChatTemplate>(),
                make_minicpm5_tool_call_codec(), capabilities);
}

} // namespace celeg
