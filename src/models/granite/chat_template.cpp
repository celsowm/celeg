#include "celeg/models/granite/chat_template.hpp"

#include "celeg/text/detail/chat_template_support.hpp"

#include <stdexcept>

namespace celeg {
namespace {

const char* role_tag(ChatRole role) {
    switch (role) {
        case ChatRole::System: return "system";
        case ChatRole::Developer: return "developer";
        case ChatRole::User: return "user";
        case ChatRole::Assistant: return "assistant";
        case ChatRole::Tool:
            throw std::invalid_argument("GraniteInstructChatTemplate does not support tool messages");
    }
    throw std::invalid_argument("unknown chat role");
}

void append_message(std::string& out, ChatRole role, std::string_view content) {
    out += "<|start_of_role|>";
    out += role_tag(role);
    out += "<|end_of_role|>";
    out += content;
    out += "<|end_of_text|>\n";
}

std::string tools_block(std::span<const ChatToolDefinition> tools) {
    if (tools.empty()) return {};
    std::string out =
        "You are a helpful assistant with access to the following tools. You may call one or more tools to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
    for (const ChatToolDefinition& tool : tools) {
        out += "\n{\"type\":\"function\",\"function\":{\"name\":";
        out += chat_template_detail::json_string(tool.function.name);
        out += ",\"description\":" + chat_template_detail::json_string(tool.function.description);
        out += ",\"parameters\":" +
            (tool.function.parameters.serialized.empty() ? "{}" : tool.function.parameters.serialized);
        out += "}}";
    }
    return out +
        "\n</tools>\n\nFor each tool call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>.";
}

} // namespace

std::string GraniteInstructChatTemplate::format(
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool add_generation_prompt,
    const ChatTemplateOptions& options) const {
    chat_template_detail::reject_unhandled_options(options, "granite-instruct");
    std::string out = "<|startoftext|>";
    const bool has_system_message = !messages.empty() && messages.front().role == ChatRole::System;
    const std::string tool_block = tools_block(tools);
    if (!has_system_message) {
        append_message(out, ChatRole::System,
            "Knowledge Cutoff Date: April 2024. You are Granite, developed by IBM. You are a helpful AI assistant." + tool_block);
    }
    for (std::size_t index = 0; index < messages.size(); ++index) {
        const ChatMessage& message = messages[index];
        std::string content = message.content.value_or("");
        if (index == 0 && message.role == ChatRole::System && !tool_block.empty()) {
            content += "\n\n" + tool_block;
        }
        if (message.role == ChatRole::Tool) {
            out += "<|start_of_role|>user<|end_of_role|>\n<tool_response>\n" + content +
                "\n</tool_response><|end_of_text|>\n";
            continue;
        }
        out += "<|start_of_role|>";
        out += role_tag(message.role);
        out += "<|end_of_role|>" + content;
        if (message.role == ChatRole::Assistant) {
            for (const ToolCall& call : message.tool_calls) {
                if (!content.empty() || &call != &message.tool_calls.front()) out += "\n";
                out += "<tool_call>\n{\"name\": ";
                out += chat_template_detail::json_string(call.name);
                out += ", \"arguments\": " + call.arguments + "}\n</tool_call>";
            }
        }
        out += "<|end_of_text|>\n";
    }
    if (add_generation_prompt) out += "<|start_of_role|>assistant<|end_of_role|>";
    return out;
}

void add_granite_chat_profile(ChatProfileCatalog& catalog) {
    catalog.add("granite-instruct", std::make_unique<GraniteInstructChatTemplate>());
}

} // namespace celeg
