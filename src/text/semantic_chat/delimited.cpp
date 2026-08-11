#include "celeg/text/semantic_chat_templates.hpp"

#include "celeg/text/chat_template.hpp"
#include "celeg/text/detail/chat_template_support.hpp"

#include <stdexcept>

namespace celeg {
namespace {

const char* role_tag(ChatRole role) {
    switch (role) {
        case ChatRole::System:
        case ChatRole::Developer: return "system";
        case ChatRole::User: return "user";
        case ChatRole::Assistant: return "assistant";
        case ChatRole::Tool: return "tool";
    }
    throw std::invalid_argument("unknown chat role");
}

} // namespace

std::string DelimitedChatTemplate::format(
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool add_generation_prompt,
    const ChatTemplateOptions& options) const {
    chat_template_detail::reject_unhandled_options(options, "chat:delimited");
    std::string out = "<|startoftext|>";
    std::string system_tools;
    if (!tools.empty()) {
        system_tools = "List of tools: <|tool_list_start|>[";
        for (std::size_t i = 0; i < tools.size(); ++i) {
            if (i != 0) system_tools += ", ";
            system_tools += "{\"type\":\"function\",\"function\":{\"name\":";
            system_tools += chat_template_detail::json_string(tools[i].function.name);
            system_tools += ",\"description\":" +
                chat_template_detail::json_string(tools[i].function.description);
            system_tools += ",\"parameters\":" +
                (tools[i].function.parameters.serialized.empty() ? "{}" :
                 tools[i].function.parameters.serialized);
            system_tools += "}}";
        }
        system_tools += "]<|tool_list_end|>";
    }
    for (const ChatMessage& message : messages) {
        if (message.role == ChatRole::System && !system_tools.empty()) {
            const std::string content = message.content.value_or("");
            system_tools = content + (content.empty() ? "" : "\n") + system_tools;
            continue;
        }
        out += "<|im_start|>";
        out += role_tag(message.role);
        out += "\n";
        const std::string content = message.content.value_or("");
        out += content;
        if (message.role == ChatRole::Tool) {
            out = out.substr(0, out.size() - content.size()) +
                "<|tool_response_start|>" + content + "<|tool_response_end|>";
        }
        for (const auto& call : message.tool_calls) {
            out += "<|tool_call_start|>[" + call.name + "(" + call.arguments +
                ")]<|tool_call_end|>";
        }
        out += "<|im_end|>\n";
    }
    if (!system_tools.empty()) {
        out = "<|startoftext|><|im_start|>system\n" + system_tools +
            "<|im_end|>\n" + out.substr(std::string("<|startoftext|>").size());
    }
    if (add_generation_prompt) out += "<|im_start|>assistant\n";
    return out;
}

void add_delimited_chat_template(ChatTemplateCatalog& catalog) {
    ChatCapabilities capabilities;
    capabilities.assistant_tool_calls = true;
    capabilities.parallel_tool_calls = true;
    capabilities.native_tool_call_codec = true;
    capabilities.roles.developer = true;
    capabilities.roles.tool = true;
    catalog.add("chat:delimited", std::make_unique<DelimitedChatTemplate>(),
                make_delimited_tool_codec(), capabilities);
}

} // namespace celeg
