#include "celeg/text/semantic_chat_templates.hpp"

#include "celeg/text/detail/chat_template_support.hpp"

namespace celeg {

namespace {
const char* role_name(ChatRole role) {
    switch (role) {
        case ChatRole::System: return "SYSTEM";
        case ChatRole::Developer: return "DEVELOPER";
        case ChatRole::User: return "HUMAN";
        case ChatRole::Assistant: return "ASSISTANT";
        case ChatRole::Tool: return "OBSERVATION";
    }
    return "UNKNOWN";
}

bool has_thinking_marker(std::string_view content) {
    return content.find("detailed thinking on") != std::string_view::npos ||
           content.find("detailed thinking off") != std::string_view::npos;
}
}

std::string TaggedRoleChatTemplate::format(
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool add_generation_prompt,
    const ChatTemplateOptions& options) const {
    const bool thinking = options.enable_thinking.value_or(true);
    std::string out = "<role>SYSTEM</role>";
    const auto codec = make_tagged_role_tool_codec();
    const bool first_is_system = !messages.empty() &&
        messages.front().role == ChatRole::System;
    const std::string first_system_content = first_is_system
        ? messages.front().content.value_or("") : "";
    if (!tools.empty()) {
        if (first_is_system) out += first_system_content + "\n";
        out += codec->render_tool_definitions(tools, options.tool_choice);
        if (first_is_system && has_thinking_marker(first_system_content)) {
            out += "<|role_end|>";
        } else {
            out += "detailed thinking " + std::string(thinking ? "on" : "off") +
                   "<|role_end|>";
        }
    } else if (first_is_system) {
        if (has_thinking_marker(first_system_content)) {
            out += first_system_content + "<|role_end|>";
        } else {
            out += first_system_content + "\n";
            out += "detailed thinking " + std::string(thinking ? "on" : "off") +
                   "<|role_end|>";
        }
    } else {
        out += "detailed thinking " + std::string(thinking ? "on" : "off") +
               "<|role_end|>";
    }
    for (std::size_t index = 0; index < messages.size(); ++index) {
        const ChatMessage& message = messages[index];
        if (message.role == ChatRole::System && index == 0) continue;
        if (message.role == ChatRole::Tool) {
            if (index == 0 || messages[index - 1].role != ChatRole::Tool) {
                out += "<role>OBSERVATION</role>";
            }
            out += "\n" + codec->render_tool_result(message);
            if (index + 1 == messages.size() || messages[index + 1].role != ChatRole::Tool) {
                out += "<|role_end|>";
            }
            continue;
        }
        const std::string content = message.content.value_or("");
        if (message.role == ChatRole::Assistant) {
            out += "<role>ASSISTANT</role>\n<think></think>" + content;
            if (!message.tool_calls.empty()) {
                if (!content.empty()) out += "\n";
                out += codec->render_assistant_tool_calls(message.tool_calls);
            }
        } else {
            out += "<role>" + std::string(role_name(message.role)) + "</role>" + content;
        }
        out += "<|role_end|>";
    }
    if (add_generation_prompt) {
        out += "<role>ASSISTANT</role>\n";
        out += thinking ? "<think>" : "<think></think>";
    }
    return out;
}

void add_tagged_role_chat_template(ChatTemplateCatalog& catalog) {
    ChatCapabilities capabilities;
    capabilities.assistant_tool_calls = true;
    capabilities.parallel_tool_calls = true;
    capabilities.native_tool_call_codec = true;
    capabilities.roles.developer = true;
    capabilities.roles.tool = true;
    catalog.add("chat:tagged-role", std::make_unique<TaggedRoleChatTemplate>(),
                make_tagged_role_tool_codec(), capabilities);
}

} // namespace celeg
