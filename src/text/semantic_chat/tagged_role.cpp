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
}

std::string TaggedRoleChatTemplate::format(
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool add_generation_prompt,
    const ChatTemplateOptions& options) const {
    const bool thinking = options.enable_thinking.value_or(true);
    std::string out = "<role>SYSTEM</role>";
    const auto codec = make_tagged_role_tool_codec();
    if (!tools.empty()) out += codec->render_tool_definitions(tools, {});
    out += "detailed thinking " + std::string(thinking ? "on" : "off") + "<|role_end|>";
    for (const ChatMessage& message : messages) {
        if (message.role == ChatRole::Tool) {
            out += "<role>OBSERVATION</role>\n" + codec->render_tool_result(message) + "<|role_end|>";
            continue;
        }
        out += "<role>" + std::string(role_name(message.role)) + "</role>" +
               message.content.value_or("");
        if (message.role == ChatRole::Assistant && !message.tool_calls.empty()) {
            out += codec->render_assistant_tool_calls(message.tool_calls);
        }
        out += "<|role_end|>";
    }
    if (add_generation_prompt) {
        out += "<role>ASSISTANT</role>\n<think>";
        if (!thinking) out += "</think>";
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
