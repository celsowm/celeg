#include "celeg/text/semantic_chat_templates.hpp"

#include "celeg/text/detail/chat_template_support.hpp"

#include <stdexcept>

namespace celeg {

std::string ThinkingRoleChatTemplate::format(
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool add_generation_prompt,
    const ChatTemplateOptions& options) const {
    if (!tools.empty()) {
        throw std::invalid_argument("thinking-role tool-call formatting is not implemented");
    }
    const bool thinking = options.enable_thinking.value_or(true);
    std::string out;
    for (const ChatMessage& message : messages) {
        if (message.role == ChatRole::Tool || message.role == ChatRole::Developer) {
            throw std::invalid_argument("thinking-role profile does not support this chat role");
        }
        const char* role = message.role == ChatRole::System ? "system" :
                           message.role == ChatRole::User ? "user" : "assistant";
        out += "<|im_start|>" + std::string(role) + "\n" +
               message.content.value_or("") + "<|im_end|>\n";
    }
    if (add_generation_prompt) {
        out += "<|im_start|>assistant\n";
        out += thinking ? "<think>\n" : "<think></think>";
    }
    return out;
}

void add_thinking_role_chat_template(ChatTemplateCatalog& catalog) {
    ChatCapabilities capabilities;
    capabilities.roles.developer = false;
    capabilities.roles.tool = false;
    catalog.add("chat:thinking-role", std::make_unique<ThinkingRoleChatTemplate>(),
                nullptr, capabilities);
}

} // namespace celeg
