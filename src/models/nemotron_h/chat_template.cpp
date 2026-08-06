#include "celeg/models/nemotron_h/chat_template.hpp"

#include "celeg/text/detail/chat_template_support.hpp"

#include <stdexcept>

namespace celeg {

std::string NemotronHInstructChatTemplate::format(
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool add_generation_prompt,
    const ChatTemplateOptions& options) const {
    if (!tools.empty()) {
        throw std::invalid_argument("Nemotron-H tool-call formatting is not implemented");
    }
    const bool thinking = options.enable_thinking.value_or(true);
    std::string out;
    for (const ChatMessage& message : messages) {
        if (message.role == ChatRole::Tool || message.role == ChatRole::Developer) {
            throw std::invalid_argument("Nemotron-H profile does not support this chat role");
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

void add_nemotron_h_chat_profile(ChatProfileCatalog& catalog) {
    ChatCapabilities capabilities;
    capabilities.roles.developer = false;
    capabilities.roles.tool = false;
    catalog.add("nemotron-h-instruct", std::make_unique<NemotronHInstructChatTemplate>(),
                nullptr, capabilities);
}

} // namespace celeg
