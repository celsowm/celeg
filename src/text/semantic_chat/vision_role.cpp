#include "celeg/text/semantic_chat_templates.hpp"

#include "celeg/text/detail/chat_template_support.hpp"

#include <stdexcept>

namespace celeg {

std::string VisionRoleChatTemplate::format(
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool add_generation_prompt,
    const ChatTemplateOptions& options) const {
    chat_template_detail::reject_unhandled_options(options, "chat:vision-role");
    if (!tools.empty()) throw std::invalid_argument("vision-role tool-call formatting is not implemented");
    std::string out;
    for (const ChatMessage& message : messages) {
        if (message.role == ChatRole::Tool || message.role == ChatRole::Developer) {
            throw std::invalid_argument("vision-role profile does not support this chat role");
        }
        std::string content = message.content.value_or("");
        constexpr std::string_view generic_marker = "<|image|>";
        constexpr std::string_view vision_marker = "<|vision_start|><|image_pad|><|vision_end|>";
        for (std::size_t found = content.find(generic_marker); found != std::string::npos;
             found = content.find(generic_marker, found + vision_marker.size())) {
            content.replace(found, generic_marker.size(), vision_marker);
        }
        const char* role = message.role == ChatRole::System ? "system" :
                           message.role == ChatRole::User ? "user" : "assistant";
        out += "<|im_start|>" + std::string(role) + "\n" + content + "<|im_end|>\n";
    }
    if (add_generation_prompt) out += "<|im_start|>assistant\n";
    return out;
}

void add_vision_role_chat_template(ChatTemplateCatalog& catalog) {
    ChatCapabilities capabilities;
    capabilities.vision = true;
    capabilities.image_marker = "<|image_pad|>";
    capabilities.roles.developer = false;
    capabilities.roles.tool = false;
    catalog.add("chat:vision-role", std::make_unique<VisionRoleChatTemplate>(),
                nullptr, capabilities);
}

} // namespace celeg
