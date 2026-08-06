#include "celeg/models/qwen35/chat_template.hpp"

#include "celeg/text/detail/chat_template_support.hpp"

#include <stdexcept>

namespace celeg {

std::string Qwen35InstructChatTemplate::format(
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool add_generation_prompt,
    const ChatTemplateOptions& options) const {
    chat_template_detail::reject_unhandled_options(options, "qwen35-instruct");
    if (!tools.empty()) throw std::invalid_argument("Qwen 3.5 tool-call formatting is not implemented");
    std::string out;
    for (const ChatMessage& message : messages) {
        if (message.role == ChatRole::Tool || message.role == ChatRole::Developer) {
            throw std::invalid_argument("Qwen 3.5 profile does not support this chat role");
        }
        std::string content = message.content.value_or("");
        constexpr std::string_view generic_marker = "<|image|>";
        constexpr std::string_view qwen_marker = "<|vision_start|><|image_pad|><|vision_end|>";
        for (std::size_t found = content.find(generic_marker); found != std::string::npos;
             found = content.find(generic_marker, found + qwen_marker.size())) {
            content.replace(found, generic_marker.size(), qwen_marker);
        }
        const char* role = message.role == ChatRole::System ? "system" :
                           message.role == ChatRole::User ? "user" : "assistant";
        out += "<|im_start|>" + std::string(role) + "\n" + content + "<|im_end|>\n";
    }
    if (add_generation_prompt) out += "<|im_start|>assistant\n";
    return out;
}

void add_qwen35_chat_profile(ChatProfileCatalog& catalog) {
    ChatCapabilities capabilities;
    capabilities.vision = true;
    capabilities.image_marker = "<|image_pad|>";
    capabilities.roles.developer = false;
    capabilities.roles.tool = false;
    catalog.add("qwen35-instruct", std::make_unique<Qwen35InstructChatTemplate>(),
                nullptr, capabilities);
}

} // namespace celeg
