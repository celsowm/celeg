#include "celeg/models/muse_glimmer/chat_template.hpp"

#include "celeg/text/detail/chat_template_support.hpp"

#include <stdexcept>

namespace celeg {

std::string MuseGlimmerChatTemplate::format(
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool add_generation_prompt,
    const ChatTemplateOptions& options) const {
    chat_template_detail::reject_unhandled_options(options, "muse-glimmer");
    if (!tools.empty()) {
        throw std::invalid_argument(
            "Muse Glimmer ATEM tool formatting requires structured tool-call support");
    }
    std::string out = "<|begin_of_text|>";
    bool has_system = false;
    for (const ChatMessage& message : messages) {
        if (message.role == ChatRole::System) has_system = true;
    }
    if (!has_system) {
        out += "<|start|>system<|message|>You are a helpful AI assistant."
               "\nKnowledge cutoff: 2026-01-04.\n\n"
               "Reasoning strength: high.\n\n"
               "# Valid recipients: \"self\", \"user\".<|eot|>";
    }
    for (const ChatMessage& message : messages) {
        if (message.role == ChatRole::Tool || message.role == ChatRole::Developer) {
            throw std::invalid_argument("Muse Glimmer profile does not support this chat role");
        }
        const char* role = message.role == ChatRole::System ? "system" :
                           message.role == ChatRole::User ? "user" : "assistant";
        out += "<|start|>" + std::string(role) + "<|message|>" +
               message.content.value_or("") + "<|eot|>";
    }
    if (add_generation_prompt) out += "<|start|>assistant";
    return out;
}

void add_muse_glimmer_chat_profile(ChatProfileCatalog& catalog) {
    ChatCapabilities capabilities;
    capabilities.vision = true;
    capabilities.video = true;
    capabilities.image_marker = "<|patch|>";
    capabilities.video_marker = "<|video|>";
    capabilities.roles.developer = false;
    capabilities.roles.tool = false;
    catalog.add("muse-glimmer", std::make_unique<MuseGlimmerChatTemplate>(), nullptr,
                capabilities);
}

} // namespace celeg
