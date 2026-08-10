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
        std::string content = message.content.value_or("");
        constexpr std::string_view generic_image_marker = "<|image|>";
        constexpr std::string_view muse_image_marker = "<|patch|>";
        for (std::size_t found = content.find(generic_image_marker);
             found != std::string::npos;
             found = content.find(generic_image_marker, found + muse_image_marker.size())) {
            content.replace(found, generic_image_marker.size(), muse_image_marker);
        }
        const char* role = message.role == ChatRole::System ? "system" :
                           message.role == ChatRole::User ? "user" : "assistant";
        out += "<|start|>" + std::string(role) + "<|message|>" +
               content + "<|eot|>";
    }
    if (add_generation_prompt) out += "<|start|>assistant";
    return out;
}

void add_muse_glimmer_chat_profile(ChatProfileCatalog& catalog) {
    ChatCapabilities capabilities;
    capabilities.vision = true;
    // The current serving contract accepts image data URLs only. Keep the
    // capability catalog honest until a temporal sampling/provider path is
    // added; the text and image tiers remain independently usable.
    capabilities.video = false;
    capabilities.image_marker = "<|patch|>";
    capabilities.roles.developer = false;
    capabilities.roles.tool = false;
    catalog.add("muse-glimmer", std::make_unique<MuseGlimmerChatTemplate>(), nullptr,
                capabilities);
}

} // namespace celeg
