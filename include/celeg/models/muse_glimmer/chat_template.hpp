#pragma once

#include "celeg/text/chat_template.hpp"

namespace celeg {

class MuseGlimmerChatTemplate final : public IChatTemplate {
public:
    using IChatTemplate::format;
    std::string format(std::span<const ChatMessage> messages,
                       std::span<const ChatToolDefinition> tools,
                       bool add_generation_prompt,
                       const ChatTemplateOptions& options) const override;
};

void add_muse_glimmer_chat_profile(ChatProfileCatalog& catalog);

} // namespace celeg
