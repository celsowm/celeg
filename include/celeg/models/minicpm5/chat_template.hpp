#pragma once

#include "celeg/text/chat_template.hpp"

namespace celeg {

std::unique_ptr<IChatToolCallCodec> make_minicpm5_tool_call_codec();

class MiniCpm5InstructChatTemplate final : public IChatTemplate {
public:
    using IChatTemplate::format;
    std::string format(std::span<const ChatMessage> messages,
                       std::span<const ChatToolDefinition> tools,
                       bool add_generation_prompt,
                       const ChatTemplateOptions& options) const override;
};

void add_minicpm5_chat_profile(ChatProfileCatalog& catalog);

} // namespace celeg
