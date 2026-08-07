#pragma once

#include "celeg/text/chat_template.hpp"

namespace celeg {

std::unique_ptr<IChatToolCallCodec> make_smollm3_tool_call_codec();

class SmolLm3InstructChatTemplate final : public IChatTemplate {
public:
    using IChatTemplate::format;
    explicit SmolLm3InstructChatTemplate(bool enable_thinking = true)
        : enable_thinking_(enable_thinking) {}

    std::string format(std::span<const ChatMessage> messages,
                       std::span<const ChatToolDefinition> tools,
                       bool add_generation_prompt,
                       const ChatTemplateOptions& options) const override;

private:
    std::string format_with_thinking(std::span<const ChatMessage> messages,
                                     std::span<const ChatToolDefinition> tools,
                                     bool add_generation_prompt,
                                     bool enable_thinking) const;
    bool enable_thinking_;
};

void add_smollm3_chat_profile(ChatProfileCatalog& catalog);

} // namespace celeg
