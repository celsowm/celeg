#pragma once

#include "celeg/text/chat_profile.hpp"

#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace celeg {

enum class ChatRole {
    System,
    Developer,
    User,
    Assistant,
    Tool,
};

struct ChatMessage {
    ChatRole role;
    std::string content;
};

// Interface Segregation Principle: tokenizer chat formatting depends only on
// this narrow interface, not on model metadata.
class IChatTemplate {
public:
    virtual ~IChatTemplate() = default;

    // Formats a full multi-turn conversation. Throws std::invalid_argument
    // for roles the template does not (yet) support, e.g. Tool.
    virtual std::string format(std::span<const ChatMessage> messages,
                               bool add_generation_prompt) const = 0;
    virtual ChatTemplateKind kind() const = 0;
};

// Instruct chat template:
//   <|startoftext|>(<|im_start|>{role}\n{content}<|im_end|>\n)*
//   [<|im_start|>assistant\n]
// System and Developer messages both render under the "system" role tag;
// This profile has no separate developer-turn syntax.
class Lfm2InstructChatTemplate final : public IChatTemplate {
public:
    std::string format(std::span<const ChatMessage> messages,
                       bool add_generation_prompt) const override;
    ChatTemplateKind kind() const override { return ChatTemplateKind::Lfm2Instruct; }
};

// Factory: returns the chat template implementation for a given kind.
std::unique_ptr<IChatTemplate> make_chat_template(ChatTemplateKind kind);
std::unique_ptr<IChatTemplate> make_chat_template(std::string_view profile_id);

} // namespace celeg
