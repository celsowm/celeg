#pragma once

#include "celeg/text/chat_profile.hpp"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace celeg {

using ChatToolDefinition = ToolDefinition;

// Interface Segregation Principle: tokenizer chat formatting depends only on
// this narrow interface, not on model metadata.
class IChatTemplate {
public:
    virtual ~IChatTemplate() = default;

    // Formats a full multi-turn conversation.
    std::string format(std::span<const ChatMessage> messages,
                       bool add_generation_prompt) const {
        return format(messages, std::span<const ChatToolDefinition>{},
                      add_generation_prompt);
    }
    virtual std::string format(std::span<const ChatMessage> messages,
                               std::span<const ChatToolDefinition> tools,
                               bool add_generation_prompt) const = 0;
};

// Instruct chat template:
//   <|startoftext|>(<|im_start|>{role}\n{content}<|im_end|>\n)*
//   [<|im_start|>assistant\n]
// System and Developer messages both render under the "system" role tag;
// This profile has no separate developer-turn syntax.
class Lfm2InstructChatTemplate final : public IChatTemplate {
public:
    std::string format(std::span<const ChatMessage> messages,
                       std::span<const ChatToolDefinition> tools,
                       bool add_generation_prompt) const override;
};

std::string render_chat(std::span<const ChatMessage> messages,
                        const IChatTemplate& chat_template,
                        bool add_generation_prompt = true);
std::string render_chat(std::span<const ChatMessage> messages,
                        std::span<const ChatToolDefinition> tools,
                        const IChatTemplate& chat_template,
                        bool add_generation_prompt = true);

class ChatProfileCatalog {
public:
    void add(std::string profile_id, std::unique_ptr<IChatTemplate> chat_template,
             std::unique_ptr<IChatToolCallCodec> tool_call_codec = nullptr,
             ChatCapabilities capabilities = {});
    void freeze();
    const IChatTemplate& find(std::string_view profile_id) const;
    ChatCapabilities capabilities(std::string_view profile_id) const;

private:
    struct Entry {
        std::unique_ptr<IChatTemplate> chat_template;
        std::unique_ptr<IChatToolCallCodec> tool_call_codec;
        ChatCapabilities capabilities;
    };
    std::unordered_map<std::string, Entry> entries_;
    bool frozen_ = false;
};

ChatProfileCatalog make_chat_profile_catalog();

// Granite instruct chat template:
//   (<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n)*
//   [<|start_of_role|>assistant<|end_of_role|>]
// When no system message is supplied, Granite's standard assistant system
// message is inserted before the conversation.
class GraniteInstructChatTemplate final : public IChatTemplate {
public:
    std::string format(std::span<const ChatMessage> messages,
                       std::span<const ChatToolDefinition> tools,
                       bool add_generation_prompt) const override;
};

class Gemma4InstructChatTemplate final : public IChatTemplate {
public:
    std::string format(std::span<const ChatMessage> messages,
                       std::span<const ChatToolDefinition> tools,
                       bool add_generation_prompt) const override;
};

class MiniCpm5InstructChatTemplate final : public IChatTemplate {
public:
    std::string format(std::span<const ChatMessage> messages,
                       std::span<const ChatToolDefinition> tools,
                       bool add_generation_prompt) const override;
};


} // namespace celeg
