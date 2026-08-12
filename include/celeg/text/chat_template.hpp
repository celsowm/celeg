#pragma once

#include "celeg/text/chat_contract.hpp"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace celeg {

using ChatToolDefinition = ToolDefinition;

struct ChatTemplateOptions {
    std::optional<bool> enable_thinking;
    ToolChoice tool_choice{};
};

// Interface Segregation Principle: tokenizer chat formatting depends only on
// this narrow interface, not on model metadata.
class IChatTemplate {
public:
    virtual ~IChatTemplate() = default;

    // Formats a full multi-turn conversation. Options are part of the
    // contract so a renderer cannot silently discard a caller's request.
    std::string format(std::span<const ChatMessage> messages,
                       bool add_generation_prompt) const {
        return format(messages, std::span<const ChatToolDefinition>{},
                      add_generation_prompt, {});
    }
    std::string format(std::span<const ChatMessage> messages,
                       std::span<const ChatToolDefinition> tools,
                       bool add_generation_prompt) const {
        return format(messages, tools, add_generation_prompt, {});
    }
    virtual std::string format(std::span<const ChatMessage> messages,
                               std::span<const ChatToolDefinition> tools,
                               bool add_generation_prompt,
                               const ChatTemplateOptions& options) const = 0;
};

std::string render_chat(std::span<const ChatMessage> messages,
                        const IChatTemplate& chat_template,
                        bool add_generation_prompt = true);
std::string render_chat(std::span<const ChatMessage> messages,
                        std::span<const ChatToolDefinition> tools,
                        const IChatTemplate& chat_template,
                        bool add_generation_prompt = true);
std::string render_chat(std::span<const ChatMessage> messages,
                        std::span<const ChatToolDefinition> tools,
                        const IChatTemplate& chat_template,
                        bool add_generation_prompt,
                        const ChatTemplateOptions& options);

class ChatTemplateCatalog {
public:
    void add(std::string template_id, std::unique_ptr<IChatTemplate> chat_template,
             std::unique_ptr<IChatToolCallCodec> tool_call_codec = nullptr,
             ChatCapabilities capabilities = {});
    void freeze();
    const IChatTemplate& find(std::string_view template_id) const;
    ChatCapabilities capabilities(std::string_view template_id) const;
    const IChatToolCallCodec* tool_codec(std::string_view template_id) const;

private:
    struct Entry {
        std::unique_ptr<IChatTemplate> chat_template;
        std::shared_ptr<const IChatToolCallCodec> tool_call_codec;
        ChatCapabilities capabilities;
    };
    std::unordered_map<std::string, Entry> entries_;
    bool frozen_ = false;
};

ChatTemplateCatalog make_chat_template_catalog();

} // namespace celeg
