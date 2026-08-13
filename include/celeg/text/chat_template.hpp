#pragma once

#include "celeg/text/chat_contract.hpp"
#include "celeg/checkpoint/metadata.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace celeg {

class ITokenizer;

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

// A model-scoped interaction contract compiled from checkpoint template
// metadata.  It deliberately carries no model, repository, or architecture
// identity: the rendered wire format is selected solely by template source or
// by tokenizer evidence when the source is absent.
class ResolvedChatTemplate final : public IChatTemplate {
public:
    using IChatTemplate::format;

    std::string format(std::span<const ChatMessage> messages,
                       std::span<const ChatToolDefinition> tools,
                       bool add_generation_prompt,
                       const ChatTemplateOptions& options) const override;

    const std::string& source_origin() const noexcept { return source_origin_; }
    const std::string& fingerprint() const noexcept { return fingerprint_; }
    ChatCapabilities capabilities() const noexcept { return capabilities_; }
    const IChatToolCallCodec* tool_codec() const noexcept { return tool_codec_.get(); }

private:
    friend ResolvedChatTemplate resolve_chat_template(
        const CheckpointMetadata&, const ITokenizer&,
        const std::optional<std::filesystem::path>&);

    bool include_bos_ = false;
    bool tools_in_system_ = false;
    std::string source_origin_;
    std::string fingerprint_;
    ChatCapabilities capabilities_;
    std::shared_ptr<const IChatToolCallCodec> tool_codec_;
};

// Resolves a checkpoint's interaction format.  An override wins over
// checkpoint metadata; source-less checkpoints may use only a conservative
// tokenizer-evidence inference.  Templates that are present but outside the
// deterministic supported subset fail instead of silently changing protocol.
ResolvedChatTemplate resolve_chat_template(
    const CheckpointMetadata& metadata,
    const ITokenizer& tokenizer,
    const std::optional<std::filesystem::path>& override_file = std::nullopt);

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
