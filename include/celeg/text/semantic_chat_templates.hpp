#pragma once

#include "celeg/text/chat_template.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace celeg {

enum class ChatInstructionKind : std::uint8_t {
    EmitLiteral,
    EmitMessageRole,
    EmitMessageContent,
    EmitToolDefinitions,
    EmitAssistantToolCalls,
    EmitToolResult,
    EmitGenerationPrompt,
};

struct ChatInstruction {
    ChatInstructionKind kind = ChatInstructionKind::EmitLiteral;
    std::string value;
    friend bool operator==(const ChatInstruction&, const ChatInstruction&) = default;
};

// The immutable result of importing a chat-template source. Importers may
// produce different syntax trees, but runtime formatting consumes this one
// semantic instruction vocabulary.
struct ChatTemplateProgram {
    std::vector<ChatInstruction> instructions;
    std::string source;

    void validate() const;
    std::string fingerprint() const;
};

class ProgramChatTemplate final : public IChatTemplate {
public:
    using IChatTemplate::format;
    explicit ProgramChatTemplate(ChatTemplateProgram program)
        : program_(std::move(program)) { program_.validate(); }
    std::string format(std::span<const ChatMessage>, std::span<const ChatToolDefinition>,
                       bool, const ChatTemplateOptions&) const override;
    const ChatTemplateProgram& program() const noexcept { return program_; }
private:
    ChatTemplateProgram program_;
};

// These names describe wire-level behavior, not checkpoint families.  A
// checkpoint importer selects one of these programs from its interaction
// evidence and the runtime only executes the selected semantics.
class DelimitedChatTemplate final : public IChatTemplate {
public:
    using IChatTemplate::format;
    std::string format(std::span<const ChatMessage>, std::span<const ChatToolDefinition>,
                       bool, const ChatTemplateOptions&) const override;
};

class RoleEnvelopeChatTemplate final : public IChatTemplate {
public:
    using IChatTemplate::format;
    std::string format(std::span<const ChatMessage>, std::span<const ChatToolDefinition>,
                       bool, const ChatTemplateOptions&) const override;
};

class TurnChatTemplate final : public IChatTemplate {
public:
    using IChatTemplate::format;
    std::string format(std::span<const ChatMessage>, std::span<const ChatToolDefinition>,
                       bool, const ChatTemplateOptions&) const override;
};

class ThinkingFunctionChatTemplate final : public IChatTemplate {
public:
    using IChatTemplate::format;
    std::string format(std::span<const ChatMessage>, std::span<const ChatToolDefinition>,
                       bool, const ChatTemplateOptions&) const override;
};

class ReasoningXmlChatTemplate final : public IChatTemplate {
public:
    using IChatTemplate::format;
    std::string format(std::span<const ChatMessage>, std::span<const ChatToolDefinition>,
                       bool, const ChatTemplateOptions&) const override;
};

class MetadataThinkingChatTemplate final : public IChatTemplate {
public:
    using IChatTemplate::format;
    explicit MetadataThinkingChatTemplate(bool enable_thinking = true)
        : enable_thinking_(enable_thinking) {}
    std::string format(std::span<const ChatMessage>, std::span<const ChatToolDefinition>,
                       bool, const ChatTemplateOptions&) const override;
private:
    std::string format_with_thinking(std::span<const ChatMessage>,
                                     std::span<const ChatToolDefinition>,
                                     bool, bool) const;
    bool enable_thinking_;
};

class VisionRoleChatTemplate final : public IChatTemplate {
public:
    using IChatTemplate::format;
    std::string format(std::span<const ChatMessage>, std::span<const ChatToolDefinition>,
                       bool, const ChatTemplateOptions&) const override;
};

class PatchRoleChatTemplate final : public IChatTemplate {
public:
    using IChatTemplate::format;
    std::string format(std::span<const ChatMessage>, std::span<const ChatToolDefinition>,
                       bool, const ChatTemplateOptions&) const override;
};

class ThinkingRoleChatTemplate final : public IChatTemplate {
public:
    using IChatTemplate::format;
    std::string format(std::span<const ChatMessage>, std::span<const ChatToolDefinition>,
                       bool, const ChatTemplateOptions&) const override;
};

class TaggedRoleChatTemplate final : public IChatTemplate {
public:
    using IChatTemplate::format;
    std::string format(std::span<const ChatMessage>, std::span<const ChatToolDefinition>,
                       bool, const ChatTemplateOptions&) const override;
};

std::unique_ptr<IChatToolCallCodec> make_delimited_tool_codec();
std::unique_ptr<IChatToolCallCodec> make_native_tag_tool_codec();
std::unique_ptr<IChatToolCallCodec> make_function_xml_tool_codec();
std::unique_ptr<IChatToolCallCodec> make_parameter_xml_tool_codec();
std::unique_ptr<IChatToolCallCodec> make_tagged_json_tool_codec();
std::unique_ptr<IChatToolCallCodec> make_tagged_role_tool_codec();

void add_delimited_chat_template(ChatTemplateCatalog&);
void add_role_envelope_chat_template(ChatTemplateCatalog&);
void add_turn_chat_template(ChatTemplateCatalog&);
void add_thinking_function_chat_template(ChatTemplateCatalog&);
void add_reasoning_xml_chat_template(ChatTemplateCatalog&);
void add_metadata_thinking_chat_template(ChatTemplateCatalog&);
void add_vision_role_chat_template(ChatTemplateCatalog&);
void add_patch_role_chat_template(ChatTemplateCatalog&);
void add_thinking_role_chat_template(ChatTemplateCatalog&);
void add_tagged_role_chat_template(ChatTemplateCatalog&);

} // namespace celeg
