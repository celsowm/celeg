#pragma once

#include "celeg/checkpoint/metadata.hpp"
#include "celeg/text/chat_contract.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace celeg {

class ITokenizer;
class InteractionRenderProgram;

using ChatToolDefinition = ToolDefinition;

struct ChatTemplateOptions {
    std::optional<bool> enable_thinking;
    ToolChoice tool_choice{};
};

struct ToolCallGrammar {
    std::string opening;
    std::string closing;
    std::string call_prefix;
    std::string call_suffix;
    std::string required_prefix;
    bool parallel_calls = false;
    bool xml_parameters = false;

    ToolParseResult parse(std::string_view generated) const;
};

class ResolvedInteraction final {
public:
    std::string format(std::span<const ChatMessage> messages,
                       bool add_generation_prompt = true) const;
    std::string format(std::span<const ChatMessage> messages,
                       std::span<const ChatToolDefinition> tools,
                       bool add_generation_prompt = true,
                       const ChatTemplateOptions& options = {}) const;

    const std::string& source_origin() const noexcept { return source_origin_; }
    const std::string& fingerprint() const noexcept { return fingerprint_; }
    const std::vector<std::string>& diagnostics() const noexcept { return diagnostics_; }
    ChatCapabilities capabilities() const noexcept { return capabilities_; }
    const std::optional<ToolCallGrammar>& tool_call_grammar() const noexcept {
        return tool_call_grammar_;
    }
    ToolParseResult parse_tool_calls(std::string_view generated) const;
    std::string forced_tool_call_prefix(std::span<const ChatToolDefinition> tools,
                                        const ToolChoice& choice) const;

private:
    friend ResolvedInteraction resolve_interaction(
        const CheckpointMetadata&, const ITokenizer&,
        const std::optional<std::filesystem::path>&,
        const std::optional<std::filesystem::path>&);

    std::shared_ptr<const InteractionRenderProgram> render_program_;
    std::string source_origin_;
    std::string fingerprint_;
    std::string bos_token_;
    std::vector<std::string> diagnostics_;
    ChatCapabilities capabilities_;
    std::optional<ToolCallGrammar> tool_call_grammar_;
};

ResolvedInteraction resolve_interaction(
    const CheckpointMetadata& metadata,
    const ITokenizer& tokenizer,
    const std::optional<std::filesystem::path>& override_file = std::nullopt,
    const std::optional<std::filesystem::path>& companion_file = std::nullopt);

std::string render_chat(std::span<const ChatMessage> messages,
                        const ResolvedInteraction& interaction,
                        bool add_generation_prompt = true);
std::string render_chat(std::span<const ChatMessage> messages,
                        std::span<const ChatToolDefinition> tools,
                        const ResolvedInteraction& interaction,
                        bool add_generation_prompt = true,
                        const ChatTemplateOptions& options = {});

}
