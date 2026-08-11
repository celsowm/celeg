#include "celeg/text/chat_template.hpp"
#include "celeg/text/semantic_chat_templates.hpp"

#include <functional>
#include <stdexcept>

namespace celeg {

void ChatTemplateProgram::validate() const {
    if (instructions.empty()) {
        throw std::invalid_argument("chat template program has no instructions");
    }
    for (const ChatInstruction& instruction : instructions) {
        if (instruction.kind == ChatInstructionKind::EmitLiteral && instruction.value.empty()) {
            throw std::invalid_argument("chat template program contains an empty literal");
        }
    }
}

std::string ChatTemplateProgram::fingerprint() const {
    std::size_t hash = std::hash<std::string_view>{}(source);
    for (const ChatInstruction& instruction : instructions) {
        hash ^= static_cast<std::size_t>(instruction.kind) +
            0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
        hash ^= std::hash<std::string>{}(instruction.value) +
            0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    }
    return std::to_string(hash);
}

std::string ProgramChatTemplate::format(
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool add_generation_prompt,
    const ChatTemplateOptions& options) const {
    if (options.enable_thinking.has_value()) {
        throw std::invalid_argument("chat template program does not expose thinking options");
    }
    std::string output;
    for (const ChatInstruction& instruction : program_.instructions) {
        if (instruction.kind == ChatInstructionKind::EmitLiteral) {
            output += instruction.value;
            continue;
        }
        if (instruction.kind == ChatInstructionKind::EmitGenerationPrompt) {
            if (add_generation_prompt) output += instruction.value;
            continue;
        }
        if (instruction.kind == ChatInstructionKind::EmitToolDefinitions) {
            for (const ChatToolDefinition& tool : tools) {
                output += instruction.value + tool.function.name;
            }
            continue;
        }
        for (const ChatMessage& message : messages) {
            switch (instruction.kind) {
                case ChatInstructionKind::EmitMessageRole:
                    output += instruction.value + std::to_string(static_cast<int>(message.role));
                    break;
                case ChatInstructionKind::EmitMessageContent:
                    output += instruction.value + message.content.value_or("");
                    break;
                case ChatInstructionKind::EmitToolResult:
                    if (message.role == ChatRole::Tool) {
                        output += instruction.value + message.content.value_or("");
                    }
                    break;
                case ChatInstructionKind::EmitAssistantToolCalls:
                    if (message.role == ChatRole::Assistant) {
                        for (const ToolCall& call : message.tool_calls) {
                            output += instruction.value + call.name + call.arguments;
                        }
                    }
                    break;
                default:
                    break;
            }
        }
    }
    return output;
}

std::string render_chat(std::span<const ChatMessage> messages,
                        const IChatTemplate& chat_template,
                        bool add_generation_prompt) {
    return chat_template.format(messages, add_generation_prompt);
}

std::string render_chat(std::span<const ChatMessage> messages,
                        std::span<const ChatToolDefinition> tools,
                        const IChatTemplate& chat_template,
                        bool add_generation_prompt) {
    return chat_template.format(messages, tools, add_generation_prompt);
}

std::string render_chat(std::span<const ChatMessage> messages,
                        std::span<const ChatToolDefinition> tools,
                        const IChatTemplate& chat_template,
                        bool add_generation_prompt,
                        const ChatTemplateOptions& options) {
    return chat_template.format(messages, tools, add_generation_prompt, options);
}

void ChatTemplateCatalog::add(std::string template_id,
                             std::unique_ptr<IChatTemplate> chat_template,
                             std::unique_ptr<IChatToolCallCodec> tool_call_codec,
                             ChatCapabilities capabilities) {
    if (frozen_) throw std::logic_error("chat profile catalog is frozen");
    if (template_id.empty() || !chat_template) {
        throw std::invalid_argument("chat profile requires an id and template");
    }
    if (capabilities.native_tool_call_codec && !tool_call_codec) {
        throw std::invalid_argument("chat profile declares a native tool codec but provides none");
    }
    if (capabilities.parallel_tool_calls &&
        (!tool_call_codec || !tool_call_codec->supports_parallel_calls())) {
        throw std::invalid_argument("chat profile declares parallel tool calls without codec support");
    }
    if (tool_call_codec && !capabilities.native_tool_call_codec) {
        throw std::invalid_argument("chat profile provides a tool codec without enabling its capability");
    }
    std::shared_ptr<const IChatToolCallCodec> shared_codec(std::move(tool_call_codec));
    Entry entry{std::move(chat_template), std::move(shared_codec), capabilities};
    if (!entries_.emplace(std::move(template_id), std::move(entry)).second) {
        throw std::invalid_argument("duplicate chat profile");
    }
}

void ChatTemplateCatalog::freeze() { frozen_ = true; }

const IChatTemplate& ChatTemplateCatalog::find(std::string_view template_id) const {
    const auto it = entries_.find(std::string(template_id));
    if (it == entries_.end()) throw std::invalid_argument("unknown chat template: " + std::string(template_id));
    return *it->second.chat_template;
}

ChatCapabilities ChatTemplateCatalog::capabilities(std::string_view template_id) const {
    const auto it = entries_.find(std::string(template_id));
    if (it == entries_.end()) throw std::invalid_argument("unknown chat template: " + std::string(template_id));
    return it->second.capabilities;
}

const IChatToolCallCodec* ChatTemplateCatalog::tool_codec(
    std::string_view template_id) const {
    const auto it = entries_.find(std::string(template_id));
    if (it == entries_.end()) throw std::invalid_argument(
        "unknown chat template: " + std::string(template_id));
    return it->second.tool_call_codec.get();
}

} // namespace celeg
