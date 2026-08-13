#include "celeg/text/chat_template.hpp"
#include "celeg/text/semantic_chat_templates.hpp"
#include "celeg/text/tokenizer.hpp"

#include <fstream>
#include <cstdint>
#include <functional>
#include <sstream>
#include <stdexcept>

namespace celeg {

namespace {

std::string read_template_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::invalid_argument("cannot read --chat-template-file: " + path.string());
    }
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::string fingerprint(std::string_view source) {
    // Stable FNV-1a rather than std::hash, whose result is implementation-defined.
    std::uint64_t value = 1469598103934665603ull;
    for (const unsigned char byte : source) {
        value ^= byte;
        value *= 1099511628211ull;
    }
    std::ostringstream out;
    out << std::hex << value;
    return out.str();
}

bool has_forbidden_jinja(std::string_view source, std::string& construct) {
    static constexpr std::string_view forbidden[] = {
        "{% include", "{% import", "{% from", "__", "eval(", "open(", "read("};
    for (const std::string_view candidate : forbidden) {
        if (source.find(candidate) != std::string_view::npos) {
            construct = std::string(candidate);
            return true;
        }
    }
    return false;
}

std::string role_name(ChatRole role) {
    switch (role) {
        case ChatRole::System: return "system";
        case ChatRole::Developer: return "developer";
        case ChatRole::User: return "user";
        case ChatRole::Assistant: return "assistant";
        case ChatRole::Tool: return "tool";
    }
    throw std::invalid_argument("unknown chat role");
}

std::string json_string(std::string_view value) {
    std::string out{"\""};
    for (const unsigned char character : value) {
        switch (character) {
            case '\\': out += "\\\\"; break;
            case '\"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += static_cast<char>(character); break;
        }
    }
    return out + '"';
}

std::string render_tools(std::span<const ChatToolDefinition> tools) {
    std::string out = "List of tools: [";
    for (std::size_t i = 0; i < tools.size(); ++i) {
        if (i != 0) out += ", ";
        const auto& function = tools[i].function;
        out += "{\"type\":\"function\",\"function\":{\"name\":";
        out += json_string(function.name);
        out += ",\"description\":" + json_string(function.description);
        out += ",\"parameters\":" +
            (function.parameters.serialized.empty() ? "{}" : function.parameters.serialized);
        out += "}}";
    }
    return out + ']';
}

bool lfm_delimited_evidence(const ITokenizer& tokenizer) {
    return tokenizer.token_id("<|startoftext|>").has_value() &&
        tokenizer.token_id("<|im_start|>").has_value() &&
        tokenizer.token_id("<|im_end|>").has_value();
}

} // namespace

ResolvedChatTemplate resolve_chat_template(
    const CheckpointMetadata& metadata,
    const ITokenizer& tokenizer,
    const std::optional<std::filesystem::path>& override_file) {
    std::string source;
    std::string origin;
    if (override_file) {
        source = read_template_file(*override_file);
        origin = "override:" + override_file->string();
    } else {
        const auto source_at = [&](std::string_view key) -> std::string {
            const auto it = metadata.values.find(std::string(key));
            if (it == metadata.values.end() || !std::holds_alternative<std::string>(it->second)) {
                return {};
            }
            return std::get<std::string>(it->second);
        };
        source = source_at("chat_template");
        if (source.empty()) source = source_at("tokenizer.chat_template");
        origin = source.empty() ? "tokenizer-inference" : "checkpoint-metadata";
    }

    ResolvedChatTemplate result;
    result.source_origin_ = origin;
    if (!source.empty()) {
        std::string forbidden;
        if (has_forbidden_jinja(source, forbidden)) {
            throw std::invalid_argument("UnsupportedChatTemplateConstruct at " +
                origin + ": " + forbidden);
        }
        // This first compiler target is the deterministic role-delimited
        // protocol emitted by current HF chat templates.  Parsing source is
        // intentionally evidence based: no model name can select behavior.
        if (source.find("<|im_start|>") == std::string::npos ||
            source.find("<|im_end|>") == std::string::npos ||
            source.find("add_generation_prompt") == std::string::npos) {
            throw std::invalid_argument("UnsupportedChatTemplateConstruct at " + origin +
                ": expected deterministic role-delimited generation template");
        }
        result.include_bos_ = source.find("bos_token") != std::string::npos ||
            source.find("<|startoftext|>") != std::string::npos;
        result.tools_in_system_ = source.find("List of tools:") != std::string::npos;
        result.fingerprint_ = fingerprint(source);
    } else {
        if (!lfm_delimited_evidence(tokenizer)) {
            throw std::invalid_argument(
                "no chat template metadata and tokenizer does not prove a role-delimited "
                "protocol; supply --chat-template-file");
        }
        result.include_bos_ = true;
        result.tools_in_system_ = tokenizer.token_id("<|tool_call_start|>").has_value() &&
            tokenizer.token_id("<|tool_call_end|>").has_value();
        result.fingerprint_ = fingerprint("inferred:<|startoftext|><|im_start|><|im_end|>");
    }
    result.capabilities_.roles.developer = true;
    result.capabilities_.roles.tool = true;
    result.capabilities_.assistant_tool_calls = result.tools_in_system_;
    result.capabilities_.parallel_tool_calls = result.tools_in_system_;
    result.capabilities_.native_tool_call_codec = result.tools_in_system_;
    if (result.tools_in_system_) {
        result.tool_codec_ = std::shared_ptr<const IChatToolCallCodec>(
            make_delimited_tool_codec().release());
    }
    return result;
}

std::string ResolvedChatTemplate::format(
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool add_generation_prompt,
    const ChatTemplateOptions& options) const {
    validate_conversation(messages);
    if (options.enable_thinking.has_value()) {
        throw std::invalid_argument("the resolved chat template has no thinking toggle");
    }
    if (!tools.empty() && !tools_in_system_) {
        throw std::invalid_argument("the resolved chat template does not define tool rendering");
    }
    std::string output = include_bos_ ? "<|startoftext|>" : "";
    bool wrote_tools = false;
    for (const ChatMessage& message : messages) {
        if (!tools.empty() && !wrote_tools && message.role != ChatRole::System) {
            output += "<|im_start|>system\n" + render_tools(tools) + "<|im_end|>\n";
            wrote_tools = true;
        }
        if (message.role == ChatRole::System && !tools.empty() && !wrote_tools) {
            const std::string content = message.content.value_or("");
            output += "<|im_start|>system\n" + content +
                (content.empty() ? "" : "\n") + render_tools(tools) + "<|im_end|>\n";
            wrote_tools = true;
            continue;
        }
        output += "<|im_start|>" + role_name(message.role) + "\n";
        output += message.content.value_or("");
        for (const ToolCall& call : message.tool_calls) {
            output += "<|tool_call_start|>[" + call.name + "(" + call.arguments +
                ")]<|tool_call_end|>";
        }
        output += "<|im_end|>\n";
    }
    if (!tools.empty() && !wrote_tools) {
        output += "<|im_start|>system\n" + render_tools(tools) + "<|im_end|>\n";
    }
    if (add_generation_prompt) output += "<|im_start|>assistant\n";
    return output;
}

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
