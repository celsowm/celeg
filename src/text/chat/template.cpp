#include "celeg/text/chat_template.hpp"
#include "celeg/text/tokenizer.hpp"
#include "template/program.hpp"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace celeg {
namespace {

std::string read_template_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::invalid_argument(
            "cannot read --chat-template-file: " + path.string());
    }
    return {
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>(),
    };
}

std::string fingerprint(std::string_view source) {
    std::uint64_t value = 1469598103934665603ull;
    for (const unsigned char byte : source) {
        value ^= byte;
        value *= 1099511628211ull;
    }
    std::ostringstream out;
    out << std::hex << value;
    return out.str();
}

bool inferred_delimited_evidence(const ITokenizer& tokenizer) {
    return tokenizer.token_id("<|startoftext|>").has_value() &&
           tokenizer.token_id("<|im_start|>").has_value() &&
           tokenizer.token_id("<|im_end|>").has_value() &&
           tokenizer.token_id("assistant").has_value();
}

std::string inferred_delimited_source() {
    return "{{ bos_token }}{% for message in messages %}"
           "<|im_start|>{{ message.role }}\\n{{ message.content }}"
           "{% for call in message.tool_calls %}"
           "<|tool_call_start|>[{{ call.function.name }}"
           "({{ call.function.arguments }})]<|tool_call_end|>"
           "{% endfor %}<|im_end|>\\n{% endfor %}"
           "{% if add_generation_prompt %}<|im_start|>assistant\\n{% endif %}";
}

}

ResolvedInteraction resolve_interaction(
    const CheckpointMetadata& metadata,
    const ITokenizer& tokenizer,
    const std::optional<std::filesystem::path>& override_file,
    const std::optional<std::filesystem::path>& companion_file) {
    std::string source;
    std::string origin;

    if (override_file) {
        source = read_template_file(*override_file);
        origin = "override:" + override_file->string();
    } else {
        source = metadata.string_or(
            "chat_template",
            metadata.string_or("tokenizer.chat_template", {}));
        if (source.empty() &&
            companion_file &&
            std::filesystem::is_regular_file(*companion_file)) {
            source = read_template_file(*companion_file);
            origin =
                "checkpoint-companion:" + companion_file->string();
        } else {
            origin = source.empty()
                ? "tokenizer-inference"
                : "checkpoint-metadata";
        }
    }

    ResolvedInteraction result;
    result.source_origin_ = origin;

    if (source.empty()) {
        if (!inferred_delimited_evidence(tokenizer)) {
            throw std::invalid_argument(
                "no chat template metadata and tokenizer does not structurally "
                "prove BOS, role delimiters, turn terminator, and assistant "
                "generation prefix; supply --chat-template-file");
        }
        source = inferred_delimited_source();
        result.diagnostics_.push_back(
            "inferred role-delimited interaction from tokenizer evidence: "
            "<|startoftext|>, <|im_start|>, <|im_end|>, assistant");
    }

    if (source.find("__") != std::string::npos ||
        source.find("{% include") != std::string::npos ||
        source.find("{% import") != std::string::npos ||
        source.find("{% from") != std::string::npos) {
        throw std::invalid_argument(
            "UnsupportedChatTemplateConstruct at " + origin +
            ": imports, introspection, and I/O are not permitted");
    }

    std::string normalized_source = source;
    for (std::size_t position = 0;
         (position = normalized_source.find("[::-1]", position)) !=
             std::string::npos;) {
        normalized_source.replace(
            position,
            std::string_view("[::-1]").size(),
            "|reverse");
        position += std::string_view("|reverse").size();
    }

    auto nodes = chat_template_detail::parse_template(
        normalized_source,
        origin);
    chat_template_detail::validate_program(nodes, origin);

    result.render_program_ =
        std::make_shared<InteractionRenderProgram>(std::move(nodes));
    result.fingerprint_ = fingerprint(source);
    result.bos_token_ = tokenizer.decode(
        std::vector<std::int32_t>{tokenizer.bos_id()},
        false);
    result.capabilities_.roles.developer =
        source.find("developer") != std::string::npos;
    result.capabilities_.roles.tool =
        source.find("tool") != std::string::npos;
    result.tool_call_grammar_ =
        chat_template_detail::derive_tool_grammar(source);

    result.capabilities_.assistant_tool_calls =
        result.tool_call_grammar_.has_value();
    result.capabilities_.parallel_tool_calls =
        result.tool_call_grammar_ &&
        result.tool_call_grammar_->parallel_calls;
    result.capabilities_.native_tool_call_codec =
        result.tool_call_grammar_.has_value();
    return result;
}

std::string ResolvedInteraction::format(
    std::span<const ChatMessage> messages,
    bool add_generation_prompt) const {
    return format(
        messages,
        {},
        add_generation_prompt,
        {});
}

std::string ResolvedInteraction::format(
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool add_generation_prompt,
    const ChatTemplateOptions& options) const {
    if (!render_program_) {
        throw std::logic_error(
            "resolved interaction has no render program");
    }
    validate_conversation(messages);
    return render_program_->render(
        messages,
        tools,
        add_generation_prompt,
        options,
        source_origin_,
        bos_token_);
}

ToolParseResult ResolvedInteraction::parse_tool_calls(
    std::string_view generated) const {
    if (!tool_call_grammar_) {
        return ToolParseResult{
            .status = ToolParseStatus::NotToolCall,
            .assistant_text = std::string(generated),
            .calls = {},
            .error = {},
            .consumed_bytes = 0,
        };
    }
    return tool_call_grammar_->parse(generated);
}

std::string ResolvedInteraction::forced_tool_call_prefix(
    std::span<const ChatToolDefinition>,
    const ToolChoice& choice) const {
    if (!tool_call_grammar_ ||
        (choice.mode != ToolChoiceMode::Required &&
         choice.mode != ToolChoiceMode::Specific)) {
        return {};
    }
    return tool_call_grammar_->required_prefix;
}

std::string render_chat(
    std::span<const ChatMessage> messages,
    const ResolvedInteraction& interaction,
    bool add_generation_prompt) {
    return interaction.format(messages, add_generation_prompt);
}

std::string render_chat(
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    const ResolvedInteraction& interaction,
    bool add_generation_prompt,
    const ChatTemplateOptions& options) {
    return interaction.format(
        messages,
        tools,
        add_generation_prompt,
        options);
}

}

#include "template/parser.cpp"
#include "template/validation.cpp"
#include "template/runtime/values.cpp"
#include "template/runtime/context.cpp"
#include "template/runtime/expression.cpp"
#include "template/runtime/filters.cpp"
#include "template/runtime/calls.cpp"
#include "template/runtime/render.cpp"
#include "template/runtime/runtime.cpp"
#include "template/tool_codec.cpp"
