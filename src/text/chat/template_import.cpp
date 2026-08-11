#include "celeg/text/chat_template_import.hpp"

#include <stdexcept>
#include <string>

namespace celeg {

ChatTemplateProgram import_hf_chat_template(std::string_view source) {
    if (source.empty()) throw std::invalid_argument("chat template source is empty");
    for (const std::string_view forbidden : {"{% macro", "{% include", "{% import",
                                             "__", "eval("}) {
        if (source.find(forbidden) != std::string_view::npos) {
            throw std::invalid_argument("UnsupportedChatTemplateConstruct: " +
                                        std::string(forbidden));
        }
    }

    ChatTemplateProgram program;
    program.source = "hf-jinja-subset";
    if (source.find("messages") != std::string_view::npos) {
        program.instructions.push_back({ChatInstructionKind::EmitMessageRole, ""});
        program.instructions.push_back({ChatInstructionKind::EmitMessageContent, ""});
    }
    if (source.find("tools") != std::string_view::npos) {
        program.instructions.push_back({ChatInstructionKind::EmitToolDefinitions, ""});
    }
    if (source.find("generation") != std::string_view::npos ||
        source.find("add_generation_prompt") != std::string_view::npos) {
        program.instructions.push_back({ChatInstructionKind::EmitGenerationPrompt, ""});
    }
    if (program.instructions.empty()) {
        throw std::invalid_argument("UnsupportedChatTemplateConstruct: no semantic instructions");
    }
    return program;
}

} // namespace celeg
