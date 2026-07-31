#include "celeg/text/chat_template.hpp"

#include <stdexcept>

namespace celeg {

namespace {

const char* role_tag(ChatRole role) {
    switch (role) {
        case ChatRole::System: return "system";
        case ChatRole::Developer: return "system";
        case ChatRole::User: return "user";
        case ChatRole::Assistant: return "assistant";
        case ChatRole::Tool:
            throw std::invalid_argument(
                "Lfm2InstructChatTemplate does not support tool messages");
    }
    throw std::invalid_argument("unknown chat role");
}

const char* granite_role_tag(ChatRole role) {
    switch (role) {
        case ChatRole::System: return "system";
        case ChatRole::Developer: return "developer";
        case ChatRole::User: return "user";
        case ChatRole::Assistant: return "assistant";
        case ChatRole::Tool:
            throw std::invalid_argument(
                "GraniteInstructChatTemplate does not support tool messages");
    }
    throw std::invalid_argument("unknown chat role");
}

void append_granite_message(std::string& out, ChatRole role, std::string_view content) {
    out += "<|start_of_role|>";
    out += granite_role_tag(role);
    out += "<|end_of_role|>";
    out += content;
    out += "<|end_of_text|>\n";
}

} // namespace

std::string Lfm2InstructChatTemplate::format(std::span<const ChatMessage> messages,
                                             bool add_generation_prompt) const {
    std::string out = "<|startoftext|>";
    for (const ChatMessage& message : messages) {
        out += "<|im_start|>";
        out += role_tag(message.role);
        out += "\n";
        out += message.content;
        out += "<|im_end|>\n";
    }
    if (add_generation_prompt) out += "<|im_start|>assistant\n";
    return out;
}

std::string GraniteInstructChatTemplate::format(std::span<const ChatMessage> messages,
                                                bool add_generation_prompt) const {
    std::string out;
    const bool has_system_message = !messages.empty() &&
        messages.front().role == ChatRole::System;
    if (!has_system_message) {
        append_granite_message(
            out, ChatRole::System,
            "Knowledge Cutoff Date: April 2024. You are Granite, developed by IBM. "
            "You are a helpful AI assistant.");
    }
    for (const ChatMessage& message : messages) {
        append_granite_message(out, message.role, message.content);
    }
    if (add_generation_prompt) {
        out += "<|start_of_role|>assistant<|end_of_role|>";
    }
    return out;
}

std::unique_ptr<IChatTemplate> make_chat_template(ChatTemplateKind kind) {
    switch (kind) {
        case ChatTemplateKind::Lfm2Instruct:
            return std::make_unique<Lfm2InstructChatTemplate>();
        case ChatTemplateKind::GraniteInstruct:
            return std::make_unique<GraniteInstructChatTemplate>();
    }
    throw std::invalid_argument("unsupported chat template kind");
}

std::unique_ptr<IChatTemplate> make_chat_template(std::string_view profile_id) {
    if (profile_id == "lfm2-instruct" || profile_id == "lfm2") {
        return make_chat_template(ChatTemplateKind::Lfm2Instruct);
    }
    if (profile_id == "granite-instruct" || profile_id == "granite") {
        return make_chat_template(ChatTemplateKind::GraniteInstruct);
    }
    throw std::invalid_argument("unknown chat profile: " + std::string(profile_id));
}

} // namespace celeg
