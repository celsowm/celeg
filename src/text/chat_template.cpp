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

std::unique_ptr<IChatTemplate> make_chat_template(ChatTemplateKind kind) {
    switch (kind) {
        case ChatTemplateKind::Lfm2Instruct:
            return std::make_unique<Lfm2InstructChatTemplate>();
    }
    throw std::invalid_argument("unsupported chat template kind");
}

} // namespace celeg
