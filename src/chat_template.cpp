#include "lfm/chat_template.hpp"

#include <stdexcept>

namespace lfm {

std::string Lfm2InstructChatTemplate::format(std::string_view user_prompt,
                                             std::string_view system_prompt) const {
    std::string out = "<|startoftext|>";
    if (!system_prompt.empty()) {
        out += "<|im_start|>system\n";
        out += system_prompt;
        out += "<|im_end|>\n";
    }
    out += "<|im_start|>user\n";
    out += user_prompt;
    out += "<|im_end|>\n<|im_start|>assistant\n";
    return out;
}

std::unique_ptr<IChatTemplate> make_chat_template(ChatTemplateKind kind) {
    switch (kind) {
        case ChatTemplateKind::Lfm2Instruct:
            return std::make_unique<Lfm2InstructChatTemplate>();
    }
    throw std::invalid_argument("unsupported chat template kind");
}

} // namespace lfm
