#pragma once

#include "lfm/model_variant.hpp"

#include <memory>
#include <string>
#include <string_view>

namespace lfm {

// Interface Segregation Principle: tokenizer chat formatting depends only on
// this narrow interface, not on the variant metadata.
class IChatTemplate {
public:
    virtual ~IChatTemplate() = default;
    virtual std::string format(std::string_view user_prompt,
                               std::string_view system_prompt) const = 0;
    virtual ChatTemplateKind kind() const = 0;
};

// LFM2 Instruct chat template:
//   <|startoftext|>[<|im_start|>system\n{system}<|im_end|>\n]
//   <|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n
class Lfm2InstructChatTemplate final : public IChatTemplate {
public:
    std::string format(std::string_view user_prompt,
                       std::string_view system_prompt) const override;
    ChatTemplateKind kind() const override { return ChatTemplateKind::Lfm2Instruct; }
};

// Factory: returns the chat template implementation for a given kind.
std::unique_ptr<IChatTemplate> make_chat_template(ChatTemplateKind kind);

} // namespace lfm
