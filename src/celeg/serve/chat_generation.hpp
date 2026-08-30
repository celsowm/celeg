#pragma once

#include "celeg/serve/types.hpp"
#include "celeg/text/chat_template.hpp"
#include "celeg/text/tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace celeg::serve {

struct ChatGenerationDelta {
    std::string text;
    std::vector<ToolCall> tool_calls;
    std::string error;
    FinishReason finish_reason = FinishReason::None;
    bool finished = false;
};

class ChatGenerationInterpreter {
public:
    ChatGenerationInterpreter(const ITokenizer& tokenizer,
                              const ResolvedInteraction& interaction);

    ChatGenerationDelta consume(std::span<const std::int32_t> tokens, bool finished);

private:
    const ITokenizer& tokenizer_;
    const ResolvedInteraction& interaction_;
    std::string generated_text_;
    std::string emitted_text_;
    std::size_t emitted_calls_ = 0;
};

}
