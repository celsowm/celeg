#pragma once

#include "celeg/serve/types.hpp"
#include "celeg/text/chat_profile.hpp"
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

// Interprets model text above the backend event stream incrementally. Only an
// incomplete UTF-8 suffix is withheld from the client.
class ChatGenerationInterpreter {
public:
    ChatGenerationInterpreter(const BpeTokenizer& tokenizer,
                              const IChatToolCallCodec* tool_codec);

    ChatGenerationDelta consume(std::span<const std::int32_t> tokens, bool finished);

private:
    const BpeTokenizer& tokenizer_;
    const IChatToolCallCodec* tool_codec_;
    std::string generated_text_;
    std::string emitted_text_;
    std::size_t emitted_calls_ = 0;
};

} // namespace celeg::serve
