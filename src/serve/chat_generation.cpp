#include "celeg/serve/chat_generation.hpp"
#include "celeg/text/utf8.hpp"

#include <algorithm>

namespace celeg::serve {

using celeg::text::complete_utf8_prefix;

ChatGenerationInterpreter::ChatGenerationInterpreter(
    const ITokenizer& tokenizer, const ResolvedInteraction& interaction)
    : tokenizer_(tokenizer), interaction_(interaction) {}

ChatGenerationDelta ChatGenerationInterpreter::consume(
    std::span<const std::int32_t> tokens, bool finished) {
    for (const std::int32_t token : tokens) generated_text_ += tokenizer_.decode_token(token, false);
    const std::string& generated = generated_text_;
    const auto parsed = interaction_.parse_tool_calls(generated);
    const auto& calls = parsed.calls;
    const std::string visible = parsed.status == ToolParseStatus::NotToolCall
        ? generated : parsed.assistant_text;
    const std::size_t safe_size = finished ? visible.size() : complete_utf8_prefix(visible);
    const std::string_view safe_visible(visible.data(), safe_size);

    ChatGenerationDelta delta;
    if (safe_visible.size() >= emitted_text_.size() &&
        safe_visible.compare(0, emitted_text_.size(), emitted_text_) == 0) {
        delta.text = std::string(safe_visible.substr(emitted_text_.size()));
    } else {
        delta.text = std::string(safe_visible);
    }
    emitted_text_ = std::string(safe_visible);
    if (calls.size() > emitted_calls_) {
        delta.tool_calls.assign(calls.begin() + static_cast<std::ptrdiff_t>(emitted_calls_), calls.end());
        emitted_calls_ = calls.size();
    }
    delta.finished = finished;
    if (finished) {
        if (parsed.status == ToolParseStatus::Invalid || parsed.status == ToolParseStatus::Incomplete) {
            delta.finish_reason = FinishReason::Error;
            delta.error = parsed.error.empty() ? "incomplete or malformed tool call" : parsed.error;
        } else {
            delta.finish_reason = !calls.empty() ? FinishReason::ToolCalls : FinishReason::Stop;
        }
    }
    return delta;
}

}
