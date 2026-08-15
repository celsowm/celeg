#include "celeg/serve/chat_generation.hpp"

#include <algorithm>

namespace {

std::size_t complete_utf8_prefix(std::string_view text) {
    std::size_t cursor = 0;
    while (cursor < text.size()) {
        const auto lead = static_cast<unsigned char>(text[cursor]);
        std::size_t width = 1;
        if ((lead & 0x80u) == 0) width = 1;
        else if ((lead & 0xe0u) == 0xc0u) width = 2;
        else if ((lead & 0xf0u) == 0xe0u) width = 3;
        else if ((lead & 0xf8u) == 0xf0u) width = 4;
        else { ++cursor; continue; }
        if (cursor + width > text.size()) break;
        bool valid = true;
        for (std::size_t i = 1; i < width; ++i) {
            if ((static_cast<unsigned char>(text[cursor + i]) & 0xc0u) != 0x80u) {
                valid = false;
                break;
            }
        }
        if (!valid) { ++cursor; continue; }
        cursor += width;
    }
    return cursor;
}

}

namespace celeg::serve {

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
