#pragma once

#include <cstddef>
#include <string_view>

namespace celeg::text {

/// Length of the longest prefix of `text` that contains no truncated
/// trailing UTF-8 sequence. Byte-level BPE tokenizers routinely split a
/// single multi-byte codepoint across two or more token ids, so decoding
/// and emitting tokens one at a time can produce an incomplete trailing
/// sequence that only becomes valid once the next token arrives -- callers
/// that stream decoded text should hold back the bytes after this prefix
/// until more tokens are available.
inline std::size_t complete_utf8_prefix(std::string_view text) {
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

}  // namespace celeg::text
