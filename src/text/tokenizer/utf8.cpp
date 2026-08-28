#include "detail.hpp"

#include <cctype>

namespace celeg::tokenizer_detail {

void append_utf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7F) out.push_back(static_cast<char>(cp));
    else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

std::pair<uint32_t, size_t> next_cp(std::string_view text, size_t offset) {
    const uint8_t c0 = static_cast<uint8_t>(text[offset]);
    if (c0 < 0x80) return {c0, 1};
    const auto continuation = [&](size_t index) {
        return index < text.size() &&
            (static_cast<uint8_t>(text[index]) & 0xC0) == 0x80;
    };
    if ((c0 >> 5) == 0x6 && continuation(offset + 1)) {
        const uint32_t cp = static_cast<uint32_t>(((c0 & 0x1F) << 6) |
            (static_cast<uint8_t>(text[offset + 1]) & 0x3F));
        return cp >= 0x80 ? std::pair<uint32_t, size_t>{cp, 2} : std::pair<uint32_t, size_t>{0xFFFD, 1};
    }
    if ((c0 >> 4) == 0xE && continuation(offset + 1) && continuation(offset + 2)) {
        const uint32_t cp = static_cast<uint32_t>(((c0 & 0x0F) << 12) |
            ((static_cast<uint8_t>(text[offset + 1]) & 0x3F) << 6) |
            (static_cast<uint8_t>(text[offset + 2]) & 0x3F));
        return cp >= 0x800 && !(cp >= 0xD800 && cp <= 0xDFFF)
            ? std::pair<uint32_t, size_t>{cp, 3} : std::pair<uint32_t, size_t>{0xFFFD, 1};
    }
    if ((c0 >> 3) == 0x1E && continuation(offset + 1) && continuation(offset + 2) && continuation(offset + 3)) {
        const uint32_t cp = static_cast<uint32_t>(((c0 & 0x07) << 18) |
            ((static_cast<uint8_t>(text[offset + 1]) & 0x3F) << 12) |
            ((static_cast<uint8_t>(text[offset + 2]) & 0x3F) << 6) |
            (static_cast<uint8_t>(text[offset + 3]) & 0x3F));
        return cp >= 0x10000 && cp <= 0x10FFFF
            ? std::pair<uint32_t, size_t>{cp, 4} : std::pair<uint32_t, size_t>{0xFFFD, 1};
    }
    return {0xFFFD, 1};
}

bool in_range(uint32_t cp, uint32_t first, uint32_t last) {
    return cp >= first && cp <= last;
}

bool is_space_cp(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
           cp == '\f' || cp == '\v' || cp == 0x0085 || cp == 0x00A0 ||
           cp == 0x1680 || in_range(cp, 0x2000, 0x200A) || cp == 0x2028 ||
           cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

bool is_number_cp(uint32_t cp) {
    static constexpr uint32_t ranges[][2] = {
        {0x0030, 0x0039}, {0x0660, 0x0669}, {0x06F0, 0x06F9},
        {0x07C0, 0x07C9}, {0x0966, 0x096F}, {0x09E6, 0x09EF},
        {0x0A66, 0x0A6F}, {0x0AE6, 0x0AEF}, {0x0B66, 0x0B6F},
        {0x0BE6, 0x0BEF}, {0x0C66, 0x0C6F}, {0x0CE6, 0x0CEF},
        {0x0D66, 0x0D6F}, {0x0DE6, 0x0DEF}, {0x0E50, 0x0E59},
        {0x0ED0, 0x0ED9}, {0x0F20, 0x0F29}, {0x1040, 0x1049},
        {0x17E0, 0x17E9}, {0x1810, 0x1819}, {0x1946, 0x194F},
        {0x19D0, 0x19D9}, {0x1A80, 0x1A89}, {0x1A90, 0x1A99},
        {0x1B50, 0x1B59}, {0x1BB0, 0x1BB9}, {0x1C40, 0x1C49},
        {0x1C50, 0x1C59}, {0xA620, 0xA629}, {0xA8D0, 0xA8D9},
        {0xA900, 0xA909}, {0xA9D0, 0xA9D9}, {0xA9F0, 0xA9F9},
        {0xAA50, 0xAA59}, {0xABF0, 0xABF9}, {0xFF10, 0xFF19},
    };
    for (const auto& range : ranges) {
        if (in_range(cp, range[0], range[1])) return true;
    }
    return false;
}

bool is_punctuation_or_symbol_cp(uint32_t cp) {
    if (cp < 128) return !std::isalnum(static_cast<unsigned char>(cp));
    if (in_range(cp, 0x2000, 0x206F) || in_range(cp, 0x20A0, 0x20CF) ||
        in_range(cp, 0x2100, 0x214F) || in_range(cp, 0x2190, 0x2BFF) ||
        in_range(cp, 0x2E00, 0x2E7F) || in_range(cp, 0x3001, 0x303F) ||
        in_range(cp, 0xFE10, 0xFE1F) || in_range(cp, 0xFE30, 0xFE6F) ||
        in_range(cp, 0xFF01, 0xFF0F) || in_range(cp, 0xFF1A, 0xFF20) ||
        in_range(cp, 0xFF3B, 0xFF40) || in_range(cp, 0xFF5B, 0xFF65) ||
        in_range(cp, 0x1F000, 0x1FAFF)) {
        return true;
    }
    switch (cp) {
        case 0x00A1: case 0x00A2: case 0x00A3: case 0x00A4: case 0x00A5:
        case 0x00A6: case 0x00A7: case 0x00A9: case 0x00AB: case 0x00AC:
        case 0x00AE: case 0x00B0: case 0x00B1: case 0x00B6: case 0x00B7:
        case 0x00BB: case 0x00BF: case 0x00D7: case 0x00F7:
        case 0x060C: case 0x061B: case 0x061F: case 0x066A: case 0x066B:
        case 0x066C: case 0x066D: case 0x06D4:
            return true;
        default:
            return false;
    }
}

int category(uint32_t cp) {
    if (is_space_cp(cp)) return 0;
    if (is_number_cp(cp)) return 2;
    if (is_punctuation_or_symbol_cp(cp)) return 3;
    return 1;
}

bool ascii_case_equal(std::string_view text, size_t offset,
                      std::string_view expected) {
    if (offset + expected.size() > text.size()) return false;
    for (size_t i = 0; i < expected.size(); ++i) {
        unsigned char a = static_cast<unsigned char>(text[offset + i]);
        unsigned char b = static_cast<unsigned char>(expected[i]);
        if (a >= 'A' && a <= 'Z') a = static_cast<unsigned char>(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = static_cast<unsigned char>(b - 'A' + 'a');
        if (a != b) return false;
    }
    return true;
}

std::string pair_key(const std::string& a, const std::string& b) {
    std::string out;
    out.reserve(a.size() + b.size() + 1);
    out += a;
    out.push_back('\0');
    out += b;
    return out;
}

}
