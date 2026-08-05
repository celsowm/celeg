#pragma once

#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>

namespace celeg {

// Protocol-neutral JSON string quoting. Bytes forming valid UTF-8 are kept
// intact; JSON controls are escaped explicitly and no parser is hidden here.
inline std::string json_quote(std::string_view value) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('"');
    for (const unsigned char byte : value) {
        switch (byte) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (byte < 0x20) {
                    out += "\\u00";
                    out.push_back(hex[byte >> 4]);
                    out.push_back(hex[byte & 0x0f]);
                } else {
                    out.push_back(static_cast<char>(byte));
                }
        }
    }
    out.push_back('"');
    return out;
}

inline std::string remove_tagged_blocks(std::string_view text,
                                        std::string_view start,
                                        std::string_view end) {
    std::string result(text);
    for (std::size_t pos = 0;
         (pos = result.find(start, pos)) != std::string::npos;) {
        const std::size_t finish = result.find(end, pos + start.size());
        if (finish == std::string::npos) {
            result.erase(pos);
            break;
        }
        result.erase(pos, finish + end.size() - pos);
    }
    return result;
}

} // namespace celeg
