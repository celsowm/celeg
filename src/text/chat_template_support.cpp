#include "celeg/text/detail/chat_template_support.hpp"

#include <stdexcept>

namespace celeg::chat_template_detail {

std::string json_string(std::string_view value) {
    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('"');
    for (const char ch : value) {
        if (ch == '\\' || ch == '"') out.push_back('\\');
        if (ch == '\n') { out += "\\n"; continue; }
        if (ch == '\r') { out += "\\r"; continue; }
        if (ch == '\t') { out += "\\t"; continue; }
        out.push_back(ch);
    }
    out.push_back('"');
    return out;
}

void reject_unhandled_options(const ChatTemplateOptions& options,
                              std::string_view profile_id) {
    if (options.enable_thinking) {
        throw std::invalid_argument(
            "chat profile '" + std::string(profile_id) +
            "' does not support enable_thinking");
    }
}

} // namespace celeg::chat_template_detail
