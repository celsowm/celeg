#pragma once

#include "celeg/text/chat_template.hpp"

#include <string>
#include <string_view>

namespace celeg::chat_template_detail {

std::string json_string(std::string_view value);
void reject_unhandled_options(const ChatTemplateOptions& options,
                              std::string_view profile_id);

} // namespace celeg::chat_template_detail
