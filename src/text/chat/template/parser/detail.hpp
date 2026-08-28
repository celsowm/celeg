#pragma once

#include "../program.hpp"

#include <string>
#include <string_view>

namespace celeg::chat_template_detail {

Expression parse_expression(std::string_view source);
std::string trim(std::string_view input);

}
