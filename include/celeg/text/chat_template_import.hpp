#pragma once

#include "celeg/text/semantic_chat_templates.hpp"

#include <string_view>

namespace celeg {

// Imports the deterministic, side-effect-free subset shared by tokenizer
// configuration files. Unsupported Jinja constructs fail at load time.
ChatTemplateProgram import_hf_chat_template(std::string_view source);

} // namespace celeg
