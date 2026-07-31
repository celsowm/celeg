#pragma once

#include "App.h"
#include "celeg/text/tokenizer.hpp"

#include <cstddef>

namespace celeg::app::serve {

// POST /tokenize -- vLLM/SGLang-style tokenization endpoint.
void register_tokenize_route(uWS::App& app, const celeg::BpeTokenizer& tokenizer,
                             std::size_t max_model_len);

} // namespace celeg::app::serve
