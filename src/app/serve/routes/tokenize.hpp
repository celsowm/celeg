#pragma once

#include "App.h"
#include "lfm/text/tokenizer.hpp"

#include <cstddef>

namespace lfm::app::serve {

// POST /tokenize -- vLLM/SGLang-style tokenization endpoint.
void register_tokenize_route(uWS::App& app, const lfm::BpeTokenizer& tokenizer,
                             std::size_t max_model_len);

} // namespace lfm::app::serve
