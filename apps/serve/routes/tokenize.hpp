#pragma once

#include "App.h"
#include "celeg/text/tokenizer.hpp"

#include <cstddef>

namespace celeg::app::serve {

void register_tokenize_route(uWS::App& app, const celeg::ITokenizer& tokenizer,
                             const celeg::ResolvedInteraction& interaction,
                             std::size_t max_model_len);

}
