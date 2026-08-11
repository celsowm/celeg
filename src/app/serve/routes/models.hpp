#pragma once

#include "App.h"
#include "celeg/text/chat_template.hpp"

#include <string>

namespace celeg::app::serve {

// GET /v1/models -- OpenAI-style single-model listing.
void register_models_route(uWS::App& app, const std::string& model_name,
                           std::size_t context_window,
                           const celeg::ChatCapabilities& capabilities);

} // namespace celeg::app::serve
