#pragma once

#include "App.h"

#include <string>

namespace celeg::app::serve {

// GET /v1/models -- OpenAI-style single-model listing.
void register_models_route(uWS::App& app, const std::string& model_name);

} // namespace celeg::app::serve
