#pragma once

#include "App.h"

#include <string>

namespace lfm::app::serve {

// Registers the human-facing surface: GET / (a tiny try-it-yourself page),
// GET /openapi.json (the OpenAPI 3.1 spec), and GET /docs + /docs/:file
// (vendored Swagger UI that renders it).
void register_docs_routes(uWS::App& app, const std::string& model_name);

} // namespace lfm::app::serve
