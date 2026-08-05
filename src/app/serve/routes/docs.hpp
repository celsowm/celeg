#pragma once

#include "App.h"

#include <filesystem>
#include <string>

namespace celeg::app::serve {

// GET /openapi.json and GET /docs + /docs/:file (vendored Swagger UI).
void register_docs_routes(uWS::App& app, const std::string& model_name,
                          const std::filesystem::path& asset_dir);

} // namespace celeg::app::serve
