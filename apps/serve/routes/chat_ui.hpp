#pragma once

#include "App.h"

#include <filesystem>

namespace celeg::app::serve {

void register_chat_ui_routes(uWS::App& app, const std::filesystem::path& asset_dir);

}
