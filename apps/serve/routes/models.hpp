#pragma once

#include "App.h"
#include "celeg/text/chat_template.hpp"

#include <string>

namespace celeg::app::serve {

void register_models_route(uWS::App& app, const std::string& model_name,
                           std::size_t context_window,
                           const celeg::ChatCapabilities& capabilities);

}
