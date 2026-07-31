#pragma once

#include "App.h"

namespace celeg::app::serve {

// GET /healthz (legacy, plain-text "ok") and GET /health (vLLM/SGLang-style
// bare 200, empty body).
void register_health_routes(uWS::App& app);

} // namespace celeg::app::serve
