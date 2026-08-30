#include "health.hpp"

namespace celeg::app::serve {

void register_health_routes(uWS::App& app) {
    app.get("/healthz", [](auto* res, auto* /*req*/) { res->end("ok"); });
    app.get("/health", [](auto* res, auto* /*req*/) { res->writeStatus("200 OK")->end(); });
}

}
