#include "models.hpp"

namespace lfm::app::serve {

void register_models_route(uWS::App& app, const std::string& model_name) {
    app.get("/v1/models", [model_name](auto* res, auto* /*req*/) {
        res->writeHeader("Content-Type", "application/json")
            ->end("{\"object\":\"list\",\"data\":[{\"id\":\"" + model_name + "\",\"object\":\"model\"}]}");
    });
}

} // namespace lfm::app::serve
