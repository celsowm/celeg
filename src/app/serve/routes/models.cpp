#include "models.hpp"

#include "celeg/serve/protocol/chat.hpp"
#include "celeg/serve/protocol/json.hpp"

namespace celeg::app::serve {

void register_models_route(uWS::App& app, const std::string& model_name,
                           std::size_t context_window,
                           const celeg::ChatCapabilities& capabilities) {
    namespace protocol = celeg::serve::protocol;
    protocol::ModelListDto models;
    protocol::ModelDto model;
    model.id = model_name;
    model.celeg.context_window = context_window;
    model.celeg.capabilities.vision = capabilities.vision;
    model.celeg.capabilities.tool_calls = capabilities.assistant_tool_calls;
    model.celeg.capabilities.parallel_tool_calls = capabilities.parallel_tool_calls;
    models.data.push_back(std::move(model));
    const std::string json = protocol::to_json(models);
    app.get("/v1/models", [json](auto* res, auto* /*req*/) {
        res->writeHeader("Content-Type", "application/json")
            ->end(json);
    });
}

}
