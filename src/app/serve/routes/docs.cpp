#include "docs.hpp"

#include "../static_assets.hpp"
#include "celeg/serve/protocol/json.hpp"
#include "celeg/serve/protocol/mapping.hpp"
#include "celeg/serve/protocol/openapi.hpp"

#include <filesystem>
#include <string>
#include <unordered_map>

#ifndef CELEG_SERVE_STATIC_DIR
#error "CELEG_SERVE_STATIC_DIR must be defined by the build (see CMakeLists.txt)"
#endif
#ifndef CELEG_SWAGGER_UI_DIR
#error "CELEG_SWAGGER_UI_DIR must be defined by the build (see CMakeLists.txt)"
#endif

namespace celeg::app::serve {

namespace {

namespace protocol = celeg::serve::protocol;

// Known Swagger UI dist assets served under /docs/<name>. Kept as an
// explicit allowlist (rather than general static-file serving) so a request
// path can never escape CELEG_SWAGGER_UI_DIR.
const std::unordered_map<std::string, std::string>& swagger_ui_assets() {
    static const std::unordered_map<std::string, std::string> assets = {
        {"swagger-ui-bundle.js", "application/javascript"},
        {"swagger-ui-standalone-preset.js", "application/javascript"},
        {"swagger-ui.css", "text/css"},
    };
    return assets;
}

} // namespace

void register_docs_routes(uWS::App& app, const std::string& model_name) {
    static const std::string index_html =
        read_required_file(std::filesystem::path(CELEG_SERVE_STATIC_DIR) / "index.html");
    static const std::string docs_html =
        read_required_file(std::filesystem::path(CELEG_SERVE_STATIC_DIR) / "docs.html");

    app.get("/", [](auto* res, auto* /*req*/) {
        res->writeHeader("Content-Type", "text/html; charset=utf-8")->end(index_html);
    });

    app.get("/openapi.json", [model_name](auto* res, auto* /*req*/) {
        res->writeHeader("Content-Type", "application/json")
            ->end(protocol::build_openapi_spec(model_name));
    });

    app.get("/docs", [](auto* res, auto* /*req*/) {
        res->writeHeader("Content-Type", "text/html; charset=utf-8")->end(docs_html);
    });

    app.get("/docs/:file", [](auto* res, auto* req) {
        const std::string file(req->getParameter(0));
        const auto& assets = swagger_ui_assets();
        const auto asset = assets.find(file);
        if (asset == assets.end()) {
            res->writeStatus("404 Not Found")
                ->writeHeader("Content-Type", "application/json")
                ->end(protocol::to_json(protocol::error_response("documentation asset not found")));
            return;
        }
        const auto contents = read_file(std::filesystem::path(CELEG_SWAGGER_UI_DIR) / file);
        if (!contents) {
            res->writeStatus("404 Not Found")
                ->writeHeader("Content-Type", "application/json")
                ->end(protocol::to_json(protocol::error_response("documentation asset not found")));
            return;
        }
        res->writeHeader("Content-Type", asset->second)->end(*contents);
    });
}

} // namespace celeg::app::serve
