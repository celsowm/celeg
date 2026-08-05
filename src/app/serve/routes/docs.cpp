#include "docs.hpp"

#include "../static_assets.hpp"
#include "celeg/serve/protocol/json.hpp"
#include "celeg/serve/protocol/mapping.hpp"
#include "celeg/serve/protocol/openapi.hpp"

#include <filesystem>
#include <string>
#include <unordered_map>

namespace celeg::app::serve {

namespace {

namespace protocol = celeg::serve::protocol;

// Known Swagger UI assets served under /docs/<name>. Kept as an explicit
// allowlist so request paths can never escape the packaged asset directory.
const std::unordered_map<std::string, std::string>& swagger_ui_assets() {
    static const std::unordered_map<std::string, std::string> assets = {
        {"swagger-ui-bundle.js", "application/javascript"},
        {"swagger-ui-standalone-preset.js", "application/javascript"},
        {"swagger-ui.css", "text/css"},
    };
    return assets;
}

} // namespace

void register_docs_routes(uWS::App& app, const std::string& model_name,
                          const std::filesystem::path& asset_dir) {
    const std::string docs_html = read_required_file(asset_dir / "index.html");

    app.get("/openapi.json", [model_name](auto* res, auto* /*req*/) {
        res->writeHeader("Content-Type", "application/json")
            ->end(protocol::build_openapi_spec(model_name));
    });

    app.get("/docs", [docs_html](auto* res, auto* /*req*/) {
        res->writeHeader("Content-Type", "text/html; charset=utf-8")
            ->writeHeader("X-Content-Type-Options", "nosniff")
            ->writeHeader("Referrer-Policy", "no-referrer")
            ->writeHeader("Content-Security-Policy",
                          "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                          "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                          "object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            ->end(docs_html);
    });

    app.get("/docs/:file", [asset_dir](auto* res, auto* req) {
        const std::string file(req->getParameter(0));
        const auto& assets = swagger_ui_assets();
        const auto asset = assets.find(file);
        if (asset == assets.end()) {
            res->writeStatus("404 Not Found")
                ->writeHeader("Content-Type", "application/json")
                ->end(protocol::to_json(protocol::error_response("documentation asset not found")));
            return;
        }
        const auto contents = read_file(asset_dir / file);
        if (!contents) {
            res->writeStatus("404 Not Found")
                ->writeHeader("Content-Type", "application/json")
                ->end(protocol::to_json(protocol::error_response("documentation asset not found")));
            return;
        }
        res->writeHeader("Content-Type", asset->second)
            ->writeHeader("X-Content-Type-Options", "nosniff")
            ->writeHeader("Cache-Control", "public, max-age=86400")
            ->end(*contents);
    });
}

} // namespace celeg::app::serve
