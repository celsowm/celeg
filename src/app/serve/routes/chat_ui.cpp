#include "chat_ui.hpp"

#include "../static_assets.hpp"
#include "celeg/serve/protocol/json.hpp"
#include "celeg/serve/protocol/mapping.hpp"

#include <memory>
#include <string>
#include <unordered_map>

namespace celeg::app::serve {

namespace {

namespace protocol = celeg::serve::protocol;

struct Asset {
    std::string content_type;
    std::string contents;
};

void write_security_headers(uWS::HttpResponse<false>* res) {
    res->writeHeader("X-Content-Type-Options", "nosniff")
        ->writeHeader("Referrer-Policy", "no-referrer")
        ->writeHeader("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        ->writeHeader("Cross-Origin-Resource-Policy", "same-origin");
}

} // namespace

void register_chat_ui_routes(uWS::App& app, const std::filesystem::path& asset_dir) {
    const auto index_html = std::make_shared<const std::string>(
        read_required_file(asset_dir / "index.html"));
    auto assets = std::make_shared<std::unordered_map<std::string, Asset>>();
    for (const auto& entry : std::filesystem::directory_iterator(asset_dir / "assets")) {
        if (!entry.is_regular_file()) continue;
        const std::string extension = entry.path().extension().string();
        const std::string content_type = extension == ".js" ? "application/javascript" :
            extension == ".css" ? "text/css" : "";
        if (content_type.empty()) continue;
        assets->emplace(entry.path().filename().string(),
                        Asset{content_type, read_required_file(entry.path())});
    }

    app.get("/", [index_html](auto* res, auto* /*req*/) {
        write_security_headers(res);
        res->writeHeader("Content-Type", "text/html; charset=utf-8")
            ->writeHeader("Cache-Control", "no-cache")
            ->writeHeader("Content-Security-Policy",
                          "default-src 'self'; script-src 'self'; style-src 'self'; "
                          "img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; "
                          "object-src 'none'; base-uri 'none'; form-action 'self'; "
                          "frame-ancestors 'none'")
            ->end(*index_html);
    });

    app.get("/assets/:file", [assets](auto* res, auto* req) {
        const auto asset = assets->find(std::string(req->getParameter(0)));
        if (asset == assets->end()) {
            res->writeStatus("404 Not Found")
                ->writeHeader("Content-Type", "application/json")
                ->end(protocol::to_json(protocol::error_response("chat asset not found")));
            return;
        }
        write_security_headers(res);
        res->writeHeader("Content-Type", asset->second.content_type)
            ->writeHeader("Cache-Control", "public, max-age=31536000, immutable")
            ->end(asset->second.contents);
    });
}

} // namespace celeg::app::serve
