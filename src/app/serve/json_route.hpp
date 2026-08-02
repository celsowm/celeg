#pragma once

#include "celeg/serve/protocol/mapping.hpp"
#include "celeg/serve/protocol/json.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace celeg::app::serve {

// Wires up the "buffer the full request body, parse it as Req, hand it to
// fn(), write the result as JSON, or write a 400 on any thrown exception"
// shape shared by every synchronous JSON POST endpoint. Callers only supply
// the DTO types and the business-logic call itself.
template <typename Req, typename Resp, typename Res, typename Fn>
void handle_json_post(Res* res, Fn&& fn) {
    auto body = std::make_shared<std::string>();
    res->onAborted([]() {});
    res->onData([res, body, fn = std::forward<Fn>(fn)](std::string_view chunk, bool last) mutable {
        body->append(chunk);
        if (!last) return;

        try {
            const Req request = celeg::serve::protocol::from_json<Req>(*body);
            const Resp response = fn(request);
            res->writeHeader("Content-Type", "application/json")
                ->end(celeg::serve::protocol::to_json(response));
        } catch (const std::exception& error) {
            res->writeStatus("400 Bad Request")
                ->writeHeader("Content-Type", "application/json")
                ->end(celeg::serve::protocol::to_json(
                    celeg::serve::protocol::error_response(error.what())));
        }
    });
}

} // namespace celeg::app::serve
