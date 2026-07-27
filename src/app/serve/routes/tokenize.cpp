#include "tokenize.hpp"

#include "../json_route.hpp"
#include "lfm/serve/protocol/chat.hpp"
#include "lfm/serve/protocol/mapping.hpp"

namespace lfm::app::serve {

namespace {
namespace protocol = lfm::serve::protocol;
} // namespace

void register_tokenize_route(uWS::App& app, const lfm::BpeTokenizer& tokenizer,
                             std::size_t max_model_len) {
    app.post("/tokenize", [&tokenizer, max_model_len](auto* res, auto* /*req*/) {
        handle_json_post<protocol::TokenizeRequest, protocol::TokenizeResponse>(
            res, [&tokenizer, max_model_len](const protocol::TokenizeRequest& request) {
                return protocol::to_tokenize_response(request, tokenizer, max_model_len);
            });
    });
}

} // namespace lfm::app::serve
