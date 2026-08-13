#include "tokenize.hpp"

#include "../json_route.hpp"
#include "celeg/serve/protocol/chat.hpp"
#include "celeg/serve/protocol/mapping.hpp"

namespace celeg::app::serve {

namespace {
namespace protocol = celeg::serve::protocol;
} // namespace

void register_tokenize_route(uWS::App& app, const celeg::ITokenizer& tokenizer,
                             const celeg::ResolvedInteraction& interaction,
                             std::size_t max_model_len) {
    app.post("/tokenize", [&tokenizer, &interaction, max_model_len](auto* res, auto* /*req*/) {
        handle_json_post<protocol::TokenizeRequest, protocol::TokenizeResponse>(
            res, [&tokenizer, &interaction, max_model_len](const protocol::TokenizeRequest& request) {
                return protocol::to_tokenize_response(request, tokenizer, interaction, max_model_len);
            });
    });
}

} // namespace celeg::app::serve
