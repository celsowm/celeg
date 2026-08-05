#pragma once

#include "App.h"
#include "celeg/serve/generation_dispatcher.hpp"
#include "celeg/serve/types.hpp"
#include "celeg/text/tokenizer.hpp"

#include <cstdint>
#include <cstddef>
#include <span>
#include <string>

namespace celeg::app::serve {

// POST /v1/chat/completions -- OpenAI-compatible, supports both a single
// JSON response and (stream: true) a Server-Sent Events response.
void register_chat_completions_route(uWS::App& app,
                                     celeg::serve::GenerationDispatcher& dispatcher,
                                     celeg::serve::IRequestService& service,
                                     const celeg::BpeTokenizer& tokenizer,
                                     const celeg::IChatTemplate& chat_template,
                                     const celeg::ChatCapabilities& capabilities,
                                     const std::string& model_name,
                                     std::span<const std::int32_t> eos_token_ids,
                                     std::size_t max_context_tokens,
                                     uWS::Loop* loop);

} // namespace celeg::app::serve
