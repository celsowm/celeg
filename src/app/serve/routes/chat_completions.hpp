#pragma once

#include "App.h"
#include "celeg/serve/generation_dispatcher.hpp"
#include "celeg/serve/types.hpp"
#include "celeg/text/tokenizer.hpp"

#include <cstdint>
#include <string>

namespace celeg::app::serve {

// POST /v1/chat/completions -- OpenAI-compatible, supports both a single
// JSON response and (stream: true) a Server-Sent Events response.
void register_chat_completions_route(uWS::App& app,
                                     celeg::serve::GenerationDispatcher& dispatcher,
                                     celeg::serve::IInferenceService& service,
                                     const celeg::BpeTokenizer& tokenizer,
                                     const std::string& model_name,
                                     std::int32_t eos_token_id,
                                     uWS::Loop* loop);

} // namespace celeg::app::serve
