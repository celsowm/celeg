#pragma once

#include "App.h"
#include "lfm/serve/generation_dispatcher.hpp"
#include "lfm/serve/types.hpp"
#include "lfm/text/tokenizer.hpp"

#include <cstdint>
#include <string>

namespace lfm::app::serve {

// POST /v1/chat/completions -- OpenAI-compatible, supports both a single
// JSON response and (stream: true) a Server-Sent Events response.
void register_chat_completions_route(uWS::App& app,
                                     lfm::serve::GenerationDispatcher& dispatcher,
                                     lfm::serve::IInferenceService& service,
                                     const lfm::BpeTokenizer& tokenizer,
                                     const std::string& model_name,
                                     std::int32_t eos_token_id,
                                     uWS::Loop* loop);

} // namespace lfm::app::serve
