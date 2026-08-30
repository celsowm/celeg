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

void register_chat_completions_route(uWS::App& app,
                                     celeg::serve::GenerationDispatcher& dispatcher,
                                     celeg::serve::IRequestService& service,
                                     const celeg::ITokenizer& tokenizer,
                                     const celeg::ResolvedInteraction& interaction,
                                     const std::string& model_name,
                                     std::span<const std::int32_t> eos_token_ids,
                                     std::size_t max_context_tokens,
                                     uWS::Loop* loop);

}
