#pragma once

#include "celeg/serve/protocol/chat.hpp"
#include "celeg/serve/chat_generation.hpp"
#include "celeg/serve/types.hpp"
#include "celeg/text/tokenizer.hpp"

#include <cstdint>
#include <span>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace celeg::serve::protocol {

// Converts an OpenAI-style role string ("system"/"developer"/"user"/
// "assistant"/"tool") to celeg::ChatRole. Throws std::invalid_argument for any
// other value.
ChatRole role_from_string(const std::string& role);
std::string role_to_string(ChatRole role);

void validate_chat_request(const ChatCompletionRequest& request,
                           const ChatCapabilities& capabilities);
ErrorResponseDto error_response(std::string message,
                                std::optional<std::string> param = std::nullopt);

// Builds a backend GenerateRequest from a chat completion request: renders
// the message list through the tokenizer's chat template and tokenizes it.
// Stop token ids are passed through unchanged (callers own the model's token ids).
GenerateRequest to_generate_request(const ChatCompletionRequest& request,
                                    const celeg::BpeTokenizer& tokenizer,
                                    const celeg::IChatTemplate& chat_template,
                                    const celeg::ChatCapabilities& capabilities,
                                    std::span<const std::int32_t> eos_token_ids,
                                    const celeg::ChatTemplateOptions& template_options = {});

// Maps a backend FinishReason to the OpenAI wire string. Returns "" for
// FinishReason::None (request still in progress).
std::string finish_reason_to_string(FinishReason reason);

// Builds the final, non-streaming response once a request has finished.
ChatCompletionResponse to_chat_completion_response(const std::string& id,
                                                   const std::string& model,
                                                   std::int64_t created,
                                                   std::size_t prompt_token_count,
                                                   const std::vector<std::int32_t>& completion_tokens,
                                                   FinishReason reason,
                                                   const celeg::BpeTokenizer& tokenizer,
                                                   const celeg::ChatCapabilities& capabilities);

// Builds one SSE chunk for a batch of newly generated tokens. include_role
// should be true only for the first chunk of a stream. finish_reason is
// unset for every chunk except the last, which carries no token text.
ChatCompletionChunk to_chat_completion_chunk(const std::string& id,
                                             const std::string& model,
                                             std::int64_t created,
                                             const std::vector<std::int32_t>& new_tokens,
                                             bool include_role,
                                             std::optional<FinishReason> finish,
                                             const celeg::BpeTokenizer& tokenizer,
                                             const celeg::ChatCapabilities& capabilities,
                                             std::string_view accumulated_text = {});

ChatCompletionChunk to_chat_completion_chunk(const std::string& id,
                                             const std::string& model,
                                             std::int64_t created,
                                             const celeg::serve::ChatGenerationDelta& delta,
                                             bool include_role,
                                             std::optional<FinishReason> finish);

// Tokenizes a /tokenize request: renders `messages` through the chat template
// if given, otherwise tokenizes `prompt` directly. Throws std::invalid_argument
// if neither (or both) are given. max_model_len is the server's configured
// context window, echoed back as-is.
TokenizeResponse to_tokenize_response(const TokenizeRequest& request,
                                      const celeg::BpeTokenizer& tokenizer,
                                      const celeg::IChatTemplate& chat_template,
                                      std::size_t max_model_len);

} // namespace celeg::serve::protocol
