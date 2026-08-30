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

ChatRole role_from_string(const std::string& role);
std::string role_to_string(ChatRole role);

void validate_chat_request(const ChatCompletionRequest& request,
                           const ChatCapabilities& capabilities);
ErrorResponseDto error_response(std::string message,
                                std::optional<std::string> param = std::nullopt);

GenerateRequest to_generate_request(const ChatCompletionRequest& request,
                                    const celeg::ITokenizer& tokenizer,
                                    const celeg::ResolvedInteraction& interaction,
                                    std::span<const std::int32_t> eos_token_ids,
                                    const celeg::ChatTemplateOptions& template_options = {},
                                    std::size_t max_context_tokens = 0);

std::string finish_reason_to_string(FinishReason reason);

ChatCompletionResponse to_chat_completion_response(const std::string& id,
                                                   const std::string& model,
                                                   std::int64_t created,
                                                   std::size_t prompt_token_count,
                                                   const std::vector<std::int32_t>& completion_tokens,
                                                   FinishReason reason,
                                                   const celeg::ITokenizer& tokenizer,
                                                   const celeg::ResolvedInteraction& interaction);

ChatCompletionChunk to_chat_completion_chunk(const std::string& id,
                                             const std::string& model,
                                             std::int64_t created,
                                             const std::vector<std::int32_t>& new_tokens,
                                             bool include_role,
                                             std::optional<FinishReason> finish,
                                             const celeg::ITokenizer& tokenizer,
                                             const celeg::ResolvedInteraction& interaction,
                                             std::string_view accumulated_text = {});

ChatCompletionChunk to_chat_completion_chunk(const std::string& id,
                                             const std::string& model,
                                             std::int64_t created,
                                             const celeg::serve::ChatGenerationDelta& delta,
                                             bool include_role,
                                             std::optional<FinishReason> finish);

TokenizeResponse to_tokenize_response(const TokenizeRequest& request,
                                      const celeg::ITokenizer& tokenizer,
                                      const celeg::ResolvedInteraction& interaction,
                                      std::size_t max_model_len);

}
