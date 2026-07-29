#include "lfm/serve/protocol/mapping.hpp"

#include <stdexcept>

namespace lfm::serve::protocol {

ChatRole role_from_string(const std::string& role) {
    if (role == "system") return ChatRole::System;
    if (role == "developer") return ChatRole::Developer;
    if (role == "user") return ChatRole::User;
    if (role == "assistant") return ChatRole::Assistant;
    if (role == "tool") return ChatRole::Tool;
    throw std::invalid_argument("unknown chat role: " + role);
}

std::string role_to_string(ChatRole role) {
    switch (role) {
        case ChatRole::System: return "system";
        case ChatRole::Developer: return "developer";
        case ChatRole::User: return "user";
        case ChatRole::Assistant: return "assistant";
        case ChatRole::Tool: return "tool";
    }
    throw std::invalid_argument("unknown chat role");
}

GenerateRequest to_generate_request(const ChatCompletionRequest& request,
                                    const lfm::BpeTokenizer& tokenizer,
                                    std::int32_t eos_token_id) {
    if (request.messages.empty()) {
        throw std::invalid_argument("messages must not be empty");
    }

    std::vector<lfm::ChatMessage> messages;
    messages.reserve(request.messages.size());
    for (const ChatMessageDto& message : request.messages) {
        messages.push_back({role_from_string(message.role), message.content});
    }

    const std::string prompt_text = tokenizer.format_chat(messages, /*add_generation_prompt=*/true);

    GenerateRequest generate_request;
    generate_request.prompt_tokens = tokenizer.encode(prompt_text, /*add_bos=*/false);
    generate_request.eos_token_id = eos_token_id;
    if (request.max_tokens && *request.max_tokens <= 0) {
        throw std::invalid_argument("max_tokens must be positive");
    }
    generate_request.max_output_tokens =
        request.max_tokens ? static_cast<std::size_t>(*request.max_tokens) : 128;
    if (request.temperature) generate_request.generation.temperature = static_cast<float>(*request.temperature);
    if (request.top_p) generate_request.generation.top_p = static_cast<float>(*request.top_p);
    if (request.top_k) generate_request.generation.top_k = *request.top_k;
    if (request.seed) generate_request.generation.seed = *request.seed;
    return generate_request;
}

std::string finish_reason_to_string(FinishReason reason) {
    switch (reason) {
        case FinishReason::None: return "";
        case FinishReason::Stop: return "stop";
        case FinishReason::Length: return "length";
        case FinishReason::Cancelled: return "cancelled";
        case FinishReason::Error: return "error";
    }
    return "";
}

ChatCompletionResponse to_chat_completion_response(const std::string& id,
                                                   const std::string& model,
                                                   std::int64_t created,
                                                   std::size_t prompt_token_count,
                                                   const std::vector<std::int32_t>& completion_tokens,
                                                   FinishReason reason,
                                                   const lfm::BpeTokenizer& tokenizer) {
    ChatCompletionResponse response;
    response.id = id;
    response.model = model;
    response.created = created;

    ChatCompletionChoice choice;
    choice.index = 0;
    choice.message.content = tokenizer.decode(completion_tokens);
    choice.finish_reason = finish_reason_to_string(reason);
    response.choices.push_back(std::move(choice));

    response.usage.prompt_tokens = prompt_token_count;
    response.usage.completion_tokens = completion_tokens.size();
    response.usage.total_tokens = prompt_token_count + completion_tokens.size();
    return response;
}

ChatCompletionChunk to_chat_completion_chunk(const std::string& id,
                                             const std::string& model,
                                             std::int64_t created,
                                             const std::vector<std::int32_t>& new_tokens,
                                             bool include_role,
                                             std::optional<FinishReason> finish,
                                             const lfm::BpeTokenizer& tokenizer) {
    ChatCompletionChunk chunk;
    chunk.id = id;
    chunk.model = model;
    chunk.created = created;

    ChatCompletionChunkChoice choice;
    choice.index = 0;
    if (include_role) choice.delta.role = "assistant";
    if (!new_tokens.empty()) choice.delta.content = tokenizer.decode(new_tokens);
    if (finish) choice.finish_reason = finish_reason_to_string(*finish);
    chunk.choices.push_back(std::move(choice));
    return chunk;
}

TokenizeResponse to_tokenize_response(const TokenizeRequest& request,
                                      const lfm::BpeTokenizer& tokenizer,
                                      std::size_t max_model_len) {
    if (static_cast<bool>(request.prompt) == static_cast<bool>(request.messages)) {
        throw std::invalid_argument("exactly one of prompt or messages must be set");
    }

    const bool add_special_tokens = request.add_special_tokens.value_or(true);

    TokenizeResponse response;
    if (request.prompt) {
        response.tokens = tokenizer.encode(*request.prompt, /*add_bos=*/add_special_tokens);
    } else {
        std::vector<lfm::ChatMessage> messages;
        messages.reserve(request.messages->size());
        for (const ChatMessageDto& message : *request.messages) {
            messages.push_back({role_from_string(message.role), message.content});
        }
        const std::string prompt_text =
            tokenizer.format_chat(messages, /*add_generation_prompt=*/true);
        // The chat template already embeds the model's special tokens, so BOS
        // is never added again here regardless of add_special_tokens (which
        // only applies to the raw-prompt path above).
        response.tokens = tokenizer.encode(prompt_text, /*add_bos=*/false);
    }
    response.count = response.tokens.size();
    response.max_model_len = max_model_len;
    return response;
}

} // namespace lfm::serve::protocol
