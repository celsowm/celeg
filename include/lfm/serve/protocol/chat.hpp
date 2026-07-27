#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace lfm::serve::protocol {

// OpenAI /v1/chat/completions request/response DTOs. Field names match the
// JSON wire format exactly so Glaze's automatic reflection can (de)serialize
// them without glz::meta specializations.

struct ChatMessageDto {
    std::string role;
    std::string content;
};

struct ChatCompletionRequest {
    std::string model;
    std::vector<ChatMessageDto> messages;
    std::optional<double> temperature;
    std::optional<double> top_p;
    std::optional<int> top_k;
    std::optional<int> max_tokens;
    std::optional<bool> stream;
    std::optional<std::uint64_t> seed;
};

struct ChatCompletionResponseMessage {
    std::string role = "assistant";
    std::string content;
};

struct ChatCompletionChoice {
    int index = 0;
    ChatCompletionResponseMessage message;
    std::string finish_reason;
};

struct Usage {
    std::size_t prompt_tokens = 0;
    std::size_t completion_tokens = 0;
    std::size_t total_tokens = 0;
};

struct ChatCompletionResponse {
    std::string id;
    std::string object = "chat.completion";
    std::int64_t created = 0;
    std::string model;
    std::vector<ChatCompletionChoice> choices;
    Usage usage;
};

// Server-Sent Events chunk for stream=true, one per "data: " line.
struct ChatCompletionChunkDelta {
    std::optional<std::string> role;
    std::optional<std::string> content;
};

struct ChatCompletionChunkChoice {
    int index = 0;
    ChatCompletionChunkDelta delta;
    std::optional<std::string> finish_reason;
};

struct ChatCompletionChunk {
    std::string id;
    std::string object = "chat.completion.chunk";
    std::int64_t created = 0;
    std::string model;
    std::vector<ChatCompletionChunkChoice> choices;
};

} // namespace lfm::serve::protocol
