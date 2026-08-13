#pragma once

#include "celeg/text/tool_call.hpp"

#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace celeg {

enum class ChatRole {
    System,
    Developer,
    User,
    Assistant,
    Tool,
};

struct ChatMessage {
    ChatRole role;
    std::optional<std::string> content;
    std::vector<ToolCall> tool_calls;
    std::optional<std::string> tool_call_id;
    std::optional<std::string> name;
    std::optional<std::string> reasoning_content;
};

inline void validate_conversation(std::span<const ChatMessage> messages) {
    std::vector<std::string> call_ids;
    for (const ChatMessage& message : messages) {
        if (message.role == ChatRole::Tool) {
            if (!message.tool_call_id || message.tool_call_id->empty()) {
                throw std::invalid_argument("tool messages require tool_call_id");
            }
            if (!message.content) throw std::invalid_argument("tool messages require content");
            bool found = false;
            for (const std::string& id : call_ids) found = found || id == *message.tool_call_id;
            if (!found) throw std::invalid_argument("tool message references an unknown tool call id");
        } else if (message.role == ChatRole::Assistant) {
            if (!message.content && !message.reasoning_content && message.tool_calls.empty()) {
                throw std::invalid_argument("assistant messages require content, reasoning_content, or tool_calls");
            }
            call_ids.clear();
            for (const ToolCall& call : message.tool_calls) {
                if (call.id.empty()) throw std::invalid_argument("assistant tool call ids must not be empty");
                for (const std::string& prior : call_ids) {
                    if (prior == call.id) throw std::invalid_argument("assistant tool call IDs must be unique");
                }
                call_ids.push_back(call.id);
            }
        } else {
            if (!message.content) throw std::invalid_argument("message content is required");
            call_ids.clear();
        }
    }
}

} // namespace celeg
