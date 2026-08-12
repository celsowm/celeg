#pragma once

#include "celeg/text/tool_call.hpp"
#include "celeg/text/conversation.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace celeg {

class IChatToolCallCodec {
public:
    virtual ~IChatToolCallCodec() = default;
    virtual bool supports_parallel_calls() const noexcept = 0;
    virtual std::string render_tool_definitions(
        std::span<const ToolDefinition> tools, const ToolChoice& choice) const = 0;
    virtual std::string render_assistant_tool_calls(
        std::span<const ToolCall> calls) const = 0;
    // Returns the protocol prefix that must begin a required/specific call.
    // An empty result means that this codec does not provide prefix forcing.
    virtual std::string forced_tool_call_prefix(
        std::span<const ToolDefinition> tools, const ToolChoice& choice) const = 0;
    virtual std::string render_tool_result(const ChatMessage& message) const = 0;
    virtual ToolParseResult parse_generation(std::string_view generated_text) const = 0;
};

struct ChatRoleCapabilities {
    bool system = true;
    bool developer = false;
    bool user = true;
    bool assistant = true;
    bool tool = false;
};

struct ChatCapabilities {
    bool vision = false;
    bool video = false;
    // Token text whose single occurrence is replaced by visual embeddings.
    // It belongs to the resolved interaction contract, not to protocol
    // mapping or a backend.
    std::string image_marker = "<|image|>";
    std::string video_marker = "<|video|>";
    bool assistant_tool_calls = false;
    bool parallel_tool_calls = false;
    bool native_tool_call_codec = false;
    ChatRoleCapabilities roles;
};

} // namespace celeg
