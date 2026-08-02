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
    virtual std::string render_tool_result(const ChatMessage& message) const = 0;
    virtual ToolParseResult parse_generation(std::string_view generated_text) const = 0;
};

std::unique_ptr<IChatToolCallCodec> make_lfm2_tool_call_codec();
std::unique_ptr<IChatToolCallCodec> make_gemma4_tool_call_codec();

struct ChatCapabilities {
    bool developer_messages = false;
    bool tool_messages = false;
    bool assistant_tool_calls = false;
    bool parallel_tool_calls = false;
    bool native_tool_call_codec = false;
    const IChatToolCallCodec* tool_call_codec = nullptr;
};

struct ChatProfile {
    std::string id;
    ChatCapabilities capabilities;
};


} // namespace celeg
