#pragma once

#include "celeg/text/tool_call.hpp"
#include "celeg/text/conversation.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace celeg {

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
