#include "celeg/models/gemma4/chat_template.hpp"

#include "celeg/text/chat_profile.hpp"
#include "celeg/text/detail/chat_template_support.hpp"

namespace celeg {

std::string Gemma4InstructChatTemplate::format(
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool add_generation_prompt,
    const ChatTemplateOptions& options) const {
    chat_template_detail::reject_unhandled_options(options, "gemma4-instruct");
    std::string out = "<bos>";
    if (!tools.empty()) {
        out += "<|turn>system\n";
        for (const auto& tool : tools) {
            out += "declaration:" + tool.function.name + "{description:<|\"|>" +
                tool.function.description + "<|\"|>,parameters:" +
                (tool.function.parameters.serialized.empty() ? "{}" : tool.function.parameters.serialized) + "}\n";
        }
        out += "<turn|>\n";
    }
    for (const ChatMessage& message : messages) {
        const char* role = nullptr;
        switch (message.role) {
            case ChatRole::System: role = "system"; break;
            case ChatRole::User: role = "user"; break;
            case ChatRole::Assistant: role = "model"; break;
            case ChatRole::Developer: role = "system"; break;
            case ChatRole::Tool: role = "tool"; break;
        }
        if (message.role == ChatRole::Tool) {
            out += "<|tool_response>response:" + message.tool_call_id.value_or("unknown") +
                "{value:" + message.content.value_or("") + "}<tool_response|>\n";
            continue;
        }
        out += "<|turn>" + std::string(role) + "\n" + message.content.value_or("");
        for (const auto& call : message.tool_calls) {
            out += "<|tool_call>call:" + call.name + call.arguments + "<tool_call|>";
        }
        out += "<turn|>\n";
    }
    if (add_generation_prompt) out += "<|turn>model\n";
    return out;
}

void add_gemma4_chat_profile(ChatProfileCatalog& catalog) {
    ChatCapabilities capabilities;
    capabilities.vision = true;
    capabilities.assistant_tool_calls = true;
    capabilities.parallel_tool_calls = true;
    capabilities.native_tool_call_codec = true;
    capabilities.roles.developer = true;
    capabilities.roles.tool = true;
    catalog.add("gemma4-instruct", std::make_unique<Gemma4InstructChatTemplate>(),
                make_gemma4_tool_call_codec(), capabilities);
}

} // namespace celeg
