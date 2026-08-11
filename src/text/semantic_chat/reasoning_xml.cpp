#include "celeg/text/semantic_chat_templates.hpp"

#include "celeg/text/chat_template.hpp"
#include <stdexcept>

namespace celeg {
namespace {

constexpr std::string_view kDefaultSystem =
    "你是南北阁，一款由BOSS直聘自主研发并训练的专业大语言模型。";

const char* role_name(ChatRole role) {
    switch (role) {
    case ChatRole::System: return "system";
    case ChatRole::Developer: return "system";
    case ChatRole::User: return "user";
    case ChatRole::Assistant: return "assistant";
    case ChatRole::Tool: return "user";
    }
    throw std::invalid_argument("unknown reasoning chat role");
}

} // namespace

std::string ReasoningXmlChatTemplate::format(
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool add_generation_prompt,
    const ChatTemplateOptions& options) const {
    const bool thinking = options.enable_thinking.value_or(true);
    std::string out;
    bool wrote_system = false;
    if (!messages.empty() && messages.front().role == ChatRole::System) {
        out += "<|im_start|>system\n" + messages.front().content.value_or("");
        if (!tools.empty()) {
            out += "\n" + make_parameter_xml_tool_codec()->render_tool_definitions(tools, {});
        }
        out += "<|im_end|>\n";
        wrote_system = true;
    } else if (!tools.empty()) {
        out += "<|im_start|>system\n" + std::string(kDefaultSystem) + "\n";
        out += make_parameter_xml_tool_codec()->render_tool_definitions(tools, {});
        out += "<|im_end|>\n";
        wrote_system = true;
    } else if (messages.empty() || messages.front().role != ChatRole::System) {
        out += "<|im_start|>system\n" + std::string(kDefaultSystem) + "<|im_end|>\n";
        wrote_system = true;
    }
    const std::size_t begin = wrote_system && !messages.empty() &&
        messages.front().role == ChatRole::System ? 1 : 0;
    const auto codec = make_parameter_xml_tool_codec();
    for (std::size_t index = begin; index < messages.size(); ++index) {
        const ChatMessage& message = messages[index];
        if (message.role == ChatRole::Tool) {
            out += "<|im_start|>user\n" + codec->render_tool_result(message) +
                "<|im_end|>\n";
            continue;
        }
        out += "<|im_start|>" + std::string(role_name(message.role)) + "\n" +
            message.content.value_or("");
        if (message.role == ChatRole::Assistant && !message.tool_calls.empty()) {
            out += codec->render_assistant_tool_calls(message.tool_calls);
        }
        out += "<|im_end|>\n";
    }
    if (add_generation_prompt) {
        out += "<|im_start|>assistant\n<think>\n";
        if (!thinking) out += "\n</think>\n\n";
    }
    return out;
}

void add_reasoning_xml_chat_template(ChatTemplateCatalog& catalog) {
    ChatCapabilities capabilities;
    capabilities.assistant_tool_calls = true;
    capabilities.parallel_tool_calls = true;
    capabilities.native_tool_call_codec = true;
    capabilities.roles.developer = true;
    capabilities.roles.tool = true;
    catalog.add("chat:reasoning-xml", std::make_unique<ReasoningXmlChatTemplate>(),
                make_parameter_xml_tool_codec(), capabilities);
}

} // namespace celeg
