#include "celeg/text/semantic_chat_templates.hpp"

#include "celeg/text/chat_template.hpp"
#include "celeg/text/detail/chat_template_support.hpp"

#include <cctype>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace celeg {
namespace {

std::string today() {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    std::ostringstream out;
    out << std::put_time(&local, "%d %B %Y");
    return out.str();
}

std::string remove_flag(std::string value, std::string_view flag) {
    for (std::size_t pos = 0; (pos = value.find(flag, pos)) != std::string::npos;) value.erase(pos, flag.size());
    return value;
}

std::string tools_block(std::span<const ChatToolDefinition> tools) {
    if (tools.empty()) return {};
    std::string out = "### Tools\n\nYou may call one or more functions to assist with the user query.\n"
        "You are provided with function signatures within <tools> XML tags:\n<tools>\n";
    for (const ChatToolDefinition& tool : tools) {
        out += "{\"name\":" + chat_template_detail::json_string(tool.function.name);
        out += ",\"description\":" + chat_template_detail::json_string(tool.function.description);
        out += ",\"parameters\":" +
            (tool.function.parameters.serialized.empty() ? "{}" : tool.function.parameters.serialized) + "}\n";
    }
    return out + "</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call> XML tags.\n";
}

} // namespace

std::string MetadataThinkingChatTemplate::format(
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool add_generation_prompt,
    const ChatTemplateOptions& options) const {
    return format_with_thinking(messages, tools, add_generation_prompt,
        options.enable_thinking.value_or(enable_thinking_));
}

std::string MetadataThinkingChatTemplate::format_with_thinking(
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool add_generation_prompt,
    bool default_thinking) const {
    bool thinking = default_thinking;
    std::string system_message;
    if (!messages.empty() && messages.front().role == ChatRole::System) {
        system_message = messages.front().content.value_or("");
        if (system_message.find("/no_think") != std::string::npos) thinking = false;
        else if (system_message.find("/think") != std::string::npos) thinking = true;
    }
    const std::string reasoning_mode = thinking ? "/think" : "/no_think";
    std::string custom_instructions = remove_flag(remove_flag(system_message, "/no_think"), "/think");
    const bool system_override = custom_instructions.find("/system_override") != std::string::npos;
    custom_instructions = remove_flag(custom_instructions, "/system_override");
    while (!custom_instructions.empty() && std::isspace(static_cast<unsigned char>(custom_instructions.back()))) custom_instructions.pop_back();

    std::string out = "<|im_start|>system\n";
    if (system_override) {
        out += custom_instructions;
    } else {
        out += "## Metadata\n\nKnowledge Cutoff Date: June 2025\nToday Date: " + today() +
            "\nReasoning Mode: " + reasoning_mode + "\n\n## Custom Instructions\n\n";
        if (!custom_instructions.empty()) out += custom_instructions + "\n\n";
        else if (thinking) out += "You are a helpful AI assistant named SmolLM, trained by Hugging Face. Think carefully before providing a precise and accurate solution.\n\n";
        else out += "You are a helpful AI assistant named SmolLM, trained by Hugging Face.\n\n";
        out += tools_block(tools);
    }
    out += "<|im_end|>\n";
    const std::size_t first_message = (!messages.empty() && messages.front().role == ChatRole::System) ? 1 : 0;
    for (std::size_t index = first_message; index < messages.size(); ++index) {
        const ChatMessage& message = messages[index];
        const std::string content = message.content.value_or("");
        if (message.role == ChatRole::Tool || message.role == ChatRole::User) {
            out += "<|im_start|>user\n" + content + "<|im_end|>\n";
        } else if (message.role == ChatRole::Assistant) {
            out += "<|im_start|>assistant\n";
            if (!thinking) out += "<think>\n\n</think>\n";
            out += content;
            for (const ToolCall& call : message.tool_calls) {
                out += "<tool_call>{\"name\":" + chat_template_detail::json_string(call.name) +
                    ",\"arguments\":" + (call.arguments.empty() ? "{}" : call.arguments) + "}</tool_call>";
            }
            out += "<|im_end|>\n";
        }
    }
    if (add_generation_prompt) {
        out += "<|im_start|>assistant\n";
        if (!thinking) out += "<think>\n\n</think>\n";
    }
    return out;
}

void add_metadata_thinking_chat_template(ChatTemplateCatalog& catalog) {
    ChatCapabilities capabilities;
    capabilities.assistant_tool_calls = true;
    capabilities.parallel_tool_calls = true;
    capabilities.native_tool_call_codec = true;
    capabilities.roles.developer = true;
    capabilities.roles.tool = true;
    catalog.add("chat:metadata-thinking", std::make_unique<MetadataThinkingChatTemplate>(),
                make_tagged_json_tool_codec(), capabilities);
}

} // namespace celeg
