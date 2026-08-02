#include "celeg/text/chat_template.hpp"

#include <stdexcept>

namespace celeg {

std::string render_chat(std::span<const ChatMessage> messages,
                        const IChatTemplate& chat_template,
                        bool add_generation_prompt) {
    return chat_template.format(messages, add_generation_prompt);
}

std::string render_chat(std::span<const ChatMessage> messages,
                        std::span<const ChatToolDefinition> tools,
                        const IChatTemplate& chat_template,
                        bool add_generation_prompt) {
    return chat_template.format(messages, tools, add_generation_prompt);
}

namespace {

const char* role_tag(ChatRole role) {
    switch (role) {
        case ChatRole::System: return "system";
        case ChatRole::Developer: return "system";
        case ChatRole::User: return "user";
        case ChatRole::Assistant: return "assistant";
        case ChatRole::Tool: return "tool";
    }
    throw std::invalid_argument("unknown chat role");
}

const char* granite_role_tag(ChatRole role) {
    switch (role) {
        case ChatRole::System: return "system";
        case ChatRole::Developer: return "developer";
        case ChatRole::User: return "user";
        case ChatRole::Assistant: return "assistant";
        case ChatRole::Tool:
            throw std::invalid_argument(
                "GraniteInstructChatTemplate does not support tool messages");
    }
    throw std::invalid_argument("unknown chat role");
}

void append_granite_message(std::string& out, ChatRole role, std::string_view content) {
    out += "<|start_of_role|>";
    out += granite_role_tag(role);
    out += "<|end_of_role|>";
    out += content;
    out += "<|end_of_text|>\n";
}

std::string json_string(std::string_view value) {
    std::string out = "\"";
    for (const char ch : value) {
        if (ch == '\\' || ch == '"') out.push_back('\\');
        if (ch == '\n') { out += "\\n"; continue; }
        if (ch == '\r') { out += "\\r"; continue; }
        if (ch == '\t') { out += "\\t"; continue; }
        out.push_back(ch);
    }
    out.push_back('"');
    return out;
}

std::string granite_tools_block(std::span<const ChatToolDefinition> tools) {
    if (tools.empty()) return {};
    std::string out =
        "You are a helpful assistant with access to the following tools. You may call one or more tools to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
    for (const ChatToolDefinition& tool : tools) {
        out += "\n{\"type\":\"function\",\"function\":{\"name\":";
        out += json_string(tool.function.name);
        out += ",\"description\":";
        out += json_string(tool.function.description);
        out += ",\"parameters\":";
        out += tool.function.parameters.serialized.empty() ? "{}" : tool.function.parameters.serialized;
        out += "}}";
    }
    out += "\n</tools>\n\nFor each tool call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>.";
    return out;
}

} // namespace

std::string Lfm2InstructChatTemplate::format(std::span<const ChatMessage> messages,
                                             std::span<const ChatToolDefinition> tools,
                                             bool add_generation_prompt) const {
    std::string out = "<|startoftext|>";
    std::string system_tools;
    if (!tools.empty()) {
        system_tools = "List of tools: <|tool_list_start|>[";
        for (std::size_t i = 0; i < tools.size(); ++i) {
            if (i != 0) system_tools += ", ";
            system_tools += "{\"type\":\"function\",\"function\":{\"name\":";
            system_tools += json_string(tools[i].function.name);
            system_tools += ",\"description\":" + json_string(tools[i].function.description);
            system_tools += ",\"parameters\":" + (tools[i].function.parameters.serialized.empty() ? "{}" : tools[i].function.parameters.serialized);
            system_tools += "}}";
        }
        system_tools += "]<|tool_list_end|>";
    }
    for (const ChatMessage& message : messages) {
        if (message.role == ChatRole::System && !system_tools.empty()) {
            const std::string content = message.content.value_or("");
            system_tools = content + (content.empty() ? "" : "\n") + system_tools;
            continue;
        }
        out += "<|im_start|>";
        out += role_tag(message.role);
        out += "\n";
        const std::string content = message.content.value_or("");
        out += content;
        if (message.role == ChatRole::Tool) out = out.substr(0, out.size() - content.size()) + "<|tool_response_start|>" + content + "<|tool_response_end|>";
        for (const auto& call : message.tool_calls) {
            out += "<|tool_call_start|>[" + call.name + "(" + call.arguments + ")]<|tool_call_end|>";
        }
        out += "<|im_end|>\n";
    }
    if (!system_tools.empty()) out = "<|startoftext|><|im_start|>system\n" + system_tools + "<|im_end|>\n" + out.substr(std::string("<|startoftext|>").size());
    if (add_generation_prompt) out += "<|im_start|>assistant\n";
    return out;
}

std::string GraniteInstructChatTemplate::format(std::span<const ChatMessage> messages,
                                                std::span<const ChatToolDefinition> tools,
                                                bool add_generation_prompt) const {
    // Granite's Hub chat template omits BOS because Transformers adds the
    // configured BOS token when tokenizing the rendered template. Keep that
    // tokenizer-side behavior explicit in the native text representation.
    std::string out = "<|startoftext|>";
    const bool has_system_message = !messages.empty() &&
        messages.front().role == ChatRole::System;
    const std::string tool_block = granite_tools_block(tools);
    if (!has_system_message) {
        append_granite_message(
            out, ChatRole::System,
            "Knowledge Cutoff Date: April 2024. You are Granite, developed by IBM. "
            "You are a helpful AI assistant." + tool_block);
    }
    for (size_t index = 0; index < messages.size(); ++index) {
        const ChatMessage& message = messages[index];
        std::string content = message.content.value_or("");
        if (index == 0 && message.role == ChatRole::System && !tool_block.empty()) {
            content += "\n\n";
            content += tool_block;
        }
        if (message.role == ChatRole::Tool) {
            out += "<|start_of_role|>user<|end_of_role|>\n<tool_response>\n";
            out += content;
            out += "\n</tool_response><|end_of_text|>\n";
            continue;
        }
        out += "<|start_of_role|>";
        out += granite_role_tag(message.role);
        out += "<|end_of_role|>";
        out += content;
        if (message.role == ChatRole::Assistant) {
            for (const ToolCall& call : message.tool_calls) {
                if (!content.empty() || &call != &message.tool_calls.front()) out += "\n";
                out += "<tool_call>\n{\"name\": ";
                out += json_string(call.name);
                out += ", \"arguments\": ";
                out += call.arguments;
                out += "}\n</tool_call>";
            }
        }
        out += "<|end_of_text|>\n";
    }
    if (add_generation_prompt) {
        out += "<|start_of_role|>assistant<|end_of_role|>";
    }
    return out;
}

std::string Gemma4InstructChatTemplate::format(std::span<const ChatMessage> messages,
                                               std::span<const ChatToolDefinition> tools,
                                               bool add_generation_prompt) const {
    std::string out = "<bos>";
    if (!tools.empty()) {
        out += "<|turn>system\n";
        for (const auto& tool : tools) {
            out += "declaration:" + tool.function.name + "{description:<|\"|>" + tool.function.description + "<|\"|>,parameters:";
            out += tool.function.parameters.serialized.empty() ? "{}" : tool.function.parameters.serialized;
            out += "}\n";
        }
        out += "<turn|>\n";
    }
    for (const ChatMessage& message : messages) {
        const char* role = nullptr;
        switch (message.role) {
            case ChatRole::System: role = "system"; break;
            case ChatRole::User: role = "user"; break;
            case ChatRole::Assistant: role = "model"; break;
            case ChatRole::Developer:
                role = "system"; break;
            case ChatRole::Tool:
                role = "tool"; break;
        }
        if (message.role == ChatRole::Tool) {
            out += "<|tool_response>response:" + message.tool_call_id.value_or("unknown") + "{value:" + message.content.value_or("") + "}<tool_response|>\n";
            continue;
        }
        out += "<|turn>";
        out += role;
        out += "\n";
        out += message.content.value_or("");
        for (const auto& call : message.tool_calls) {
            out += "<|tool_call>call:" + call.name + call.arguments + "<tool_call|>";
        }
        out += "<turn|>\n";
    }
    if (add_generation_prompt) out += "<|turn>model\n";
    return out;
}

void ChatProfileCatalog::add(std::string profile_id,
                              std::unique_ptr<IChatTemplate> chat_template,
                              std::unique_ptr<IChatToolCallCodec> tool_call_codec,
                              ChatCapabilities capabilities) {
    if (frozen_) throw std::logic_error("chat profile catalog is frozen");
    if (profile_id.empty() || !chat_template) {
        throw std::invalid_argument("chat profile requires an id and template");
    }
    Entry entry{std::move(chat_template), std::move(tool_call_codec), capabilities};
    entry.capabilities.tool_call_codec = entry.tool_call_codec.get();
    if (!entries_.emplace(std::move(profile_id), std::move(entry)).second) {
        throw std::invalid_argument("duplicate chat profile");
    }
}

void ChatProfileCatalog::freeze() {
    frozen_ = true;
}

const IChatTemplate& ChatProfileCatalog::find(std::string_view profile_id) const {
    const auto it = entries_.find(std::string(profile_id));
    if (it == entries_.end()) throw std::invalid_argument(
        "unknown chat profile: " + std::string(profile_id));
    return *it->second.chat_template;
}

ChatCapabilities ChatProfileCatalog::capabilities(std::string_view profile_id) const {
    const auto it = entries_.find(std::string(profile_id));
    if (it == entries_.end()) throw std::invalid_argument(
        "unknown chat profile: " + std::string(profile_id));
    return it->second.capabilities;
}

ChatProfileCatalog make_chat_profile_catalog() {
    ChatProfileCatalog catalog;
    catalog.add("lfm2-instruct", std::make_unique<Lfm2InstructChatTemplate>(),
                make_lfm2_tool_call_codec(),
                ChatCapabilities{true, true, true, true, true});
    catalog.add("granite-instruct", std::make_unique<GraniteInstructChatTemplate>(),
                nullptr,
                ChatCapabilities{false, false, false, false, false});
    catalog.add("gemma4-instruct", std::make_unique<Gemma4InstructChatTemplate>(),
                make_gemma4_tool_call_codec(),
                ChatCapabilities{true, true, true, true, true});
    catalog.freeze();
    return catalog;
}

} // namespace celeg
