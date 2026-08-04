#include "celeg/text/chat_template.hpp"

#include <cctype>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <utility>
#include <vector>
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

std::string render_chat(std::span<const ChatMessage> messages,
                        std::span<const ChatToolDefinition> tools,
                        const IChatTemplate& chat_template,
                        bool add_generation_prompt,
                        const ChatTemplateOptions& options) {
    return chat_template.format(messages, tools, add_generation_prompt, options);
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

std::string minicpm5_tools_block(std::span<const ChatToolDefinition> tools) {
    if (tools.empty()) return {};
    std::string out =
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n<tools>\n";
    for (const ChatToolDefinition& tool : tools) {
        out += "{\"type\":\"function\",\"function\":{\"name\":";
        out += json_string(tool.function.name);
        out += ",\"description\":" + json_string(tool.function.description);
        out += ",\"parameters\":";
        out += tool.function.parameters.serialized.empty() ? "{}" : tool.function.parameters.serialized;
        out += "}}\n";
    }
    out += "</tools>\n\nFor each function call, return the function name and arguments inside <function></function> XML tags.\n";
    return out;
}

const char* minicpm5_role_tag(ChatRole role) {
    switch (role) {
        case ChatRole::System:
        case ChatRole::Developer: return "system";
        case ChatRole::User:
        case ChatRole::Tool: return "user";
        case ChatRole::Assistant: return "assistant";
    }
    throw std::invalid_argument("unknown chat role");
}

std::vector<std::pair<std::string, std::string>> minicpm5_object_members(std::string_view object) {
    std::vector<std::pair<std::string, std::string>> members;
    if (object.size() < 2 || object.front() != '{' || object.back() != '}') return members;
    std::size_t cursor = 1;
    while (cursor + 1 < object.size()) {
        while (cursor + 1 < object.size() &&
               (std::isspace(static_cast<unsigned char>(object[cursor])) || object[cursor] == ',')) ++cursor;
        if (cursor + 1 >= object.size() || object[cursor] != '"') break;
        const std::size_t key_begin = ++cursor;
        while (cursor < object.size() && object[cursor] != '"') ++cursor;
        if (cursor >= object.size()) break;
        const std::string key(object.substr(key_begin, cursor - key_begin));
        ++cursor;
        while (cursor < object.size() && (std::isspace(static_cast<unsigned char>(object[cursor])) || object[cursor] == ':')) ++cursor;
        const std::size_t value_begin = cursor;
        int depth = 0;
        bool quoted = false;
        bool escaped = false;
        for (; cursor < object.size(); ++cursor) {
            const char ch = object[cursor];
            if (escaped) { escaped = false; continue; }
            if (quoted && ch == '\\') { escaped = true; continue; }
            if (ch == '"') { quoted = !quoted; continue; }
            if (quoted) continue;
            if (ch == '{' || ch == '[') ++depth;
            else if (ch == '}' || ch == ']') {
                if (ch == '}' && depth == 0) break;
                --depth;
            } else if (ch == ',' && depth == 0) break;
        }
        std::string value(object.substr(value_begin, cursor - value_begin));
        while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
        members.emplace_back(key, std::move(value));
        if (cursor < object.size() && object[cursor] == ',') ++cursor;
    }
    return members;
}

std::string smollm3_json_string(std::string_view value) {
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

std::string smollm3_today() {
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

std::string smollm3_remove_flag(std::string value, std::string_view flag) {
    for (std::size_t pos = 0; (pos = value.find(flag, pos)) != std::string::npos;) {
        value.erase(pos, flag.size());
    }
    return value;
}

std::string smollm3_tools_block(std::span<const ChatToolDefinition> tools) {
    if (tools.empty()) return {};
    std::string out = "### Tools\n\nYou may call one or more functions to assist with the user query.\n"
        "You are provided with function signatures within <tools> XML tags:\n<tools>\n";
    for (const ChatToolDefinition& tool : tools) {
        out += "{\"name\":" + smollm3_json_string(tool.function.name);
        out += ",\"description\":" + smollm3_json_string(tool.function.description);
        out += ",\"parameters\":" +
            (tool.function.parameters.serialized.empty() ? "{}" : tool.function.parameters.serialized);
        out += "}\n";
    }
    return out + "</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call> XML tags.\n";
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

std::string MiniCpm5InstructChatTemplate::format(
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool add_generation_prompt) const {
    std::string out = "<bos>";
    const std::string tool_block = minicpm5_tools_block(tools);
    std::size_t first_message = 0;
    if (!tool_block.empty()) {
        out += "<|im_start|>system\n";
        if (!messages.empty() && messages.front().role == ChatRole::System) {
            out += messages.front().content.value_or("");
            out += "\n\n";
            first_message = 1;
        }
        out += tool_block;
        out += "<|im_end|>\n";
    }
    for (std::size_t index = first_message; index < messages.size(); ++index) {
        const ChatMessage& message = messages[index];
        if (message.role == ChatRole::Tool) {
            out += "<|im_start|>user\n<tool_response>\n";
            out += message.content.value_or("");
            out += "\n</tool_response><|im_end|>\n";
            continue;
        }
        out += "<|im_start|>";
        out += minicpm5_role_tag(message.role);
        out += "\n";
        out += message.content.value_or("");
        if (message.role == ChatRole::Assistant) {
            for (const ToolCall& call : message.tool_calls) {
                out += "<function name=\"" + call.name + "\">";
                for (const auto& pair : minicpm5_object_members(call.arguments)) {
                    out += "<param name=\"" + pair.first + "\">" + pair.second + "</param>";
                }
                out += "</function>";
            }
        }
        out += "<|im_end|>\n";
    }
    if (add_generation_prompt) {
        // The native profile selects enable_thinking=false, matching the
        // official template's empty think block before the answer.
        out += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    }
    return out;
}

std::string SmolLm3InstructChatTemplate::format(
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool add_generation_prompt) const {
    return format_with_thinking(messages, tools, add_generation_prompt, enable_thinking_);
}

std::string SmolLm3InstructChatTemplate::format(
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool add_generation_prompt,
    const ChatTemplateOptions& options) const {
    return format_with_thinking(messages, tools, add_generation_prompt,
                                options.enable_thinking.value_or(enable_thinking_));
}

std::string SmolLm3InstructChatTemplate::format_with_thinking(
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
    std::string custom_instructions = smollm3_remove_flag(
        smollm3_remove_flag(system_message, "/no_think"), "/think");
    const bool system_override = custom_instructions.find("/system_override") != std::string::npos;
    custom_instructions = smollm3_remove_flag(custom_instructions, "/system_override");
    while (!custom_instructions.empty() &&
           (custom_instructions.back() == ' ' || custom_instructions.back() == '\n' ||
            custom_instructions.back() == '\r' || custom_instructions.back() == '\t')) {
        custom_instructions.pop_back();
    }

    std::string out = "<|im_start|>system\n";
    if (system_override) {
        out += custom_instructions;
    } else {
        out += "## Metadata\n\nKnowledge Cutoff Date: June 2025\nToday Date: ";
        out += smollm3_today();
        out += "\nReasoning Mode: " + reasoning_mode + "\n\n## Custom Instructions\n\n";
        if (!custom_instructions.empty()) out += custom_instructions + "\n\n";
        else if (thinking) {
            out += "You are a helpful AI assistant named SmolLM, trained by Hugging Face. "
                   "Think carefully before providing a precise and accurate solution.\n\n";
        } else {
            out += "You are a helpful AI assistant named SmolLM, trained by Hugging Face.\n\n";
        }
        out += smollm3_tools_block(tools);
    }
    out += "<|im_end|>\n";

    const std::size_t first_message = (!messages.empty() &&
        messages.front().role == ChatRole::System) ? 1 : 0;
    for (std::size_t index = first_message; index < messages.size(); ++index) {
        const ChatMessage& message = messages[index];
        const std::string content = message.content.value_or("");
        if (message.role == ChatRole::Tool) {
            out += "<|im_start|>user\n" + content + "<|im_end|>\n";
        } else if (message.role == ChatRole::User) {
            out += "<|im_start|>user\n" + content + "<|im_end|>\n";
        } else if (message.role == ChatRole::Assistant) {
            out += "<|im_start|>assistant\n";
            if (!thinking) out += "<think>\n\n</think>\n";
            out += content;
            for (const ToolCall& call : message.tool_calls) {
                out += "<tool_call>{\"name\":" + smollm3_json_string(call.name) +
                    ",\"arguments\":" + (call.arguments.empty() ? "{}" : call.arguments) +
                    "}</tool_call>";
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

void ChatProfileCatalog::add(std::string profile_id,
                              std::unique_ptr<IChatTemplate> chat_template,
                              std::unique_ptr<IChatToolCallCodec> tool_call_codec,
                              ChatCapabilities capabilities) {
    if (frozen_) throw std::logic_error("chat profile catalog is frozen");
    if (profile_id.empty() || !chat_template) {
        throw std::invalid_argument("chat profile requires an id and template");
    }
    if (capabilities.native_tool_call_codec && !tool_call_codec) {
        throw std::invalid_argument(
            "chat profile declares a native tool codec but provides none");
    }
    if (capabilities.parallel_tool_calls &&
        (!tool_call_codec || !tool_call_codec->supports_parallel_calls())) {
        throw std::invalid_argument(
            "chat profile declares parallel tool calls without codec support");
    }
    if (tool_call_codec && !capabilities.native_tool_call_codec) {
        throw std::invalid_argument(
            "chat profile provides a tool codec without enabling its capability");
    }
    std::shared_ptr<const IChatToolCallCodec> shared_codec(std::move(tool_call_codec));
    Entry entry{std::move(chat_template), std::move(shared_codec), capabilities};
    entry.capabilities.tool_call_codec = entry.tool_call_codec;
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
    add_builtin_chat_profiles(catalog);
    catalog.freeze();
    return catalog;
}

void add_builtin_chat_profiles(ChatProfileCatalog& catalog) {
    const auto tool_roles = [] {
        ChatRoleCapabilities roles;
        roles.developer = true;
        roles.tool = true;
        return roles;
    };
    const auto basic_roles = [] {
        return ChatRoleCapabilities{};
    };

    ChatCapabilities lfm2_capabilities;
    lfm2_capabilities.assistant_tool_calls = true;
    lfm2_capabilities.parallel_tool_calls = true;
    lfm2_capabilities.native_tool_call_codec = true;
    lfm2_capabilities.roles = tool_roles();
    catalog.add("lfm2-instruct", std::make_unique<Lfm2InstructChatTemplate>(),
                make_lfm2_tool_call_codec(),
                lfm2_capabilities);

    ChatCapabilities granite_capabilities;
    granite_capabilities.roles = basic_roles();
    catalog.add("granite-instruct", std::make_unique<GraniteInstructChatTemplate>(),
                nullptr,
                granite_capabilities);

    ChatCapabilities gemma_capabilities;
    gemma_capabilities.vision = true;
    gemma_capabilities.assistant_tool_calls = true;
    gemma_capabilities.parallel_tool_calls = true;
    gemma_capabilities.native_tool_call_codec = true;
    gemma_capabilities.roles = tool_roles();
    catalog.add("gemma4-instruct", std::make_unique<Gemma4InstructChatTemplate>(),
                make_gemma4_tool_call_codec(),
                gemma_capabilities);

    ChatCapabilities minicpm5_capabilities;
    minicpm5_capabilities.assistant_tool_calls = true;
    minicpm5_capabilities.parallel_tool_calls = true;
    minicpm5_capabilities.native_tool_call_codec = true;
    minicpm5_capabilities.roles = tool_roles();
    catalog.add("minicpm5-instruct", std::make_unique<MiniCpm5InstructChatTemplate>(),
                make_minicpm5_tool_call_codec(),
                minicpm5_capabilities);

    ChatCapabilities smollm3_capabilities;
    smollm3_capabilities.assistant_tool_calls = true;
    smollm3_capabilities.parallel_tool_calls = true;
    smollm3_capabilities.native_tool_call_codec = true;
    smollm3_capabilities.roles = tool_roles();
    catalog.add("smollm3-instruct", std::make_unique<SmolLm3InstructChatTemplate>(),
                make_smollm3_tool_call_codec(),
                smollm3_capabilities);
}

} // namespace celeg
