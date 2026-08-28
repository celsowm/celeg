#include "renderer.hpp"

namespace celeg::chat_template_detail {

std::string role_name(ChatRole role) {
    switch (role) {
    case ChatRole::System:
        return "system";
    case ChatRole::Developer:
        return "developer";
    case ChatRole::User:
        return "user";
    case ChatRole::Assistant:
        return "assistant";
    case ChatRole::Tool:
        return "tool";
    }
    throw std::invalid_argument("unknown chat role");
}

TemplateValue message_values(std::span<const ChatMessage> messages) {
    ValueList rendered;
    rendered.reserve(messages.size());
    for (const ChatMessage& message : messages) {
        ValueObject value;
        value.emplace("role", role_name(message.role));
        value.emplace("content", message.content.value_or(""));
        value.emplace("tool_call_id", message.tool_call_id.value_or(""));
        value.emplace("name", message.name.value_or(""));
        value.emplace(
            "reasoning_content",
            message.reasoning_content.value_or(""));
        value.emplace("thinking", message.reasoning_content.value_or(""));

        ValueList calls;
        for (const ToolCall& call : message.tool_calls) {
            ValueObject function;
            function.emplace("name", call.name);
            function.emplace("arguments", call.arguments);

            ValueObject call_value;
            call_value.emplace("id", call.id);
            call_value.emplace("name", call.name);
            call_value.emplace("arguments", call.arguments);
            call_value.emplace("type", "function");
            call_value.emplace(
                "function",
                TemplateValue{std::move(function)});
            try {
                call_value["arguments"] =
                    template_value_from_json(Json::parse(call.arguments));
                auto& callable =
                    std::get<ValueObject>(call_value["function"].value);
                callable["arguments"] = call_value["arguments"];
            } catch (const std::exception&) {
            }
            calls.emplace_back(std::move(call_value));
        }
        value.emplace("tool_calls", TemplateValue{std::move(calls)});
        rendered.emplace_back(std::move(value));
    }
    return TemplateValue{std::move(rendered)};
}

TemplateValue tool_values(std::span<const ChatToolDefinition> tools) {
    ValueList rendered;
    rendered.reserve(tools.size());
    for (const ChatToolDefinition& tool : tools) {
        ValueObject function;
        function.emplace("name", tool.function.name);
        function.emplace("description", tool.function.description);
        function.emplace(
            "parameters",
            RawJson{
                tool.function.parameters.serialized.empty()
                    ? "{}"
                    : tool.function.parameters.serialized});
        function.emplace("strict", tool.function.strict);

        ValueObject value;
        value.emplace("type", tool.type);
        value.emplace(
            "function",
            TemplateValue{std::move(function)});
        rendered.emplace_back(std::move(value));
    }
    return TemplateValue{std::move(rendered)};
}

TemplateValue tool_choice_value(const ToolChoice& choice) {
    ValueObject value;
    switch (choice.mode) {
    case ToolChoiceMode::None:
        value.emplace("mode", "none");
        break;
    case ToolChoiceMode::Auto:
        value.emplace("mode", "auto");
        break;
    case ToolChoiceMode::Required:
        value.emplace("mode", "required");
        break;
    case ToolChoiceMode::Specific:
        value.emplace("mode", "specific");
        break;
    }
    value.emplace("function_name", choice.function_name);
    return TemplateValue{std::move(value)};
}

}
