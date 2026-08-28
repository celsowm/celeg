#pragma once

#include "../program.hpp"

namespace celeg::chat_template_detail {

struct MacroDefinition {
    const TemplateNode* node = nullptr;
};

struct RenderState {
    std::vector<ValueObject> scopes;
    std::map<std::string, MacroDefinition, std::less<>> macros;
    std::string origin;
    int current_line = 1;
    bool generation_prompt = false;
};

std::string role_name(ChatRole role);
TemplateValue message_values(std::span<const ChatMessage> messages);
TemplateValue tool_values(std::span<const ChatToolDefinition> tools);
TemplateValue tool_choice_value(const ToolChoice& choice);

class Renderer {
public:
    explicit Renderer(const std::vector<TemplateNode>& nodes);

    std::string render(
        std::span<const ChatMessage> messages,
        std::span<const ChatToolDefinition> tools,
        bool generation_prompt,
        const ChatTemplateOptions& options,
        std::string_view origin,
        std::string_view bos_token) const;

private:
    TemplateValue eval(
        const Expression& expression,
        RenderState& state) const;
    TemplateValue lookup(
        const std::string& name,
        const RenderState& state) const;
    TemplateValue index(
        const Expression& expression,
        RenderState& state) const;
    TemplateValue slice(
        const Expression& expression,
        RenderState& state) const;
    TemplateValue binary(
        const Expression& expression,
        RenderState& state) const;
    static TemplateValue test(
        std::string_view name,
        const TemplateValue& value);
    TemplateValue filter(
        const Expression& expression,
        RenderState& state) const;
    TemplateValue call(
        const Expression& expression,
        RenderState& state) const;
    void render_nodes(
        const std::vector<TemplateNode>& nodes,
        RenderState& state,
        std::string& output,
        bool generation_prompt) const;
    void render_loop(
        const TemplateNode& node,
        RenderState& state,
        std::string& output,
        bool generation_prompt) const;
    void assign(
        const std::string& target,
        TemplateValue value,
        RenderState& state) const;
    static std::int64_t require_integer(
        const TemplateValue& value,
        const RenderState& state,
        std::string_view message);
    static void replace_all(
        std::string& output,
        const std::string& from,
        const std::string& to);
    [[noreturn]] static void throw_error(
        const RenderState& state,
        const std::string& message);

    const std::vector<TemplateNode>& nodes_;
};

}
