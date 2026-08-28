#include "renderer.hpp"

namespace celeg::chat_template_detail {

std::string render_program(
    const std::vector<TemplateNode>& nodes,
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool generation_prompt,
    const ChatTemplateOptions& options,
    std::string_view origin,
    std::string_view bos_token) {
    return Renderer(nodes).render(
        messages,
        tools,
        generation_prompt,
        options,
        origin,
        bos_token);
}

}

namespace celeg {

InteractionRenderProgram::InteractionRenderProgram(
    std::vector<chat_template_detail::TemplateNode> nodes)
    : nodes_(std::move(nodes)) {}

std::string InteractionRenderProgram::render(
    std::span<const ChatMessage> messages,
    std::span<const ChatToolDefinition> tools,
    bool generation_prompt,
    const ChatTemplateOptions& options,
    std::string_view origin,
    std::string_view bos_token) const {
    return chat_template_detail::render_program(
        nodes_,
        messages,
        tools,
        generation_prompt,
        options,
        origin,
        bos_token);
}

}
