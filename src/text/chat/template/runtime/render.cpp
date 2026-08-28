#include "renderer.hpp"

namespace celeg::chat_template_detail {

Renderer::Renderer(const std::vector<TemplateNode>& nodes)
        : nodes_(nodes) {}

std::string Renderer::render(
        std::span<const ChatMessage> messages,
        std::span<const ChatToolDefinition> tools,
        bool generation_prompt,
        const ChatTemplateOptions& options,
        std::string_view origin,
        std::string_view bos_token) const {
        RenderState state;
        state.origin = origin;
        state.generation_prompt = generation_prompt;

        ValueObject root;
        root.emplace("bos_token", std::string(bos_token));
        root.emplace("eos_token", "<|im_end|>");
        root.emplace("messages", message_values(messages));
        root.emplace("tools", tool_values(tools));
        root.emplace(
            "add_generation_prompt",
            TemplateValue{generation_prompt});
        if (options.enable_thinking.has_value()) {
            root.emplace(
                "enable_thinking",
                TemplateValue{*options.enable_thinking});
        }
        root.emplace("keep_past_thinking", TemplateValue{false});
        root.emplace(
            "tool_choice",
            tool_choice_value(options.tool_choice));
        state.scopes.push_back(std::move(root));

        std::string output;
        render_nodes(nodes_, state, output, generation_prompt);
        return output;
    }

void Renderer::render_nodes(
        const std::vector<TemplateNode>& nodes,
        RenderState& state,
        std::string& output,
        bool generation_prompt) const {
        for (const TemplateNode& node : nodes) {
            state.current_line = node.line;
            switch (node.kind) {
            case TemplateNode::Kind::Text:
                output += node.text;
                break;

            case TemplateNode::Kind::Output:
                output += stringify(eval(node.expression, state));
                break;

            case TemplateNode::Kind::Set:
                if (node.body.empty()) {
                    assign(
                        node.text,
                        eval(node.expression, state),
                        state);
                } else {
                    std::string captured;
                    render_nodes(node.body, state, captured, generation_prompt);
                    assign(
                        node.text,
                        TemplateValue{std::move(captured)},
                        state);
                }
                break;

            case TemplateNode::Kind::Macro:
                state.macros[node.text] = {&node};
                break;

            case TemplateNode::Kind::If: {
                bool rendered = false;
                for (const auto& [condition, body] : node.branches) {
                    if (truthy(eval(condition, state))) {
                        render_nodes(body, state, output, generation_prompt);
                        rendered = true;
                        break;
                    }
                }
                if (!rendered) {
                    render_nodes(node.otherwise, state, output, generation_prompt);
                }
                break;
            }

            case TemplateNode::Kind::For:
                render_loop(node, state, output, generation_prompt);
                break;

            case TemplateNode::Kind::Generation:
                if (generation_prompt) {
                    render_nodes(node.body, state, output, generation_prompt);
                }
                break;
            }
        }
    }

void Renderer::render_loop(
        const TemplateNode& node,
        RenderState& state,
        std::string& output,
        bool generation_prompt) const {
        const TemplateValue sequence = eval(node.expression, state);
        const auto* values = std::get_if<ValueList>(&sequence.value);
        if (!values) {
            throw_error(state, "Jinja for requires a list");
        }

        bool rendered_any = false;
        for (std::size_t index = 0;
             index < values->size();
             ++index) {
            ValueObject scope;
            const std::size_t comma = node.text.find(',');
            if (comma == std::string::npos) {
                scope.emplace(node.text, (*values)[index]);
            } else {
                const std::string first =
                    trim(node.text.substr(0, comma));
                const std::string second =
                    trim(node.text.substr(comma + 1));
                const auto* pair =
                    std::get_if<ValueList>(&(*values)[index].value);
                if (!pair || pair->size() != 2) {
                    throw_error(
                        state,
                        "Jinja destructuring loop requires pairs");
                }
                scope.emplace(first, (*pair)[0]);
                scope.emplace(second, (*pair)[1]);
            }

            ValueObject loop;
            loop.emplace(
                "index0",
                static_cast<std::int64_t>(index));
            loop.emplace(
                "index",
                static_cast<std::int64_t>(index + 1));
            loop.emplace("first", index == 0);
            loop.emplace(
                "last",
                index + 1 == values->size());
            loop.emplace(
                "length",
                static_cast<std::int64_t>(values->size()));
            if (index != 0) {
                loop.emplace("previtem", (*values)[index - 1]);
            }
            if (index + 1 < values->size()) {
                loop.emplace("nextitem", (*values)[index + 1]);
            }
            scope.emplace("loop", TemplateValue{std::move(loop)});

            state.scopes.push_back(std::move(scope));
            const bool selected =
                !node.condition ||
                truthy(eval(*node.condition, state));
            if (selected) {
                render_nodes(node.body, state, output, generation_prompt);
            }
            state.scopes.pop_back();
            rendered_any = rendered_any || selected;
        }

        if (!rendered_any) {
            render_nodes(node.otherwise, state, output, generation_prompt);
        }
    }

void Renderer::assign(
        const std::string& target,
        TemplateValue value,
        RenderState& state) const {
        const std::size_t dot = target.find('.');
        if (dot == std::string::npos) {
            state.scopes.back()[target] = std::move(value);
            return;
        }

        const std::string root = target.substr(0, dot);
        TemplateValue* current = nullptr;
        for (auto scope = state.scopes.rbegin();
             scope != state.scopes.rend() && !current;
             ++scope) {
            const auto found = scope->find(root);
            if (found != scope->end()) {
                current = &found->second;
            }
        }
        if (!current) {
            throw_error(
                state,
                "undefined Jinja assignment target '" + root + "'");
        }

        std::size_t begin = dot + 1;
        for (;;) {
            const std::size_t next = target.find('.', begin);
            const std::string member = target.substr(
                begin,
                next == std::string::npos
                    ? std::string::npos
                    : next - begin);
            auto* object =
                std::get_if<ValueObject>(&current->value);
            if (!object) {
                throw_error(
                    state,
                    "Jinja assignment target is not an object");
            }
            if (next == std::string::npos) {
                (*object)[member] = std::move(value);
                return;
            }
            current = &(*object)[member];
            begin = next + 1;
        }
    }

std::int64_t Renderer::require_integer(
        const TemplateValue& value,
        const RenderState& state,
        std::string_view message) {
        const auto* integer =
            std::get_if<std::int64_t>(&value.value);
        if (!integer) {
            throw_error(state, std::string(message));
        }
        return *integer;
    }

void Renderer::replace_all(
        std::string& output,
        const std::string& from,
        const std::string& to) {
        std::size_t offset = 0;
        while (!from.empty() &&
               (offset = output.find(from, offset)) !=
                   std::string::npos) {
            output.replace(offset, from.size(), to);
            offset += to.size();
        }
    }

[[noreturn]] void Renderer::throw_error(
        const RenderState& state,
        const std::string& message) {
        throw std::invalid_argument(
            "UnsupportedChatTemplateConstruct at " + state.origin + ":" +
            std::to_string(state.current_line) + ": " + message);
    }


}
