#include "renderer.hpp"

namespace celeg::chat_template_detail {

TemplateValue Renderer::call(
        const Expression& expression,
        RenderState& state) const {
        if (expression.text == "__list") {
            ValueList values;
            for (const Expression& child : expression.children) {
                values.push_back(eval(child, state));
            }
            return TemplateValue{std::move(values)};
        }
        if (expression.children.empty()) {
            throw_error(state, "unsupported Jinja call");
        }

        const Expression& target = expression.children.front();
        if (target.kind == Expression::Kind::Name) {
            if (target.text == "namespace") {
                ValueObject members;
                for (std::size_t index = 1;
                     index < expression.children.size();
                     ++index) {
                    const Expression& argument =
                        expression.children[index];
                    if (argument.kind != Expression::Kind::Binary ||
                        !argument.text.starts_with("__keyword:") ||
                        argument.children.size() != 1) {
                        throw_error(
                            state,
                            "namespace() requires named arguments");
                    }
                    members.emplace(
                        argument.text.substr(
                            std::string_view("__keyword:").size()),
                        eval(argument.children.front(), state));
                }
                return TemplateValue{std::move(members)};
            }

            if (target.text == "range") {
                if (expression.children.size() != 2 &&
                    expression.children.size() != 3) {
                    throw_error(
                        state,
                        "range() requires one or two arguments");
                }
                const std::int64_t start =
                    expression.children.size() == 2
                        ? 0
                        : require_integer(
                              eval(expression.children.at(1), state),
                              state,
                              "range() requires integer arguments");
                const std::int64_t end = require_integer(
                    eval(expression.children.back(), state),
                    state,
                    "range() requires integer arguments");
                ValueList values;
                for (std::int64_t value = start;
                     value < end;
                     ++value) {
                    values.emplace_back(value);
                }
                return TemplateValue{std::move(values)};
            }

            if (target.text == "raise_exception") {
                throw_error(
                    state,
                    stringify(eval(
                        expression.children.at(1),
                        state)));
            }

            if (target.text == "strftime_now") {
                if (expression.children.size() != 2) {
                    throw_error(
                        state,
                        "strftime_now() requires a format string");
                }
                const std::string format = stringify(
                    eval(expression.children.at(1), state));
                const std::time_t now =
                    std::chrono::system_clock::to_time_t(
                        std::chrono::system_clock::now());
                std::tm components{};
#ifdef _WIN32
                localtime_s(&components, &now);
#else
                localtime_r(&now, &components);
#endif
                std::string formatted(format.size() + 64, '\0');
                const std::size_t written = std::strftime(
                    formatted.data(),
                    formatted.size(),
                    format.c_str(),
                    &components);
                if (written == 0) {
                    throw_error(state, "invalid strftime format");
                }
                formatted.resize(written);
                return TemplateValue{std::move(formatted)};
            }

            const auto macro = state.macros.find(target.text);
            if (macro != state.macros.end()) {
                const auto& parameters = macro->second.node->parameters;
                if (expression.children.size() - 1 > parameters.size()) {
                    throw_error(
                        state,
                        "wrong argument count for Jinja macro '" +
                            target.text + "'");
                }

                ValueObject arguments;
                std::size_t positional = 0;
                for (std::size_t index = 1;
                     index < expression.children.size();
                     ++index) {
                    const Expression& argument = expression.children[index];
                    if (argument.kind == Expression::Kind::Binary &&
                        argument.text.starts_with("__keyword:")) {
                        const std::string name = argument.text.substr(
                            std::string_view("__keyword:").size());
                        const auto parameter = std::find_if(
                            parameters.begin(), parameters.end(),
                            [&](const MacroParameter& candidate) {
                                return candidate.name == name;
                            });
                        if (parameter == parameters.end() || arguments.contains(name)) {
                            throw_error(state, "invalid Jinja macro keyword");
                        }
                        arguments.emplace(name, eval(argument.children.front(), state));
                    } else {
                        while (positional < parameters.size() &&
                               arguments.contains(parameters[positional].name)) {
                            ++positional;
                        }
                        if (positional >= parameters.size()) {
                            throw_error(state, "too many positional Jinja macro arguments");
                        }
                        arguments.emplace(
                            parameters[positional].name,
                            eval(argument, state));
                        ++positional;
                    }
                }
                for (const MacroParameter& parameter : parameters) {
                    if (!arguments.contains(parameter.name)) {
                        arguments.emplace(
                            parameter.name,
                            parameter.default_value
                                ? eval(*parameter.default_value, state)
                                : TemplateValue{});
                    }
                }
                state.scopes.push_back(std::move(arguments));
                std::string output;
                render_nodes(
                    macro->second.node->body,
                    state,
                    output,
                    state.generation_prompt);
                state.scopes.pop_back();
                return TemplateValue{std::move(output)};
            }
        }

        if (target.kind == Expression::Kind::Access) {
            const TemplateValue object =
                eval(target.children.at(0), state);
            const std::string member = target.text;
            const std::string text = stringify(object);

            if (member == "strip") {
                return TemplateValue{trim(text)};
            }
            if (member == "lstrip" || member == "rstrip") {
                std::string output = text;
                if (member == "lstrip") {
                    while (!output.empty() &&
                           std::isspace(static_cast<unsigned char>(
                               output.front()))) {
                        output.erase(output.begin());
                    }
                } else {
                    while (!output.empty() &&
                           std::isspace(static_cast<unsigned char>(
                               output.back()))) {
                        output.pop_back();
                    }
                }
                return TemplateValue{std::move(output)};
            }
            if (member == "lower" || member == "upper") {
                std::string output = text;
                std::transform(
                    output.begin(),
                    output.end(),
                    output.begin(),
                    [&](unsigned char character) {
                        return static_cast<char>(
                            member == "lower"
                                ? std::tolower(character)
                                : std::toupper(character));
                    });
                return TemplateValue{std::move(output)};
            }
            if (member == "items") {
                const auto* object_values =
                    std::get_if<ValueObject>(&object.value);
                if (!object_values) {
                    throw_error(
                        state,
                        "items() requires an object");
                }
                ValueList pairs;
                for (const auto& [key, value] : *object_values) {
                    pairs.emplace_back(
                        ValueList{TemplateValue{key}, value});
                }
                return TemplateValue{std::move(pairs)};
            }
            if (member == "split") {
                const std::string separator =
                    expression.children.size() > 1
                        ? stringify(eval(
                              expression.children.at(1),
                              state))
                        : " ";
                ValueList pieces;
                std::size_t offset = 0;
                while (offset <= text.size()) {
                    const std::size_t found =
                        text.find(separator, offset);
                    pieces.emplace_back(text.substr(
                        offset,
                        found == std::string::npos
                            ? std::string::npos
                            : found - offset));
                    if (found == std::string::npos ||
                        separator.empty()) {
                        break;
                    }
                    offset = found + separator.size();
                }
                return TemplateValue{std::move(pieces)};
            }
            if (member == "replace") {
                if (expression.children.size() != 3) {
                    throw_error(
                        state,
                        "replace() requires old and new strings");
                }
                std::string output = text;
                const std::string from = stringify(
                    eval(expression.children.at(1), state));
                const std::string to = stringify(
                    eval(expression.children.at(2), state));
                replace_all(output, from, to);
                return TemplateValue{std::move(output)};
            }
            if (member == "startswith" ||
                member == "endswith") {
                const std::string needle = stringify(
                    eval(expression.children.at(1), state));
                return TemplateValue{
                    member == "startswith"
                        ? text.starts_with(needle)
                        : text.ends_with(needle)};
            }

            if (member == "get") {
                const TemplateValue key = eval(
                    expression.children.at(1), state);
                const TemplateValue fallback =
                    expression.children.size() > 2
                        ? eval(expression.children.at(2), state)
                        : TemplateValue{};
                if (const auto* obj =
                        std::get_if<ValueObject>(&object.value)) {
                    const auto found = obj->find(stringify(key));
                    if (found != obj->end()) {
                        return found->second;
                    }
                }
                return fallback;
            }
        }

        throw_error(state, "unsupported Jinja call");
    }


}
