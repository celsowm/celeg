#include "renderer.hpp"

namespace celeg::chat_template_detail {

TemplateValue Renderer::filter(
        const Expression& expression,
        RenderState& state) const {
        TemplateValue input =
            eval(expression.children.at(0), state);
        const auto argument = [&](std::size_t index) {
            return eval(expression.children.at(index), state);
        };

        if (expression.text == "tojson") {
            return TemplateValue{to_json(input)};
        }
        if (expression.text == "string" ||
            expression.text == "safe") {
            return TemplateValue{stringify(input)};
        }
        if (expression.text == "trim") {
            return TemplateValue{trim(stringify(input))};
        }
        if (expression.text == "lower" ||
            expression.text == "upper") {
            std::string out = stringify(input);
            std::transform(
                out.begin(),
                out.end(),
                out.begin(),
                [&](unsigned char character) {
                    return static_cast<char>(
                        expression.text == "lower"
                            ? std::tolower(character)
                            : std::toupper(character));
                });
            return TemplateValue{std::move(out)};
        }
        if (expression.text == "length") {
            if (const auto* list =
                    std::get_if<ValueList>(&input.value)) {
                return TemplateValue{
                    static_cast<std::int64_t>(list->size())};
            }
            if (const auto* object =
                    std::get_if<ValueObject>(&input.value)) {
                return TemplateValue{
                    static_cast<std::int64_t>(object->size())};
            }
            return TemplateValue{
                static_cast<std::int64_t>(
                    stringify(input).size())};
        }
        if (expression.text == "default") {
            return truthy(input) ? input : argument(1);
        }
        if (expression.text == "replace") {
            std::string out = stringify(input);
            const std::string from = stringify(argument(1));
            const std::string to = stringify(argument(2));
            replace_all(out, from, to);
            return TemplateValue{std::move(out)};
        }
        if (expression.text == "join") {
            const std::string separator =
                expression.children.size() > 1
                    ? stringify(argument(1))
                    : "";
            std::string out;
            if (const auto* list =
                    std::get_if<ValueList>(&input.value)) {
                for (std::size_t index = 0;
                     index < list->size();
                     ++index) {
                    if (index != 0) {
                        out += separator;
                    }
                    out += stringify((*list)[index]);
                }
            }
            return TemplateValue{std::move(out)};
        }
        if (expression.text == "dictsort") {
            const auto* object = std::get_if<ValueObject>(&input.value);
            if (!object) {
                throw_error(state, "dictsort filter requires an object");
            }
            ValueList pairs;
            pairs.reserve(object->size());
            for (const auto& [key, value] : *object) {
                pairs.emplace_back(ValueList{TemplateValue{key}, value});
            }
            return TemplateValue{std::move(pairs)};
        }
        if (expression.text == "map") {
            const auto* list = std::get_if<ValueList>(&input.value);
            if (!list || expression.children.size() < 2) {
                throw_error(state, "map filter requires a list and filter name");
            }
            const std::string operation = stringify(argument(1));
            ValueList mapped;
            mapped.reserve(list->size());
            for (const TemplateValue& value : *list) {
                const std::string text = stringify(value);
                if (operation == "upper") {
                    std::string transformed = text;
                    std::transform(
                        transformed.begin(), transformed.end(), transformed.begin(),
                        [](unsigned char character) {
                            return static_cast<char>(std::toupper(character));
                        });
                    mapped.emplace_back(std::move(transformed));
                } else if (operation == "lower") {
                    std::string transformed = text;
                    std::transform(
                        transformed.begin(), transformed.end(), transformed.begin(),
                        [](unsigned char character) {
                            return static_cast<char>(std::tolower(character));
                        });
                    mapped.emplace_back(std::move(transformed));
                } else {
                    throw_error(
                        state,
                        "unsupported map filter operation '" + operation + "'");
                }
            }
            return TemplateValue{std::move(mapped)};
        }
        if (expression.text == "items") {
            const auto* object =
                std::get_if<ValueObject>(&input.value);
            if (!object) {
                throw_error(
                    state,
                    "items filter requires an object");
            }
            ValueList pairs;
            pairs.reserve(object->size());
            for (const auto& [key, value] : *object) {
                pairs.emplace_back(
                    ValueList{TemplateValue{key}, value});
            }
            return TemplateValue{std::move(pairs)};
        }
        if (expression.text == "list") {
            return input;
        }
        if (expression.text == "min" ||
            expression.text == "max") {
            const auto* list =
                std::get_if<ValueList>(&input.value);
            if (!list || list->empty()) {
                throw_error(
                    state,
                    expression.text +
                        " filter requires a non-empty list");
            }
            const auto compare = [](
                                     const TemplateValue& left,
                                     const TemplateValue& right) {
                if (std::holds_alternative<std::int64_t>(
                        left.value) &&
                    std::holds_alternative<std::int64_t>(
                        right.value)) {
                    return std::get<std::int64_t>(left.value) <
                           std::get<std::int64_t>(right.value);
                }
                return stringify(left) < stringify(right);
            };
            TemplateValue selected = list->front();
            for (const TemplateValue& candidate : *list) {
                if ((expression.text == "min" &&
                     compare(candidate, selected)) ||
                    (expression.text == "max" &&
                     compare(selected, candidate))) {
                    selected = candidate;
                }
            }
            return selected;
        }
        if (expression.text == "reverse") {
            if (const auto* list =
                    std::get_if<ValueList>(&input.value)) {
                ValueList reversed = *list;
                std::reverse(reversed.begin(), reversed.end());
                return TemplateValue{std::move(reversed)};
            }
            throw_error(state, "reverse filter requires a list");
        }

        throw_error(
            state,
            "unsupported Jinja filter '" + expression.text + "'");
    }


}
