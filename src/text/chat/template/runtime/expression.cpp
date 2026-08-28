#include "renderer.hpp"

namespace celeg::chat_template_detail {

TemplateValue Renderer::eval(
        const Expression& expression,
        RenderState& state) const {
        switch (expression.kind) {
        case Expression::Kind::Literal:
            return expression.literal;

        case Expression::Kind::Name:
            return lookup(expression.text, state);

        case Expression::Kind::Unary:
            return TemplateValue{
                !truthy(eval(expression.children.at(0), state))};

        case Expression::Kind::Conditional:
            return truthy(eval(expression.children.at(0), state))
                ? eval(expression.children.at(1), state)
                : eval(expression.children.at(2), state);

        case Expression::Kind::Access: {
            const TemplateValue object =
                eval(expression.children.at(0), state);
            return member_of(object, expression.text)
                .value_or(TemplateValue{});
        }

        case Expression::Kind::Index:
            return index(expression, state);

        case Expression::Kind::Slice:
            return slice(expression, state);

        case Expression::Kind::Binary:
            return binary(expression, state);

        case Expression::Kind::Filter:
            return filter(expression, state);

        case Expression::Kind::Call:
            return call(expression, state);
        }
        throw_error(state, "invalid Jinja expression");
    }

TemplateValue Renderer::lookup(
        const std::string& name,
        const RenderState& state) const {
        for (auto scope = state.scopes.rbegin();
             scope != state.scopes.rend();
             ++scope) {
            const auto found = scope->find(name);
            if (found != scope->end()) {
                return found->second;
            }
        }
        return {};
    }

TemplateValue Renderer::index(
        const Expression& expression,
        RenderState& state) const {
        const TemplateValue object =
            eval(expression.children.at(0), state);
        const TemplateValue key =
            eval(expression.children.at(1), state);

        if (const auto* list = std::get_if<ValueList>(&object.value)) {
            const auto* integer =
                std::get_if<std::int64_t>(&key.value);
            if (!integer || *integer < 0 ||
                static_cast<std::size_t>(*integer) >= list->size()) {
                throw_error(state, "Jinja list index is out of range");
            }
            return (*list)[static_cast<std::size_t>(*integer)];
        }

        if (const auto* members =
                std::get_if<ValueObject>(&object.value)) {
            const std::string name = stringify(key);
            const auto found = members->find(name);
            if (found == members->end()) {
                throw_error(
                    state,
                    "undefined Jinja key '" + name + "'");
            }
            return found->second;
        }

        throw_error(state, "Jinja index applied to a non-container");
    }

TemplateValue Renderer::slice(
        const Expression& expression,
        RenderState& state) const {
        const TemplateValue object =
            eval(expression.children.at(0), state);
        const auto* list = std::get_if<ValueList>(&object.value);
        if (!list) {
            throw_error(state, "Jinja slice applied to a non-list");
        }

        const auto optional_integer =
            [&](std::size_t child) -> std::optional<std::int64_t> {
            if (child >= expression.children.size()) {
                return std::nullopt;
            }
            const TemplateValue value =
                eval(expression.children[child], state);
            if (std::holds_alternative<std::monostate>(value.value)) {
                return std::nullopt;
            }
            const auto* integer =
                std::get_if<std::int64_t>(&value.value);
            if (!integer) {
                throw_error(
                    state,
                    "Jinja slice bounds require integers");
            }
            return *integer;
        };

        const auto start = optional_integer(1);
        const auto end = optional_integer(2);
        const auto step = optional_integer(3);
        const std::int64_t stride = step.value_or(1);
        if (stride == 0) {
            throw_error(state, "Jinja slice step cannot be zero");
        }

        const auto bound = [&](std::int64_t raw) {
            const std::int64_t size =
                static_cast<std::int64_t>(list->size());
            const std::int64_t adjusted =
                raw < 0 ? raw + size : raw;
            return static_cast<std::size_t>(
                std::clamp(adjusted, std::int64_t{0}, size));
        };

        ValueList selected;
        if (stride > 0) {
            const std::size_t first = bound(start.value_or(0));
            const std::size_t last = bound(end.value_or(
                static_cast<std::int64_t>(list->size())));
            for (std::size_t position = first;
                 position < last;
                 position += static_cast<std::size_t>(stride)) {
                selected.push_back((*list)[position]);
            }
        } else {
            std::int64_t position = start.value_or(
                static_cast<std::int64_t>(list->size()) - 1);
            const std::int64_t last = end.value_or(-1);
            if (position < 0) {
                position += static_cast<std::int64_t>(list->size());
            }
            for (; position > last && position >= 0; position += stride) {
                selected.push_back(
                    (*list)[static_cast<std::size_t>(position)]);
            }
        }
        return TemplateValue{std::move(selected)};
    }

TemplateValue Renderer::binary(
        const Expression& expression,
        RenderState& state) const {
        const TemplateValue left =
            eval(expression.children.at(0), state);

        if (expression.text == "and") {
            return truthy(left)
                ? TemplateValue{
                      truthy(eval(expression.children.at(1), state))}
                : TemplateValue{false};
        }
        if (expression.text == "or") {
            return truthy(left)
                ? TemplateValue{true}
                : TemplateValue{
                      truthy(eval(expression.children.at(1), state))};
        }

        if (expression.text.starts_with("is ")) {
            return test(expression.text, left);
        }

        const TemplateValue right =
            eval(expression.children.at(1), state);

        if (expression.text == "+" &&
            std::holds_alternative<std::int64_t>(left.value) &&
            std::holds_alternative<std::int64_t>(right.value)) {
            return TemplateValue{
                std::get<std::int64_t>(left.value) +
                std::get<std::int64_t>(right.value)};
        }
        if (expression.text == "-") {
            const auto* l =
                std::get_if<std::int64_t>(&left.value);
            const auto* r =
                std::get_if<std::int64_t>(&right.value);
            if (!l || !r) {
                throw_error(
                    state,
                    "Jinja subtraction requires integers");
            }
            return TemplateValue{*l - *r};
        }
        if (expression.text == "~" || expression.text == "+") {
            return TemplateValue{
                stringify(left) + stringify(right)};
        }
        if (expression.text == "==") {
            return TemplateValue{
                stringify(left) == stringify(right)};
        }
        if (expression.text == "!=") {
            return TemplateValue{
                stringify(left) != stringify(right)};
        }
        if (expression.text == "in" ||
            expression.text == "not in") {
            bool found = false;
            if (const auto* list =
                    std::get_if<ValueList>(&right.value)) {
                found = std::any_of(
                    list->begin(),
                    list->end(),
                    [&](const TemplateValue& item) {
                        return stringify(item) == stringify(left);
                    });
            } else {
                found =
                    stringify(right).find(stringify(left)) !=
                    std::string::npos;
            }
            return TemplateValue{
                expression.text == "in" ? found : !found};
        }

        const std::string l = stringify(left);
        const std::string r = stringify(right);
        if (expression.text == ">") {
            return TemplateValue{l > r};
        }
        if (expression.text == "<") {
            return TemplateValue{l < r};
        }
        if (expression.text == ">=") {
            return TemplateValue{l >= r};
        }
        if (expression.text == "<=") {
            return TemplateValue{l <= r};
        }

        throw_error(
            state,
            "unsupported Jinja operator '" + expression.text + "'");
    }

TemplateValue Renderer::test(
        std::string_view name,
        const TemplateValue& value) {
        if (name == "is defined") {
            return TemplateValue{
                !std::holds_alternative<std::monostate>(value.value)};
        }
        if (name == "is not defined") {
            return TemplateValue{
                std::holds_alternative<std::monostate>(value.value)};
        }
        if (name == "is undefined") {
            return TemplateValue{
                std::holds_alternative<std::monostate>(value.value)};
        }
        if (name == "is not undefined") {
            return TemplateValue{
                !std::holds_alternative<std::monostate>(value.value)};
        }
        if (name == "is none") {
            return TemplateValue{
                std::holds_alternative<std::monostate>(value.value)};
        }
        if (name == "is not none") {
            return TemplateValue{
                !std::holds_alternative<std::monostate>(value.value)};
        }
        if (name == "is string") {
            return TemplateValue{
                std::holds_alternative<std::string>(value.value)};
        }
        if (name == "is not string") {
            return TemplateValue{
                !std::holds_alternative<std::string>(value.value)};
        }
        if (name == "is boolean") {
            return TemplateValue{
                std::holds_alternative<bool>(value.value)};
        }
        if (name == "is true") {
            return TemplateValue{truthy(value)};
        }
        if (name == "is false") {
            return TemplateValue{!truthy(value)};
        }
        if (name == "is iterable") {
            return TemplateValue{
                std::holds_alternative<ValueList>(value.value) ||
                std::holds_alternative<ValueObject>(value.value) ||
                std::holds_alternative<std::string>(value.value)};
        }
        if (name == "is mapping") {
            return TemplateValue{
                std::holds_alternative<ValueObject>(value.value)};
        }
        if (name == "is not mapping") {
            return TemplateValue{
                !std::holds_alternative<ValueObject>(value.value)};
        }
        if (name == "is sequence") {
            return TemplateValue{
                std::holds_alternative<ValueList>(value.value)};
        }
        if (name == "is not sequence") {
            return TemplateValue{
                !std::holds_alternative<ValueList>(value.value)};
        }
        return TemplateValue{false};
    }


}
