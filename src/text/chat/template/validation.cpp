#include "program.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace celeg::chat_template_detail {
namespace {

[[noreturn]] void fail_validation(
    std::string_view origin,
    int line,
    const std::string& message) {
    throw std::invalid_argument(
        "UnsupportedChatTemplateConstruct at " + std::string(origin) + ":" +
        std::to_string(line) + ": " + message);
}

bool supported_name(
    std::string_view name,
    const std::vector<std::string>& macros) {
    return name == "namespace" || name == "range" ||
           name == "raise_exception" ||
           std::find(macros.begin(), macros.end(), name) != macros.end();
}

bool supported_member_call(std::string_view member) {
    return member == "strip" || member == "lstrip" || member == "rstrip" ||
           member == "lower" || member == "upper" || member == "items" ||
           member == "split" || member == "replace" ||
           member == "startswith" || member == "endswith" || member == "get";
}

bool supported_filter(std::string_view filter) {
    static constexpr std::array<std::string_view, 15> kFilters{
        "tojson",
        "string",
        "safe",
        "trim",
        "lower",
        "upper",
        "length",
        "default",
        "replace",
        "join",
        "list",
        "items",
        "min",
        "max",
        "reverse",
    };
    return std::find(kFilters.begin(), kFilters.end(), filter) !=
           kFilters.end();
}

void validate_expression(
    const Expression& expression,
    std::string_view origin,
    int line,
    const std::vector<std::string>& macros) {
    const auto validate_children = [&] {
        for (const Expression& child : expression.children) {
            validate_expression(child, origin, line, macros);
        }
    };

    switch (expression.kind) {
    case Expression::Kind::Literal:
    case Expression::Kind::Name:
        return;

    case Expression::Kind::Unary:
        if (expression.text != "not") {
            fail_validation(
                origin,
                line,
                "unsupported Jinja unary operator '" + expression.text + "'");
        }
        validate_children();
        return;

    case Expression::Kind::Binary: {
        static constexpr std::array<std::string_view, 22> kAllowed{
            "and",
            "or",
            "+",
            "-",
            "~",
            "==",
            "!=",
            "in",
            "not in",
            ">",
            "<",
            ">=",
            "<=",
            "is defined",
            "is undefined",
            "is none",
            "is not none",
            "is string",
            "is not string",
            "is true",
            "is false",
            "is iterable",
        };
        static constexpr std::array<std::string_view, 4> kContainerTests{
            "is mapping",
            "is not mapping",
            "is sequence",
            "is not sequence",
        };
        const bool allowed =
            expression.text.starts_with("__keyword:") ||
            std::find(kAllowed.begin(), kAllowed.end(), expression.text) !=
                kAllowed.end() ||
            std::find(
                kContainerTests.begin(),
                kContainerTests.end(),
                expression.text) != kContainerTests.end();
        if (!allowed) {
            fail_validation(
                origin,
                line,
                "unsupported Jinja operator '" + expression.text + "'");
        }
        validate_children();
        return;
    }

    case Expression::Kind::Conditional:
    case Expression::Kind::Access:
    case Expression::Kind::Index:
    case Expression::Kind::Slice:
        validate_children();
        return;

    case Expression::Kind::Filter:
        if (!supported_filter(expression.text)) {
            fail_validation(
                origin,
                line,
                "unsupported Jinja filter '" + expression.text + "'");
        }
        validate_children();
        return;

    case Expression::Kind::Call:
        if (expression.text == "__list") {
            validate_children();
            return;
        }
        if (expression.children.empty()) {
            fail_validation(origin, line, "Jinja call has no target");
        }
        if (const Expression& target = expression.children.front();
            target.kind == Expression::Kind::Name) {
            if (!supported_name(target.text, macros)) {
                fail_validation(
                    origin,
                    line,
                    "unsupported Jinja call '" + target.text + "'");
            }
        } else if (target.kind == Expression::Kind::Access) {
            if (!supported_member_call(target.text)) {
                fail_validation(
                    origin,
                    line,
                    "unsupported Jinja member call '" + target.text + "'");
            }
        } else {
            fail_validation(
                origin,
                line,
                "dynamic Jinja calls are not permitted");
        }
        validate_children();
        return;
    }
}

void collect_macros(
    const std::vector<TemplateNode>& nodes,
    std::vector<std::string>& macros) {
    for (const TemplateNode& node : nodes) {
        if (node.kind == TemplateNode::Kind::Macro) {
            macros.push_back(node.text);
        }
        for (const auto& [unused_condition, body] : node.branches) {
            (void)unused_condition;
            collect_macros(body, macros);
        }
        collect_macros(node.body, macros);
        collect_macros(node.otherwise, macros);
    }
}

void validate_nodes(
    const std::vector<TemplateNode>& nodes,
    std::string_view origin,
    const std::vector<std::string>& macros) {
    for (const TemplateNode& node : nodes) {
        switch (node.kind) {
        case TemplateNode::Kind::Text:
            break;

        case TemplateNode::Kind::Output:
            validate_expression(node.expression, origin, node.line, macros);
            break;

        case TemplateNode::Kind::Set:
            if (node.body.empty()) {
                validate_expression(node.expression, origin, node.line, macros);
            }
            validate_nodes(node.body, origin, macros);
            break;

        case TemplateNode::Kind::Macro:
            validate_nodes(node.body, origin, macros);
            break;

        case TemplateNode::Kind::Generation:
            validate_nodes(node.body, origin, macros);
            break;

        case TemplateNode::Kind::If:
            for (const auto& [condition, body] : node.branches) {
                validate_expression(condition, origin, node.line, macros);
                validate_nodes(body, origin, macros);
            }
            validate_nodes(node.otherwise, origin, macros);
            break;

        case TemplateNode::Kind::For:
            validate_expression(node.expression, origin, node.line, macros);
            if (node.condition) {
                validate_expression(*node.condition, origin, node.line, macros);
            }
            validate_nodes(node.body, origin, macros);
            validate_nodes(node.otherwise, origin, macros);
            break;
        }
    }
}

}

void validate_program(
    const std::vector<TemplateNode>& nodes,
    std::string_view origin) {
    std::vector<std::string> macros;
    collect_macros(nodes, macros);
    validate_nodes(nodes, origin, macros);
}

}
