#include "detail.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <utility>

namespace celeg::chat_template_detail {
namespace {

class TemplateParser {
public:
    TemplateParser(std::string_view source, std::string origin)
        : source_(source), origin_(std::move(origin)) {}

    std::vector<TemplateNode> parse() {
        auto result = parse_nodes({});
        if (pending_.has_value()) {
            fail("unexpected Jinja block terminator '" + pending_->first + "'");
        }
        return result;
    }

private:
    using Terminators = std::vector<std::string>;

    bool at_terminator(const Terminators& terms, std::string_view tag) const {
        const std::string head{tag.substr(0, tag.find_first_of(" \t"))};
        return std::find(terms.begin(), terms.end(), head) != terms.end();
    }

    std::vector<TemplateNode> parse_nodes(const Terminators& terminators) {
        std::vector<TemplateNode> nodes;
        while (position_ < source_.size()) {
            const std::size_t opening = source_.find("{", position_);
            if (opening == std::string_view::npos) {
                append_text(nodes, source_.substr(position_));
                position_ = source_.size();
                break;
            }

            const bool trim_left =
                source_.substr(opening, 3) == "{{-" ||
                source_.substr(opening, 3) == "{%-";
            if (opening > position_) {
                append_text(nodes, source_.substr(position_, opening - position_));
            }
            if (trim_left) {
                trim_trailing_text(nodes);
            }

            if (source_.substr(opening, 2) == "{{") {
                const std::size_t close = source_.find("}}", opening + 2);
                if (close == std::string_view::npos) {
                    fail("unterminated Jinja output");
                }
                TemplateNode output;
                output.kind = TemplateNode::Kind::Output;
                output.line = line_at(opening);
                output.expression = parse_expression(
                    strip_control(source_.substr(
                        opening + 2, close - opening - 2)));
                nodes.push_back(std::move(output));
                position_ = close + 2;
                trim_following_whitespace(
                    close > opening + 2 && source_[close - 1] == '-');
                continue;
            }

            if (source_.substr(opening, 2) == "{#") {
                const std::size_t close = source_.find("#}", opening + 2);
                if (close == std::string_view::npos) {
                    fail("unterminated Jinja comment");
                }
                position_ = close + 2;
                continue;
            }

            if (source_.substr(opening, 2) != "{%") {
                append_text(nodes, source_.substr(opening, 1));
                position_ = opening + 1;
                continue;
            }

            const std::size_t close = source_.find("%}", opening + 2);
            if (close == std::string_view::npos) {
                fail("unterminated Jinja block");
            }
            const int line = line_at(opening);
            const std::string tag = trim(strip_control(source_.substr(
                opening + 2, close - opening - 2)));
            position_ = close + 2;
            trim_following_whitespace(
                close > opening + 2 && source_[close - 1] == '-');

            if (at_terminator(terminators, tag)) {
                pending_ = std::pair{tag, line};
                return nodes;
            }

            if (tag.starts_with("if ")) {
                TemplateNode branch;
                branch.kind = TemplateNode::Kind::If;
                branch.line = line;
                branch.branches.push_back({
                    parse_expression(tag.substr(3)),
                    parse_nodes({"elif", "else", "endif"}),
                });
                while (pending_ && pending_->first.starts_with("elif ")) {
                    const std::string else_if = pending_->first;
                    pending_.reset();
                    branch.branches.push_back({
                        parse_expression(else_if.substr(5)),
                        parse_nodes({"elif", "else", "endif"}),
                    });
                }
                if (pending_ && pending_->first == "else") {
                    pending_.reset();
                    branch.otherwise = parse_nodes({"endif"});
                }
                require_pending("endif", "missing endif");
                nodes.push_back(std::move(branch));
                continue;
            }

            if (tag.starts_with("for ")) {
                const std::string declaration = tag.substr(4);
                const std::size_t in = declaration.find(" in ");
                if (in == std::string::npos) {
                    fail("for requires 'name in expression'");
                }
                const std::string name = trim(declaration.substr(0, in));
                if (name.empty() ||
                    name.find_first_not_of(
                        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_, ") !=
                        std::string::npos) {
                    fail("unsupported Jinja loop target");
                }

                const std::string remainder = declaration.substr(in + 4);
                const std::size_t predicate = remainder.find(" if ");
                TemplateNode loop;
                loop.kind = TemplateNode::Kind::For;
                loop.line = line;
                loop.text = name;
                loop.expression = parse_expression(
                    predicate == std::string::npos
                        ? remainder
                        : remainder.substr(0, predicate));
                if (predicate != std::string::npos) {
                    loop.condition =
                        parse_expression(remainder.substr(predicate + 4));
                }
                loop.body = parse_nodes({"else", "endfor"});
                if (pending_ && pending_->first == "else") {
                    pending_.reset();
                    loop.otherwise = parse_nodes({"endfor"});
                }
                require_pending("endfor", "missing endfor");
                nodes.push_back(std::move(loop));
                continue;
            }

            if (tag.starts_with("set ")) {
                const std::string assignment = tag.substr(4);
                const std::size_t equals = assignment.find('=');
                if (equals == std::string::npos) {
                    const std::string name = trim(assignment);
                    if (!valid_identifier(name, false)) {
                        fail("unsupported Jinja set target");
                    }
                    TemplateNode set;
                    set.kind = TemplateNode::Kind::Set;
                    set.line = line;
                    set.text = name;
                    set.body = parse_nodes({"endset"});
                    require_pending("endset", "missing endset");
                    nodes.push_back(std::move(set));
                    continue;
                }

                const std::string name = trim(assignment.substr(0, equals));
                if (!valid_identifier(name, true)) {
                    fail("unsupported Jinja set target");
                }
                TemplateNode set;
                set.kind = TemplateNode::Kind::Set;
                set.line = line;
                set.text = name;
                set.expression =
                    parse_expression(assignment.substr(equals + 1));
                nodes.push_back(std::move(set));
                continue;
            }

            if (tag == "generation") {
                TemplateNode generation;
                generation.kind = TemplateNode::Kind::Generation;
                generation.line = line;
                generation.body = parse_nodes({"endgeneration"});
                require_pending("endgeneration",
                                "missing endgeneration");
                nodes.push_back(std::move(generation));
                continue;
            }

            if (tag.starts_with("macro ")) {
                const std::string signature = tag.substr(6);
                const std::size_t open = signature.find('(');
                const std::size_t close_paren = signature.rfind(')');
                if (open == std::string::npos ||
                    close_paren != signature.size() - 1) {
                    fail("invalid macro declaration");
                }

                TemplateNode macro;
                macro.kind = TemplateNode::Kind::Macro;
                macro.line = line;
                macro.text = trim(signature.substr(0, open));
                const std::string parameters =
                    trim(signature.substr(open + 1, close_paren - open - 1));
                if (!parameters.empty()) {
                    std::size_t begin = 0;
                    while (begin < parameters.size()) {
                        const std::size_t comma = parameters.find(',', begin);
                        const std::string parameter = trim(parameters.substr(
                            begin,
                            comma == std::string::npos
                                ? std::string::npos
                                : comma - begin));
                        const std::size_t equals = parameter.find('=');
                        MacroParameter parsed;
                        parsed.name = trim(parameter.substr(
                            0, equals == std::string::npos
                                ? std::string::npos
                                : equals));
                        if (equals != std::string::npos) {
                            parsed.default_value = parse_expression(
                                parameter.substr(equals + 1));
                        }
                        macro.parameters.push_back(std::move(parsed));
                        if (macro.parameters.back().name.empty()) {
                            fail("empty macro parameter");
                        }
                        if (comma == std::string::npos) {
                            break;
                        }
                        begin = comma + 1;
                    }
                }
                macro.body = parse_nodes({"endmacro"});
                require_pending("endmacro", "missing endmacro");
                nodes.push_back(std::move(macro));
                continue;
            }

            if (tag == "raw" || tag == "endraw" ||
                tag.starts_with("include") || tag.starts_with("import") ||
                tag.starts_with("from ")) {
                fail("unsupported Jinja construct '" + tag + "'");
            }
            fail("unsupported Jinja block '" + tag + "'");
        }
        return nodes;
    }

    void require_pending(std::string_view expected, std::string_view message) {
        if (!pending_ || pending_->first != expected) {
            fail(std::string(message));
        }
        pending_.reset();
    }

    static bool valid_identifier(std::string_view name, bool allow_dot) {
        if (name.empty()) {
            return false;
        }
        const std::string_view allowed = allow_dot
            ? "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_."
            : "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_";
        return name.find_first_not_of(allowed) == std::string::npos;
    }

    void append_text(std::vector<TemplateNode>& nodes, std::string_view text) {
        if (text.empty()) {
            return;
        }
        TemplateNode node;
        node.kind = TemplateNode::Kind::Text;
        node.line = line_at(position_);
        node.text = std::string(text);
        nodes.push_back(std::move(node));
    }

    static void trim_trailing_text(std::vector<TemplateNode>& nodes) {
        if (nodes.empty() || nodes.back().kind != TemplateNode::Kind::Text) {
            return;
        }
        std::string& text = nodes.back().text;
        while (!text.empty() &&
               std::isspace(static_cast<unsigned char>(text.back()))) {
            text.pop_back();
        }
        if (text.empty()) {
            nodes.pop_back();
        }
    }

    void trim_following_whitespace(bool requested) {
        if (!requested) {
            return;
        }
        while (position_ < source_.size() &&
               std::isspace(static_cast<unsigned char>(source_[position_]))) {
            ++position_;
        }
    }

    static std::string_view strip_control(std::string_view control) {
        if (!control.empty() && control.front() == '-') {
            control.remove_prefix(1);
        }
        if (!control.empty() && control.back() == '-') {
            control.remove_suffix(1);
        }
        return control;
    }

    Expression parse_expression(std::string_view expression) {
        try {
            return chat_template_detail::parse_expression(trim(expression));
        } catch (const std::exception& error) {
            fail(error.what());
        }
    }

    int line_at(std::size_t offset) const {
        return 1 + static_cast<int>(std::count(
            source_.begin(),
            source_.begin() + static_cast<std::ptrdiff_t>(offset),
            '\n'));
    }

    [[noreturn]] void fail(const std::string& message) const {
        throw std::invalid_argument(
            "UnsupportedChatTemplateConstruct at " + origin_ + ":" +
            std::to_string(line_at(position_)) + ": " + message);
    }

    std::string_view source_;
    std::string origin_;
    std::size_t position_ = 0;
    std::optional<std::pair<std::string, int>> pending_;
};

}

std::string trim(std::string_view input) {
        std::size_t first = 0;
        while (first < input.size() &&
               std::isspace(static_cast<unsigned char>(input[first]))) {
            ++first;
        }
        std::size_t last = input.size();
        while (last > first &&
               std::isspace(static_cast<unsigned char>(input[last - 1]))) {
            --last;
        }
        return std::string(input.substr(first, last - first));
}

std::vector<TemplateNode> parse_template(
    std::string_view source,
    std::string origin) {
    return TemplateParser(source, std::move(origin)).parse();
}

}
