#include "program.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace celeg::chat_template_detail {
namespace {

struct LexerToken {
    enum class Kind {
        End,
        Identifier,
        String,
        Number,
        Symbol,
    };

    Kind kind = Kind::End;
    std::string text;
};

class ExpressionLexer {
public:
    explicit ExpressionLexer(std::string_view source) : source_(source) {
        next();
    }

    const LexerToken& current() const {
        return current_;
    }

    void next() {
        skip_space();
        if (position_ == source_.size()) {
            current_ = {};
            return;
        }

        const char first = source_[position_];
        if (std::isalpha(static_cast<unsigned char>(first)) || first == '_') {
            const std::size_t start = position_++;
            while (position_ < source_.size() &&
                   (std::isalnum(static_cast<unsigned char>(source_[position_])) ||
                    source_[position_] == '_')) {
                ++position_;
            }
            current_ = {
                LexerToken::Kind::Identifier,
                std::string(source_.substr(start, position_ - start)),
            };
            return;
        }

        if (std::isdigit(static_cast<unsigned char>(first))) {
            const std::size_t start = position_++;
            while (position_ < source_.size() &&
                   std::isdigit(static_cast<unsigned char>(source_[position_]))) {
                ++position_;
            }
            current_ = {
                LexerToken::Kind::Number,
                std::string(source_.substr(start, position_ - start)),
            };
            return;
        }

        if (first == '\'' || first == '"') {
            const char quote = first;
            ++position_;
            std::string decoded;
            while (position_ < source_.size() && source_[position_] != quote) {
                if (source_[position_] == '\\' && position_ + 1 < source_.size()) {
                    const char escaped = source_[position_ + 1];
                    switch (escaped) {
                    case 'n':
                        decoded += '\n';
                        break;
                    case 'r':
                        decoded += '\r';
                        break;
                    case 't':
                        decoded += '\t';
                        break;
                    default:
                        decoded += escaped;
                        break;
                    }
                    position_ += 2;
                } else {
                    decoded += source_[position_++];
                }
            }
            if (position_ == source_.size()) {
                throw std::invalid_argument("unterminated Jinja string literal");
            }
            ++position_;
            current_ = {LexerToken::Kind::String, std::move(decoded)};
            return;
        }

        const std::string_view two = source_.substr(position_, 2);
        if (two == "==" || two == "!=" || two == ">=" || two == "<=" ||
            two == "//") {
            position_ += 2;
            current_ = {LexerToken::Kind::Symbol, std::string(two)};
            return;
        }

        ++position_;
        current_ = {LexerToken::Kind::Symbol, std::string(1, first)};
    }

private:
    void skip_space() {
        while (position_ < source_.size() &&
               std::isspace(static_cast<unsigned char>(source_[position_]))) {
            ++position_;
        }
    }

    std::string_view source_;
    std::size_t position_ = 0;
    LexerToken current_;
};

class ExpressionParser {
public:
    explicit ExpressionParser(std::string_view source) : lexer_(source) {}

    Expression parse() {
        Expression result = parse_conditional();
        if (lexer_.current().kind != LexerToken::Kind::End) {
            throw std::invalid_argument(
                "unsupported Jinja expression near '" + lexer_.current().text + "'");
        }
        return result;
    }

private:
    bool consume(std::string_view symbol) {
        if (lexer_.current().text != symbol) {
            return false;
        }
        lexer_.next();
        return true;
    }

    bool keyword_argument_ahead() const {
        if (lexer_.current().kind != LexerToken::Kind::Identifier) {
            return false;
        }
        ExpressionLexer probe = lexer_;
        probe.next();
        return probe.current().text == "=";
    }

    static Expression make_binary(std::string op, Expression left, Expression right) {
        Expression result;
        result.kind = Expression::Kind::Binary;
        result.text = std::move(op);
        result.children.push_back(std::move(left));
        result.children.push_back(std::move(right));
        return result;
    }

    Expression parse_conditional() {
        Expression value = parse_or();
        if (lexer_.current().kind == LexerToken::Kind::Identifier &&
            lexer_.current().text == "if") {
            lexer_.next();
            Expression condition = parse_or();
            if (lexer_.current().kind != LexerToken::Kind::Identifier ||
                lexer_.current().text != "else") {
                throw std::invalid_argument(
                    "Jinja conditional expression requires else");
            }
            lexer_.next();

            Expression result;
            result.kind = Expression::Kind::Conditional;
            result.children.push_back(std::move(condition));
            result.children.push_back(std::move(value));
            result.children.push_back(parse_conditional());
            return result;
        }
        return value;
    }

    Expression parse_or() {
        Expression left = parse_and();
        while (lexer_.current().kind == LexerToken::Kind::Identifier &&
               lexer_.current().text == "or") {
            lexer_.next();
            left = make_binary("or", std::move(left), parse_and());
        }
        return left;
    }

    Expression parse_and() {
        Expression left = parse_compare();
        while (lexer_.current().kind == LexerToken::Kind::Identifier &&
               lexer_.current().text == "and") {
            lexer_.next();
            left = make_binary("and", std::move(left), parse_compare());
        }
        return left;
    }

    Expression parse_compare() {
        Expression left = parse_concat();
        for (;;) {
            std::string op;
            if (lexer_.current().text == "==" || lexer_.current().text == "!=" ||
                lexer_.current().text == ">" || lexer_.current().text == "<" ||
                lexer_.current().text == ">=" || lexer_.current().text == "<=") {
                op = lexer_.current().text;
                lexer_.next();
            } else if (lexer_.current().kind == LexerToken::Kind::Identifier &&
                       lexer_.current().text == "in") {
                op = "in";
                lexer_.next();
            } else if (lexer_.current().kind == LexerToken::Kind::Identifier &&
                       lexer_.current().text == "not") {
                lexer_.next();
                if (lexer_.current().text != "in") {
                    throw std::invalid_argument("unsupported Jinja 'not' expression");
                }
                op = "not in";
                lexer_.next();
            } else if (lexer_.current().kind == LexerToken::Kind::Identifier &&
                       lexer_.current().text == "is") {
                lexer_.next();
                bool negated = false;
                if (lexer_.current().kind == LexerToken::Kind::Identifier &&
                    lexer_.current().text == "not") {
                    negated = true;
                    lexer_.next();
                }
                if (lexer_.current().kind != LexerToken::Kind::Identifier) {
                    throw std::invalid_argument("missing Jinja test name");
                }
                op = negated ? "is not " : "is ";
                op += lexer_.current().text;
                lexer_.next();

                Expression tested;
                tested.kind = Expression::Kind::Binary;
                tested.text = std::move(op);
                tested.children.push_back(std::move(left));
                left = std::move(tested);
                continue;
            } else {
                break;
            }
            left = make_binary(std::move(op), std::move(left), parse_concat());
        }
        return left;
    }

    Expression parse_concat() {
        Expression left = parse_unary();
        while (lexer_.current().text == "~" || lexer_.current().text == "+" ||
               lexer_.current().text == "-") {
            const std::string op = lexer_.current().text;
            lexer_.next();
            left = make_binary(op, std::move(left), parse_unary());
        }
        return left;
    }

    Expression parse_unary() {
        if (lexer_.current().kind == LexerToken::Kind::Identifier &&
            lexer_.current().text == "not") {
            lexer_.next();
            Expression result;
            result.kind = Expression::Kind::Unary;
            result.text = "not";
            result.children.push_back(parse_unary());
            return result;
        }
        return parse_filter();
    }

    Expression parse_filter() {
        Expression value = parse_postfix();
        while (consume("|")) {
            if (lexer_.current().kind != LexerToken::Kind::Identifier) {
                throw std::invalid_argument("missing Jinja filter name");
            }

            Expression filtered;
            filtered.kind = Expression::Kind::Filter;
            filtered.text = lexer_.current().text;
            lexer_.next();
            filtered.children.push_back(std::move(value));

            if (consume("(") && !consume(")")) {
                do {
                    filtered.children.push_back(parse_argument(
                        "lost Jinja filter keyword assignment"));
                } while (consume(","));
                if (!consume(")")) {
                    throw std::invalid_argument(
                        "unterminated Jinja filter arguments");
                }
            }
            value = std::move(filtered);
        }
        return value;
    }

    Expression parse_postfix() {
        Expression value = parse_primary();
        for (;;) {
            if (consume(".")) {
                if (lexer_.current().kind != LexerToken::Kind::Identifier) {
                    throw std::invalid_argument("missing Jinja member name");
                }
                Expression access;
                access.kind = Expression::Kind::Access;
                access.text = lexer_.current().text;
                lexer_.next();
                access.children.push_back(std::move(value));
                value = std::move(access);
                continue;
            }

            if (consume("[")) {
                Expression container = std::move(value);
                const bool omitted_start = lexer_.current().text == ":";
                Expression first;
                if (!omitted_start) {
                    first = parse_or();
                }

                if (consume(":")) {
                    Expression slice;
                    slice.kind = Expression::Kind::Slice;
                    slice.children.push_back(std::move(container));
                    slice.children.push_back(std::move(first));
                    if (lexer_.current().text != "]" &&
                        lexer_.current().text != ":") {
                        slice.children.push_back(parse_or());
                    } else {
                        slice.children.emplace_back();
                    }
                    if (consume(":")) {
                        if (lexer_.current().text != "]") {
                            slice.children.push_back(parse_or());
                        } else {
                            slice.children.emplace_back();
                        }
                    }
                    if (!consume("]")) {
                        throw std::invalid_argument("unterminated Jinja slice");
                    }
                    value = std::move(slice);
                } else {
                    Expression index;
                    index.kind = Expression::Kind::Index;
                    index.children.push_back(std::move(container));
                    index.children.push_back(std::move(first));
                    if (!consume("]")) {
                        throw std::invalid_argument("unterminated Jinja index");
                    }
                    value = std::move(index);
                }
                continue;
            }

            if (consume("(")) {
                Expression call;
                call.kind = Expression::Kind::Call;
                call.children.push_back(std::move(value));
                if (!consume(")")) {
                    do {
                        call.children.push_back(
                            parse_argument("lost Jinja keyword assignment"));
                    } while (consume(","));
                    if (!consume(")")) {
                        throw std::invalid_argument("unterminated Jinja call");
                    }
                }
                value = std::move(call);
                continue;
            }

            break;
        }
        return value;
    }

    Expression parse_argument(std::string_view lost_assignment) {
        if (!keyword_argument_ahead()) {
            return parse_conditional();
        }

        const std::string name = lexer_.current().text;
        lexer_.next();
        if (!consume("=")) {
            throw std::logic_error(std::string(lost_assignment));
        }
        Expression keyword;
        keyword.kind = Expression::Kind::Binary;
        keyword.text = "__keyword:" + name;
        keyword.children.push_back(parse_conditional());
        return keyword;
    }

    Expression parse_primary() {
        if (consume("-")) {
            if (lexer_.current().kind != LexerToken::Kind::Number) {
                throw std::invalid_argument("unsupported Jinja unary minus");
            }
            Expression result;
            result.kind = Expression::Kind::Literal;
            result.literal = TemplateValue{
                -static_cast<std::int64_t>(std::stoll(lexer_.current().text))};
            lexer_.next();
            return result;
        }

        if (lexer_.current().kind == LexerToken::Kind::String) {
            Expression result;
            result.kind = Expression::Kind::Literal;
            result.literal = TemplateValue{lexer_.current().text};
            lexer_.next();
            return result;
        }

        if (lexer_.current().kind == LexerToken::Kind::Number) {
            Expression result;
            result.kind = Expression::Kind::Literal;
            result.literal = TemplateValue{
                static_cast<std::int64_t>(std::stoll(lexer_.current().text))};
            lexer_.next();
            return result;
        }

        if (lexer_.current().kind == LexerToken::Kind::Identifier) {
            const std::string name = lexer_.current().text;
            lexer_.next();
            Expression result;
            result.kind = Expression::Kind::Name;
            result.text = name;
            if (name == "true" || name == "True") {
                result.kind = Expression::Kind::Literal;
                result.literal = TemplateValue{true};
            } else if (name == "false" || name == "False") {
                result.kind = Expression::Kind::Literal;
                result.literal = TemplateValue{false};
            } else if (name == "none" || name == "None" || name == "null") {
                result.kind = Expression::Kind::Literal;
            }
            return result;
        }

        if (consume("(")) {
            Expression value = parse_conditional();
            if (!consume(")")) {
                throw std::invalid_argument("unterminated Jinja group");
            }
            return value;
        }

        if (consume("[")) {
            Expression list;
            list.kind = Expression::Kind::Call;
            list.text = "__list";
            if (!consume("]")) {
                do {
                    list.children.push_back(parse_or());
                } while (consume(","));
                if (!consume("]")) {
                    throw std::invalid_argument("unterminated Jinja list");
                }
            }
            return list;
        }

        throw std::invalid_argument(
            "unsupported Jinja expression near '" + lexer_.current().text + "'");
    }

    ExpressionLexer lexer_;
};

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
                        macro.parameters.push_back(trim(parameters.substr(
                            begin,
                            comma == std::string::npos
                                ? std::string::npos
                                : comma - begin)));
                        if (macro.parameters.back().empty()) {
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
            return ExpressionParser(trim(expression)).parse();
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

} // namespace

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

} // namespace celeg::chat_template_detail
