#include "detail.hpp"
#include "lexer.hpp"

#include <stdexcept>
#include <utility>

namespace celeg::chat_template_detail {
namespace {

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

            Expression result;
            result.kind = Expression::Kind::Conditional;
            result.children.push_back(std::move(condition));
            result.children.push_back(std::move(value));
            if (lexer_.current().kind == LexerToken::Kind::Identifier &&
                lexer_.current().text == "else") {
                lexer_.next();
                result.children.push_back(parse_conditional());
            } else {
                result.children.push_back(Expression{});
            }
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
            while (lexer_.current().kind == LexerToken::Kind::String) {
                std::get<std::string>(result.literal.value) +=
                    lexer_.current().text;
                lexer_.next();
            }
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

}

Expression parse_expression(std::string_view source) {
    return ExpressionParser(source).parse();
}

}
