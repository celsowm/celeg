#include "celeg/text/chat_template.hpp"
#include "celeg/text/tokenizer.hpp"
#include "celeg/checkpoint/formats/json.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <variant>

namespace celeg {
namespace {

struct RawJson { std::string value; };
struct TemplateValue;
using ValueList = std::vector<TemplateValue>;
using ValueObject = std::map<std::string, TemplateValue, std::less<>>;

struct TemplateValue {
    using Storage = std::variant<std::monostate, bool, std::int64_t, std::string,
                                 RawJson, ValueList, ValueObject>;
    Storage value;

    TemplateValue() = default;
    TemplateValue(bool input) : value(input) {}
    TemplateValue(std::int64_t input) : value(input) {}
    TemplateValue(std::string input) : value(std::move(input)) {}
    TemplateValue(const char* input) : value(std::string(input)) {}
    TemplateValue(RawJson input) : value(std::move(input)) {}
    TemplateValue(ValueList input) : value(std::move(input)) {}
    TemplateValue(ValueObject input) : value(std::move(input)) {}
};

TemplateValue template_value_from_json(const Json& input) {
    if (input.is_null()) return {};
    if (input.is_bool()) return TemplateValue{input.as_bool()};
    if (input.is_number()) return TemplateValue{input.as_i64()};
    if (input.is_string()) return TemplateValue{input.as_string()};
    if (input.is_array()) {
        ValueList values;
        for (const Json& member : input.as_array()) values.push_back(template_value_from_json(member));
        return TemplateValue{std::move(values)};
    }
    ValueObject values;
    for (const auto& [key, member] : input.as_object()) values.emplace(key, template_value_from_json(member));
    return TemplateValue{std::move(values)};
}

std::string json_string(std::string_view input) {
    std::string out{"\""};
    for (const unsigned char character : input) {
        switch (character) {
        case '\\': out += "\\\\"; break;
        case '\"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (character < 0x20) {
                constexpr char hex[] = "0123456789abcdef";
                out += "\\u00";
                out += hex[(character >> 4) & 0xf];
                out += hex[character & 0xf];
            } else {
                out += static_cast<char>(character);
            }
        }
    }
    return out + '"';
}

std::string stringify(const TemplateValue& value) {
    if (const auto* text = std::get_if<std::string>(&value.value)) return *text;
    if (const auto* raw = std::get_if<RawJson>(&value.value)) return raw->value;
    if (const auto* integer = std::get_if<std::int64_t>(&value.value)) return std::to_string(*integer);
    if (const auto* boolean = std::get_if<bool>(&value.value)) return *boolean ? "true" : "false";
    return {};
}

bool truthy(const TemplateValue& value) {
    if (const auto* boolean = std::get_if<bool>(&value.value)) return *boolean;
    if (const auto* integer = std::get_if<std::int64_t>(&value.value)) return *integer != 0;
    if (const auto* text = std::get_if<std::string>(&value.value)) return !text->empty();
    if (const auto* list = std::get_if<ValueList>(&value.value)) return !list->empty();
    if (const auto* object = std::get_if<ValueObject>(&value.value)) return !object->empty();
    return false;
}

std::string to_json(const TemplateValue& value) {
    if (std::holds_alternative<std::monostate>(value.value)) return "null";
    if (const auto* boolean = std::get_if<bool>(&value.value)) return *boolean ? "true" : "false";
    if (const auto* integer = std::get_if<std::int64_t>(&value.value)) return std::to_string(*integer);
    if (const auto* text = std::get_if<std::string>(&value.value)) return json_string(*text);
    if (const auto* raw = std::get_if<RawJson>(&value.value)) return raw->value;
    if (const auto* list = std::get_if<ValueList>(&value.value)) {
        std::string out = "[";
        for (std::size_t index = 0; index < list->size(); ++index) {
            if (index != 0) out += ',';
            out += to_json((*list)[index]);
        }
        return out + ']';
    }
    const auto& object = std::get<ValueObject>(value.value);
    std::string out = "{";
    bool first = true;
    for (const auto& [key, member] : object) {
        if (!first) out += ',';
        first = false;
        out += json_string(key) + ':' + to_json(member);
    }
    return out + '}';
}

std::optional<TemplateValue> member_of(const TemplateValue& value, std::string_view key) {
    if (const auto* object = std::get_if<ValueObject>(&value.value)) {
        const auto found = object->find(key);
        if (found != object->end()) return found->second;
    }
    if (key == "length") {
        if (const auto* list = std::get_if<ValueList>(&value.value)) return TemplateValue{static_cast<std::int64_t>(list->size())};
        if (const auto* text = std::get_if<std::string>(&value.value)) return TemplateValue{static_cast<std::int64_t>(text->size())};
    }
    return std::nullopt;
}

struct Expression {
    enum class Kind { Literal, Name, Unary, Binary, Conditional, Access, Index, Slice, Call, Filter };
    Kind kind = Kind::Literal;
    std::string text;
    TemplateValue literal;
    std::vector<Expression> children;
};

struct TemplateNode {
    enum class Kind { Text, Output, If, For, Set, Macro };
    Kind kind = Kind::Text;
    int line = 1;
    std::string text;
    Expression expression;
    std::optional<Expression> condition;
    std::vector<TemplateNode> body;
    std::vector<std::pair<Expression, std::vector<TemplateNode>>> branches;
    std::vector<TemplateNode> otherwise;
    std::vector<std::string> parameters;
};

struct LexerToken {
    enum class Kind { End, Identifier, String, Number, Symbol };
    Kind kind = Kind::End;
    std::string text;
};

class ExpressionLexer {
public:
    explicit ExpressionLexer(std::string_view source) : source_(source) { next(); }
    const LexerToken& current() const { return current_; }
    void next() {
        skip_space();
        if (position_ == source_.size()) { current_ = {}; return; }
        const char first = source_[position_];
        if (std::isalpha(static_cast<unsigned char>(first)) || first == '_') {
            const std::size_t start = position_++;
            while (position_ < source_.size() &&
                   (std::isalnum(static_cast<unsigned char>(source_[position_])) || source_[position_] == '_')) ++position_;
            current_ = {LexerToken::Kind::Identifier, std::string(source_.substr(start, position_ - start))};
            return;
        }
        if (std::isdigit(static_cast<unsigned char>(first))) {
            const std::size_t start = position_++;
            while (position_ < source_.size() && std::isdigit(static_cast<unsigned char>(source_[position_]))) ++position_;
            current_ = {LexerToken::Kind::Number, std::string(source_.substr(start, position_ - start))};
            return;
        }
        if (first == '\'' || first == '\"') {
            const char quote = first;
            ++position_;
            std::string decoded;
            while (position_ < source_.size() && source_[position_] != quote) {
                if (source_[position_] == '\\' && position_ + 1 < source_.size()) {
                    const char escaped = source_[position_ + 1];
                    switch (escaped) {
                    case 'n': decoded += '\n'; break;
                    case 'r': decoded += '\r'; break;
                    case 't': decoded += '\t'; break;
                    default: decoded += escaped; break;
                    }
                    position_ += 2;
                } else {
                    decoded += source_[position_++];
                }
            }
            if (position_ == source_.size()) throw std::invalid_argument("unterminated Jinja string literal");
            ++position_;
            current_ = {LexerToken::Kind::String, std::move(decoded)};
            return;
        }
        const std::string_view two = source_.substr(position_, 2);
        if (two == "==" || two == "!=" || two == ">=" || two == "<=" || two == "//") {
            position_ += 2;
            current_ = {LexerToken::Kind::Symbol, std::string(two)};
            return;
        }
        ++position_;
        current_ = {LexerToken::Kind::Symbol, std::string(1, first)};
    }
private:
    void skip_space() { while (position_ < source_.size() && std::isspace(static_cast<unsigned char>(source_[position_]))) ++position_; }
    std::string_view source_;
    std::size_t position_ = 0;
    LexerToken current_;
};

class ExpressionParser {
public:
    explicit ExpressionParser(std::string_view source) : lexer_(source) {}
    Expression parse() {
        Expression result = parse_conditional();
        if (lexer_.current().kind != LexerToken::Kind::End) throw std::invalid_argument("unsupported Jinja expression near '" + lexer_.current().text + "'");
        return result;
    }
private:
    bool keyword_argument_ahead() const {
        if (lexer_.current().kind != LexerToken::Kind::Identifier) return false;
        ExpressionLexer probe = lexer_;
        probe.next();
        return probe.current().text == "=";
    }
    bool consume(std::string_view symbol) {
        if (lexer_.current().text != symbol) return false;
        lexer_.next();
        return true;
    }
    Expression make_binary(std::string op, Expression left, Expression right) {
        Expression result; result.kind = Expression::Kind::Binary; result.text = std::move(op);
        result.children.push_back(std::move(left)); result.children.push_back(std::move(right)); return result;
    }
    Expression parse_or() {
        Expression left = parse_and();
        while (lexer_.current().kind == LexerToken::Kind::Identifier && lexer_.current().text == "or") {
            lexer_.next(); left = make_binary("or", std::move(left), parse_and());
        }
        return left;
    }
    Expression parse_conditional() {
        Expression value = parse_or();
        if (lexer_.current().kind == LexerToken::Kind::Identifier && lexer_.current().text == "if") {
            lexer_.next();
            Expression condition = parse_or();
            if (lexer_.current().kind != LexerToken::Kind::Identifier || lexer_.current().text != "else") {
                throw std::invalid_argument("Jinja conditional expression requires else");
            }
            lexer_.next();
            Expression result; result.kind = Expression::Kind::Conditional;
            result.children.push_back(std::move(condition));
            result.children.push_back(std::move(value));
            result.children.push_back(parse_conditional());
            return result;
        }
        return value;
    }
    Expression parse_and() {
        Expression left = parse_compare();
        while (lexer_.current().kind == LexerToken::Kind::Identifier && lexer_.current().text == "and") {
            lexer_.next(); left = make_binary("and", std::move(left), parse_compare());
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
                op = lexer_.current().text; lexer_.next();
            } else if (lexer_.current().kind == LexerToken::Kind::Identifier &&
                       lexer_.current().text == "in") {
                op = "in"; lexer_.next();
            } else if (lexer_.current().kind == LexerToken::Kind::Identifier &&
                       lexer_.current().text == "not") {
                lexer_.next();
                if (lexer_.current().text != "in") throw std::invalid_argument("unsupported Jinja 'not' expression");
                op = "not in"; lexer_.next();
            } else if (lexer_.current().kind == LexerToken::Kind::Identifier &&
                       lexer_.current().text == "is") {
                lexer_.next();
                if (lexer_.current().kind != LexerToken::Kind::Identifier) throw std::invalid_argument("missing Jinja test name");
                op = "is " + lexer_.current().text; lexer_.next();
                if (lexer_.current().kind == LexerToken::Kind::Identifier &&
                    (lexer_.current().text == "defined" || lexer_.current().text == "undefined")) {
                    op += " " + lexer_.current().text;
                    lexer_.next();
                }
                if (op == "is not") {
                    if (lexer_.current().kind != LexerToken::Kind::Identifier) throw std::invalid_argument("missing Jinja test name");
                    op += " " + lexer_.current().text; lexer_.next();
                }
                Expression tested; tested.kind = Expression::Kind::Binary; tested.text = std::move(op);
                tested.children.push_back(std::move(left));
                left = std::move(tested);
                continue;
            } else break;
            left = make_binary(std::move(op), std::move(left), parse_concat());
        }
        return left;
    }
    Expression parse_concat() {
        Expression left = parse_unary();
        while (lexer_.current().text == "~" || lexer_.current().text == "+" || lexer_.current().text == "-") {
            const std::string op = lexer_.current().text; lexer_.next();
            left = make_binary(op, std::move(left), parse_unary());
        }
        return left;
    }
    Expression parse_unary() {
        if (lexer_.current().kind == LexerToken::Kind::Identifier && lexer_.current().text == "not") {
            lexer_.next(); Expression result; result.kind = Expression::Kind::Unary; result.text = "not";
            result.children.push_back(parse_unary()); return result;
        }
        return parse_filter();
    }
    Expression parse_filter() {
        Expression value = parse_postfix();
        while (consume("|")) {
            if (lexer_.current().kind != LexerToken::Kind::Identifier) throw std::invalid_argument("missing Jinja filter name");
            Expression filtered; filtered.kind = Expression::Kind::Filter; filtered.text = lexer_.current().text;
            lexer_.next(); filtered.children.push_back(std::move(value));
            if (consume("(")) {
                if (!consume(")")) {
                    do {
                        if (keyword_argument_ahead()) {
                            const std::string name = lexer_.current().text;
                            lexer_.next();
                            if (!consume("=")) throw std::logic_error("lost Jinja filter keyword assignment");
                            Expression keyword; keyword.kind = Expression::Kind::Binary; keyword.text = "__keyword:" + name;
                            keyword.children.push_back(parse_conditional());
                            filtered.children.push_back(std::move(keyword));
                        } else {
                            filtered.children.push_back(parse_conditional());
                        }
                    } while (consume(","));
                    if (!consume(")")) throw std::invalid_argument("unterminated Jinja filter arguments");
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
                if (lexer_.current().kind != LexerToken::Kind::Identifier) throw std::invalid_argument("missing Jinja member name");
                Expression access; access.kind = Expression::Kind::Access; access.text = lexer_.current().text;
                lexer_.next(); access.children.push_back(std::move(value)); value = std::move(access);
            } else if (consume("[")) {
                Expression index; index.kind = Expression::Kind::Index; index.children.push_back(std::move(value));
                const bool omitted_start = lexer_.current().text == ":";
                Expression first;
                if (!omitted_start) first = parse_or();
                if (consume(":")) {
                    Expression slice; slice.kind = Expression::Kind::Slice; slice.children.push_back(std::move(value));
                    slice.children.push_back(std::move(first));
                    if (lexer_.current().text != "]" && lexer_.current().text != ":") slice.children.push_back(parse_or());
                    else slice.children.emplace_back();
                    if (consume(":")) {
                        if (lexer_.current().text != "]") slice.children.push_back(parse_or());
                        else slice.children.emplace_back();
                    }
                    if (!consume("]")) throw std::invalid_argument("unterminated Jinja slice");
                    value = std::move(slice);
                } else {
                    index.children.push_back(std::move(first));
                    if (!consume("]")) throw std::invalid_argument("unterminated Jinja index");
                    value = std::move(index);
                }
            } else if (consume("(")) {
                Expression call; call.kind = Expression::Kind::Call; call.children.push_back(std::move(value));
                if (!consume(")")) {
                    do {
                        if (keyword_argument_ahead()) {
                            const std::string name = lexer_.current().text;
                            lexer_.next();
                            if (!consume("=")) throw std::logic_error("lost Jinja keyword assignment");
                            Expression keyword; keyword.kind = Expression::Kind::Binary; keyword.text = "__keyword:" + name;
                            keyword.children.push_back(parse_conditional());
                            call.children.push_back(std::move(keyword));
                        } else {
                            call.children.push_back(parse_conditional());
                        }
                    } while (consume(","));
                    if (!consume(")")) throw std::invalid_argument("unterminated Jinja call");
                }
                value = std::move(call);
            } else break;
        }
        return value;
    }
    Expression parse_primary() {
        if (consume("-")) {
            if (lexer_.current().kind != LexerToken::Kind::Number) throw std::invalid_argument("unsupported Jinja unary minus");
            Expression result; result.kind = Expression::Kind::Literal;
            result.literal = TemplateValue{-static_cast<std::int64_t>(std::stoll(lexer_.current().text))};
            lexer_.next(); return result;
        }
        if (lexer_.current().kind == LexerToken::Kind::String) {
            Expression result; result.kind = Expression::Kind::Literal; result.literal = TemplateValue{lexer_.current().text}; lexer_.next(); return result;
        }
        if (lexer_.current().kind == LexerToken::Kind::Number) {
            Expression result; result.kind = Expression::Kind::Literal; result.literal = TemplateValue{static_cast<std::int64_t>(std::stoll(lexer_.current().text))}; lexer_.next(); return result;
        }
        if (lexer_.current().kind == LexerToken::Kind::Identifier) {
            const std::string name = lexer_.current().text; lexer_.next();
            Expression result; result.kind = Expression::Kind::Name; result.text = name;
            if (name == "true" || name == "True") { result.kind = Expression::Kind::Literal; result.literal = TemplateValue{true}; }
            else if (name == "false" || name == "False") { result.kind = Expression::Kind::Literal; result.literal = TemplateValue{false}; }
            else if (name == "none" || name == "None" || name == "null") { result.kind = Expression::Kind::Literal; }
            return result;
        }
        if (consume("(")) {
            Expression value = parse_conditional();
            if (!consume(")")) throw std::invalid_argument("unterminated Jinja group");
            return value;
        }
        if (consume("[")) {
            Expression list; list.kind = Expression::Kind::Call; list.text = "__list";
            if (!consume("]")) {
                do { list.children.push_back(parse_or()); } while (consume(","));
                if (!consume("]")) throw std::invalid_argument("unterminated Jinja list");
            }
            return list;
        }
        throw std::invalid_argument("unsupported Jinja expression near '" + lexer_.current().text + "'");
    }
    ExpressionLexer lexer_;
};

std::string trim(std::string_view input) {
    std::size_t first = 0;
    while (first < input.size() && std::isspace(static_cast<unsigned char>(input[first]))) ++first;
    std::size_t last = input.size();
    while (last > first && std::isspace(static_cast<unsigned char>(input[last - 1]))) --last;
    return std::string(input.substr(first, last - first));
}

class TemplateParser {
public:
    TemplateParser(std::string_view source, std::string origin) : source_(source), origin_(std::move(origin)) {}
    std::vector<TemplateNode> parse() {
        auto result = parse_nodes({});
        if (pending_.has_value()) fail("unexpected Jinja block terminator '" + pending_->first + "'");
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
                append_text(nodes, source_.substr(position_)); position_ = source_.size(); break;
            }
            const bool trim_left = source_.substr(opening, 3) == "{{-" || source_.substr(opening, 3) == "{%-";
            if (opening > position_) append_text(nodes, source_.substr(position_, opening - position_));
            if (trim_left) trim_trailing_text(nodes);
            if (source_.substr(opening, 2) == "{{") {
                const std::size_t close = source_.find("}}", opening + 2);
                if (close == std::string_view::npos) fail("unterminated Jinja output");
                TemplateNode output; output.kind = TemplateNode::Kind::Output; output.line = line_at(opening);
                output.expression = parse_expression(strip_control(source_.substr(opening + 2, close - opening - 2)));
                nodes.push_back(std::move(output)); position_ = close + 2;
                trim_following_whitespace(close > opening + 2 && source_[close - 1] == '-');
                continue;
            }
            if (source_.substr(opening, 2) == "{#") {
                const std::size_t close = source_.find("#}", opening + 2);
                if (close == std::string_view::npos) fail("unterminated Jinja comment");
                position_ = close + 2; continue;
            }
            if (source_.substr(opening, 2) != "{%") {
                append_text(nodes, source_.substr(opening, 1)); position_ = opening + 1; continue;
            }
            const std::size_t close = source_.find("%}", opening + 2);
            if (close == std::string_view::npos) fail("unterminated Jinja block");
            const int line = line_at(opening);
            const std::string tag = trim(strip_control(source_.substr(opening + 2, close - opening - 2)));
            position_ = close + 2;
            trim_following_whitespace(close > opening + 2 && source_[close - 1] == '-');
            if (at_terminator(terminators, tag)) { pending_ = std::pair{tag, line}; return nodes; }
            if (tag.starts_with("if ")) {
                TemplateNode branch; branch.kind = TemplateNode::Kind::If; branch.line = line;
                branch.branches.push_back({parse_expression(tag.substr(3)), parse_nodes({"elif", "else", "endif"})});
                while (pending_ && pending_->first.starts_with("elif ")) {
                    const std::string else_if = pending_->first; pending_.reset();
                    branch.branches.push_back({parse_expression(else_if.substr(5)), parse_nodes({"elif", "else", "endif"})});
                }
                if (pending_ && pending_->first == "else") { pending_.reset(); branch.otherwise = parse_nodes({"endif"}); }
                if (!pending_ || pending_->first != "endif") fail("missing endif");
                pending_.reset(); nodes.push_back(std::move(branch)); continue;
            }
            if (tag.starts_with("for ")) {
                const std::string declaration = tag.substr(4);
                const std::size_t in = declaration.find(" in ");
                if (in == std::string::npos) fail("for requires 'name in expression'");
                const std::string name = trim(declaration.substr(0, in));
                if (name.empty() || name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_, ") != std::string::npos) fail("unsupported Jinja loop target");
                const std::string remainder = declaration.substr(in + 4);
                const std::size_t predicate = remainder.find(" if ");
                TemplateNode loop; loop.kind = TemplateNode::Kind::For; loop.line = line; loop.text = name;
                loop.expression = parse_expression(predicate == std::string::npos ? remainder : remainder.substr(0, predicate));
                if (predicate != std::string::npos) loop.condition = parse_expression(remainder.substr(predicate + 4));
                loop.body = parse_nodes({"else", "endfor"});
                if (pending_ && pending_->first == "else") { pending_.reset(); loop.otherwise = parse_nodes({"endfor"}); }
                if (!pending_ || pending_->first != "endfor") fail("missing endfor");
                pending_.reset(); nodes.push_back(std::move(loop)); continue;
            }
            if (tag.starts_with("set ")) {
                const std::string assignment = tag.substr(4);
                const std::size_t equals = assignment.find('=');
                if (equals == std::string::npos) {
                    const std::string name = trim(assignment);
                    if (name.empty() || name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") != std::string::npos) fail("unsupported Jinja set target");
                    TemplateNode set; set.kind = TemplateNode::Kind::Set; set.line = line; set.text = name;
                    set.body = parse_nodes({"endset"});
                    if (!pending_ || pending_->first != "endset") fail("missing endset");
                    pending_.reset(); nodes.push_back(std::move(set)); continue;
                }
                const std::string name = trim(assignment.substr(0, equals));
                if (name.empty() || name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.") != std::string::npos) fail("unsupported Jinja set target");
                TemplateNode set; set.kind = TemplateNode::Kind::Set; set.line = line; set.text = name;
                set.expression = parse_expression(assignment.substr(equals + 1)); nodes.push_back(std::move(set)); continue;
            }
            if (tag.starts_with("macro ")) {
                const std::string signature = tag.substr(6);
                const std::size_t open = signature.find('('), close_paren = signature.rfind(')');
                if (open == std::string::npos || close_paren != signature.size() - 1) fail("invalid macro declaration");
                TemplateNode macro; macro.kind = TemplateNode::Kind::Macro; macro.line = line; macro.text = trim(signature.substr(0, open));
                const std::string parameters = trim(signature.substr(open + 1, close_paren - open - 1));
                if (!parameters.empty()) {
                    std::size_t begin = 0;
                    while (begin < parameters.size()) {
                        const std::size_t comma = parameters.find(',', begin);
                        macro.parameters.push_back(trim(parameters.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin)));
                        if (macro.parameters.back().empty()) fail("empty macro parameter");
                        if (comma == std::string::npos) break;
                        begin = comma + 1;
                    }
                }
                macro.body = parse_nodes({"endmacro"});
                if (!pending_ || pending_->first != "endmacro") fail("missing endmacro");
                pending_.reset(); nodes.push_back(std::move(macro)); continue;
            }
            if (tag == "raw" || tag == "endraw" || tag.starts_with("include") || tag.starts_with("import") || tag.starts_with("from ")) {
                fail("unsupported Jinja construct '" + tag + "'");
            }
            fail("unsupported Jinja block '" + tag + "'");
        }
        return nodes;
    }
    void append_text(std::vector<TemplateNode>& nodes, std::string_view text) {
        if (text.empty()) return;
        TemplateNode node; node.kind = TemplateNode::Kind::Text; node.line = line_at(position_); node.text = std::string(text); nodes.push_back(std::move(node));
    }
    void trim_trailing_text(std::vector<TemplateNode>& nodes) {
        if (nodes.empty() || nodes.back().kind != TemplateNode::Kind::Text) return;
        std::string& text = nodes.back().text;
        while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) text.pop_back();
        if (text.empty()) nodes.pop_back();
    }
    void trim_following_whitespace(bool requested) {
        if (!requested) return;
        while (position_ < source_.size() && std::isspace(static_cast<unsigned char>(source_[position_]))) ++position_;
    }
    static std::string_view strip_control(std::string_view control) {
        if (!control.empty() && control.front() == '-') control.remove_prefix(1);
        if (!control.empty() && control.back() == '-') control.remove_suffix(1);
        return control;
    }
    Expression parse_expression(std::string_view expression) {
        try { return ExpressionParser(trim(expression)).parse(); }
        catch (const std::exception& error) { fail(error.what()); }
    }
    int line_at(std::size_t offset) const {
        return 1 + static_cast<int>(std::count(source_.begin(), source_.begin() + static_cast<std::ptrdiff_t>(offset), '\n'));
    }
    [[noreturn]] void fail(const std::string& message) const {
        throw std::invalid_argument("UnsupportedChatTemplateConstruct at " + origin_ + ":" + std::to_string(line_at(position_)) + ": " + message);
    }
    std::string_view source_;
    std::string origin_;
    std::size_t position_ = 0;
    std::optional<std::pair<std::string, int>> pending_;
};

[[noreturn]] void fail_validation(std::string_view origin, int line,
                                  const std::string& message) {
    throw std::invalid_argument("UnsupportedChatTemplateConstruct at " +
                                std::string(origin) + ":" +
                                std::to_string(line) + ": " + message);
}

bool supported_name(std::string_view name,
                    const std::vector<std::string>& macros) {
    return name == "namespace" || name == "range" ||
           name == "raise_exception" ||
           std::find(macros.begin(), macros.end(), name) != macros.end();
}

bool supported_member_call(std::string_view member) {
    return member == "strip" || member == "lstrip" || member == "rstrip" ||
           member == "lower" || member == "upper" || member == "items" ||
           member == "split" || member == "replace" || member == "startswith" ||
           member == "endswith";
}

bool supported_filter(std::string_view filter) {
    return filter == "tojson" || filter == "string" || filter == "safe" ||
           filter == "trim" || filter == "lower" || filter == "upper" ||
           filter == "length" || filter == "default" || filter == "replace" ||
           filter == "join" || filter == "list" || filter == "items" ||
           filter == "min" ||
           filter == "max" || filter == "reverse";
}

void validate_expression(const Expression& expression, std::string_view origin,
                         int line, const std::vector<std::string>& macros) {
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
            fail_validation(origin, line, "unsupported Jinja unary operator '" + expression.text + "'");
        }
        validate_children();
        return;
    case Expression::Kind::Binary: {
        static constexpr std::string_view allowed[] = {
            "and", "or", "+", "-", "~", "==", "!=", "in", "not in",
            ">", "<", ">=", "<=", "is defined", "is undefined", "is none",
            "is not none", "is string", "is not string", "is true", "is false",
            "is iterable", "is mapping", "is not mapping", "is sequence", "is not sequence"};
        if (!expression.text.starts_with("__keyword:") &&
            std::find(std::begin(allowed), std::end(allowed), expression.text) == std::end(allowed)) {
            fail_validation(origin, line, "unsupported Jinja operator '" + expression.text + "'");
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
            fail_validation(origin, line, "unsupported Jinja filter '" + expression.text + "'");
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
        const Expression& target = expression.children.front();
        if (target.kind == Expression::Kind::Name) {
            if (!supported_name(target.text, macros)) {
                fail_validation(origin, line, "unsupported Jinja call '" + target.text + "'");
            }
        } else if (target.kind == Expression::Kind::Access) {
            if (!supported_member_call(target.text)) {
                fail_validation(origin, line, "unsupported Jinja member call '" + target.text + "'");
            }
        } else {
            fail_validation(origin, line, "dynamic Jinja calls are not permitted");
        }
        validate_children();
        return;
    }
}

void collect_macros(const std::vector<TemplateNode>& nodes,
                    std::vector<std::string>& macros) {
    for (const TemplateNode& node : nodes) {
        if (node.kind == TemplateNode::Kind::Macro) macros.push_back(node.text);
        for (const auto& [unused_condition, body] : node.branches) {
            (void)unused_condition;
            collect_macros(body, macros);
        }
        collect_macros(node.body, macros);
        collect_macros(node.otherwise, macros);
    }
}

void validate_nodes(const std::vector<TemplateNode>& nodes,
                    std::string_view origin, const std::vector<std::string>& macros) {
    for (const TemplateNode& node : nodes) {
        switch (node.kind) {
        case TemplateNode::Kind::Text:
            break;
        case TemplateNode::Kind::Output:
            validate_expression(node.expression, origin, node.line, macros);
            break;
        case TemplateNode::Kind::Set:
            if (node.body.empty()) validate_expression(node.expression, origin, node.line, macros);
            validate_nodes(node.body, origin, macros);
            break;
        case TemplateNode::Kind::Macro:
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
            if (node.condition) validate_expression(*node.condition, origin, node.line, macros);
            validate_nodes(node.body, origin, macros);
            validate_nodes(node.otherwise, origin, macros);
            break;
        }
    }
}

void validate_program(const std::vector<TemplateNode>& nodes,
                      std::string_view origin) {
    std::vector<std::string> macros;
    collect_macros(nodes, macros);
    validate_nodes(nodes, origin, macros);
}

struct MacroDefinition { const TemplateNode* node = nullptr; };
struct RenderState {
    std::vector<ValueObject> scopes;
    std::map<std::string, MacroDefinition, std::less<>> macros;
    std::string origin;
    int current_line = 1;
};

class InteractionRenderProgramImpl {
public:
    explicit InteractionRenderProgramImpl(std::vector<TemplateNode> nodes) : nodes_(std::move(nodes)) {}
    std::string render(std::span<const ChatMessage> messages,
                       std::span<const ChatToolDefinition> tools,
                       bool generation_prompt,
                       const ChatTemplateOptions& options,
                       std::string_view origin,
                       std::string_view bos_token) const {
        RenderState state;
        state.origin = origin;
        ValueObject root;
        root.emplace("bos_token", std::string(bos_token));
        root.emplace("eos_token", "<|im_end|>");
        root.emplace("messages", message_values(messages));
        root.emplace("tools", tool_values(tools));
        root.emplace("add_generation_prompt", TemplateValue{generation_prompt});
        root.emplace("enable_thinking", TemplateValue{options.enable_thinking.value_or(false)});
        root.emplace("keep_past_thinking", TemplateValue{false});
        root.emplace("tool_choice", tool_choice_value(options.tool_choice));
        state.scopes.push_back(std::move(root));
        std::string output;
        render_nodes(nodes_, state, output);
        return output;
    }
private:
    static TemplateValue message_values(std::span<const ChatMessage> messages) {
        ValueList rendered;
        rendered.reserve(messages.size());
        for (const ChatMessage& message : messages) {
            ValueObject value;
            value.emplace("role", role_name(message.role));
            value.emplace("content", message.content.value_or(""));
            value.emplace("tool_call_id", message.tool_call_id.value_or(""));
            value.emplace("name", message.name.value_or(""));
            value.emplace("reasoning_content", message.reasoning_content.value_or(""));
            value.emplace("thinking", message.reasoning_content.value_or(""));
            ValueList calls;
            for (const ToolCall& call : message.tool_calls) {
                ValueObject function;
                function.emplace("name", call.name);
                function.emplace("arguments", call.arguments);
                ValueObject call_value;
                call_value.emplace("id", call.id);
                call_value.emplace("name", call.name);
                call_value.emplace("arguments", call.arguments);
                call_value.emplace("type", "function");
                call_value.emplace("function", TemplateValue{std::move(function)});
                try {
                    call_value["arguments"] = template_value_from_json(Json::parse(call.arguments));
                    auto& callable = std::get<ValueObject>(call_value["function"].value);
                    callable["arguments"] = call_value["arguments"];
                } catch (const std::exception&) {
                    // The protocol layer validates request-side JSON. Keep a raw
                    // string for non-server callers so the renderer remains
                    // deterministic and can surface its own grammar error.
                }
                calls.emplace_back(std::move(call_value));
            }
            value.emplace("tool_calls", TemplateValue{std::move(calls)});
            rendered.emplace_back(std::move(value));
        }
        return TemplateValue{std::move(rendered)};
    }
    static TemplateValue tool_values(std::span<const ChatToolDefinition> tools) {
        ValueList rendered;
        rendered.reserve(tools.size());
        for (const ChatToolDefinition& tool : tools) {
            ValueObject function;
            function.emplace("name", tool.function.name);
            function.emplace("description", tool.function.description);
            function.emplace("parameters", RawJson{tool.function.parameters.serialized.empty() ? "{}" : tool.function.parameters.serialized});
            function.emplace("strict", tool.function.strict);
            ValueObject value;
            value.emplace("type", tool.type);
            value.emplace("function", TemplateValue{std::move(function)});
            rendered.emplace_back(std::move(value));
        }
        return TemplateValue{std::move(rendered)};
    }
    static TemplateValue tool_choice_value(const ToolChoice& choice) {
        ValueObject value;
        switch (choice.mode) {
        case ToolChoiceMode::None: value.emplace("mode", "none"); break;
        case ToolChoiceMode::Auto: value.emplace("mode", "auto"); break;
        case ToolChoiceMode::Required: value.emplace("mode", "required"); break;
        case ToolChoiceMode::Specific: value.emplace("mode", "specific"); break;
        }
        value.emplace("function_name", choice.function_name);
        return TemplateValue{std::move(value)};
    }
    static std::string role_name(ChatRole role) {
        switch (role) {
        case ChatRole::System: return "system";
        case ChatRole::Developer: return "developer";
        case ChatRole::User: return "user";
        case ChatRole::Assistant: return "assistant";
        case ChatRole::Tool: return "tool";
        }
        throw std::invalid_argument("unknown chat role");
    }
    TemplateValue eval(const Expression& expression, RenderState& state) const {
        switch (expression.kind) {
        case Expression::Kind::Literal: return expression.literal;
        case Expression::Kind::Name: return lookup(expression.text, state);
        case Expression::Kind::Unary: return TemplateValue{!truthy(eval(expression.children[0], state))};
        case Expression::Kind::Conditional:
            return truthy(eval(expression.children[0], state))
                ? eval(expression.children[1], state) : eval(expression.children[2], state);
        case Expression::Kind::Access: {
            const TemplateValue object = eval(expression.children[0], state);
            const auto member = member_of(object, expression.text);
            return member.value_or(TemplateValue{});
        }
        case Expression::Kind::Index: {
            const TemplateValue object = eval(expression.children[0], state);
            const TemplateValue index = eval(expression.children[1], state);
            if (const auto* list = std::get_if<ValueList>(&object.value)) {
                const auto* integer = std::get_if<std::int64_t>(&index.value);
                if (!integer || *integer < 0 || static_cast<std::size_t>(*integer) >= list->size()) throw_error(state, "Jinja list index is out of range");
                return (*list)[static_cast<std::size_t>(*integer)];
            }
            if (const auto* object_members = std::get_if<ValueObject>(&object.value)) {
                const auto found = object_members->find(stringify(index));
                if (found == object_members->end()) throw_error(state, "undefined Jinja key '" + stringify(index) + "'");
                return found->second;
            }
            throw_error(state, "Jinja index applied to a non-container");
        }
        case Expression::Kind::Slice: {
            const TemplateValue object = eval(expression.children[0], state);
            const TemplateValue start_value = eval(expression.children[1], state);
            const std::optional<std::int64_t> start = std::holds_alternative<std::monostate>(start_value.value)
                ? std::nullopt : std::optional{std::get<std::int64_t>(start_value.value)};
            const TemplateValue end_value = expression.children.size() >= 3
                ? eval(expression.children[2], state) : TemplateValue{};
            const std::optional<std::int64_t> end = std::holds_alternative<std::monostate>(end_value.value)
                ? std::nullopt : std::optional{std::get<std::int64_t>(end_value.value)};
            const std::optional<std::int64_t> step = expression.children.size() >= 4 &&
                !std::holds_alternative<std::monostate>(expression.children[3].literal.value)
                ? std::optional{std::get<std::int64_t>(eval(expression.children[3], state).value)} : std::nullopt;
            if (const auto* list = std::get_if<ValueList>(&object.value)) {
                const auto bound = [&](std::int64_t index) {
                    return static_cast<std::size_t>(std::clamp(index < 0 ? index + static_cast<std::int64_t>(list->size()) : index,
                                                               std::int64_t{0}, static_cast<std::int64_t>(list->size())));
                };
                const std::int64_t stride = step.value_or(1);
                if (stride == 0) throw_error(state, "Jinja slice step cannot be zero");
                ValueList selected;
                if (stride > 0) {
                    const std::size_t first = bound(start.value_or(0)), last = bound(end.value_or(static_cast<std::int64_t>(list->size())));
                    for (std::size_t index = first; index < last; index += static_cast<std::size_t>(stride)) selected.push_back((*list)[index]);
                } else {
                    std::int64_t index = start.value_or(static_cast<std::int64_t>(list->size()) - 1);
                    const std::int64_t last = end.value_or(-1);
                    if (index < 0) index += static_cast<std::int64_t>(list->size());
                    for (; index > last && index >= 0; index += stride) selected.push_back((*list)[static_cast<std::size_t>(index)]);
                }
                return TemplateValue{std::move(selected)};
            }
            throw_error(state, "Jinja slice applied to a non-list");
        }
        case Expression::Kind::Binary: return binary(expression, state);
        case Expression::Kind::Filter: return filter(expression, state);
        case Expression::Kind::Call: return call(expression, state);
        }
        throw_error(state, "invalid Jinja expression");
    }
    TemplateValue lookup(const std::string& name, const RenderState& state) const {
        for (auto scope = state.scopes.rbegin(); scope != state.scopes.rend(); ++scope) {
            const auto found = scope->find(name);
            if (found != scope->end()) return found->second;
        }
        // Optional render controls are Undefined in Jinja: false in
        // predicates, empty when rendered, and visible through `is defined`.
        return {};
    }
    TemplateValue binary(const Expression& expression, RenderState& state) const {
        const TemplateValue left = eval(expression.children[0], state);
        if (expression.text == "and") return truthy(left) ? TemplateValue{truthy(eval(expression.children[1], state))} : TemplateValue{false};
        if (expression.text == "or") return truthy(left) ? TemplateValue{true} : TemplateValue{truthy(eval(expression.children[1], state))};
        if (expression.text == "is defined") return TemplateValue{!std::holds_alternative<std::monostate>(left.value)};
        if (expression.text == "is none") return TemplateValue{std::holds_alternative<std::monostate>(left.value)};
        if (expression.text == "is not none") return TemplateValue{!std::holds_alternative<std::monostate>(left.value)};
        if (expression.text == "is string") return TemplateValue{std::holds_alternative<std::string>(left.value)};
        if (expression.text == "is not string") return TemplateValue{!std::holds_alternative<std::string>(left.value)};
        if (expression.text == "is true") return TemplateValue{truthy(left)};
        if (expression.text == "is false") return TemplateValue{!truthy(left)};
        if (expression.text == "is iterable") return TemplateValue{std::holds_alternative<ValueList>(left.value) || std::holds_alternative<ValueObject>(left.value) || std::holds_alternative<std::string>(left.value)};
        if (expression.text == "is mapping") return TemplateValue{std::holds_alternative<ValueObject>(left.value)};
        if (expression.text == "is not mapping") return TemplateValue{!std::holds_alternative<ValueObject>(left.value)};
        if (expression.text == "is sequence") return TemplateValue{std::holds_alternative<ValueList>(left.value)};
        if (expression.text == "is not sequence") return TemplateValue{!std::holds_alternative<ValueList>(left.value)};
        if (expression.text == "is undefined") return TemplateValue{std::holds_alternative<std::monostate>(left.value)};
        const TemplateValue right = eval(expression.children[1], state);
        if (expression.text == "+" && std::holds_alternative<std::int64_t>(left.value) &&
            std::holds_alternative<std::int64_t>(right.value)) {
            return TemplateValue{std::get<std::int64_t>(left.value) + std::get<std::int64_t>(right.value)};
        }
        if (expression.text == "-") {
            if (!std::holds_alternative<std::int64_t>(left.value) || !std::holds_alternative<std::int64_t>(right.value)) {
                throw_error(state, "Jinja subtraction requires integers");
            }
            return TemplateValue{std::get<std::int64_t>(left.value) - std::get<std::int64_t>(right.value)};
        }
        if (expression.text == "~" || expression.text == "+") return TemplateValue{stringify(left) + stringify(right)};
        if (expression.text == "==") return TemplateValue{stringify(left) == stringify(right)};
        if (expression.text == "!=") return TemplateValue{stringify(left) != stringify(right)};
        if (expression.text == "in" || expression.text == "not in") {
            bool found = false;
            if (const auto* list = std::get_if<ValueList>(&right.value)) {
                found = std::any_of(list->begin(), list->end(), [&](const TemplateValue& item) { return stringify(item) == stringify(left); });
            } else found = stringify(right).find(stringify(left)) != std::string::npos;
            return TemplateValue{expression.text == "in" ? found : !found};
        }
        const std::string l = stringify(left), r = stringify(right);
        if (expression.text == ">") return TemplateValue{l > r};
        if (expression.text == "<") return TemplateValue{l < r};
        if (expression.text == ">=") return TemplateValue{l >= r};
        if (expression.text == "<=") return TemplateValue{l <= r};
        throw_error(state, "unsupported Jinja operator '" + expression.text + "'");
    }
    TemplateValue filter(const Expression& expression, RenderState& state) const {
        TemplateValue input = eval(expression.children[0], state);
        const auto argument = [&](std::size_t index) { return eval(expression.children.at(index), state); };
        if (expression.text == "tojson") return TemplateValue{to_json(input)};
        if (expression.text == "string" || expression.text == "safe") return TemplateValue{stringify(input)};
        if (expression.text == "trim") return TemplateValue{trim(stringify(input))};
        if (expression.text == "lower" || expression.text == "upper") {
            std::string out = stringify(input);
            std::transform(out.begin(), out.end(), out.begin(), [&](unsigned char c) {
                return static_cast<char>(expression.text == "lower" ? std::tolower(c) : std::toupper(c));
            });
            return TemplateValue{std::move(out)};
        }
        if (expression.text == "length") {
            if (const auto* list = std::get_if<ValueList>(&input.value)) return TemplateValue{static_cast<std::int64_t>(list->size())};
            if (const auto* object = std::get_if<ValueObject>(&input.value)) return TemplateValue{static_cast<std::int64_t>(object->size())};
            return TemplateValue{static_cast<std::int64_t>(stringify(input).size())};
        }
        if (expression.text == "default") return truthy(input) ? input : argument(1);
        if (expression.text == "replace") {
            std::string out = stringify(input), from = stringify(argument(1)), to = stringify(argument(2));
            std::size_t offset = 0; while (!from.empty() && (offset = out.find(from, offset)) != std::string::npos) { out.replace(offset, from.size(), to); offset += to.size(); }
            return TemplateValue{std::move(out)};
        }
        if (expression.text == "join") {
            const std::string separator = expression.children.size() > 1 ? stringify(argument(1)) : "";
            std::string out;
            if (const auto* list = std::get_if<ValueList>(&input.value)) {
                for (std::size_t index = 0; index < list->size(); ++index) { if (index) out += separator; out += stringify((*list)[index]); }
            }
            return TemplateValue{std::move(out)};
        }
        if (expression.text == "items") {
            const auto* object = std::get_if<ValueObject>(&input.value);
            if (!object) throw_error(state, "items filter requires an object");
            ValueList pairs;
            pairs.reserve(object->size());
            for (const auto& [key, value] : *object) {
                pairs.emplace_back(ValueList{TemplateValue{key}, value});
            }
            return TemplateValue{std::move(pairs)};
        }
        if (expression.text == "list") return input;
        if (expression.text == "min" || expression.text == "max") {
            const auto* list = std::get_if<ValueList>(&input.value);
            if (!list || list->empty()) throw_error(state, expression.text + " filter requires a non-empty list");
            const auto compare = [&](const TemplateValue& left, const TemplateValue& right) {
                if (std::holds_alternative<std::int64_t>(left.value) && std::holds_alternative<std::int64_t>(right.value)) {
                    return std::get<std::int64_t>(left.value) < std::get<std::int64_t>(right.value);
                }
                return stringify(left) < stringify(right);
            };
            TemplateValue selected = list->front();
            for (const TemplateValue& candidate : *list) {
                if ((expression.text == "min" && compare(candidate, selected)) ||
                    (expression.text == "max" && compare(selected, candidate))) selected = candidate;
            }
            return selected;
        }
        if (expression.text == "reverse") {
            if (const auto* list = std::get_if<ValueList>(&input.value)) {
                ValueList reversed = *list;
                std::reverse(reversed.begin(), reversed.end());
                return TemplateValue{std::move(reversed)};
            }
            throw_error(state, "reverse filter requires a list");
        }
        throw_error(state, "unsupported Jinja filter '" + expression.text + "'");
    }
    TemplateValue call(const Expression& expression, RenderState& state) const {
        if (expression.text == "__list") {
            ValueList values; for (const Expression& child : expression.children) values.push_back(eval(child, state)); return TemplateValue{std::move(values)};
        }
        const Expression& target = expression.children.front();
        if (target.kind == Expression::Kind::Name) {
            if (target.text == "namespace") {
                ValueObject members;
                for (std::size_t index = 1; index < expression.children.size(); ++index) {
                    const Expression& argument = expression.children[index];
                    if (argument.kind != Expression::Kind::Binary || !argument.text.starts_with("__keyword:") || argument.children.size() != 1) {
                        throw_error(state, "namespace() requires named arguments");
                    }
                    members.emplace(argument.text.substr(std::string("__keyword:").size()), eval(argument.children[0], state));
                }
                return TemplateValue{std::move(members)};
            }
            if (target.text == "range") {
                const std::int64_t start = expression.children.size() == 2 ? 0 : std::get<std::int64_t>(eval(expression.children[1], state).value);
                const std::int64_t end = std::get<std::int64_t>(eval(expression.children.back(), state).value);
                ValueList values; for (std::int64_t value = start; value < end; ++value) values.emplace_back(value); return TemplateValue{std::move(values)};
            }
            if (target.text == "raise_exception") throw_error(state, stringify(eval(expression.children.at(1), state)));
            const auto macro = state.macros.find(target.text);
            if (macro != state.macros.end()) {
                if (expression.children.size() - 1 > macro->second.node->parameters.size()) throw_error(state, "wrong argument count for Jinja macro '" + target.text + "'");
                ValueObject arguments;
                for (std::size_t index = 0; index < macro->second.node->parameters.size(); ++index) {
                    arguments.emplace(macro->second.node->parameters[index],
                        index + 1 < expression.children.size() ? eval(expression.children[index + 1], state) : TemplateValue{});
                }
                state.scopes.push_back(std::move(arguments)); std::string output; render_nodes(macro->second.node->body, state, output); state.scopes.pop_back(); return TemplateValue{std::move(output)};
            }
        }
        if (target.kind == Expression::Kind::Access) {
            const TemplateValue object = eval(target.children[0], state);
            const std::string member = target.text;
            const std::string text = stringify(object);
            if (member == "strip") return TemplateValue{trim(text)};
            if (member == "lstrip" || member == "rstrip") {
                std::string output = text;
                if (member == "lstrip") while (!output.empty() && std::isspace(static_cast<unsigned char>(output.front()))) output.erase(output.begin());
                else while (!output.empty() && std::isspace(static_cast<unsigned char>(output.back()))) output.pop_back();
                return TemplateValue{std::move(output)};
            }
            if (member == "lower" || member == "upper") {
                std::string output = text; std::transform(output.begin(), output.end(), output.begin(), [&](unsigned char c) { return static_cast<char>(member == "lower" ? std::tolower(c) : std::toupper(c)); }); return TemplateValue{std::move(output)};
            }
            if (member == "items") {
                const auto* object_values = std::get_if<ValueObject>(&object.value); if (!object_values) throw_error(state, "items() requires an object");
                ValueList pairs; for (const auto& [key, value] : *object_values) pairs.emplace_back(ValueList{TemplateValue{key}, value}); return TemplateValue{std::move(pairs)};
            }
            if (member == "split") {
                const std::string separator = expression.children.size() > 1 ? stringify(eval(expression.children[1], state)) : " ";
                ValueList pieces;
                std::size_t offset = 0;
                while (offset <= text.size()) {
                    const std::size_t found = text.find(separator, offset);
                    pieces.emplace_back(text.substr(offset, found == std::string::npos ? std::string::npos : found - offset));
                    if (found == std::string::npos || separator.empty()) break;
                    offset = found + separator.size();
                }
                return TemplateValue{std::move(pieces)};
            }
            if (member == "replace") {
                if (expression.children.size() != 3) {
                    throw_error(state, "replace() requires old and new strings");
                }
                std::string output = text;
                const std::string from = stringify(eval(expression.children[1], state));
                const std::string to = stringify(eval(expression.children[2], state));
                std::size_t offset = 0;
                while (!from.empty() && (offset = output.find(from, offset)) != std::string::npos) {
                    output.replace(offset, from.size(), to);
                    offset += to.size();
                }
                return TemplateValue{std::move(output)};
            }
            if (member == "startswith" || member == "endswith") {
                const std::string needle = stringify(eval(expression.children.at(1), state));
                const bool matches = member == "startswith" ? text.starts_with(needle) : text.ends_with(needle);
                return TemplateValue{matches};
            }
        }
        throw_error(state, "unsupported Jinja call");
    }
    void render_nodes(const std::vector<TemplateNode>& nodes, RenderState& state, std::string& output) const {
        for (const TemplateNode& node : nodes) {
            state.current_line = node.line;
            switch (node.kind) {
            case TemplateNode::Kind::Text: output += node.text; break;
            case TemplateNode::Kind::Output: output += stringify(eval(node.expression, state)); break;
            case TemplateNode::Kind::Set:
                if (node.body.empty()) assign(node.text, eval(node.expression, state), state);
                else { std::string captured; render_nodes(node.body, state, captured); assign(node.text, TemplateValue{std::move(captured)}, state); }
                break;
            case TemplateNode::Kind::Macro: state.macros[node.text] = {&node}; break;
            case TemplateNode::Kind::If:
                for (const auto& [condition, body] : node.branches) if (truthy(eval(condition, state))) { render_nodes(body, state, output); goto done_if; }
                render_nodes(node.otherwise, state, output);
            done_if: break;
            case TemplateNode::Kind::For: {
                const TemplateValue sequence = eval(node.expression, state);
                const auto* values = std::get_if<ValueList>(&sequence.value);
                if (!values) throw_error(state, "Jinja for requires a list");
                bool rendered_any = false;
                for (std::size_t index = 0; index < values->size(); ++index) {
                    ValueObject scope;
                    const std::size_t comma = node.text.find(',');
                    if (comma == std::string::npos) {
                        scope.emplace(node.text, (*values)[index]);
                    } else {
                        const std::string first = trim(node.text.substr(0, comma));
                        const std::string second = trim(node.text.substr(comma + 1));
                        const auto* pair = std::get_if<ValueList>(&(*values)[index].value);
                        if (!pair || pair->size() != 2) throw_error(state, "Jinja destructuring loop requires pairs");
                        scope.emplace(first, (*pair)[0]); scope.emplace(second, (*pair)[1]);
                    }
                    ValueObject loop; loop.emplace("index0", static_cast<std::int64_t>(index)); loop.emplace("index", static_cast<std::int64_t>(index + 1));
                    loop.emplace("first", index == 0); loop.emplace("last", index + 1 == values->size()); loop.emplace("length", static_cast<std::int64_t>(values->size()));
                    if (index != 0) loop.emplace("previtem", (*values)[index - 1]);
                    if (index + 1 < values->size()) loop.emplace("nextitem", (*values)[index + 1]);
                    scope.emplace("loop", TemplateValue{std::move(loop)});
                    state.scopes.push_back(std::move(scope));
                    const bool selected = !node.condition || truthy(eval(*node.condition, state));
                    if (selected) render_nodes(node.body, state, output);
                    state.scopes.pop_back();
                    rendered_any = rendered_any || selected;
                }
                if (!rendered_any) render_nodes(node.otherwise, state, output);
                break;
            }
            }
        }
    }
    void assign(const std::string& target, TemplateValue value, RenderState& state) const {
        const std::size_t dot = target.find('.');
        if (dot == std::string::npos) { state.scopes.back()[target] = std::move(value); return; }
        const std::string root = target.substr(0, dot);
        TemplateValue* current = nullptr;
        for (auto scope = state.scopes.rbegin(); scope != state.scopes.rend() && !current; ++scope) {
            const auto found = scope->find(root);
            if (found != scope->end()) current = &found->second;
        }
        if (!current) throw_error(state, "undefined Jinja assignment target '" + root + "'");
        std::size_t begin = dot + 1;
        for (;;) {
            const std::size_t next = target.find('.', begin);
            const std::string member = target.substr(begin, next == std::string::npos ? std::string::npos : next - begin);
            auto* object = std::get_if<ValueObject>(&current->value);
            if (!object) throw_error(state, "Jinja assignment target is not an object");
            if (next == std::string::npos) { (*object)[member] = std::move(value); return; }
            current = &(*object)[member];
            begin = next + 1;
        }
    }
    [[noreturn]] static void throw_error(const RenderState& state, const std::string& message) {
        throw std::invalid_argument("UnsupportedChatTemplateConstruct at " + state.origin + ":" +
                                    std::to_string(state.current_line) + ": " + message);
    }
    std::vector<TemplateNode> nodes_;
};

std::string read_template_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::invalid_argument("cannot read --chat-template-file: " + path.string());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::string fingerprint(std::string_view source) {
    std::uint64_t value = 1469598103934665603ull;
    for (const unsigned char byte : source) { value ^= byte; value *= 1099511628211ull; }
    std::ostringstream out; out << std::hex << value; return out.str();
}

bool inferred_delimited_evidence(const ITokenizer& tokenizer) {
    return tokenizer.token_id("<|startoftext|>").has_value() &&
           tokenizer.token_id("<|im_start|>").has_value() &&
           tokenizer.token_id("<|im_end|>").has_value() &&
           tokenizer.token_id("assistant").has_value();
}

std::string inferred_delimited_source() {
    return "{{ bos_token }}{% for message in messages %}<|im_start|>{{ message.role }}\\n{{ message.content }}{% for call in message.tool_calls %}<|tool_call_start|>[{{ call.function.name }}({{ call.function.arguments }})]<|tool_call_end|>{% endfor %}<|im_end|>\\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\\n{% endif %}";
}

std::optional<ToolCallGrammar> derive_tool_grammar(std::string_view source) {
    const std::string_view opening = "<|tool_call_start|>";
    const std::string_view closing = "<|tool_call_end|>";
    const std::size_t begin = source.find(opening), end = source.find(closing);
    if (begin == std::string_view::npos || end == std::string_view::npos || begin >= end) {
        if (source.find("<tool_call>") == std::string_view::npos ||
            source.find("<function=") == std::string_view::npos ||
            source.find("</function>") == std::string_view::npos) return std::nullopt;
        ToolCallGrammar grammar;
        grammar.opening = "<tool_call>";
        grammar.closing = "</tool_call>";
        grammar.call_prefix = "<function=";
        grammar.call_suffix = "</function>";
        grammar.parallel_calls = source.find("for tool_call in message.tool_calls") != std::string_view::npos;
        grammar.xml_parameters = true;
        return grammar;
    }
    const std::string_view body = source.substr(begin + opening.size(), end - begin - opening.size());
    if (body.find("arguments") == std::string_view::npos || body.find("name") == std::string_view::npos) return std::nullopt;
    ToolCallGrammar grammar; grammar.opening = std::string(opening); grammar.closing = std::string(closing);
    grammar.call_prefix = body.find('[') != std::string_view::npos ? "[" : "";
    grammar.call_suffix = body.find(']') != std::string_view::npos ? "]" : "";
    grammar.parallel_calls = source.find("tool_calls") != std::string_view::npos;
    return grammar;
}

} // namespace

class InteractionRenderProgram final {
public:
    explicit InteractionRenderProgram(std::vector<TemplateNode> nodes) : implementation_(std::move(nodes)) {}
    std::string render(std::span<const ChatMessage> messages,
                       std::span<const ChatToolDefinition> tools,
                       bool generation_prompt,
                       const ChatTemplateOptions& options,
                       std::string_view origin,
                       std::string_view bos_token) const {
        return implementation_.render(messages, tools, generation_prompt, options, origin, bos_token);
    }
private:
    InteractionRenderProgramImpl implementation_;
};

ToolParseResult ToolCallGrammar::parse(std::string_view generated) const {
    ToolParseResult result;
    const std::size_t first = generated.find(opening);
    if (first == std::string_view::npos) { result.assistant_text = std::string(generated); return result; }
    result.assistant_text = std::string(generated.substr(0, first));
    std::size_t cursor = first;
    std::size_t call_index = 0;
    while (cursor < generated.size()) {
        if (!generated.substr(cursor).starts_with(opening)) { result.status = ToolParseStatus::Invalid; result.error = "unexpected text between tool calls"; return result; }
        cursor += opening.size();
        const std::size_t end = generated.find(closing, cursor);
        if (end == std::string_view::npos) { result.status = ToolParseStatus::Incomplete; result.consumed_bytes = cursor; return result; }
        std::string payload(generated.substr(cursor, end - cursor));
        if (xml_parameters) {
            if (!payload.starts_with(call_prefix)) {
                result.status = ToolParseStatus::Invalid;
                result.error = "tool call has no function tag";
                return result;
            }
            const std::size_t name_end = payload.find('>');
            const std::size_t function_end = payload.rfind(call_suffix);
            if (name_end == std::string::npos || function_end == std::string::npos || function_end < name_end) {
                result.status = ToolParseStatus::Incomplete;
                result.consumed_bytes = cursor;
                return result;
            }
            const std::string name = payload.substr(call_prefix.size(), name_end - call_prefix.size());
            std::string arguments = "{";
            std::size_t parameter = name_end + 1;
            bool first_parameter = true;
            while (parameter < function_end) {
                const std::size_t tag = payload.find("<parameter=", parameter);
                if (tag == std::string::npos || tag >= function_end) break;
                const std::size_t key_end = payload.find('>', tag);
                const std::size_t value_end = payload.find("</parameter>", key_end);
                if (key_end == std::string::npos || value_end == std::string::npos || value_end > function_end) {
                    result.status = ToolParseStatus::Incomplete;
                    result.consumed_bytes = cursor + tag;
                    return result;
                }
                if (!first_parameter) arguments += ',';
                first_parameter = false;
                arguments += json_string(payload.substr(tag + std::string_view("<parameter=").size(), key_end - tag - std::string_view("<parameter=").size()));
                arguments += ':' + json_string(trim(payload.substr(key_end + 1, value_end - key_end - 1)));
                parameter = value_end + std::string_view("</parameter>").size();
            }
            arguments += '}';
            result.calls.push_back({"call_" + std::to_string(call_index++), name, std::move(arguments)});
            cursor = end + closing.size();
            if (!parallel_calls) break;
            continue;
        }
        if (!call_prefix.empty()) {
            if (!payload.starts_with(call_prefix) || !payload.ends_with(call_suffix)) { result.status = ToolParseStatus::Invalid; result.error = "tool call delimiters do not match grammar"; return result; }
            payload = payload.substr(call_prefix.size(), payload.size() - call_prefix.size() - call_suffix.size());
        }
        const std::size_t open = payload.find('('), close = payload.rfind(')');
        if (open == std::string::npos || close != payload.size() - 1 || open == 0) { result.status = ToolParseStatus::Invalid; result.error = "tool call is not name(arguments)"; return result; }
        result.calls.push_back({"call_" + std::to_string(call_index++), payload.substr(0, open), payload.substr(open + 1, close - open - 1)});
        cursor = end + closing.size();
        if (!parallel_calls) break;
    }
    result.status = result.calls.empty() ? ToolParseStatus::NotToolCall : ToolParseStatus::Complete;
    result.consumed_bytes = cursor;
    return result;
}

ResolvedInteraction resolve_interaction(const CheckpointMetadata& metadata,
                                        const ITokenizer& tokenizer,
                                        const std::optional<std::filesystem::path>& override_file,
                                        const std::optional<std::filesystem::path>& companion_file) {
    std::string source;
    std::string origin;
    if (override_file) {
        source = read_template_file(*override_file);
        origin = "override:" + override_file->string();
    } else {
        source = metadata.string_or("chat_template", metadata.string_or("tokenizer.chat_template", {}));
        if (source.empty() && companion_file && std::filesystem::is_regular_file(*companion_file)) {
            source = read_template_file(*companion_file);
            origin = "checkpoint-companion:" + companion_file->string();
        } else {
            origin = source.empty() ? "tokenizer-inference" : "checkpoint-metadata";
        }
    }
    ResolvedInteraction result;
    result.source_origin_ = origin;
    if (source.empty()) {
        if (!inferred_delimited_evidence(tokenizer)) {
            throw std::invalid_argument("no chat template metadata and tokenizer does not structurally prove BOS, role delimiters, turn terminator, and assistant generation prefix; supply --chat-template-file");
        }
        source = inferred_delimited_source();
        result.diagnostics_.push_back("inferred role-delimited interaction from tokenizer evidence: <|startoftext|>, <|im_start|>, <|im_end|>, assistant");
    }
    if (source.find("__") != std::string::npos || source.find("{% include") != std::string::npos ||
        source.find("{% import") != std::string::npos || source.find("{% from") != std::string::npos) {
        throw std::invalid_argument("UnsupportedChatTemplateConstruct at " + origin + ": imports, introspection, and I/O are not permitted");
    }
    // Jinja's reverse slice is a pure list operation; normalize it to the
    // equivalent safe filter so the compiler needs no Python-style stride
    // semantics at the template boundary.
    std::string normalized_source = source;
    for (std::size_t position = 0; (position = normalized_source.find("[::-1]", position)) != std::string::npos;) {
        normalized_source.replace(position, std::string_view("[::-1]").size(), "|reverse");
        position += std::string_view("|reverse").size();
    }
    const std::vector<TemplateNode> nodes = TemplateParser(normalized_source, origin).parse();
    // Validation walks every branch before a model can render.  A supplied
    // template therefore cannot hide an unsupported construct behind a false
    // condition and silently fall through to a different formatting path.
    validate_program(nodes, origin);
    result.render_program_ = std::make_shared<InteractionRenderProgram>(nodes);
    result.fingerprint_ = fingerprint(source);
    result.bos_token_ = tokenizer.decode(std::vector<std::int32_t>{tokenizer.bos_id()}, false);
    result.capabilities_.roles.developer = source.find("developer") != std::string::npos;
    result.capabilities_.roles.tool = source.find("tool") != std::string::npos;
    result.tool_call_grammar_ = derive_tool_grammar(source);
    // Rendering a tools declaration is not sufficient to expose an API tool
    // capability.  The completion side must have a statically proven grammar
    // so callers never receive opaque model text as a purported tool call.
    result.capabilities_.assistant_tool_calls = result.tool_call_grammar_.has_value();
    result.capabilities_.parallel_tool_calls = result.tool_call_grammar_ && result.tool_call_grammar_->parallel_calls;
    result.capabilities_.native_tool_call_codec = result.tool_call_grammar_.has_value();
    return result;
}

std::string ResolvedInteraction::format(std::span<const ChatMessage> messages,
                                        bool add_generation_prompt) const {
    return format(messages, {}, add_generation_prompt, {});
}

std::string ResolvedInteraction::format(std::span<const ChatMessage> messages,
                                        std::span<const ChatToolDefinition> tools,
                                        bool add_generation_prompt,
                                        const ChatTemplateOptions& options) const {
    if (!render_program_) throw std::logic_error("resolved interaction has no render program");
    validate_conversation(messages);
    return render_program_->render(messages, tools, add_generation_prompt, options, source_origin_, bos_token_);
}

ToolParseResult ResolvedInteraction::parse_tool_calls(std::string_view generated) const {
    if (!tool_call_grammar_) return {ToolParseStatus::NotToolCall, std::string(generated)};
    return tool_call_grammar_->parse(generated);
}

std::string ResolvedInteraction::forced_tool_call_prefix(std::span<const ChatToolDefinition>, const ToolChoice& choice) const {
    if (!tool_call_grammar_ || (choice.mode != ToolChoiceMode::Required && choice.mode != ToolChoiceMode::Specific)) return {};
    return tool_call_grammar_->required_prefix;
}

std::string render_chat(std::span<const ChatMessage> messages,
                        const ResolvedInteraction& interaction,
                        bool add_generation_prompt) {
    return interaction.format(messages, add_generation_prompt);
}

std::string render_chat(std::span<const ChatMessage> messages,
                        std::span<const ChatToolDefinition> tools,
                        const ResolvedInteraction& interaction,
                        bool add_generation_prompt,
                        const ChatTemplateOptions& options) {
    return interaction.format(messages, tools, add_generation_prompt, options);
}

} // namespace celeg
