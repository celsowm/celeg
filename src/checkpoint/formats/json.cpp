#include "celeg/checkpoint/formats/json.hpp"

#include <charconv>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace celeg {
namespace {

void append_utf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    Json parse() {
        skip_ws();
        Json result = parse_value();
        skip_ws();
        if (pos_ != text_.size()) fail("trailing characters");
        return result;
    }

private:
    [[noreturn]] void fail(const std::string& message) const {
        throw std::runtime_error("JSON parse error at byte " + std::to_string(pos_) + ": " + message);
    }

    void skip_ws() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    }

    char take() {
        if (pos_ >= text_.size()) fail("unexpected end of input");
        return text_[pos_++];
    }

    bool consume(char expected) {
        if (pos_ < text_.size() && text_[pos_] == expected) {
            ++pos_;
            return true;
        }
        return false;
    }

    Json parse_value() {
        skip_ws();
        if (pos_ >= text_.size()) fail("expected value");
        switch (text_[pos_]) {
            case 'n': parse_literal("null"); return Json(nullptr);
            case 't': parse_literal("true"); return Json(true);
            case 'f': parse_literal("false"); return Json(false);
            case '"': return Json(parse_string());
            case '[': return Json(parse_array());
            case '{': return Json(parse_object());
            default: return Json(parse_number());
        }
    }

    void parse_literal(std::string_view literal) {
        if (text_.substr(pos_, literal.size()) != literal) fail("invalid literal");
        pos_ += literal.size();
    }

    uint32_t parse_hex4() {
        if (pos_ + 4 > text_.size()) fail("incomplete unicode escape");
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = text_[pos_++];
            value <<= 4;
            if (c >= '0' && c <= '9') value |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') value |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') value |= static_cast<uint32_t>(c - 'A' + 10);
            else fail("invalid unicode escape");
        }
        return value;
    }

    std::string parse_string() {
        if (take() != '"') fail("expected string");
        std::string out;
        while (true) {
            const char c = take();
            if (c == '"') break;
            if (static_cast<unsigned char>(c) < 0x20) fail("control character in string");
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            const char esc = take();
            switch (esc) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = parse_hex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (take() != '\\' || take() != 'u') fail("expected low surrogate");
                        const uint32_t low = parse_hex4();
                        if (low < 0xDC00 || low > 0xDFFF) fail("invalid low surrogate");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: fail("unknown escape");
            }
        }
        return out;
    }

    double parse_number() {
        const size_t start = pos_;
        if (consume('-')) {}
        if (consume('0')) {
        } else {
            if (pos_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[pos_]))) fail("invalid number");
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        }
        if (consume('.')) {
            if (pos_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[pos_]))) fail("invalid fraction");
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
            if (pos_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[pos_]))) fail("invalid exponent");
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        }
        std::string tmp(text_.substr(start, pos_ - start));
        char* end = nullptr;
        const double result = std::strtod(tmp.c_str(), &end);
        if (end != tmp.c_str() + tmp.size()) fail("invalid number");
        return result;
    }

    Json::array_t parse_array() {
        take();
        Json::array_t out;
        skip_ws();
        if (consume(']')) return out;
        while (true) {
            out.push_back(parse_value());
            skip_ws();
            if (consume(']')) break;
            if (!consume(',')) fail("expected ',' in array");
        }
        return out;
    }

    Json::object_t parse_object() {
        take();
        Json::object_t out;
        skip_ws();
        if (consume('}')) return out;
        while (true) {
            skip_ws();
            if (pos_ >= text_.size() || text_[pos_] != '"') fail("expected object key");
            std::string key = parse_string();
            skip_ws();
            if (!consume(':')) fail("expected ':' after object key");
            out.emplace(std::move(key), parse_value());
            skip_ws();
            if (consume('}')) break;
            if (!consume(',')) fail("expected ',' in object");
        }
        return out;
    }

    std::string_view text_;
    size_t pos_ = 0;
};

}

Json Json::parse(std::string_view text) { return Parser(text).parse(); }

Json Json::parse_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open JSON file: " + path);
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return parse(buffer.str());
}

bool Json::is_null() const { return std::holds_alternative<std::nullptr_t>(value_); }
bool Json::is_bool() const { return std::holds_alternative<bool>(value_); }
bool Json::is_number() const { return std::holds_alternative<double>(value_); }
bool Json::is_string() const { return std::holds_alternative<std::string>(value_); }
bool Json::is_array() const { return std::holds_alternative<array_t>(value_); }
bool Json::is_object() const { return std::holds_alternative<object_t>(value_); }

bool Json::as_bool() const { return std::get<bool>(value_); }
double Json::as_number() const { return std::get<double>(value_); }
int64_t Json::as_i64() const { return static_cast<int64_t>(as_number()); }
const std::string& Json::as_string() const { return std::get<std::string>(value_); }
const Json::array_t& Json::as_array() const { return std::get<array_t>(value_); }
const Json::object_t& Json::as_object() const { return std::get<object_t>(value_); }

bool Json::contains(std::string_view key) const {
    if (!is_object()) return false;
    return as_object().find(std::string(key)) != as_object().end();
}

const Json& Json::at(std::string_view key) const {
    const auto& object = as_object();
    const auto it = object.find(std::string(key));
    if (it == object.end()) throw std::out_of_range("missing JSON key: " + std::string(key));
    return it->second;
}

}
