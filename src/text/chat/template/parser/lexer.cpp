#include "lexer.hpp"

#include <cctype>
#include <stdexcept>
#include <utility>

namespace celeg::chat_template_detail {

ExpressionLexer::ExpressionLexer(std::string_view source) : source_(source) {
    next();
}

const LexerToken& ExpressionLexer::current() const {
    return current_;
}

void ExpressionLexer::next() {
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

void ExpressionLexer::skip_space() {
            while (position_ < source_.size() &&
                   std::isspace(static_cast<unsigned char>(source_[position_]))) {
                ++position_;
            }
        }

}
