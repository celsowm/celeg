#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace celeg::chat_template_detail {

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
    explicit ExpressionLexer(std::string_view source);

    const LexerToken& current() const;
    void next();

private:
    void skip_space();

    std::string_view source_;
    std::size_t position_ = 0;
    LexerToken current_;
};

}
