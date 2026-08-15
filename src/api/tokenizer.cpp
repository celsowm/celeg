#include "api_internal.hpp"
#include "celeg/text/tokenizer_definition.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

extern "C" {

celeg_tokenizer* celeg_tokenizer_create(const char* path) {
    if (!path || !*path) return nullptr;
    try {
        auto result = std::make_unique<celeg_tokenizer>();
        result->value = std::make_unique<celeg::BpeTokenizer>(
            celeg::load_tokenizer_definition_json(path));
        return result.release();
    } catch (const std::exception& error) {
        celeg::api::global_error = error.what();
        return nullptr;
    }
}

void celeg_tokenizer_destroy(celeg_tokenizer* tokenizer) { delete tokenizer; }

celeg_status celeg_tokenizer_encode(celeg_tokenizer* tokenizer, const char* text,
                                    int add_bos, int32_t* output, size_t capacity,
                                    size_t* required) {
    if (!tokenizer || !text || !required) return CELEG_STATUS_INVALID_ARGUMENT;
    return celeg::api::protect(tokenizer, [&] {
        const auto values = tokenizer->value->encode(text, add_bos != 0);
        *required = values.size();
        if (!output || capacity < values.size()) {
            throw std::length_error("token output buffer is too small");
        }
        std::copy(values.begin(), values.end(), output);
    });
}

celeg_status celeg_tokenizer_decode(celeg_tokenizer* tokenizer,
                                    const int32_t* tokens, size_t count,
                                    int skip_special, char* output, size_t capacity,
                                    size_t* required) {
    if (!tokenizer || (!tokens && count) || !required) {
        return CELEG_STATUS_INVALID_ARGUMENT;
    }
    return celeg::api::protect(tokenizer, [&] {
        const std::vector<int32_t> values = count == 0
            ? std::vector<int32_t>{}
            : std::vector<int32_t>(tokens, tokens + count);
        const auto text = tokenizer->value->decode(values, skip_special != 0);
        *required = text.size() + 1;
        if (!output || capacity < *required) {
            throw std::length_error("text output buffer is too small");
        }
        std::memcpy(output, text.c_str(), *required);
    });
}

const char* celeg_tokenizer_last_error(const celeg_tokenizer* tokenizer) {
    return tokenizer ? tokenizer->error.c_str() : celeg::api::global_error.c_str();
}

}
