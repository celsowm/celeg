#pragma once

#include <glaze/glaze.hpp>

#include <stdexcept>
#include <string>
#include <string_view>

namespace lfm::serve::protocol {

template <typename T>
std::string to_json(const T& value) {
    std::string out;
    if (auto ec = glz::write_json(value, out)) {
        throw std::runtime_error("JSON serialization failed: " + glz::format_error(ec, out));
    }
    return out;
}

template <typename T>
T from_json(std::string_view text) {
    T value{};
    if (auto ec = glz::read_json(value, text)) {
        throw std::runtime_error("JSON parse error: " + glz::format_error(ec, text));
    }
    return value;
}

} // namespace lfm::serve::protocol
