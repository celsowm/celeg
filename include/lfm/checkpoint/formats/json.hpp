#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace lfm {

class Json {
public:
    using array_t = std::vector<Json>;
    using object_t = std::unordered_map<std::string, Json>;
    using value_t = std::variant<std::nullptr_t, bool, double, std::string, array_t, object_t>;

    Json() : value_(nullptr) {}
    explicit Json(value_t value) : value_(std::move(value)) {}

    static Json parse(std::string_view text);
    static Json parse_file(const std::string& path);

    bool is_null() const;
    bool is_bool() const;
    bool is_number() const;
    bool is_string() const;
    bool is_array() const;
    bool is_object() const;

    bool as_bool() const;
    double as_number() const;
    int64_t as_i64() const;
    const std::string& as_string() const;
    const array_t& as_array() const;
    const object_t& as_object() const;

    bool contains(std::string_view key) const;
    const Json& at(std::string_view key) const;
    const Json& operator[](std::string_view key) const { return at(key); }

private:
    value_t value_;
};

} // namespace lfm
