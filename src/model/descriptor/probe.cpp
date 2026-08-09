#include "detail.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace celeg::descriptor_detail {

std::vector<ProbeCondition> parse_probe_conditions(const Json& value) {
    std::vector<ProbeCondition> result;
    for (const Json& item : value.as_array()) {
        ProbeCondition condition;
        condition.key = item.is_object() && item.contains("key")
            ? item.at("key").as_string() : std::string{};
        condition.json = item.is_object() && item.contains("json")
            ? item.at("json").as_string() : std::string{};
        condition.gguf = item.is_object() && item.contains("gguf")
            ? item.at("gguf").as_string() : std::string{};
        condition.equals = item.contains("equals") && item.at("equals").is_string()
            ? item.at("equals").as_string() : std::string{};
        condition.contains = item.is_object() && item.contains("contains")
            ? item.at("contains").as_string() : std::string{};
        condition.case_insensitive = item.is_object() && item.contains("case_insensitive")
            ? item.at("case_insensitive").as_bool() : false;
        if (condition.key == "integer") {
            if (!item.contains("equals") || !item.at("equals").is_number()) {
                throw std::invalid_argument("integer descriptor probe has no numeric equals");
            }
            condition.has_integer_equals = true;
            condition.integer_equals = static_cast<int>(item.at("equals").as_i64());
        }
        if (condition.key.empty() && condition.json.empty() && condition.gguf.empty()) {
            throw std::invalid_argument("descriptor probe condition has no key");
        }
        result.push_back(std::move(condition));
    }
    return result;
}

} // namespace celeg::descriptor_detail
