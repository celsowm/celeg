#include "detail.hpp"

#include <stdexcept>
#include <utility>

namespace celeg::descriptor_detail {

std::vector<BindingPattern> parse_bindings(const Json& object) {
    if (!object.is_object() || !object.contains("bindings")) {
        throw std::invalid_argument("descriptor is missing field: bindings");
    }
    std::vector<BindingPattern> result;
    for (const auto& [name, values] : object.at("bindings").as_object()) {
        BindingPattern pattern{parse_role(name), {}};
        for (const Json& candidate : values.as_array()) {
            pattern.candidates.push_back(candidate.as_string());
        }
        if (pattern.candidates.empty()) {
            throw std::invalid_argument("descriptor binding has no candidates");
        }
        result.push_back(std::move(pattern));
    }
    return result;
}

}
