#include "detail.hpp"

namespace celeg::descriptor_detail {

bool equal_text(std::string_view left, std::string_view right, bool insensitive) {
    if (!insensitive) return left == right;
    if (left.size() != right.size()) return false;
    for (size_t i = 0; i < left.size(); ++i) {
        const auto lower = [](char c) { return static_cast<char>(c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c); };
        if (lower(left[i]) != lower(right[i])) return false;
    }
    return true;
}

bool probe_condition(const CheckpointMetadata& metadata, const ProbeCondition& condition) {
    if (condition.key == "architecture_type") {
        return condition.equals.empty() || equal_text(metadata.architecture_type(), condition.equals,
                                                      condition.case_insensitive);
    }
    if (condition.key == "repository_hint") {
        return condition.contains.empty() || metadata.repository_hint.find(condition.contains) != std::string::npos;
    }
    if (condition.key == "source_format") {
        const std::string format = metadata.is_gguf() ? "gguf" : "safetensors";
        return condition.equals.empty() || equal_text(format, condition.equals,
                                                       condition.case_insensitive);
    }
    const std::string key = metadata.is_gguf()
        ? (condition.gguf.empty() ? (condition.key == "integer" ? "" : condition.key)
                                  : condition.gguf)
        : (condition.json.empty() ? (condition.key == "integer" ? "" : condition.key)
                                  : condition.json);
    if (condition.has_integer_equals) {
        return metadata.contains(key) && metadata.integer(key) == condition.integer_equals;
    }
    if (!metadata.contains(key)) return false;
    if (!condition.equals.empty()) return equal_text(metadata.string(key), condition.equals,
                                                     condition.case_insensitive);
    if (!condition.contains.empty()) return metadata.string(key).find(condition.contains) != std::string::npos;
    return true;
}


}
