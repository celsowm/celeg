#include "access.hpp"

#include "../support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <regex>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace celeg {

AttentionPatternKind parse_attention_pattern(std::string_view value,
                                              std::string_view source) {
    if (value == "full_attention" || value == "full" || value == "causal" ||
        value == "full_causal" || value == "agnes_global_attention") {
        return AttentionPatternKind::FullCausal;
    }
    if (value == "sliding_attention" || value == "sliding_window" ||
        value == "sliding") {
        return AttentionPatternKind::SlidingWindow;
    }
    /// Non-attention mixer tokens describe layers whose mixer is a convolutional
    /// or recurrent primitive (resolved by the dedicated inference rules), not an
    /// attention pattern. Treat them as "no attention pattern" rather than an
    /// error so hybrid schedules (e.g. conv/attention interleaving) resolve.
    if (value == "conv" || value == "short_convolution" ||
        value == "recurrent" || value == "mamba" || value == "mamba2" ||
        value == "gdn" || value == "gated_delta" ||
        value == "gated_delta_net" || value == "linear_attention" ||
        value == "agnes_delta_attention") {
        return AttentionPatternKind::None;
    }
    inference_detail::fail(
        ResolutionFailureKind::UnsupportedSemanticFeature,
        "unknown attention layer pattern token in " + std::string(source) + ": " +
            std::string(value));
}

const MetadataValue* metadata_alias(const CheckpointMetadata& metadata,
                                    std::string_view key) {
    if (metadata.contains(key)) {
        inference_detail::record_metadata_consumption(key);
        return &metadata.value(key);
    }
    const std::string text_key = "text_config." + std::string(key);
    if (metadata.contains(text_key)) {
        inference_detail::record_metadata_consumption(text_key);
        return &metadata.value(text_key);
    }
    return nullptr;
}

std::optional<int> integer_alias(const CheckpointMetadata& metadata,
                                 std::string_view key) {
    const MetadataValue* value = metadata_alias(metadata, key);
    if (value == nullptr) return std::nullopt;
    if (const auto* integer = std::get_if<int64_t>(value)) {
        if (*integer < std::numeric_limits<int>::min() ||
            *integer > std::numeric_limits<int>::max()) {
            inference_detail::fail(
                ResolutionFailureKind::ConflictingMetadata,
                "attention metadata integer is outside the supported range: " +
                    std::string(key));
        }
        return static_cast<int>(*integer);
    }
    inference_detail::fail(
        ResolutionFailureKind::ConflictingMetadata,
        "attention metadata key has an incompatible type: " + std::string(key));
}

std::optional<double> numeric_alias(const CheckpointMetadata& metadata,
                                    std::string_view key) {
    const MetadataValue* value = metadata_alias(metadata, key);
    if (value == nullptr) return std::nullopt;
    if (const auto* number = std::get_if<double>(value)) {
        if (!std::isfinite(*number)) {
            inference_detail::fail(
                ResolutionFailureKind::ConflictingMetadata,
                "RoPE metadata value is not finite: " + std::string(key));
        }
        return *number;
    }
    if (const auto* integer = std::get_if<int64_t>(value)) {
        return static_cast<double>(*integer);
    }
    inference_detail::fail(
        ResolutionFailureKind::ConflictingMetadata,
        "RoPE metadata key has an incompatible numeric type: " + std::string(key));
}

std::optional<std::string> string_alias(const CheckpointMetadata& metadata,
                                        std::string_view key) {
    const MetadataValue* value = metadata_alias(metadata, key);
    if (value == nullptr) return std::nullopt;
    if (const auto* text = std::get_if<std::string>(value)) return *text;
    inference_detail::fail(
        ResolutionFailureKind::ConflictingMetadata,
        "RoPE metadata key has an incompatible string type: " + std::string(key));
}

std::optional<std::string> string_aliases(
    const CheckpointMetadata& metadata,
    std::initializer_list<std::string_view> keys,
    std::string_view fact) {
    std::optional<std::string> result;
    for (const std::string_view key : keys) {
        const auto value = string_alias(metadata, key);
        if (!value.has_value()) continue;
        if (result.has_value() && *result != *value) {
            inference_detail::fail(
                ResolutionFailureKind::ConflictingMetadata,
                "conflicting RoPE metadata aliases for " + std::string(fact));
        }
        result = value;
    }
    return result;
}

std::optional<double> numeric_aliases(
    const CheckpointMetadata& metadata,
    std::initializer_list<std::string_view> keys,
    std::string_view fact) {
    std::optional<double> result;
    for (const std::string_view key : keys) {
        const auto value = numeric_alias(metadata, key);
        if (!value.has_value()) continue;
        if (result.has_value() && *result != *value) {
            inference_detail::fail(
                ResolutionFailureKind::ConflictingMetadata,
                "conflicting RoPE metadata aliases for " + std::string(fact));
        }
        result = value;
    }
    return result;
}

std::optional<int> integer_aliases(
    const CheckpointMetadata& metadata,
    std::initializer_list<std::string_view> keys,
    std::string_view fact) {
    std::optional<int> result;
    for (const std::string_view key : keys) {
        const auto value = integer_alias(metadata, key);
        if (!value.has_value()) continue;
        if (result.has_value() && *result != *value) {
            inference_detail::fail(
                ResolutionFailureKind::ConflictingMetadata,
                "conflicting RoPE metadata aliases for " + std::string(fact));
        }
        result = value;
    }
    return result;
}

}
