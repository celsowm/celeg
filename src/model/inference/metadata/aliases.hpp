#pragma once

#include "celeg/model/inference.hpp"

#include "../support.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace celeg {

template <typename T>
std::optional<T> scalar(const CheckpointMetadata& metadata, std::string_view key) {
    if (!metadata.contains(key)) return std::nullopt;
    /// Record here, not just in the `aliases` wrapper: this is the single
    /// choke point every metadata read flows through, so a present key the
    /// resolver probes is never mistaken for an unread one by the ledger.
    inference_detail::record_metadata_consumption(key);
    const MetadataValue& value = metadata.value(key);
    if constexpr (std::is_same_v<T, int>) {
        if (const auto* integer = std::get_if<std::int64_t>(&value)) {
            if (*integer < std::numeric_limits<int>::min() ||
                *integer > std::numeric_limits<int>::max()) {
                inference_detail::fail(
                    ResolutionFailureKind::ConflictingMetadata,
                    "metadata value is outside the supported integer range: " +
                        std::string(key));
            }
            return static_cast<int>(*integer);
        }
        if (const auto* number = std::get_if<double>(&value)) {
            if (!std::isfinite(*number) || *number < std::numeric_limits<int>::min() ||
                *number > std::numeric_limits<int>::max() ||
                std::floor(*number) != *number) {
                inference_detail::fail(
                    ResolutionFailureKind::ConflictingMetadata,
                    "metadata integer is invalid: " + std::string(key));
            }
            return static_cast<int>(*number);
        }
        if (const auto* boolean = std::get_if<bool>(&value)) return *boolean ? 1 : 0;
    } else if constexpr (std::is_same_v<T, float>) {
        if (const auto* number = std::get_if<double>(&value)) {
            if (!std::isfinite(*number) ||
                *number < -std::numeric_limits<float>::max() ||
                *number > std::numeric_limits<float>::max()) {
                inference_detail::fail(
                    ResolutionFailureKind::ConflictingMetadata,
                    "metadata floating-point value is invalid: " + std::string(key));
            }
            return static_cast<float>(*number);
        }
        if (const auto* integer = std::get_if<std::int64_t>(&value)) {
            return static_cast<float>(*integer);
        }
    } else if constexpr (std::is_same_v<T, double>) {
        if (const auto* number = std::get_if<double>(&value)) return *number;
        if (const auto* integer = std::get_if<std::int64_t>(&value)) {
            return static_cast<double>(*integer);
        }
    } else if constexpr (std::is_same_v<T, bool>) {
        if (const auto* boolean = std::get_if<bool>(&value)) return *boolean;
        if (const auto* integer = std::get_if<std::int64_t>(&value)) return *integer != 0;
    }
    inference_detail::fail(ResolutionFailureKind::ConflictingMetadata,
                           "metadata key has an incompatible type: " + std::string(key));
}

template <typename T>
std::optional<T> gguf_scalar_or_uniform_schedule(const CheckpointMetadata& metadata,
                                                 std::string_view key) {
    if (!metadata.contains(key)) return std::nullopt;
    /// Same choke-point recording as `scalar`: direct callers (e.g. the
    /// `<arch>.rope.dimension_count` read) consume the key too.
    inference_detail::record_metadata_consumption(key);
    const MetadataValue& value = metadata.value(key);
    if (const auto* values = std::get_if<std::vector<std::int64_t>>(&value)) {
        if (values->empty() || !std::all_of(values->begin() + 1, values->end(),
                                             [&](std::int64_t item) {
                                                 return item == values->front();
                                             })) {
            inference_detail::fail(
                ResolutionFailureKind::UnsupportedSemanticFeature,
                "GGUF metadata " + std::string(key) +
                    " varies by layer; automatic dense synthesis requires a uniform value");
        }
        if constexpr (std::is_same_v<T, int>) return static_cast<int>(values->front());
        if constexpr (std::is_same_v<T, float>) return static_cast<float>(values->front());
        if constexpr (std::is_same_v<T, double>) return static_cast<double>(values->front());
    }
    if (const auto* values = std::get_if<std::vector<double>>(&value)) {
        if (values->empty() || !std::all_of(values->begin() + 1, values->end(),
                                             [&](double item) {
                                                 return item == values->front();
                                             })) {
            inference_detail::fail(
                ResolutionFailureKind::UnsupportedSemanticFeature,
                "GGUF metadata " + std::string(key) +
                    " varies by layer; automatic dense synthesis requires a uniform value");
        }
        if constexpr (std::is_same_v<T, float>) return static_cast<float>(values->front());
        if constexpr (std::is_same_v<T, double>) return values->front();
    }
    return scalar<T>(metadata, key);
}

template <typename T>
LayerScopedValue<T> scoped_aliases(const CheckpointMetadata& metadata,
                                   std::initializer_list<std::string_view> keys,
                                   std::vector<EvidenceItem>& evidence,
                                   std::string_view fact,
                                   std::string_view gguf_suffix = {}) {
    LayerScopedValue<T> result;
    std::string source;
    const auto consider_scalar = [&](std::optional<T> value, std::string_view key) {
        if (!value.has_value()) return;
        inference_detail::record_metadata_consumption(key);
        if (result.global.has_value() && *result.global != *value) {
            inference_detail::fail(ResolutionFailureKind::ConflictingMetadata,
                                   "conflicting metadata aliases for " + std::string(fact));
        }
        result.global = value;
        source = key;
        if (!result.per_layer.empty()) {
            for (const auto& layer_value : result.per_layer) {
                if (layer_value.has_value() && *layer_value != *value) {
                    inference_detail::fail(
                        ResolutionFailureKind::ConflictingMetadata,
                        "conflicting global and layer-scoped metadata for " +
                            std::string(fact));
                }
            }
        }
    };
    const auto consider_vector = [&](const MetadataValue& metadata_value,
                                     std::string_view key) {
        inference_detail::record_metadata_consumption(key);
        std::vector<std::optional<T>> values;
        if (const auto* integers = std::get_if<std::vector<int64_t>>(&metadata_value)) {
            values.reserve(integers->size());
            for (const std::int64_t value : *integers) {
                if constexpr (std::is_same_v<T, int>) {
                    if (value < std::numeric_limits<int>::min() ||
                        value > std::numeric_limits<int>::max()) {
                        inference_detail::fail(
                            ResolutionFailureKind::ConflictingMetadata,
                            "layer-scoped integer is outside the supported range: " +
                                std::string(key));
                    }
                    values.push_back(static_cast<int>(value));
                } else {
                    values.push_back(static_cast<T>(value));
                }
            }
        } else if (const auto* numbers = std::get_if<std::vector<double>>(&metadata_value)) {
            values.reserve(numbers->size());
            for (const double value : *numbers) {
                if (!std::isfinite(value)) {
                    inference_detail::fail(ResolutionFailureKind::ConflictingMetadata,
                                           "layer-scoped number is invalid: " +
                                               std::string(key));
                }
                if constexpr (std::is_same_v<T, int>) {
                    if (std::floor(value) != value ||
                        value < std::numeric_limits<int>::min() ||
                        value > std::numeric_limits<int>::max()) {
                        inference_detail::fail(
                            ResolutionFailureKind::ConflictingMetadata,
                            "layer-scoped integer is invalid: " + std::string(key));
                    }
                    values.push_back(static_cast<int>(value));
                } else {
                    values.push_back(static_cast<T>(value));
                }
            }
        } else {
            return;
        }
        if (!result.per_layer.empty() && result.per_layer != values) {
            inference_detail::fail(ResolutionFailureKind::ConflictingMetadata,
                                   "conflicting layer-scoped metadata for " +
                                       std::string(fact));
        }
        if (result.global.has_value()) {
            for (const auto& layer_value : values) {
                if (layer_value.has_value() && *layer_value != *result.global) {
                    inference_detail::fail(
                        ResolutionFailureKind::ConflictingMetadata,
                        "conflicting global and layer-scoped metadata for " +
                            std::string(fact));
                }
            }
        }
        result.per_layer = std::move(values);
        source = key;
    };
    const auto consider = [&](std::string_view key) {
        if (!metadata.contains(key)) return;
        const MetadataValue& value = metadata.value(key);
        if (std::holds_alternative<std::vector<int64_t>>(value) ||
            std::holds_alternative<std::vector<double>>(value)) {
            consider_vector(value, key);
        } else {
            consider_scalar(scalar<T>(metadata, key), key);
        }
    };
    for (const std::string_view key : keys) {
        consider(key);
        consider("text_config." + std::string(key));
    }
    if (metadata.is_gguf() && !gguf_suffix.empty()) {
        consider(metadata.architecture_type() + "." + std::string(gguf_suffix));
    }
    if (result.has_value()) {
        evidence.push_back({EvidenceKind::AliasMetadata, source,
                            std::string(fact) + (result.per_layer.empty()
                                ? " = " + std::to_string(*result.global)
                                : " = layer-scoped schedule")});
    }
    return result;
}
template <typename T>
std::optional<T> aliases(const CheckpointMetadata& metadata,
                         std::initializer_list<std::string_view> keys,
                         std::vector<EvidenceItem>& evidence,
                         std::string_view fact,
                         std::string_view gguf_suffix = {}) {
    std::optional<T> result;
    std::string source;
    const auto consider = [&](std::optional<T> value, std::string_view key) {
        if (!value.has_value()) return;
        inference_detail::record_metadata_consumption(key);
        if (result.has_value() && *result != *value) {
            inference_detail::fail(ResolutionFailureKind::ConflictingMetadata,
                                   "conflicting metadata aliases for " + std::string(fact));
        }
        result = value;
        source = key;
    };
    for (const std::string_view key : keys) {
        consider(scalar<T>(metadata, key), key);
        const std::string component_key = "text_config." + std::string(key);
        consider(scalar<T>(metadata, component_key), component_key);
    }
    if (metadata.is_gguf() && !gguf_suffix.empty()) {
        const std::string gguf_key = metadata.architecture_type() + "." +
            std::string(gguf_suffix);
        consider(gguf_scalar_or_uniform_schedule<T>(metadata, gguf_key), gguf_key);
    }
    if (result.has_value()) {
        evidence.push_back({EvidenceKind::AliasMetadata, source,
                            std::string(fact) + " = " + std::to_string(*result)});
    }
    return result;
}

void validate_scoped_alias(const LayerScopedValue<int>& value,
                           const std::optional<int>& layer_count,
                           std::string_view fact);

std::optional<ActivationKind> feed_forward_activation(
    const CheckpointMetadata& metadata, std::vector<EvidenceItem>& evidence);

std::optional<int> tokenizer_vocabulary_size(const CheckpointMetadata& metadata,
                                             std::vector<EvidenceItem>& evidence);

std::vector<int> token_list(const CheckpointMetadata& metadata, std::string_view key);

}
