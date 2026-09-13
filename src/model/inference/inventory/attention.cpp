#include "access.hpp"
#include "detail.hpp"

#include "../normalize_internal.hpp"
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

LayerScopedValue<AttentionPatternKind> attention_pattern_metadata(
    const CheckpointMetadata& metadata,
    const std::optional<int>& layer_count,
    std::vector<EvidenceItem>& evidence) {
    LayerScopedValue<AttentionPatternKind> result;
    const MetadataValue* raw = metadata_alias(metadata, "layer_types");
    if (raw != nullptr) {
        if (const auto* value = std::get_if<std::string>(raw)) {
            result.global = parse_attention_pattern(*value, "layer_types");
        } else if (const auto* values = std::get_if<std::vector<std::string>>(raw)) {
            if (!layer_count.has_value() ||
                values->size() != static_cast<size_t>(*layer_count)) {
                inference_detail::fail(
                    ResolutionFailureKind::IncompleteLayerSchedule,
                    "attention layer schedule length does not match layer_count: layer_types");
            }
            result.per_layer.reserve(values->size());
            for (const std::string& value : *values) {
                result.per_layer.push_back(parse_attention_pattern(value, "layer_types"));
            }
        } else {
            inference_detail::fail(
                ResolutionFailureKind::ConflictingMetadata,
                "attention layer schedule has an incompatible type: layer_types");
        }
        evidence.push_back({EvidenceKind::AliasMetadata, "layer_types",
                            result.per_layer.empty()
                                ? "attention_pattern = global"
                                : "attention_pattern = layer-scoped schedule"});
    }
    /// GGUF has no `layer_types`; llama.cpp stores the sliding/full schedule
    /// as `<arch>.attention.sliding_window_pattern`, a per-layer boolean
    /// array (`flatten_json` stores it as 1/0 integers) with 1 = sliding.
    /// When `layer_types` is also present the two spellings must agree.
    if (metadata.is_gguf()) {
        const std::string pattern_key =
            metadata.architecture_type() + ".attention.sliding_window_pattern";
        const MetadataValue* pattern_raw = metadata_alias(metadata, pattern_key);
        if (pattern_raw != nullptr) {
            const auto* flags = std::get_if<std::vector<int64_t>>(pattern_raw);
            if (flags == nullptr) {
                inference_detail::fail(
                    ResolutionFailureKind::ConflictingMetadata,
                    "attention sliding schedule has an incompatible type: " + pattern_key);
            }
            if (!layer_count.has_value() ||
                flags->size() != static_cast<size_t>(*layer_count)) {
                inference_detail::fail(
                    ResolutionFailureKind::IncompleteLayerSchedule,
                    "attention sliding schedule length does not match layer_count: " +
                        pattern_key);
            }
            std::vector<std::optional<AttentionPatternKind>> schedule;
            schedule.reserve(flags->size());
            for (const int64_t flag : *flags) {
                if (flag != 0 && flag != 1) {
                    inference_detail::fail(
                        ResolutionFailureKind::ConflictingMetadata,
                        "attention sliding schedule must contain only 0 or 1: " +
                            pattern_key);
                }
                schedule.push_back(flag != 0 ? AttentionPatternKind::SlidingWindow
                                             : AttentionPatternKind::FullCausal);
            }
            if (result.has_value()) {
                for (size_t layer = 0; layer < schedule.size(); ++layer) {
                    const std::optional<AttentionPatternKind> declared =
                        layer < result.per_layer.size()
                            ? result.per_layer[layer]
                            : result.global;
                    if (declared.has_value() && *declared != schedule[layer]) {
                        inference_detail::fail(
                            ResolutionFailureKind::ConflictingMetadata,
                            "layer_types disagrees with " + pattern_key);
                    }
                }
            } else {
                result.per_layer = std::move(schedule);
            }
            evidence.push_back({EvidenceKind::AliasMetadata, pattern_key,
                                "attention_pattern = layer-scoped schedule (GGUF sliding flags)"});
        }
    }
    return result;
}

LayerScopedValue<bool> boolean_schedule_metadata(
    const CheckpointMetadata& metadata,
    std::initializer_list<std::string_view> aliases,
    const std::optional<int>& layer_count,
    std::vector<EvidenceItem>& evidence,
    std::string_view fact) {
    LayerScopedValue<bool> result;
    std::string accepted_source;
    const auto consider = [&](std::string_view key) {
        const MetadataValue* raw = metadata_alias(metadata, key);
        if (raw == nullptr) return;
        LayerScopedValue<bool> candidate;
        if (const auto* value = std::get_if<bool>(raw)) {
            candidate.global = *value;
        } else if (const auto* value = std::get_if<int64_t>(raw)) {
            if (*value != 0 && *value != 1) {
                inference_detail::fail(
                    ResolutionFailureKind::ConflictingMetadata,
                    "boolean norm metadata must be 0 or 1: " + std::string(key));
            }
            candidate.global = *value != 0;
        } else if (const auto* values = std::get_if<std::vector<int64_t>>(raw)) {
            if (!layer_count.has_value() ||
                values->size() != static_cast<size_t>(*layer_count)) {
                inference_detail::fail(
                    ResolutionFailureKind::IncompleteLayerSchedule,
                    "norm schedule length does not match layer_count: " +
                        std::string(key));
            }
            candidate.per_layer.reserve(values->size());
            for (const int64_t value : *values) {
                if (value != 0 && value != 1) {
                    inference_detail::fail(
                        ResolutionFailureKind::ConflictingMetadata,
                        "boolean norm schedule must contain only 0 or 1: " +
                            std::string(key));
                }
                candidate.per_layer.push_back(value != 0);
            }
        } else {
            inference_detail::fail(
                ResolutionFailureKind::ConflictingMetadata,
                "norm topology metadata has an incompatible type: " + std::string(key));
        }
        if (result.has_value() && !(result == candidate)) {
            inference_detail::fail(
                ResolutionFailureKind::ConflictingMetadata,
                "conflicting metadata aliases for " + std::string(fact));
        }
        result = std::move(candidate);
        accepted_source = std::string(key);
    };
    for (const std::string_view key : aliases) consider(key);
    if (result.has_value()) {
        evidence.push_back({
            EvidenceKind::AliasMetadata,
            accepted_source,
            std::string(fact) +
                (result.per_layer.empty() ? " = global" : " = layer-scoped schedule")});
    }
    return result;
}

void normalize_attention_schedule(const CheckpointMetadata& source,
                                  NormalizedModelMetadata& metadata) {
    metadata.attention.pattern = attention_pattern_metadata(
        source, metadata.core.layer_count, metadata.evidence);

    /// Some checkpoints declare two head-dim values: a base one (`head_dim`)
    /// and a wider one used only on the layers the layer_types schedule marks
    /// as full/global attention (`global_head_dim`). Expand that into a
    /// per-layer head_dim schedule so downstream resolution, which already
    /// consults `head_dim.value_for(layer)`, sees the right width for every
    /// layer instead of applying the base value everywhere.
    if (metadata.attention.head_dim.per_layer.empty() &&
        !metadata.attention.pattern.per_layer.empty()) {
        const std::optional<int> global_variant = integer_alias(source, "global_head_dim");
        const std::optional<int> base = metadata.attention.head_dim.global;
        if (global_variant.has_value() && base.has_value() && *global_variant != *base) {
            std::vector<std::optional<int>> per_layer;
            per_layer.reserve(metadata.attention.pattern.per_layer.size());
            for (const auto& pattern : metadata.attention.pattern.per_layer) {
                per_layer.push_back(
                    pattern == AttentionPatternKind::FullCausal ? *global_variant : *base);
            }
            metadata.attention.head_dim.per_layer = std::move(per_layer);
            metadata.evidence.push_back({EvidenceKind::AliasMetadata, "global_head_dim",
                "head_dim = layer-scoped schedule (full_attention layers use global_head_dim)"});
        }
    }

    std::optional<int> window;
    for (const std::string_view key : {std::string_view("sliding_window"),
                                       std::string_view("sliding_window_size")}) {
        const std::optional<int> candidate = integer_alias(source, key);
        if (!candidate.has_value()) continue;
        if (window.has_value() && *window != *candidate) {
            inference_detail::fail(
                ResolutionFailureKind::ConflictingMetadata,
                "conflicting sliding-window metadata aliases");
        }
        window = candidate;
    }
    /// GGUF stores the window under `<arch>.attention.sliding_window`
    /// (llama.cpp convention); merge it with the flat spellings, failing on
    /// disagreement rather than silently preferring one.
    if (source.is_gguf()) {
        const std::string arch_window =
            source.architecture_type() + ".attention.sliding_window";
        const std::optional<int> arch_candidate = integer_alias(source, arch_window);
        if (arch_candidate.has_value()) {
            if (window.has_value() && *window != *arch_candidate) {
                inference_detail::fail(
                    ResolutionFailureKind::ConflictingMetadata,
                    "conflicting sliding-window metadata aliases");
            }
            window = arch_candidate;
        }
    }
    if (window.has_value()) {
        if (*window <= 0) {
            inference_detail::fail(
                ResolutionFailureKind::ConflictingMetadata,
                "sliding_window must be positive");
        }
        metadata.attention.sliding_window = window;
        metadata.evidence.push_back({EvidenceKind::AliasMetadata, "sliding_window",
                                     "sliding_window = " + std::to_string(*window)});
    }

    bool needs_window = false;
    if (metadata.attention.pattern.global == AttentionPatternKind::SlidingWindow) {
        needs_window = true;
    }
    for (const auto& pattern : metadata.attention.pattern.per_layer) {
        if (pattern == AttentionPatternKind::SlidingWindow) needs_window = true;
    }
    if (needs_window && !metadata.attention.sliding_window.has_value()) {
        inference_detail::fail(
            ResolutionFailureKind::MissingRequiredMetadata,
            "sliding attention schedule requires sliding_window metadata");
    }

    expand_per_pattern_rope(source, metadata);
    apply_rope_layer_flags(source, metadata);
}

}
