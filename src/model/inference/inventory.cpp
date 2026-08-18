#include "celeg/model/inference.hpp"

#include "support.hpp"

#include <algorithm>
#include <limits>
#include <regex>
#include <unordered_map>
#include <unordered_set>

namespace celeg {
namespace {

AttentionPatternKind parse_attention_pattern(std::string_view value,
                                             std::string_view source) {
    if (value == "full_attention" || value == "full" || value == "causal" ||
        value == "full_causal") {
        return AttentionPatternKind::FullCausal;
    }
    if (value == "sliding_attention" || value == "sliding_window" ||
        value == "sliding") {
        return AttentionPatternKind::SlidingWindow;
    }
    inference_detail::fail(
        ResolutionFailureKind::UnsupportedSemanticFeature,
        "unknown attention layer pattern token in " + std::string(source) + ": " +
            std::string(value));
}

const MetadataValue* metadata_alias(const CheckpointMetadata& metadata,
                                    std::string_view key) {
    if (metadata.contains(key)) return &metadata.value(key);
    const std::string text_key = "text_config." + std::string(key);
    return metadata.contains(text_key) ? &metadata.value(text_key) : nullptr;
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

LayerScopedValue<AttentionPatternKind> attention_pattern_metadata(
    const CheckpointMetadata& metadata,
    const std::optional<int>& layer_count,
    std::vector<EvidenceItem>& evidence) {
    LayerScopedValue<AttentionPatternKind> result;
    std::string accepted_source;
    const auto consider = [&](std::string_view key) {
        const MetadataValue* raw = metadata_alias(metadata, key);
        if (raw == nullptr) return;
        LayerScopedValue<AttentionPatternKind> candidate;
        if (const auto* value = std::get_if<std::string>(raw)) {
            candidate.global = parse_attention_pattern(*value, key);
        } else if (const auto* values = std::get_if<std::vector<std::string>>(raw)) {
            if (!layer_count.has_value() ||
                values->size() != static_cast<size_t>(*layer_count)) {
                inference_detail::fail(
                    ResolutionFailureKind::IncompleteLayerSchedule,
                    "attention layer schedule length does not match layer_count: " +
                        std::string(key));
            }
            candidate.per_layer.reserve(values->size());
            for (const std::string& value : *values) {
                candidate.per_layer.push_back(parse_attention_pattern(value, key));
            }
        } else {
            inference_detail::fail(
                ResolutionFailureKind::ConflictingMetadata,
                "attention layer schedule has an incompatible type: " + std::string(key));
        }
        if (result.has_value() && !(result == candidate)) {
            inference_detail::fail(
                ResolutionFailureKind::ConflictingMetadata,
                "conflicting attention layer schedules: " + accepted_source + " and " +
                    std::string(key));
        }
        result = std::move(candidate);
        accepted_source = std::string(key);
    };
    consider("layer_types");
    consider("layer_layouts");
    if (result.has_value()) {
        evidence.push_back({EvidenceKind::AliasMetadata, accepted_source,
                            result.per_layer.empty()
                                ? "attention_pattern = global"
                                : "attention_pattern = layer-scoped schedule"});
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
}

void normalize_structural_norm_schedule(const CheckpointMetadata& source,
                                        NormalizedModelMetadata& metadata) {
    const auto layer_count = metadata.core.layer_count;
    metadata.norms.mixer_before = boolean_schedule_metadata(
        source,
        {"mixer_pre_norm", "attention_pre_norm", "pre_attention_norm"},
        layer_count, metadata.evidence, "mixer_norm.before");
    metadata.norms.mixer_after = boolean_schedule_metadata(
        source,
        {"mixer_post_norm", "attention_post_norm", "post_attention_norm"},
        layer_count, metadata.evidence, "mixer_norm.after");
    metadata.norms.feed_forward_before = boolean_schedule_metadata(
        source,
        {"ffn_pre_norm", "feed_forward_pre_norm", "pre_feed_forward_norm",
         "pre_feedforward_norm"},
        layer_count, metadata.evidence, "feed_forward_norm.before");
    metadata.norms.feed_forward_after = boolean_schedule_metadata(
        source,
        {"ffn_post_norm", "feed_forward_post_norm", "post_feed_forward_norm",
         "post_feedforward_norm"},
        layer_count, metadata.evidence, "feed_forward_norm.after");
}

}

TensorInventory::TensorInventory(std::vector<TensorInventoryEntry> entries)
    : entries_(std::move(entries)) {
    std::unordered_map<std::string, const TensorInventoryEntry*> by_name;
    by_name.reserve(entries_.size());
    for (const auto& entry : entries_) by_name.emplace(entry.name, &entry);
    std::vector<TensorInventoryEntry> derived;
    for (const auto& entry : entries_) {
        constexpr std::string_view packed_suffix = "_packed";
        if (!entry.name.ends_with(packed_suffix) || entry.dtype != TensorDType::I32 ||
            entry.shape.size() != 2) continue;
        const std::string base = entry.name.substr(
            0, entry.name.size() - packed_suffix.size());
        if (by_name.contains(base)) continue;
        const auto scale_it = by_name.find(base + "_scale");
        const auto shape_it = by_name.find(base + "_shape");
        if (scale_it == by_name.end() || shape_it == by_name.end() ||
            scale_it->second->dtype != TensorDType::BF16 ||
            scale_it->second->shape.size() != 2 ||
            shape_it->second->dtype != TensorDType::I64 ||
            shape_it->second->shape != std::vector<int64_t>{2}) continue;
        const int64_t rows = entry.shape[0];
        const int64_t packed_words = entry.shape[1];
        const int64_t scale_columns = scale_it->second->shape[1];
        if (rows <= 0 || packed_words <= 0 || scale_columns <= 0 ||
            scale_it->second->shape[0] != rows) continue;
        const int64_t cols = scale_columns == 1
            ? packed_words * 4 : scale_columns * 32;
        const int64_t expected_words = scale_columns == 1
            ? (cols + 3) / 4 : (cols + 7) / 8;
        if (expected_words != packed_words) continue;
        derived.push_back({base, {rows, cols}, TensorDType::Quantized});
    }
    entries_.insert(entries_.end(), derived.begin(), derived.end());
    std::sort(entries_.begin(), entries_.end(),
              [](const auto& left, const auto& right) { return left.name < right.name; });
    for (size_t index = 1; index < entries_.size(); ++index) {
        if (entries_[index - 1].name == entries_[index].name) {
            inference_detail::fail(
                ResolutionFailureKind::UnsupportedTensorLayout,
                "checkpoint contains duplicate tensor identity: " + entries_[index].name);
        }
    }
}

const TensorInventoryEntry* TensorInventory::find(std::string_view name) const noexcept {
    const auto it = std::lower_bound(entries_.begin(), entries_.end(), name,
        [](const auto& entry, std::string_view value) { return entry.name < value; });
    return it != entries_.end() && it->name == name ? &*it : nullptr;
}

std::vector<const TensorInventoryEntry*> TensorInventory::with_prefix(
    std::string_view prefix) const {
    std::vector<const TensorInventoryEntry*> result;
    for (const auto& entry : entries_) {
        if (entry.name.starts_with(prefix)) result.push_back(&entry);
    }
    return result;
}

TensorInventory build_tensor_inventory(const IWeightRepository& repository) {
    std::vector<TensorInventoryEntry> entries;
    for (const std::string& name : repository.names()) {
        const HostTensorView tensor = repository.tensor(name);
        if (name.empty() || tensor.shape.empty() || tensor.shape.size() > 4) {
            inference_detail::fail(
                ResolutionFailureKind::UnsupportedTensorLayout,
                "tensor inventory contains invalid tensor metadata: " + name);
        }
        entries.push_back({name, tensor.shape, tensor.dtype});
    }
    return TensorInventory(std::move(entries));
}

InferenceInput build_inference_input(const CheckpointView& checkpoint) {
    if (!checkpoint.repository) {
        inference_detail::fail(ResolutionFailureKind::MissingTensorRole,
                               "automatic checkpoint resolution requires a tensor repository");
    }
    NormalizedModelMetadata metadata = normalize_model_metadata(checkpoint.metadata);
    normalize_attention_schedule(checkpoint.metadata, metadata);
    normalize_structural_norm_schedule(checkpoint.metadata, metadata);
    return {std::move(metadata), build_tensor_inventory(*checkpoint.repository),
            checkpoint.metadata.source_format};
}

}
