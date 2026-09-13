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

struct NormLayoutFacts {
    LayerScopedValue<bool> mixer_before;
    LayerScopedValue<bool> mixer_after;
    LayerScopedValue<bool> feed_forward_before;
    LayerScopedValue<bool> feed_forward_after;
};

std::array<bool, 4> parse_norm_layout(std::string_view value) {
    if (value == "decoder_prenorm" || value == "pre_norm" || value == "prenorm") {
        return {true, false, true, false};
    }
    if (value == "decoder_postnorm" || value == "post_norm" || value == "postnorm") {
        return {false, true, false, true};
    }
    if (value == "decoder_sandwich_norm" || value == "sandwich_norm" ||
        value == "sandwich") {
        return {true, true, true, true};
    }
    inference_detail::fail(
        ResolutionFailureKind::UnsupportedSemanticFeature,
        "unknown normalization layer layout token in layer_layouts: " +
            std::string(value));
}

NormLayoutFacts norm_layout_metadata(
    const CheckpointMetadata& metadata,
    const std::optional<int>& layer_count,
    std::vector<EvidenceItem>& evidence) {
    NormLayoutFacts result;
    const MetadataValue* raw = metadata_alias(metadata, "layer_layouts");
    if (raw == nullptr) return result;

    const auto set_global = [&](const std::array<bool, 4>& values) {
        result.mixer_before.global = values[0];
        result.mixer_after.global = values[1];
        result.feed_forward_before.global = values[2];
        result.feed_forward_after.global = values[3];
    };
    const auto append = [&](const std::array<bool, 4>& values) {
        result.mixer_before.per_layer.push_back(values[0]);
        result.mixer_after.per_layer.push_back(values[1]);
        result.feed_forward_before.per_layer.push_back(values[2]);
        result.feed_forward_after.per_layer.push_back(values[3]);
    };

    if (const auto* value = std::get_if<std::string>(raw)) {
        set_global(parse_norm_layout(*value));
    } else if (const auto* values = std::get_if<std::vector<std::string>>(raw)) {
        if (!layer_count.has_value() ||
            values->size() != static_cast<size_t>(*layer_count)) {
            inference_detail::fail(
                ResolutionFailureKind::IncompleteLayerSchedule,
                "normalization layer layout length does not match layer_count: layer_layouts");
        }
        result.mixer_before.per_layer.reserve(values->size());
        result.mixer_after.per_layer.reserve(values->size());
        result.feed_forward_before.per_layer.reserve(values->size());
        result.feed_forward_after.per_layer.reserve(values->size());
        for (const std::string& value : *values) append(parse_norm_layout(value));
    } else {
        inference_detail::fail(
            ResolutionFailureKind::ConflictingMetadata,
            "normalization layer layout has an incompatible type: layer_layouts");
    }

    evidence.push_back({
        EvidenceKind::AliasMetadata,
        "layer_layouts",
        result.mixer_before.per_layer.empty()
            ? "normalization topology = global"
            : "normalization topology = layer-scoped schedule"});
    return result;
}

void merge_layout_fact(LayerScopedValue<bool>& explicit_fact,
                       const LayerScopedValue<bool>& layout_fact,
                       std::string_view fact) {
    if (!layout_fact.has_value()) return;
    if (explicit_fact.has_value() && !(explicit_fact == layout_fact)) {
        inference_detail::fail(
            ResolutionFailureKind::ConflictingMetadata,
            "explicit normalization metadata conflicts with layer_layouts for " +
                std::string(fact));
    }
    if (!explicit_fact.has_value()) explicit_fact = layout_fact;
}

void normalize_structural_norm_schedule(const CheckpointMetadata& source,
                                        NormalizedModelMetadata& metadata) {
    const auto layer_count = metadata.core.layer_count;
    metadata.norms.mixer_before = boolean_schedule_metadata(
        source,
        {"mixer_pre_norm", "attention_pre_norm", "pre_attention_norm",
         "use_pre_attn_norm"},
        layer_count, metadata.evidence, "mixer_norm.before");
    metadata.norms.mixer_after = boolean_schedule_metadata(
        source,
        {"mixer_post_norm", "attention_post_norm", "post_attention_norm",
         "use_post_attn_norm"},
        layer_count, metadata.evidence, "mixer_norm.after");
    metadata.norms.feed_forward_before = boolean_schedule_metadata(
        source,
        {"ffn_pre_norm", "feed_forward_pre_norm", "pre_feed_forward_norm",
         "pre_feedforward_norm", "use_pre_mlp_norm"},
        layer_count, metadata.evidence, "feed_forward_norm.before");
    metadata.norms.feed_forward_after = boolean_schedule_metadata(
        source,
        {"ffn_post_norm", "feed_forward_post_norm", "post_feed_forward_norm",
         "post_feedforward_norm", "use_post_mlp_norm"},
        layer_count, metadata.evidence, "feed_forward_norm.after");

    const NormLayoutFacts layout = norm_layout_metadata(
        source, layer_count, metadata.evidence);
    merge_layout_fact(metadata.norms.mixer_before, layout.mixer_before,
                      "mixer_norm.before");
    merge_layout_fact(metadata.norms.mixer_after, layout.mixer_after,
                      "mixer_norm.after");
    merge_layout_fact(metadata.norms.feed_forward_before,
                      layout.feed_forward_before,
                      "feed_forward_norm.before");
    merge_layout_fact(metadata.norms.feed_forward_after,
                      layout.feed_forward_after,
                      "feed_forward_norm.after");
}

}
