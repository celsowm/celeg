#include "celeg/model/inference.hpp"

#include "celeg/checkpoint/packed/nvfp4.hpp"
#include "support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <regex>
#include <unordered_map>
#include <unordered_set>

namespace celeg {
namespace {

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

void apply_rope_layer_flags(const CheckpointMetadata& source,
                              NormalizedModelMetadata& metadata) {
    /// Per-layer RoPE participation, mirroring the checkpoint-shipped Lizzy
    /// reference (`_get_rope_layer_flag`): `rope_layer_flags` is a boolean
    /// array (`flatten_json` stores it as 1/0 integers) and a false entry
    /// means that layer attends without rotary position. `no_rope_layer_interval`
    /// is the fallback: when set, layers past the flags array (or every
    /// layer when flags are absent/all-true) lose RoPE every Nth layer
    /// (`(index + 1) % interval == 0`). False/interval-disabled slots become
    /// `NoPositionEncodingSpec` in the layer-scoped schedule; true slots keep
    /// whatever the per-pattern expansion (or the global fallback) resolved.
    /// Length must match the layer count exactly -- a ragged array fails
    /// loudly rather than misaligning layers.
    const MetadataValue* raw = metadata_alias(source, "rope_layer_flags");
    const std::optional<int> interval = integer_alias(source, "no_rope_layer_interval");
    if (raw == nullptr && !interval.has_value()) return;
    int rope_interval = 0;
    if (interval.has_value()) {
        if (*interval <= 0) {
            inference_detail::fail(
                ResolutionFailureKind::ConflictingMetadata,
                "no_rope_layer_interval must be positive");
        }
        rope_interval = *interval;
    }
    std::vector<int64_t> flags;
    if (raw != nullptr) {
        const auto* parsed = std::get_if<std::vector<int64_t>>(raw);
        if (parsed == nullptr) {
            inference_detail::fail(
                ResolutionFailureKind::ConflictingMetadata,
                "rope_layer_flags has an incompatible type (expected boolean array)");
        }
        flags = *parsed;
    }
    const auto& layer_count = metadata.core.layer_count;
    if (!layer_count.has_value()) {
        inference_detail::fail(
            ResolutionFailureKind::MissingRequiredMetadata,
            "rope schedule requires layer_count");
    }
    if (!flags.empty() && flags.size() != static_cast<size_t>(*layer_count)) {
        inference_detail::fail(
            ResolutionFailureKind::ConflictingMetadata,
            "rope_layer_flags length does not match layer_count");
    }
    if (metadata.attention.position_encoding.per_layer.empty()) {
        metadata.attention.position_encoding.per_layer.assign(
            static_cast<size_t>(*layer_count),
            metadata.attention.position_encoding.global);
    }
    auto& per_layer = metadata.attention.position_encoding.per_layer;
    if (per_layer.size() != static_cast<size_t>(*layer_count)) {
        inference_detail::fail(
            ResolutionFailureKind::IncompleteLayerSchedule,
            "rope schedule length does not match layer_count");
    }
    const bool all_true =
        !flags.empty() && std::all_of(flags.begin(), flags.end(),
                                      [](int64_t flag) { return flag != 0; });
    for (size_t layer = 0; layer < static_cast<size_t>(*layer_count); ++layer) {
        const bool covered = layer < flags.size();
        bool rope;
        if (rope_interval > 0 && (!covered || all_true)) {
            rope = (layer + 1) % static_cast<size_t>(rope_interval) != 0;
        } else if (covered) {
            rope = flags[layer] != 0;
        } else {
            continue;
        }
        if (!rope) per_layer[layer] = NoPositionEncodingSpec{};
    }
    metadata.evidence.push_back(
        {EvidenceKind::AliasMetadata, "rope_layer_flags",
         "position_encoding = rope disabled on flagged/interval layers"});
}

void expand_per_pattern_rope(const CheckpointMetadata& source,
                               NormalizedModelMetadata& metadata) {
    /// Per-pattern RoPE fans `rope_parameters.<layer_type>.*` over the
    /// `layer_types` schedule, mirroring the `global_head_dim` idiom above:
    /// checkpoints declaring distinct thetas per attention pattern (e.g. a
    /// narrow sliding theta and a wide full theta) get a per-layer
    /// `position_encoding` schedule so each layer's kernels see their own
    /// theta instead of a first-wins global. Falls back to `global` when no
    /// nested block exists, so every existing model is bit-identical. Both
    /// thetas arrive as `int64_t` when integral-valued (`flatten_json`
    /// stores integral numbers as integers), so `numeric_alias` (which
    /// accepts either representation) is required here.
    if (metadata.attention.pattern.per_layer.empty()) return;
    const std::optional<double> full_theta =
        numeric_alias(source, "rope_parameters.full_attention.rope_theta");
    const std::optional<double> sliding_theta =
        numeric_alias(source, "rope_parameters.sliding_attention.rope_theta");
    const std::optional<double> full_rotary_partial = numeric_aliases(
        source,
        {"rope_parameters.full_attention.partial_rotary_factor",
         "rope_parameters.full_attention.rotary_fraction",
         "rope_parameters.full_attention.rotary_factor"},
        "rope_parameters.full_attention.rotary_fraction");
    const std::optional<double> sliding_rotary_partial = numeric_aliases(
        source,
        {"rope_parameters.sliding_attention.partial_rotary_factor",
         "rope_parameters.sliding_attention.rotary_fraction",
         "rope_parameters.sliding_attention.rotary_factor"},
        "rope_parameters.sliding_attention.rotary_fraction");
    const std::optional<std::string> full_rope_type = string_aliases(
        source,
        {"rope_parameters.full_attention.rope_type",
         "rope_parameters.full_attention.type"},
        "rope_parameters.full_attention.rope_type");
    const std::optional<std::string> sliding_rope_type = string_aliases(
        source,
        {"rope_parameters.sliding_attention.rope_type",
         "rope_parameters.sliding_attention.type"},
        "rope_parameters.sliding_attention.rope_type");
    const bool has_nested = full_theta.has_value() || sliding_theta.has_value() ||
        full_rotary_partial.has_value() || sliding_rotary_partial.has_value() ||
        full_rope_type.has_value() || sliding_rope_type.has_value();
    if (!has_nested) return;

    const InferredRopePosition* global_rope = std::get_if<InferredRopePosition>(
        metadata.attention.position_encoding.global.has_value()
            ? &*metadata.attention.position_encoding.global
            : nullptr);
    const std::optional<double> global_theta =
        global_rope ? std::optional<double>(global_rope->theta) : std::nullopt;
    const float global_rotary =
        global_rope ? global_rope->rotary_fraction : 1.0f;
    const RopePairingKind pairing =
        global_rope ? global_rope->pairing : RopePairingKind::SplitHalf;
    const std::vector<int> mrope_sections =
        global_rope ? global_rope->mrope_sections : std::vector<int>{};
    const bool mrope_interleaved =
        global_rope ? global_rope->mrope_interleaved : false;

    std::vector<std::optional<InferredPositionEncoding>> per_layer;
    per_layer.reserve(metadata.attention.pattern.per_layer.size());
    for (const auto& pattern : metadata.attention.pattern.per_layer) {
        const bool is_full = pattern == AttentionPatternKind::FullCausal;
        const bool is_sliding = pattern == AttentionPatternKind::SlidingWindow;
        std::optional<double> theta;
        std::optional<double> rotary;
        if (is_full) {
            theta = full_theta.has_value() ? full_theta : global_theta;
            rotary = full_rotary_partial.has_value()
                ? full_rotary_partial
                : std::optional<double>(global_rotary);
        } else if (is_sliding) {
            theta = sliding_theta.has_value() ? sliding_theta : global_theta;
            rotary = sliding_rotary_partial.has_value()
                ? sliding_rotary_partial
                : std::optional<double>(global_rotary);
        } else {
            theta = global_theta.has_value()
                ? global_theta
                : (full_theta.has_value() ? full_theta : sliding_theta);
            rotary = std::optional<double>(global_rotary);
        }
        if ((is_full || is_sliding) && !theta.has_value()) {
            inference_detail::fail(
                ResolutionFailureKind::MissingRequiredMetadata,
                "per-pattern RoPE schedule requires a theta for every attention layer");
        }
        if (!theta.has_value()) {
            per_layer.push_back(std::nullopt);
            continue;
        }
        per_layer.push_back(InferredRopePosition{
            *theta,
            rotary.has_value() ? static_cast<float>(*rotary) : global_rotary,
            pairing,
            RopeScalingSpec{},
            mrope_sections,
            mrope_interleaved});
    }
    metadata.attention.position_encoding.per_layer = std::move(per_layer);
    metadata.evidence.push_back(
        {EvidenceKind::AliasMetadata, "rope_parameters.<layer_type>",
         "position_encoding = layer-scoped schedule (per-pattern RoPE theta)"});
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

void normalize_rope_scaling(const CheckpointMetadata& source,
                            NormalizedModelMetadata& metadata) {
    /// Layer-aware RoPE scaling: flat `rope_parameters.rope_type` covers the
    /// single-theta case, while per-pattern checkpoints declare
    /// `rope_parameters.<layer_type>.rope_type` alongside their per-pattern
    /// thetas. `proportional` is a first-class scaling kind (frequencies derive
    /// from head_dim, not the rotated width); unknown types still fail by name.
    /// GGUF stores YaRN under `<arch>.rope.scaling.*` (llama.cpp convention)
    /// instead of the flat `rope_scaling.*` keys; merge both spellings,
    /// failing on disagreement rather than silently preferring one.
    const std::string arch_prefix =
        source.is_gguf() ? source.architecture_type() + ".rope.scaling." : "";
    const auto arch_numeric = [&](std::string_view suffix) -> std::optional<double> {
        if (arch_prefix.empty()) return std::nullopt;
        return numeric_alias(source, arch_prefix + std::string(suffix));
    };
    const auto arch_integer = [&](std::string_view suffix) -> std::optional<int> {
        if (arch_prefix.empty()) return std::nullopt;
        return integer_alias(source, arch_prefix + std::string(suffix));
    };
    const auto arch_string = [&](std::string_view suffix) -> std::optional<std::string> {
        if (arch_prefix.empty()) return std::nullopt;
        return string_alias(source, arch_prefix + std::string(suffix));
    };
    const auto merge_arch = [&](auto flat, auto arch, std::string_view fact) {
        if (!arch.has_value()) return flat;
        if (flat.has_value() && *flat != *arch) {
            inference_detail::fail(
                ResolutionFailureKind::ConflictingMetadata,
                "conflicting RoPE metadata aliases for " + std::string(fact));
        }
        return arch;
    };
    const auto flat_kind = merge_arch(string_aliases(
        source,
        {"rope_scaling.rope_type", "rope_scaling.type",
         "rope_parameters.rope_type", "rope_parameters.type"},
        "rope_scaling.type"), arch_string("type"), "rope_scaling.type");
    const auto full_kind = string_aliases(
        source,
        {"rope_parameters.full_attention.rope_type",
         "rope_parameters.full_attention.type"},
        "rope_parameters.full_attention.rope_type");
    const auto sliding_kind = string_aliases(
        source,
        {"rope_parameters.sliding_attention.rope_type",
         "rope_parameters.sliding_attention.type"},
        "rope_parameters.sliding_attention.rope_type");

    auto resolve_yarn = [&]() -> YarnRopeScaling {
        const auto factor = merge_arch(numeric_aliases(
            source, {"rope_scaling.factor", "rope_parameters.factor"},
            "rope_scaling.factor"), arch_numeric("factor"), "rope_scaling.factor");
        const auto original_context = merge_arch(integer_aliases(
            source,
            {"rope_scaling.original_max_position_embeddings",
             "rope_scaling.original_context",
             "rope_parameters.original_max_position_embeddings",
             "rope_parameters.original_context",
             "original_max_position_embeddings"},
            "rope_scaling.original_context"),
            arch_integer("original_context_length"), "rope_scaling.original_context");
        if (!factor.has_value() || !original_context.has_value()) {
            inference_detail::fail(
                ResolutionFailureKind::MissingRequiredMetadata,
                "YaRN scaling requires factor and original context metadata");
        }
        const auto explicit_attention_factor = merge_arch(
            merge_arch(numeric_aliases(
                source,
                {"rope_scaling.attention_factor", "rope_parameters.attention_factor"},
                "rope_scaling.attention_factor"),
                arch_numeric("attention_factor"), "rope_scaling.attention_factor"),
            arch_numeric("yarn_attn_factor"), "rope_scaling.attention_factor");
        const auto mscale = merge_arch(numeric_aliases(
            source, {"rope_scaling.mscale", "rope_parameters.mscale"},
            "rope_scaling.mscale"), arch_numeric("mscale"), "rope_scaling.mscale");
        const auto mscale_all_dim = merge_arch(numeric_aliases(
            source, {"rope_scaling.mscale_all_dim", "rope_parameters.mscale_all_dim"},
            "rope_scaling.mscale_all_dim"),
            arch_numeric("mscale_all_dim"), "rope_scaling.mscale_all_dim");
        const auto inferred_mscale = [&](double scale, double multiplier) {
            return scale <= 1.0 ? 1.0 : 0.1 * multiplier * std::log(scale) + 1.0;
        };
        double attention_factor = 1.0;
        if (explicit_attention_factor.has_value()) {
            attention_factor = *explicit_attention_factor;
        } else if (mscale.has_value() && mscale_all_dim.has_value()) {
            attention_factor =
                inferred_mscale(*factor, *mscale) /
                inferred_mscale(*factor, *mscale_all_dim);
        } else {
            attention_factor = inferred_mscale(*factor, 1.0);
        }
        const double beta_fast = merge_arch(
            merge_arch(numeric_aliases(
                source, {"rope_scaling.beta_fast", "rope_parameters.beta_fast"},
                "rope_scaling.beta_fast"),
                arch_numeric("beta_fast"), "rope_scaling.beta_fast"),
            arch_numeric("yarn_beta_fast"), "rope_scaling.beta_fast")
                                               .value_or(32.0);
        const double beta_slow = merge_arch(
            merge_arch(numeric_aliases(
                source, {"rope_scaling.beta_slow", "rope_parameters.beta_slow"},
                "rope_scaling.beta_slow"),
                arch_numeric("beta_slow"), "rope_scaling.beta_slow"),
            arch_numeric("yarn_beta_slow"), "rope_scaling.beta_slow")
                                               .value_or(1.0);
        if (!std::isfinite(*factor) || *factor < 1.0 ||
            *original_context <= 0 ||
            !std::isfinite(attention_factor) || attention_factor <= 0.0 ||
            (mscale.has_value() && !std::isfinite(*mscale)) ||
            (mscale_all_dim.has_value() && !std::isfinite(*mscale_all_dim)) ||
            !std::isfinite(beta_fast) || !std::isfinite(beta_slow) ||
            beta_fast <= 0.0 || beta_slow <= 0.0 || beta_fast < beta_slow) {
            inference_detail::fail(
                ResolutionFailureKind::ConflictingMetadata,
                "invalid YaRN scaling metadata");
        }
        YarnRopeScaling yarn;
        yarn.factor = *factor;
        yarn.original_context = *original_context;
        yarn.attention_factor = attention_factor;
        yarn.beta_fast = beta_fast;
        yarn.beta_slow = beta_slow;
        return yarn;
    };

    const auto resolve_proportional = [&]() -> ProportionalRopeScaling {
        /// HF proportional carries an optional `factor` (default 1.0) that
        /// divides the frequencies; Gemma declares none, so this is 1.0.
        const auto factor = numeric_aliases(
            source,
            {"rope_scaling.factor", "rope_parameters.factor",
             "rope_parameters.full_attention.factor",
             "rope_parameters.sliding_attention.factor"},
            "rope_scaling.factor");
        const double value = factor.value_or(1.0);
        if (!std::isfinite(value) || value <= 0.0) {
            inference_detail::fail(
                ResolutionFailureKind::ConflictingMetadata,
                "invalid proportional RoPE factor");
        }
        return ProportionalRopeScaling{value};
    };

    const auto apply_kind = [&](InferredRopePosition& rope,
                                const std::optional<std::string>& kind,
                                std::string_view source_key) {
        if (!kind.has_value() || *kind == "none" || *kind == "default") {
            rope.scaling = NoRopeScaling{};
            return;
        }
        if (*kind == "proportional") {
            rope.scaling = resolve_proportional();
            metadata.evidence.push_back({
                EvidenceKind::AliasMetadata,
                std::string(source_key),
                "rope_scaling = proportional"});
            return;
        }
        if (*kind != "yarn") {
            inference_detail::fail(
                ResolutionFailureKind::UnsupportedSemanticFeature,
                "automatic resolution does not support declared RoPE scaling type: " +
                    *kind + " (" + std::string(source_key) + ")");
        }
        rope.scaling = resolve_yarn();
        metadata.evidence.push_back({
            EvidenceKind::AliasMetadata,
            std::string(source_key),
            "rope_scaling = yarn"});
    };

    auto* global_rope = metadata.attention.position_encoding.global.has_value()
        ? std::get_if<InferredRopePosition>(
              &*metadata.attention.position_encoding.global)
        : nullptr;

    if (metadata.attention.position_encoding.per_layer.empty()) {
        if (global_rope == nullptr) return;
        apply_kind(*global_rope, flat_kind, "rope_scaling");
        return;
    }

    std::optional<YarnRopeScaling> cached_yarn;
    std::optional<ProportionalRopeScaling> cached_proportional;
    const auto& patterns = metadata.attention.pattern.per_layer;
    for (size_t index = 0;
         index < metadata.attention.position_encoding.per_layer.size(); ++index) {
        auto& slot = metadata.attention.position_encoding.per_layer[index];
        if (!slot.has_value()) continue;
        auto* rope = std::get_if<InferredRopePosition>(&*slot);
        if (rope == nullptr) continue;
        std::optional<std::string> kind = flat_kind;
        std::string_view source_key = "rope_scaling";
        if (index < patterns.size()) {
            const auto pattern = patterns[index];
            if (pattern == AttentionPatternKind::FullCausal &&
                full_kind.has_value()) {
                kind = full_kind;
                source_key = "rope_parameters.full_attention.rope_type";
            } else if (pattern == AttentionPatternKind::SlidingWindow &&
                       sliding_kind.has_value()) {
                kind = sliding_kind;
                source_key = "rope_parameters.sliding_attention.rope_type";
            }
        }
        if (!kind.has_value() || *kind == "none" || *kind == "default") {
            rope->scaling = NoRopeScaling{};
            continue;
        }
        if (*kind == "proportional") {
            if (!cached_proportional.has_value()) {
                cached_proportional = resolve_proportional();
            }
            rope->scaling = *cached_proportional;
            continue;
        }
        if (*kind != "yarn") {
            inference_detail::fail(
                ResolutionFailureKind::UnsupportedSemanticFeature,
                "automatic resolution does not support declared RoPE scaling type: " +
                    *kind + " (" + std::string(source_key) + ")");
        }
        if (!cached_yarn.has_value()) cached_yarn = resolve_yarn();
        rope->scaling = *cached_yarn;
    }
    if (global_rope != nullptr) {
        apply_kind(*global_rope, flat_kind, "rope_scaling");
    }
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
        if (!entry.name.ends_with(packed_suffix)) continue;
        const std::string base = entry.name.substr(
            0, entry.name.size() - packed_suffix.size());
        if (by_name.contains(base)) continue;
        if (entry.dtype == TensorDType::I32 && entry.shape.size() == 2) {
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
        } else if (entry.dtype == TensorDType::U8 && entry.shape.size() == 2) {
            // NVFP4-pack-quantized (compressed-tensors): "<base>_packed" is
            // [rows, cols/2] nibble-packed, alongside a per-16-block
            // "<base>_scale" and a per-tensor "<base>_global_scale". See
            // celeg/checkpoint/packed/nvfp4.hpp, which loads the same three
            // sidecars by the same naming convention once the loader binds
            // this derived entry to an actual weight.
            const auto scale_it = by_name.find(base + "_scale");
            const auto global_scale_it = by_name.find(base + "_global_scale");
            if (scale_it == by_name.end() || global_scale_it == by_name.end() ||
                scale_it->second->dtype != TensorDType::F8_E4M3 ||
                scale_it->second->shape.size() != 2 ||
                global_scale_it->second->dtype != TensorDType::F32 ||
                global_scale_it->second->shape != std::vector<int64_t>{1}) continue;
            const int64_t rows = entry.shape[0];
            const int64_t packed_cols = entry.shape[1];
            const int64_t scale_columns = scale_it->second->shape[1];
            if (rows <= 0 || packed_cols <= 0 || scale_columns <= 0 ||
                scale_it->second->shape[0] != rows) continue;
            const int64_t cols = packed_cols * 2;
            if (scale_columns * kNvfp4PackedBlockSize != cols) continue;
            derived.push_back({base, {rows, cols}, TensorDType::Quantized});
        }
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
        // Rank 5 covers the vision-tower temporal patch-embed conv
        // (SafetensorProjectionProvider requires exactly rank 5: [hidden,
        // channels, temporal, patch_h, patch_w]); nothing else in the
        // generic inference pipeline depends on this bound being tighter.
        if (name.empty() || tensor.shape.size() > 5) {
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
    inference_detail::clear_metadata_consumption();
    NormalizedModelMetadata metadata = normalize_model_metadata(checkpoint.metadata);
    normalize_attention_schedule(checkpoint.metadata, metadata);
    normalize_structural_norm_schedule(checkpoint.metadata, metadata);
    normalize_rope_scaling(checkpoint.metadata, metadata);
    /// Tensor inventory is built before the consumption-ledger gate so the
    /// gate can condition tower-absence rules (linear/vision keys) on the
    /// grammars actually present instead of guessing from config alone.
    TensorInventory inventory = build_tensor_inventory(*checkpoint.repository);
    /// Consumption ledger gate runs last, after every resolver read, so any
    /// semantic key left unread warns (fails under `CELEG_STRICT_SEMANTICS=1`)
    /// instead of silently dropping mathematics.
    inference_detail::reject_unknown_semantic_metadata(checkpoint.metadata, inventory);
    return {std::move(metadata), std::move(inventory),
            checkpoint.metadata.source_format, checkpoint.metadata.architecture_type()};
}

}
