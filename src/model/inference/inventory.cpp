#include "inventory/access.hpp"
#include "inventory/detail.hpp"

#include "celeg/model/inference.hpp"

#include "normalize_internal.hpp"
#include "support.hpp"

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
