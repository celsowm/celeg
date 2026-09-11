#include "celeg/model/inference.hpp"

#include "celeg/checkpoint/gguf_position_profile.hpp"
#include "support.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <unordered_set>

namespace celeg {
namespace {

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

void validate_scoped_alias(const LayerScopedValue<int>& value,
                           const std::optional<int>& layer_count,
                           std::string_view fact) {
    if (value.per_layer.empty()) return;
    if (!layer_count.has_value() ||
        value.per_layer.size() != static_cast<size_t>(*layer_count)) {
        inference_detail::fail(
            ResolutionFailureKind::IncompleteLayerSchedule,
            "layer-scoped metadata length does not match layer_count for " +
                std::string(fact));
    }
    for (const auto& layer_value : value.per_layer) {
        if (!layer_value.has_value()) {
            inference_detail::fail(
                ResolutionFailureKind::IncompleteLayerSchedule,
                "layer-scoped metadata has a missing layer for " + std::string(fact));
        }
    }
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

/// The checkpoint's gated feed-forward activation, from hidden_act /
/// hidden_activation. Hugging Face names the elementwise function, and for a
/// gated MLP that determines the gate: silu gives SwiGLU, the tanh-approximated
/// GELUs give GeGLU. Absent when the checkpoint names none; an unrecognized
/// name is a hard failure rather than a silent fallback, because running the
/// wrong activation produces fluent-looking nonsense instead of an error.
std::optional<ActivationKind> feed_forward_activation(
    const CheckpointMetadata& metadata, std::vector<EvidenceItem>& evidence) {
    /// `mlp_type` states the feed-forward structure explicitly: celeg builds
    /// the gated gate/up/down structure from `hidden_act`, so `gated` merely
    /// confirms it, while anything else fails loudly instead of resolving a
    /// different MLP shape as SwiGLU (verified against the checkpoint-shipped
    /// Lizzy reference: `mlp_type == "gated"` guards the gate projection).
    for (const std::string_view candidate : {
             std::string_view("mlp_type"),
             std::string_view("text_config.mlp_type")}) {
        if (!metadata.contains(candidate)) continue;
        inference_detail::record_metadata_consumption(candidate);
        const std::string mlp = metadata.string(candidate);
        if (mlp != "gated") {
            inference_detail::fail(ResolutionFailureKind::UnsupportedSemanticFeature,
                                   "unsupported feed-forward structure: " + mlp);
        }
        evidence.push_back({EvidenceKind::ExplicitMetadata, std::string(candidate),
                            "feed_forward structure = gated"});
    }
    for (const std::string_view key : {
             std::string_view("hidden_act"),
             std::string_view("hidden_activation"),
             std::string_view("text_config.hidden_act"),
             std::string_view("text_config.hidden_activation")}) {
        if (!metadata.contains(key)) continue;
        inference_detail::record_metadata_consumption(key);
        const auto* name = std::get_if<std::string>(&metadata.value(key));
        if (name == nullptr) continue;
        std::optional<ActivationKind> kind;
        if (*name == "silu" || *name == "swish" || *name == "swiglu") {
            kind = ActivationKind::SwiGLU;
        } else if (*name == "gelu_pytorch_tanh" || *name == "gelu_tanh" ||
                   *name == "gelu_new" || *name == "gelu") {
            kind = ActivationKind::GeluTanh;
        } else if (*name == "relu2" || *name == "relu_squared") {
            kind = ActivationKind::Relu2;
        } else {
            inference_detail::fail(
                ResolutionFailureKind::UnsupportedSemanticFeature,
                "unsupported feed-forward activation: " + *name);
        }
        evidence.push_back({EvidenceKind::AliasMetadata, std::string(key),
                            "feed_forward_activation = " + *name});
        return kind;
    }
    return std::nullopt;
}

std::optional<int> tokenizer_vocabulary_size(const CheckpointMetadata& metadata,
                                             std::vector<EvidenceItem>& evidence) {
    constexpr std::string_view key = "tokenizer.ggml.tokens";
    if (!metadata.contains(key)) return std::nullopt;
    inference_detail::record_metadata_consumption(key);
    const auto* tokens = std::get_if<std::vector<std::string>>(&metadata.value(key));
    if (!tokens || tokens->empty() || tokens->size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        inference_detail::fail(ResolutionFailureKind::ConflictingMetadata,
                               "tokenizer vocabulary table is invalid");
    }
    evidence.push_back({EvidenceKind::Derived, std::string(key),
                        "vocab_size = " + std::to_string(tokens->size())});
    return static_cast<int>(tokens->size());
}

std::vector<int> token_list(const CheckpointMetadata& metadata, std::string_view key) {
    std::string resolved_key(key);
    if (!metadata.contains(resolved_key)) {
        resolved_key = "text_config." + resolved_key;
        if (!metadata.contains(resolved_key)) {
            if (!metadata.is_gguf()) return {};
            resolved_key = "tokenizer.ggml." + std::string(key);
            if (!metadata.contains(resolved_key)) return {};
        }
    }
    inference_detail::record_metadata_consumption(resolved_key);
    const MetadataValue& value = metadata.value(resolved_key);
    if (const auto* integer = std::get_if<std::int64_t>(&value)) {
        if (*integer < std::numeric_limits<int>::min() ||
            *integer > std::numeric_limits<int>::max()) {
            inference_detail::fail(ResolutionFailureKind::ConflictingMetadata,
                                   "token ID is outside the supported range: " + resolved_key);
        }
        return {static_cast<int>(*integer)};
    }
    if (const auto* values = std::get_if<std::vector<std::int64_t>>(&value)) {
        std::vector<int> result;
        result.reserve(values->size());
        for (const std::int64_t item : *values) {
            if (item < std::numeric_limits<int>::min() ||
                item > std::numeric_limits<int>::max()) {
                inference_detail::fail(ResolutionFailureKind::ConflictingMetadata,
                                           "token ID is outside the supported range: " + resolved_key);
            }
            result.push_back(static_cast<int>(item));
        }
        return result;
    }
    inference_detail::fail(ResolutionFailureKind::ConflictingMetadata,
                                   "token metadata has an incompatible type: " + resolved_key);
}

}

void reject_prior_unknown_semantics(const CheckpointMetadata& metadata) {
    /// Preserve the pre-ledger hard-fail for the original trigger family so
    /// direct `normalize_model_metadata` callers (including existing tests)
    /// still fail loudly on unknown `xsa`/`qk_norm`/`rope_pair` keys without
    /// needing the full ledger (which only runs in `build_inference_input`).
    static const std::unordered_set<std::string> known = {
        "qk_norm", "query_key_norm", "use_qk_norm", "qk_norm_type", "xsa_projection",
        "xsa_projection_minimum_norm_squared", "rope_pairing", "rope_interleaved",
        "rope_theta", "rotary_fraction", "rope_scaling", "rope_parameters",
        "embedding_multiplier", "attention_multiplier", "residual_multiplier",
        "logits_multiplier", "logits_divisor", "logits_scaling",
    };
    for (const auto& [key, value] : metadata.values) {
        (void)value;
        const std::string_view semantic_key = key.starts_with("text_config.")
            ? std::string_view(key).substr(std::string_view("text_config.").size())
            : std::string_view(key);
        const bool semantic_name = semantic_key.find("xsa") != std::string::npos ||
            semantic_key.find("qk_norm") != std::string::npos ||
            semantic_key.find("rope_pair") != std::string::npos;
        if (semantic_name && !known.contains(std::string(semantic_key))) {
            inference_detail::fail(
                ResolutionFailureKind::UnsupportedSemanticFeature,
                "automatic resolution does not know the mathematics of metadata key: " + key);
        }
    }
}

NormalizedModelMetadata normalize_model_metadata(const CheckpointMetadata& metadata) {
    reject_prior_unknown_semantics(metadata);
    /// Catalog selection reads the architecture identity before any rule
    /// runs (`automatic_architecture.cpp` via `metadata.architecture_type()`
    /// and `repository_hint`), through direct accessors that bypass the alias
    /// choke point. Record those keys here so the ledger does not mistake
    /// identity for unconsumed mathematics. Recording an absent key is a
    /// no-op for the gate, which only iterates present keys.
    for (const std::string_view key : {std::string_view("model_type"),
                                        std::string_view("general.architecture"),
                                        std::string_view("general.name"),
                                        std::string_view("general.basename")}) {
        if (metadata.contains(key)) inference_detail::record_metadata_consumption(key);
    }
    NormalizedModelMetadata result;
    result.core.hidden_size = aliases<int>(metadata, {"hidden_size", "n_embd", "d_model"},
                                           result.evidence, "hidden_size", "embedding_length");
    result.core.intermediate_size = scoped_aliases<int>(
        metadata, {"intermediate_size", "n_inner", "ffn_dim"}, result.evidence,
        "intermediate_size", "feed_forward_length");
    result.core.layer_count = aliases<int>(
        metadata, {"num_hidden_layers", "n_layer", "num_layers"}, result.evidence,
        "layer_count", "block_count");
    validate_scoped_alias(result.core.intermediate_size, result.core.layer_count, "intermediate_size");
    result.core.layer_repeat_count = aliases<int>(
        metadata, {"num_loops"}, result.evidence, "layer_repeat_count", "num_loops");
    result.attention.query_heads = scoped_aliases<int>(
        metadata, {"num_attention_heads", "n_head"}, result.evidence, "query_heads",
        "attention.head_count");
    result.attention.key_value_heads = scoped_aliases<int>(
        metadata, {"num_key_value_heads", "n_kv_heads"}, result.evidence, "key_value_heads",
        "attention.head_count_kv");
    result.attention.head_dim = scoped_aliases<int>(metadata, {"head_dim"}, result.evidence, "head_dim",
                                                    "attention.key_length");
    result.mamba2.intermediate = aliases<int>(
        metadata, {"mamba_intermediate", "ssm_inner_size"}, result.evidence,
        "mamba_intermediate", "ssm.inner_size");
    result.mamba2.state_size = aliases<int>(
        metadata, {"mamba_state_size", "ssm_state_size", "state_size"}, result.evidence,
        "mamba_state_size", "ssm.state_size");
    result.mamba2.time_step_rank = aliases<int>(
        metadata, {"mamba_time_step_rank", "ssm_time_step_rank", "time_step_rank"}, result.evidence,
        "mamba_time_step_rank", "ssm.time_step_rank");
    result.mamba2.num_heads = aliases<int>(
        metadata, {"mamba_num_heads", "mamba_heads", "num_heads"}, result.evidence,
        "mamba_num_heads", "ssm.time_step_rank");
    result.mamba2.head_dim = aliases<int>(
        metadata, {"mamba_head_dim"}, result.evidence, "mamba_head_dim");
    result.mamba2.group_count = aliases<int>(
        metadata, {"n_groups", "mamba_groups"}, result.evidence,
        "mamba_group_count", "ssm.group_count");
    result.mamba2.conv_kernel = aliases<int>(
        metadata, {"conv_kernel", "mamba_conv_kernel"}, result.evidence,
        "mamba_conv_kernel", "ssm.conv_kernel");
    result.mamba2.chunk_size = aliases<int>(
        metadata, {"chunk_size", "mamba_chunk_size"}, result.evidence,
        "mamba_chunk_size", "ssm.chunk_size");
    result.mamba2.decay_encoding = metadata.is_gguf()
        ? DecayParameterEncoding::Pretransformed
        : DecayParameterEncoding::LogA;

    result.core.vocab_size = aliases<int>(metadata, {"vocab_size", "n_vocab"}, result.evidence,
                                          "vocab_size", "vocab_size");
    if (!result.core.vocab_size.has_value()) {
        result.core.vocab_size = tokenizer_vocabulary_size(metadata, result.evidence);
    }
    result.core.context_length = aliases<int>(
        metadata, {"max_position_embeddings", "max_seq_len", "context_length"},
        result.evidence, "context_length", "context_length");
    result.core.norm_epsilon = aliases<float>(
        metadata, {"norm_eps", "rms_norm_eps", "rms_norm_epsilon", "layer_norm_epsilon"},
        result.evidence, "norm_epsilon", "attention.layer_norm_rms_epsilon");
    /// `norm_type` states the normalization kind explicitly: every norm celeg
    /// binds is RMS, so `rmsnorm` merely confirms it, while anything else
    /// fails loudly instead of resolving (e.g.) a LayerNorm as RMS.
    for (const std::string_view candidate : {
             std::string_view("norm_type"),
             std::string_view("text_config.norm_type")}) {
        if (!metadata.contains(candidate)) continue;
        inference_detail::record_metadata_consumption(candidate);
        const std::string norm = metadata.string(candidate);
        if (norm != "rmsnorm") {
            inference_detail::fail(ResolutionFailureKind::UnsupportedSemanticFeature,
                                   "unsupported normalization type: " + norm);
        }
        result.evidence.push_back({EvidenceKind::ExplicitMetadata, std::string(candidate),
                                   "norm_type = rmsnorm"});
    }
    result.core.embedding_multiplier = aliases<float>(
        metadata, {"embedding_multiplier"}, result.evidence,
        "embedding_multiplier");
    result.attention.attention_multiplier = aliases<float>(
        metadata, {"attention_multiplier"}, result.evidence,
        "attention_multiplier");
    /// Suffix KV sharing (`num_kv_shared_layers`): the last N layers consume KV
    /// from an earlier publisher of their own pattern type. Absent means private KV.
    result.attention.kv_shared_layers = aliases<int>(
        metadata, {"num_kv_shared_layers", "shared_kv_suffix_layers"}, result.evidence,
        "kv_shared_layers");
    result.core.residual_multiplier = aliases<float>(
        metadata, {"residual_multiplier"}, result.evidence,
        "residual_multiplier");
    result.core.logits_multiplier = aliases<float>(
        metadata, {"logits_multiplier"}, result.evidence,
        "logits_multiplier");
    result.core.logits_divisor = aliases<float>(
        metadata, {"logits_divisor", "logits_scaling"}, result.evidence,
        "logits_divisor");
    /// `ModelGraph::final_logit_softcap` is implemented on every backend but
    /// nothing in the automatic resolver ever assigned it. One key read here
    /// feeds both the numerical policy and the graph below.
    result.core.final_logit_softcap = aliases<float>(
        metadata, {"final_logit_softcapping", "final_logit_softcap", "logit_softcapping"},
        result.evidence, "final_logit_softcap");
    result.core.feed_forward_activation =
        feed_forward_activation(metadata, result.evidence);
    result.short_conv.cache_length = aliases<int>(metadata, {"conv_L_cache"}, result.evidence,
                                                  "shortconv_cache", "shortconv.l_cache");
    /// Per-pattern thetas (`rope_parameters.<layer_type>.rope_theta`) are resolved
    /// layer-wise in `normalize_attention_schedule` once the `layer_types` schedule
    /// is known; they are intentionally absent here so that disagreeing per-pattern
    /// values fail loudly there instead of being silently first-wins here.
    std::optional<double> rope_theta = aliases<double>(
        metadata, {"rope_theta", "rope_parameters.rope_theta"}, result.evidence,
        "rope_theta", "rope.freq_base");
    /// A nested `rope_scaling.rope_theta` restates the global theta inside
    /// the scaling block (Lizzy); it must agree instead of silently winning.
    const std::optional<double> scaling_theta = scalar<double>(metadata, "rope_scaling.rope_theta");
    if (scaling_theta.has_value()) {
        if (!rope_theta.has_value() || *scaling_theta != *rope_theta) {
            inference_detail::fail(ResolutionFailureKind::ConflictingMetadata,
                                   "rope_scaling.rope_theta disagrees with rope_theta");
        }
        result.evidence.push_back({EvidenceKind::AliasMetadata, "rope_scaling.rope_theta",
                                   "rope_theta restated inside rope_scaling"});
    }
    std::optional<float> rotary_fraction = aliases<float>(
        metadata, {"rotary_fraction", "partial_rotary_factor",
                   "rope_parameters.partial_rotary_factor"}, result.evidence,
        "rotary_fraction");
    std::vector<int> mrope_sections = token_list(metadata, "rope_parameters.mrope_section");
    if (mrope_sections.empty()) mrope_sections = token_list(metadata, "mrope_section");
    if (mrope_sections.empty()) mrope_sections = token_list(metadata, "rope_parameters.mrope_sections");
    if (mrope_sections.empty()) mrope_sections = token_list(metadata, "mrope_sections");
    if (!mrope_sections.empty() && mrope_sections.size() != 3) {
        inference_detail::fail(ResolutionFailureKind::ConflictingMetadata,
                               "M-RoPE requires exactly three sections (temporal/height/width)");
    }
    const bool mrope_interleaved = aliases<bool>(
        metadata, {"mrope_interleaved", "rope_parameters.mrope_interleaved"}, result.evidence,
        "mrope_interleaved").value_or(false);
    const bool architecture_never_applies_rope = metadata.is_gguf() &&
        gguf_architecture_never_applies_rope(metadata.architecture_type());
    if (!architecture_never_applies_rope && !rotary_fraction.has_value() && metadata.is_gguf()) {
        /// GGUF stores partial rotary as an absolute dimension count
        /// ("<arch>.rope.dimension_count"), not a fraction of head_dim like the
        /// HF-config "rotary_fraction"/"partial_rotary_factor" convention does.
        const std::string rope_dim_key =
            metadata.architecture_type() + ".rope.dimension_count";
        const std::optional<int> rope_dimension_count =
            gguf_scalar_or_uniform_schedule<int>(metadata, rope_dim_key);
        if (rope_dimension_count.has_value()) {
            const std::optional<int> effective_head_dim = result.attention.head_dim.global.has_value()
                ? result.attention.head_dim.global
                : (result.core.hidden_size.has_value() && result.attention.query_heads.global.has_value() &&
                   *result.attention.query_heads.global > 0)
                      ? std::optional<int>(*result.core.hidden_size / *result.attention.query_heads.global)
                      : std::nullopt;
            if (effective_head_dim.has_value() && *effective_head_dim > 0) {
                rotary_fraction = static_cast<float>(*rope_dimension_count) /
                    static_cast<float>(*effective_head_dim);
                result.evidence.push_back({EvidenceKind::AliasMetadata, rope_dim_key,
                    "rotary_fraction = " + std::to_string(*rotary_fraction)});
            }
        }
    }
    result.core.bos_token_id = aliases<int>(metadata,
                                            {"bos_token_id", "tokenizer.ggml.bos_token_id"},
                                            result.evidence, "bos_token_id");
    result.core.pad_token_id = aliases<int>(metadata,
                                            {"pad_token_id", "tokenizer.ggml.padding_token_id"},
                                            result.evidence, "pad_token_id");
    result.attention.query_key_norm = aliases<bool>(
        metadata, {"qk_norm", "query_key_norm", "use_qk_norm"}, result.evidence,
        "query_key_norm");
    if (result.attention.query_key_norm.value_or(false)) {
        std::optional<std::string> qk_norm_type;
        std::string qk_norm_type_source;
        for (const std::string_view key : {
                 std::string_view("qk_norm_type"),
                  std::string_view("text_config.qk_norm_type")}) {
            if (!metadata.contains(key)) continue;
            inference_detail::record_metadata_consumption(key);
            const MetadataValue& raw = metadata.value(key);
            const auto* value = std::get_if<std::string>(&raw);
            if (value == nullptr) {
                inference_detail::fail(
                    ResolutionFailureKind::ConflictingMetadata,
                    "Q/K normalization type metadata is not a string: " +
                        std::string(key));
            }
            if (qk_norm_type.has_value() && *qk_norm_type != *value) {
                inference_detail::fail(
                    ResolutionFailureKind::ConflictingMetadata,
                    "conflicting Q/K normalization type metadata");
            }
            qk_norm_type = *value;
            qk_norm_type_source = std::string(key);
        }
        if (qk_norm_type.has_value()) {
            if (*qk_norm_type != "rmsnorm") {
                inference_detail::fail(
                    ResolutionFailureKind::UnsupportedSemanticFeature,
                    "unsupported Q/K normalization mathematics: " + *qk_norm_type);
            }
            result.evidence.push_back({EvidenceKind::ExplicitMetadata,
                                       qk_norm_type_source,
                                       "query_key_norm = rmsnorm"});
        }
    }
    result.core.feed_forward_auto_adjust = aliases<bool>(
        metadata, {"block_auto_adjust_ff_dim"}, result.evidence,
        "feed_forward_auto_adjust");
    result.moe.first_dense_layer = aliases<int>(
        metadata, {"first_k_dense_replace", "num_dense_layers"}, result.evidence,
        "first_dense_layer");
    result.gated_delta.conv_kernel = aliases<int>(
        metadata, {"short_conv_kernel_size", "recurrent_conv_kernel"}, result.evidence,
        "recurrent_conv_kernel");
    result.gated_delta.key_heads = aliases<int>(
        metadata, {"num_attention_heads", "num_heads_for_linear_attn"}, result.evidence,
        "recurrent_key_heads");
    result.gated_delta.value_heads = aliases<int>(
        metadata, {"num_attention_heads", "num_heads_for_linear_attn"}, result.evidence,
        "recurrent_value_heads");
    result.gated_delta.key_dim = aliases<int>(
        metadata, {"head_dim"}, result.evidence, "recurrent_key_dim");
    result.gated_delta.value_dim = aliases<int>(
        metadata, {"v_head_dim", "head_dim"}, result.evidence, "recurrent_value_dim");
    result.gated_delta.safe_decay = aliases<bool>(
        metadata, {"kda_safe_gate", "safe_decay"}, result.evidence,
        "recurrent_safe_decay");
    result.gated_delta.decay_lower_bound = aliases<float>(
        metadata, {"kda_lower_bound", "decay_lower_bound"}, result.evidence,
        "recurrent_decay_lower_bound");
    result.gated_delta.decay_encoding = metadata.is_gguf()
        ? DecayParameterEncoding::Pretransformed
        : DecayParameterEncoding::LogA;
    result.gated_delta.linear_key_heads = aliases<int>(
        metadata, {"linear_num_key_heads"}, result.evidence,
        "recurrent_linear_key_heads");
    result.gated_delta.linear_value_heads = aliases<int>(
        metadata, {"linear_num_value_heads"}, result.evidence,
        "recurrent_linear_value_heads");
    result.gated_delta.linear_key_dim = aliases<int>(
        metadata, {"linear_key_head_dim"}, result.evidence,
        "recurrent_linear_key_dim");
    result.gated_delta.linear_value_dim = aliases<int>(
        metadata, {"linear_value_head_dim"}, result.evidence,
        "recurrent_linear_value_dim");
    result.gated_delta.linear_conv_kernel = aliases<int>(
        metadata, {"linear_conv_kernel_dim"}, result.evidence,
        "recurrent_linear_conv_kernel");
    result.gated_delta.direct_projections = aliases<bool>(
        metadata, {"no_kda_lora"}, result.evidence,
        "recurrent_direct_projections");
    result.gated_delta.hybrid_group_size = aliases<int>(
        metadata, {"layer_group_size"}, result.evidence,
        "recurrent_hybrid_group_size");

    result.latent_attention.query_rank = aliases<int>(
        metadata, {"q_lora_rank"}, result.evidence, "latent_query_rank");
    result.latent_attention.kv_rank = aliases<int>(
        metadata, {"kv_lora_rank"}, result.evidence, "latent_kv_rank");
    result.latent_attention.query_head_dim = aliases<int>(
        metadata, {"qk_head_dim"}, result.evidence, "latent_query_head_dim");
    result.latent_attention.query_nope_dim = aliases<int>(
        metadata, {"qk_nope_head_dim"}, result.evidence, "latent_query_nope_dim");
    result.latent_attention.query_rope_dim = aliases<int>(
        metadata, {"qk_rope_head_dim"}, result.evidence, "latent_query_rope_dim");
    result.latent_attention.value_head_dim = aliases<int>(
        metadata, {"v_head_dim"}, result.evidence, "latent_value_head_dim");
    /// `gated_attention_proj_granularity_type` states the MLA output-gate
    /// geometry the latent rule also infers from the `g_proj` shape; the rule
    /// confronts the two and fails loudly on disagreement, so reading the
    /// string here (rather than dropping it in the ledger) is what makes a
    /// `head_wise` checkpoint verifiable instead of merely resolvable.
    for (const std::string_view candidate : {
             std::string_view("gated_attention_proj_granularity_type"),
             std::string_view("text_config.gated_attention_proj_granularity_type")}) {
        if (!metadata.contains(candidate)) continue;
        inference_detail::record_metadata_consumption(candidate);
        const std::string granularity = metadata.string(candidate);
        if (granularity != "head_wise" && granularity != "element_wise") {
            inference_detail::fail(
                ResolutionFailureKind::UnsupportedSemanticFeature,
                "unsupported attention gate granularity: " + granularity);
        }
        result.latent_attention.output_gate_granularity = granularity;
        result.evidence.push_back({EvidenceKind::ExplicitMetadata, std::string(candidate),
                                   "latent_output_gate_granularity"});
        break;
    }
    result.moe.experts = aliases<int>(
        metadata, {"num_experts"}, result.evidence, "moe_experts");
    result.moe.experts_per_token = aliases<int>(
        metadata, {"num_experts_per_tok", "experts_per_token"}, result.evidence,
        "moe_experts_per_token");
    result.moe.intermediate = aliases<int>(
        metadata, {"moe_intermediate_size"}, result.evidence, "moe_intermediate");
    result.moe.shared_intermediate = aliases<int>(
        metadata, {"moe_shared_expert_intermediate_size"}, result.evidence,
        "moe_shared_intermediate");
    result.moe.routing_groups = aliases<int>(
        metadata, {"topk_group"}, result.evidence, "moe_routing_groups");
    result.moe.total_routing_groups = aliases<int>(
        metadata, {"n_group", "num_routing_groups"}, result.evidence,
        "moe_total_routing_groups");
    result.moe.group_score_top_k = aliases<int>(
        metadata, {"routing_group_score_top_k"}, result.evidence,
        "moe_group_score_top_k");
    result.moe.normalize_topk = aliases<bool>(
        metadata, {"norm_topk_prob"}, result.evidence, "moe_normalize_topk");
    result.moe.expert_bias = aliases<bool>(
        metadata, {"moe_router_enable_expert_bias"}, result.evidence, "moe_expert_bias");
    result.moe.routed_scaling = aliases<float>(
        metadata, {"routed_scaling_factor"}, result.evidence, "moe_routed_scaling");
    for (const std::string_view candidate : {
             std::string_view("score_function"), std::string_view("scoring_func"),
             std::string_view("text_config.score_function"),
             std::string_view("text_config.scoring_func")}) {
        if (!metadata.contains(candidate)) continue;
        inference_detail::record_metadata_consumption(candidate);
        const std::string score = metadata.string(candidate);
        if (score == "sigmoid") {
            if (result.moe.score_function.has_value() &&
                *result.moe.score_function != MoeRouterScoreFunction::Sigmoid) {
                inference_detail::fail(ResolutionFailureKind::ConflictingMetadata,
                                       "conflicting MoE router score functions");
            }
            result.moe.score_function = MoeRouterScoreFunction::Sigmoid;
        } else if (score == "softmax") {
            if (result.moe.score_function.has_value() &&
                *result.moe.score_function != MoeRouterScoreFunction::Softmax) {
                inference_detail::fail(ResolutionFailureKind::ConflictingMetadata,
                                       "conflicting MoE router score functions");
            }
            result.moe.score_function = MoeRouterScoreFunction::Softmax;
        } else {
            inference_detail::fail(ResolutionFailureKind::UnsupportedSemanticFeature,
                                   "unsupported MoE router score function: " + score);
        }
        result.evidence.push_back({EvidenceKind::ExplicitMetadata, std::string(candidate),
                                   "moe_score_function"});
    }
    for (const std::string_view candidate : {
             std::string_view("topk_method"),
             std::string_view("text_config.topk_method")}) {
        if (!metadata.contains(candidate)) continue;
        inference_detail::record_metadata_consumption(candidate);
        const std::string method = metadata.string(candidate);
        if (method == "noaux_tc") {
            result.moe.selection_method = MoeRouterSelectionMethod::NoauxTc;
        } else if (method == "greedy") {
            result.moe.selection_method = MoeRouterSelectionMethod::Greedy;
        } else if (method == "topk") {
            result.moe.selection_method = MoeRouterSelectionMethod::TopK;
        } else {
            inference_detail::fail(ResolutionFailureKind::UnsupportedSemanticFeature,
                                   "unsupported MoE selection method: " + method);
        }
        if (method == "noaux_tc" && !result.moe.group_score_top_k.has_value()) {
            result.moe.group_score_top_k = 2;
        }
        result.evidence.push_back({EvidenceKind::ExplicitMetadata, std::string(candidate),
                                   "moe_selection_method"});
        break;
    }
    result.attention.xsa_projection = aliases<bool>(metadata, {"xsa_projection"}, result.evidence,
                                                    "xsa_projection");
    result.attention.xsa_minimum_norm_squared = aliases<float>(
        metadata, {"xsa_projection_minimum_norm_squared"}, result.evidence,
        "xsa_minimum_norm_squared");
    result.core.tied_embeddings = aliases<bool>(
        metadata, {"tie_word_embeddings", "tied_embeddings", "tie_embedding"}, result.evidence,
        "tied_embeddings");

    validate_scoped_alias(result.attention.query_heads, result.core.layer_count, "query_heads");
    validate_scoped_alias(result.attention.key_value_heads, result.core.layer_count, "key_value_heads");
    validate_scoped_alias(result.attention.head_dim, result.core.layer_count, "head_dim");

    const std::vector<int> eos = token_list(metadata, "eos_token_id");
    result.core.eos_token_ids = eos.empty() ? token_list(metadata, "eos_token_ids") : eos;
    if (!result.core.bos_token_id.has_value()) result.core.bos_token_id = 0;
    if (result.core.eos_token_ids.empty()) result.core.eos_token_ids = {0};
    if (!result.core.pad_token_id.has_value()) result.core.pad_token_id = 1;
    if (!result.core.norm_epsilon.has_value()) result.core.norm_epsilon = 1.0e-6f;
    /// No default for `embedding_multiplier` here: `global_facts.cpp` needs to
    /// distinguish an explicit key from absence so the per-layer-input tower
    /// can imply `sqrt(hidden)` only when no explicit multiplier is present.
    if (!result.core.residual_multiplier.has_value()) result.core.residual_multiplier = 1.0f;
    if (!result.core.logits_multiplier.has_value()) result.core.logits_multiplier = 1.0f;
    if (!result.core.logits_divisor.has_value()) result.core.logits_divisor = 1.0f;
    if (!result.attention.query_key_norm.has_value()) result.attention.query_key_norm = false;
    if (!result.attention.xsa_projection.has_value()) result.attention.xsa_projection = false;
    if (!result.attention.xsa_minimum_norm_squared.has_value()) result.attention.xsa_minimum_norm_squared = 1.0e-6f;

    /// `position_embedding_type` states the global position policy
    /// explicitly; celeg resolves RoPE everywhere by default, so `rope`
    /// merely confirms it, while anything else fails loudly instead of
    /// resolving RoPE against a checkpoint that asked for something else.
    for (const std::string_view candidate : {
             std::string_view("position_embedding_type"),
             std::string_view("text_config.position_embedding_type")}) {
        if (!metadata.contains(candidate)) continue;
        inference_detail::record_metadata_consumption(candidate);
        const std::string policy = metadata.string(candidate);
        if (policy != "rope") {
            inference_detail::fail(ResolutionFailureKind::UnsupportedSemanticFeature,
                                   "unsupported position embedding type: " + policy);
        }
        result.evidence.push_back({EvidenceKind::ExplicitMetadata, std::string(candidate),
                                   "position_embedding_type = rope"});
    }

    if (architecture_never_applies_rope) {
        /// Some GGUF architectures (mostly hybrid recurrent/attention models
        /// whose position information already flows through the recurrent
        /// state) carry vestigial RoPE hparams that the reference graph
        /// builder never actually applies to the attention layers. Applying
        /// RoPE anyway corrupts every attention layer's positional structure,
        /// so the GGUF format boundary overrides any rope metadata present.
        result.attention.position_encoding.global = NoPositionEncodingSpec{};
        result.evidence.push_back({EvidenceKind::FormatGuarantee, "architecture",
                                   metadata.architecture_type() + " does not use RoPE"});
    } else {
        RopePairingKind pairing = RopePairingKind::SplitHalf;
        std::optional<RopePairingKind> stated_pairing;
        if (metadata.contains("rope_pairing")) {
            inference_detail::record_metadata_consumption("rope_pairing");
            const std::string pairing_value = metadata.string("rope_pairing");
            if (pairing_value == "adjacent_pairs" || pairing_value == "interleaved") {
                stated_pairing = RopePairingKind::AdjacentPairs;
            } else if (pairing_value == "split_half") {
                stated_pairing = RopePairingKind::SplitHalf;
            } else {
                inference_detail::fail(ResolutionFailureKind::UnsupportedSemanticFeature,
                                       "unsupported RoPE pairing: " + pairing_value);
            }
        }
        /// `rope_interleave` is the HF convention spelling of the same
        /// pairing fact (`true` = GPT-J-style adjacent pairs, used by the
        /// shipped Ling MLA path via `apply_rotary_pos_emb_interleave`).
        if (metadata.contains("rope_interleave")) {
            inference_detail::record_metadata_consumption("rope_interleave");
            const auto* flag =
                std::get_if<bool>(&metadata.value("rope_interleave"));
            if (flag != nullptr) {
                const RopePairingKind interleave_pairing =
                    *flag ? RopePairingKind::AdjacentPairs
                          : RopePairingKind::SplitHalf;
                if (stated_pairing.has_value() &&
                    *stated_pairing != interleave_pairing) {
                    inference_detail::fail(
                        ResolutionFailureKind::ConflictingMetadata,
                        "rope_interleave disagrees with rope_pairing");
                }
                stated_pairing = interleave_pairing;
                result.evidence.push_back({
                    EvidenceKind::ExplicitMetadata, "rope_interleave",
                    *flag ? "RoPE pairing = adjacent_pairs (interleaved)"
                          : "RoPE pairing = split_half"});
            }
        }
        if (stated_pairing.has_value()) {
            pairing = *stated_pairing;
        } else if (*result.attention.xsa_projection) {
            pairing = RopePairingKind::AdjacentPairs;
            result.evidence.push_back({EvidenceKind::FormatGuarantee, "xsa_projection",
                                       "RoPE pairing = adjacent_pairs"});
        } else if (metadata.is_gguf() &&
                   gguf_architecture_uses_adjacent_rope_pairs(metadata.architecture_type())) {
            /// llama.cpp's conversion permuted this architecture's query/key
            /// rows into consecutive-pair order, so the pairing has to match the
            /// stored layout. celeg never reorders the rows back at load time --
            /// the two formulations are equivalent, and expressing it here keeps
            /// every backend on one convention.
            pairing = RopePairingKind::AdjacentPairs;
            result.evidence.push_back({EvidenceKind::FormatGuarantee,
                                       metadata.architecture_type(),
                                       "GGUF architecture stores adjacent-pair Q/K rows"});
        }
        /// Global theta absent when the checkpoint only declares per-pattern thetas
        /// (`rope_parameters.<layer_type>.rope_theta`); the per-layer schedule is
        /// fanned out in `normalize_attention_schedule` once `layer_types` is known.
        /// Leaving `global` empty here defers the missing-theta failure until every
        /// layer can be checked via `value_for(layer)`.
        if (rope_theta.has_value()) {
            result.attention.position_encoding.global = InferredRopePosition{
                *rope_theta,
                rotary_fraction.value_or(1.0f),
                pairing,
                RopeScalingSpec{},
                mrope_sections,
                mrope_interleaved};
        }
    }
    return result;
}

}
