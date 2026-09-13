#include "aliases.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace celeg {

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
