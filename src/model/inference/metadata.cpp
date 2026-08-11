#include "celeg/model/inference.hpp"

#include "support.hpp"

#include <limits>
#include <type_traits>
#include <unordered_set>

namespace celeg {
namespace {

template <typename T>
std::optional<T> scalar(const CheckpointMetadata& metadata, std::string_view key) {
    if (!metadata.contains(key)) return std::nullopt;
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
std::optional<T> aliases(const CheckpointMetadata& metadata,
                         std::initializer_list<std::string_view> keys,
                         std::vector<EvidenceItem>& evidence,
                         std::string_view fact,
                         std::string_view gguf_suffix = {}) {
    std::optional<T> result;
    std::string source;
    const auto consider = [&](std::optional<T> value, std::string_view key) {
        if (!value.has_value()) return;
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
        consider(scalar<T>(metadata, gguf_key), gguf_key);
    }
    if (result.has_value()) {
        evidence.push_back({EvidenceKind::AliasMetadata, source,
                            std::string(fact) + " = " + std::to_string(*result)});
    }
    return result;
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

void reject_unknown_semantic_metadata(const CheckpointMetadata& metadata) {
    static const std::unordered_set<std::string> known = {
        "qk_norm", "query_key_norm", "xsa_projection",
        "xsa_projection_minimum_norm_squared", "rope_pairing", "rope_interleaved",
        "rope_theta", "rotary_fraction", "rope_scaling", "rope_parameters",
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

} // namespace

NormalizedModelMetadata normalize_model_metadata(const CheckpointMetadata& metadata) {
    reject_unknown_semantic_metadata(metadata);
    NormalizedModelMetadata result;
    result.hidden_size = aliases<int>(metadata, {"hidden_size", "n_embd", "d_model"},
                                      result.evidence, "hidden_size", "embedding_length");
    result.intermediate_size = aliases<int>(
        metadata, {"intermediate_size", "n_inner", "ffn_dim"}, result.evidence,
        "intermediate_size", "feed_forward_length");
    result.layer_count = aliases<int>(
        metadata, {"num_hidden_layers", "n_layer", "num_layers"}, result.evidence,
        "layer_count", "block_count");
    result.query_heads = aliases<int>(
        metadata, {"num_attention_heads", "n_head"}, result.evidence, "query_heads",
        "attention.head_count");
    result.key_value_heads = aliases<int>(
        metadata, {"num_key_value_heads", "n_kv_heads"}, result.evidence, "key_value_heads",
        "attention.head_count_kv");
    result.head_dim = aliases<int>(metadata, {"head_dim"}, result.evidence, "head_dim",
                                   "attention.key_length");
    result.vocab_size = aliases<int>(metadata, {"vocab_size", "n_vocab"}, result.evidence,
                                     "vocab_size", "vocab_size");
    result.context_length = aliases<int>(
        metadata, {"max_position_embeddings", "max_seq_len", "context_length"},
        result.evidence, "context_length", "context_length");
    result.norm_epsilon = aliases<float>(
        metadata, {"rms_norm_eps", "rms_norm_epsilon", "layer_norm_epsilon"},
        result.evidence, "norm_epsilon", "attention.layer_norm_rms_epsilon");
    result.rope_theta = aliases<double>(metadata, {"rope_theta"}, result.evidence,
                                         "rope_theta", "rope.freq_base");
    result.rotary_fraction = aliases<float>(metadata, {"rotary_fraction"}, result.evidence,
                                            "rotary_fraction");
    result.bos_token_id = aliases<int>(metadata,
                                       {"bos_token_id", "tokenizer.ggml.bos_token_id"},
                                       result.evidence, "bos_token_id");
    result.pad_token_id = aliases<int>(metadata,
                                       {"pad_token_id", "tokenizer.ggml.padding_token_id"},
                                       result.evidence, "pad_token_id");
    result.query_key_norm = aliases<bool>(
        metadata, {"qk_norm", "query_key_norm"}, result.evidence, "query_key_norm");
    result.xsa_projection = aliases<bool>(metadata, {"xsa_projection"}, result.evidence,
                                          "xsa_projection");
    result.xsa_minimum_norm_squared = aliases<float>(
        metadata, {"xsa_projection_minimum_norm_squared"}, result.evidence,
        "xsa_minimum_norm_squared");
    result.tied_embeddings = aliases<bool>(
        metadata, {"tie_word_embeddings", "tied_embeddings"}, result.evidence,
        "tied_embeddings");

    const std::vector<int> eos = token_list(metadata, "eos_token_id");
    result.eos_token_ids = eos.empty() ? token_list(metadata, "eos_token_ids") : eos;
    if (!result.bos_token_id.has_value()) result.bos_token_id = 0;
    if (result.eos_token_ids.empty()) result.eos_token_ids = {0};
    if (!result.pad_token_id.has_value()) result.pad_token_id = 1;
    if (!result.norm_epsilon.has_value()) result.norm_epsilon = 1.0e-6f;
    if (!result.rope_theta.has_value()) result.rope_theta = 100000.0;
    if (!result.rotary_fraction.has_value()) result.rotary_fraction = 1.0f;
    if (!result.query_key_norm.has_value()) result.query_key_norm = false;
    if (!result.xsa_projection.has_value()) result.xsa_projection = false;
    if (!result.xsa_minimum_norm_squared.has_value()) result.xsa_minimum_norm_squared = 1.0e-6f;
    if (!result.tied_embeddings.has_value()) result.tied_embeddings = false;

    if (metadata.contains("rope_pairing")) {
        const std::string pairing = metadata.string("rope_pairing");
        if (pairing == "adjacent_pairs" || pairing == "interleaved") {
            result.rope_pairing = RopePairingKind::AdjacentPairs;
        } else if (pairing == "split_half") {
            result.rope_pairing = RopePairingKind::SplitHalf;
        } else {
            inference_detail::fail(ResolutionFailureKind::UnsupportedSemanticFeature,
                                   "unsupported RoPE pairing: " + pairing);
        }
    } else if (*result.xsa_projection) {
        result.rope_pairing = RopePairingKind::AdjacentPairs;
        result.evidence.push_back({EvidenceKind::FormatGuarantee, "xsa_projection",
                                   "RoPE pairing = adjacent_pairs"});
    } else {
        result.rope_pairing = RopePairingKind::SplitHalf;
    }
    return result;
}

} // namespace celeg
