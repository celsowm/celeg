#include "metadata/aliases.hpp"
#include "metadata/detail.hpp"

#include "celeg/model/inference.hpp"

#include "support.hpp"

#include <string>
#include <string_view>
#include <unordered_set>

namespace celeg {

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
    normalize_core_dims(metadata, result);
    normalize_mamba2_facts(metadata, result);
    normalize_token_vocab_policy(metadata, result);
    normalize_norm_type_gate(metadata, result);
    normalize_logit_multipliers(metadata, result);
    const RopePositionFacts rope_facts = normalize_rope_position_facts(metadata, result);
    normalize_qk_norm_facts(metadata, result);
    normalize_gated_delta_facts(metadata, result);
    normalize_latent_attention_facts(metadata, result);
    normalize_moe_facts(metadata, result);
    normalize_xsa_ties(metadata, result);
    normalize_defaults(metadata, result);
    normalize_position_embedding_gate(metadata, result);
    normalize_rope_pairing(metadata, result, rope_facts);
    return result;
}

}
