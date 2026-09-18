#include "aliases.hpp"
#include "detail.hpp"

#include "celeg/checkpoint/gguf_position_profile.hpp"
#include "../support.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace celeg {

/// RoPE theta/rotary/mrope cluster of `normalize_model_metadata`; returns the
/// local facts the pairing resolution at the end of the original body read.
RopePositionFacts normalize_rope_position_facts(const CheckpointMetadata& metadata,
                                                NormalizedModelMetadata& result) {
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
                   "rope_parameters.rotary_fraction",
                   "rope_parameters.partial_rotary_factor"}, result.evidence,
        "rotary_fraction");
    std::vector<int> mrope_sections = token_list(metadata, "rope_parameters.mrope_section");
    if (mrope_sections.empty()) mrope_sections = token_list(metadata, "mrope_section");
    if (mrope_sections.empty()) mrope_sections = token_list(metadata, "rope_parameters.mrope_sections");
    if (mrope_sections.empty()) mrope_sections = token_list(metadata, "mrope_sections");
    if (mrope_sections.empty() && metadata.is_gguf()) {
        /// GGUF stores the sections under the file's own architecture tag
        /// ("<arch>.rope.dimension_sections"), composed at runtime so no
        /// per-architecture spelling lives in tree.
        mrope_sections = token_list(
            metadata, metadata.architecture_type() + ".rope.dimension_sections");
    }
    /// Some writers append a trailing zero section for an unused fourth axis;
    /// trim only trailing zeros so a nonzero fourth axis still fails loudly
    /// below instead of being silently dropped.
    while (mrope_sections.size() > 3 && mrope_sections.back() == 0) {
        mrope_sections.pop_back();
    }
    if (!mrope_sections.empty() && mrope_sections.size() != 3) {
        inference_detail::fail(ResolutionFailureKind::ConflictingMetadata,
                               "M-RoPE requires exactly three sections (temporal/height/width)");
    }
    const std::optional<bool> explicit_interleaved = aliases<bool>(
        metadata, {"mrope_interleaved", "rope_parameters.mrope_interleaved"}, result.evidence,
        "mrope_interleaved");
    /// Interleaved section layout is the current HF/llama.cpp convention for
    /// sectioned rotary (validated against transformers text-rope
    /// recomputation); default to it when sections are present but the flag
    /// is absent, so sectioned checkpoints without the flag are not silently
    /// run chunked.
    const bool mrope_interleaved = explicit_interleaved.value_or(!mrope_sections.empty());
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
    return RopePositionFacts{rope_theta, rotary_fraction, mrope_sections,
                             mrope_interleaved, architecture_never_applies_rope};
}

/// QK-norm cluster of `normalize_model_metadata`.
void normalize_qk_norm_facts(const CheckpointMetadata& metadata, NormalizedModelMetadata& result) {
    result.attention.query_key_norm = aliases<bool>(
        metadata, {"qk_norm", "query_key_norm", "use_qk_norm"}, result.evidence,
        "query_key_norm");
    result.attention.output_gate = aliases<bool>(
        metadata, {"attn_output_gate"}, result.evidence,
        "attention_output_gate");
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
}

/// Gated-delta key cluster of `normalize_model_metadata`, plus the adjacent
/// feed-forward auto-adjust and first-dense-layer statements that started the
/// same contiguous span.
void normalize_gated_delta_facts(const CheckpointMetadata& metadata, NormalizedModelMetadata& result) {
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
        metadata, {"layer_group_size", "global_attention_interval",
                   "full_attention_interval"}, result.evidence,
        "recurrent_hybrid_group_size");
}

/// Latent-attention cluster of `normalize_model_metadata`.
void normalize_latent_attention_facts(const CheckpointMetadata& metadata,
                                      NormalizedModelMetadata& result) {
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
}

/// MoE cluster of `normalize_model_metadata`.
void normalize_moe_facts(const CheckpointMetadata& metadata, NormalizedModelMetadata& result) {
    /// Each `gguf_suffix` resolves under the file's own architecture tag
    /// ("<arch>.<suffix>"), so GGUF MoE dialects need no per-architecture
    /// spelling in tree: `expert_count`, `expert_used_count`,
    /// `expert_feed_forward_length`, `expert_shared_feed_forward_length`.
    result.moe.experts = aliases<int>(
        metadata, {"num_experts"}, result.evidence, "moe_experts",
        "expert_count");
    result.moe.experts_per_token = aliases<int>(
        metadata, {"num_experts_per_tok", "experts_per_token"}, result.evidence,
        "moe_experts_per_token", "expert_used_count");
    result.moe.intermediate = aliases<int>(
        metadata, {"moe_intermediate_size"}, result.evidence, "moe_intermediate",
        "expert_feed_forward_length");
    result.moe.shared_intermediate = aliases<int>(
        metadata, {"moe_shared_expert_intermediate_size"}, result.evidence,
        "moe_shared_intermediate", "expert_shared_feed_forward_length");
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
}

/// XSA/tied-embedding cluster of `normalize_model_metadata`, plus the scoped
/// attention-head validations that followed it in the original body.
void normalize_xsa_ties(const CheckpointMetadata& metadata, NormalizedModelMetadata& result) {
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
}

/// Defaults block of `normalize_model_metadata`.
void normalize_defaults(const CheckpointMetadata& metadata, NormalizedModelMetadata& result) {
    (void)metadata;
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
}

/// `position_embedding_type` gate of `normalize_model_metadata`.
void normalize_position_embedding_gate(const CheckpointMetadata& metadata,
                                       NormalizedModelMetadata& result) {
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
}

/// RoPE pairing resolution of `normalize_model_metadata`; the trailing
/// cluster that consumes the facts returned by `normalize_rope_position_facts`.
void normalize_rope_pairing(const CheckpointMetadata& metadata, NormalizedModelMetadata& result,
                            const RopePositionFacts& facts) {
    const std::optional<double>& rope_theta = facts.rope_theta;
    const std::optional<float>& rotary_fraction = facts.rotary_fraction;
    const std::vector<int>& mrope_sections = facts.mrope_sections;
    const bool mrope_interleaved = facts.mrope_interleaved;
    const bool architecture_never_applies_rope = facts.architecture_never_applies_rope;
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
}

}
