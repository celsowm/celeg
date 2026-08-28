#include "../canonical_internal.hpp"
#include "../rules.hpp"
#include "../support.hpp"

#include <cmath>
#include <utility>

#include "detail.hpp"

namespace celeg::inference_detail {
namespace {

const TensorInventoryEntry* find_optional_norm(
    const TensorInventory& inventory,
    const std::vector<std::string>& candidates,
    TensorRole role,
    int layer,
    int hidden) {
    const TensorInventoryEntry* match = nullptr;
    for (const std::string& candidate : candidates) {
        if (const auto* tensor = inventory.find(candidate)) {
            if (match != nullptr) {
                fail(
                    ResolutionFailureKind::AmbiguousTensorBinding,
                    "automatic resolution found multiple bindings for " +
                        std::string(tensor_role_name(role)) + " for layer " +
                        std::to_string(layer));
            }
            match = tensor;
        }
    }
    if (match != nullptr && !shape_is(*match, {hidden})) {
        fail(
            ResolutionFailureKind::ShapeConstraintViolation,
            "tensor " + match->name + " has a shape inconsistent with " +
                std::string(tensor_role_name(role)));
    }
    return match;
}

void require_explicit_norm_consistency(std::optional<bool> explicit_value,
                                       const TensorInventoryEntry* tensor,
                                       TensorRole role,
                                       int layer) {
    if (!explicit_value.has_value()) return;
    if (*explicit_value && tensor == nullptr) {
        fail(
            ResolutionFailureKind::MissingTensorRole,
            "metadata requires " + std::string(tensor_role_name(role)) +
                " but checkpoint has no matching tensor for layer " +
                std::to_string(layer));
    }
    if (!*explicit_value && tensor != nullptr) {
        fail(
            ResolutionFailureKind::ConflictingInferenceFacts,
            "metadata disables " + std::string(tensor_role_name(role)) +
                " but checkpoint tensor proves it for layer " +
                std::to_string(layer));
    }
}

bool query_key_norm_was_explicit(const NormalizedModelMetadata& metadata) {
    for (const EvidenceItem& evidence : metadata.evidence) {
        if ((evidence.kind == EvidenceKind::AliasMetadata ||
             evidence.kind == EvidenceKind::ExplicitMetadata) &&
            evidence.fact.starts_with("query_key_norm =")) {
            return true;
        }
    }
    return false;
}

}

void validate_query_key_norm_consistency(CanonicalInferenceContext& context,
                                         int layer) {
    const auto& metadata = context.input.metadata;
    if (!query_key_norm_was_explicit(metadata)) return;
    const bool ordinary_attention =
        context.facts.bindings.find(TensorRole::AttentionQuery, layer) != nullptr &&
        context.facts.bindings.find(TensorRole::AttentionKey, layer) != nullptr &&
        context.facts.bindings.find(TensorRole::AttentionValue, layer) != nullptr;
    if (!ordinary_attention) return;
    const bool requested = metadata.attention.query_key_norm.value_or(false);
    const bool query_bound =
        context.facts.bindings.find(TensorRole::AttentionQueryNorm, layer) != nullptr;
    const bool key_bound =
        context.facts.bindings.find(TensorRole::AttentionKeyNorm, layer) != nullptr;
    if (requested && (!query_bound || !key_bound)) {
        fail(
            ResolutionFailureKind::MissingTensorRole,
            "metadata requires query/key normalization but both norm tensors are not "
            "bound for layer " + std::to_string(layer));
    }
    if (!requested && (query_bound || key_bound)) {
        fail(
            ResolutionFailureKind::ConflictingInferenceFacts,
            "metadata disables query/key normalization but checkpoint tensors prove it "
            "for layer " + std::to_string(layer));
    }
}

namespace {

void bind_structural_norm(CanonicalInferenceContext& context,
                          int layer,
                          TensorRole role,
                          const TensorInventoryEntry* tensor,
                          std::optional<NormSpec>& slot) {
    if (tensor == nullptr) {
        slot.reset();
        return;
    }
    slot = NormSpec{
        context.facts.numerical_policy.norm_eps,
        context.facts.numerical_policy.norm_weight_kind};
    add_binding(
        context.facts.bindings,
        role,
        layer,
        *tensor,
        {{EvidenceKind::TensorName,
          tensor->name,
          std::string(tensor_role_name(role))}});
}

}

void infer_and_bind_layer_norms(CanonicalInferenceContext& context,
                                int layer) {
    const auto& input = context.input;
    const auto& norm_facts = input.metadata.norms;
    const int hidden = *input.metadata.core.hidden_size;
    const std::string index = std::to_string(context.physical_layer(layer));
    LayerSpec& semantic_layer =
        context.facts.graph.layers[static_cast<size_t>(layer)];

    const std::vector<std::string> mixer_before_candidates = {
        "transformer.h." + index + ".ln_1.weight",
        "model.layers." + index + ".input_layernorm.weight",
        "model.layers." + index + ".self_attn_layer_norm.weight",
        "model.language_model.layers." + index + ".input_layernorm.weight",
        "model.language_model.layers." + index + ".operator_norm.weight",
        "model.layers." + index + ".operator_norm.weight",
        "backbone.layers." + index + ".norm.weight",
        "blk." + index + ".attn_norm.weight",
    };
    const std::vector<std::string> explicit_mixer_after_candidates = {
        "model.layers." + index + ".post_attention_norm.weight",
        "model.language_model.layers." + index + ".post_attention_norm.weight",
        "model.layers." + index + ".post_attn_norm.weight",
        "model.language_model.layers." + index + ".post_attn_norm.weight",
        "layers." + index + ".post_attn_norm.weight",
        "blk." + index + ".post_attention_norm.weight",
    };
    const std::vector<std::string> post_attention_layernorm_candidates = {
        "model.layers." + index + ".post_attention_layernorm.weight",
        "model.language_model.layers." + index + ".post_attention_layernorm.weight",
    };
    const std::vector<std::string> explicit_ffn_before_candidates = {
        "transformer.h." + index + ".ln_2.weight",
        "model.language_model.layers." + index + ".ffn_norm.weight",
        "model.layers." + index + ".ffn_norm.weight",
        "blk." + index + ".ffn_norm.weight",
    };
    const std::vector<std::string> pre_feedforward_candidates = {
        "model.layers." + index + ".pre_feedforward_layernorm.weight",
        "model.language_model.layers." + index + ".pre_feedforward_layernorm.weight",
        "model.layers." + index + ".pre_feed_forward_layernorm.weight",
        "model.language_model.layers." + index + ".pre_feed_forward_layernorm.weight",
    };
    const std::vector<std::string> ffn_after_candidates = {
        "model.layers." + index + ".post_feedforward_layernorm.weight",
        "model.language_model.layers." + index + ".post_feedforward_layernorm.weight",
        "model.layers." + index + ".post_feed_forward_layernorm.weight",
        "model.language_model.layers." + index + ".post_feed_forward_layernorm.weight",
        "model.layers." + index + ".post_ffn_norm.weight",
        "model.language_model.layers." + index + ".post_ffn_norm.weight",
        "model.layers." + index + ".post_mlp_norm.weight",
        "model.language_model.layers." + index + ".post_mlp_norm.weight",
        "layers." + index + ".post_mlp_norm.weight",
        "blk." + index + ".post_ffn_norm.weight",
        "blk." + index + ".ffn_post_norm.weight",
        "blk." + index + ".post_ffw_norm.weight",
    };

    const TensorInventoryEntry* mixer_before = find_optional_norm(
        input.inventory, mixer_before_candidates,
        TensorRole::AttentionInputNorm, layer, hidden);
    const TensorInventoryEntry* explicit_mixer_after = find_optional_norm(
        input.inventory, explicit_mixer_after_candidates,
        TensorRole::AttentionPostNorm, layer, hidden);
    const TensorInventoryEntry* post_attention_layernorm = find_optional_norm(
        input.inventory, post_attention_layernorm_candidates,
        TensorRole::AttentionPostNorm, layer, hidden);
    const TensorInventoryEntry* explicit_ffn_before = find_optional_norm(
        input.inventory, explicit_ffn_before_candidates,
        TensorRole::FfnInputNorm, layer, hidden);
    const TensorInventoryEntry* pre_feedforward = find_optional_norm(
        input.inventory, pre_feedforward_candidates,
        TensorRole::FfnInputNorm, layer, hidden);
    const TensorInventoryEntry* ffn_after = find_optional_norm(
        input.inventory, ffn_after_candidates,
        TensorRole::FfnOutputNorm, layer, hidden);

    const std::optional<bool> metadata_mixer_after =
        norm_facts.mixer_after.value_for(layer);
    const std::optional<bool> metadata_ffn_before =
        norm_facts.feed_forward_before.value_for(layer);

    const TensorInventoryEntry* mixer_after = explicit_mixer_after;
    const TensorInventoryEntry* ffn_before = explicit_ffn_before;
    if (pre_feedforward != nullptr) {
        if (explicit_ffn_before != nullptr) {
            fail(
                ResolutionFailureKind::AmbiguousTensorBinding,
                "automatic resolution found multiple pre-feed-forward norms for layer " +
                    std::to_string(layer));
        }
        ffn_before = pre_feedforward;
    }

    if (post_attention_layernorm != nullptr) {
        if (metadata_mixer_after == true && metadata_ffn_before == true) {
            fail(
                ResolutionFailureKind::ConflictingMetadata,
                "one post_attention_layernorm tensor cannot satisfy both mixer-after and "
                "feed-forward-before metadata for layer " + std::to_string(layer));
        }
        const bool explicitly_mixer_after = metadata_mixer_after == true;
        const bool explicitly_ffn_before = metadata_ffn_before == true;
        const bool post_norm_topology =
            explicitly_mixer_after ||
            (!explicitly_ffn_before &&
             (pre_feedforward != nullptr || ffn_after != nullptr));
        if (post_norm_topology) {
            if (explicit_mixer_after != nullptr) {
                fail(
                    ResolutionFailureKind::AmbiguousTensorBinding,
                    "automatic resolution found multiple post-attention norms for layer " +
                        std::to_string(layer));
            }
            mixer_after = post_attention_layernorm;
        } else {
            if (explicit_ffn_before != nullptr) {
                fail(
                    ResolutionFailureKind::AmbiguousTensorBinding,
                    "post_attention_layernorm conflicts with another feed-forward input norm "
                    "for layer " + std::to_string(layer));
            }
            ffn_before = post_attention_layernorm;
        }
    }

    require_explicit_norm_consistency(
        norm_facts.mixer_before.value_for(layer), mixer_before,
        TensorRole::AttentionInputNorm, layer);
    require_explicit_norm_consistency(
        metadata_mixer_after, mixer_after,
        TensorRole::AttentionPostNorm, layer);
    require_explicit_norm_consistency(
        metadata_ffn_before, ffn_before,
        TensorRole::FfnInputNorm, layer);
    require_explicit_norm_consistency(
        norm_facts.feed_forward_after.value_for(layer), ffn_after,
        TensorRole::FfnOutputNorm, layer);

    bind_structural_norm(
        context, layer, TensorRole::AttentionInputNorm,
        mixer_before, semantic_layer.mixer_norm.before);
    bind_structural_norm(
        context, layer, TensorRole::AttentionPostNorm,
        mixer_after, semantic_layer.mixer_norm.after);

    const bool has_feed_forward =
        !std::holds_alternative<std::monostate>(semantic_layer.feed_forward);
    if (!has_feed_forward && (ffn_before != nullptr || ffn_after != nullptr)) {
        const std::string arch = context.input.architecture_type;
        if (arch.find("vision") != std::string::npos || arch.find("vl") != std::string::npos || arch.find("VL") != std::string::npos) {
            // Vision/encoder layers may have norms without FFN - allow this
            semantic_layer.feed_forward_norm = {};
        } else {
            fail(
                ResolutionFailureKind::ConflictingInferenceFacts,
                "checkpoint exposes feed-forward normalization for a layer with no "
                "feed-forward semantics: " + std::to_string(layer));
        }
    }
    if (has_feed_forward) {
        bind_structural_norm(
            context, layer, TensorRole::FfnInputNorm,
            ffn_before, semantic_layer.feed_forward_norm.before);
        bind_structural_norm(
            context, layer, TensorRole::FfnOutputNorm,
            ffn_after, semantic_layer.feed_forward_norm.after);
    } else {
        semantic_layer.feed_forward_norm = {};
    }
}

}
