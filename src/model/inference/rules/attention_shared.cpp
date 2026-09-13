#include "attention_detail.hpp"

#include "../rules.hpp"
#include "../support.hpp"

#include "celeg/model/definition.hpp"

#include <cmath>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace celeg::inference_detail {

AttentionSpec make_attention(
    const NormalizedModelMetadata& metadata,
    int layer,
    int query_heads,
    int key_value_heads,
    int head_dim,
    bool query_key_norm,
    NormGranularity query_norm_granularity,
    NormGranularity key_norm_granularity,
    NormWeightKind norm_weight_kind) {
    AttentionSpec attention;
    attention.query_heads = query_heads;
    attention.key_value_heads = key_value_heads;
    attention.head_dim = head_dim;
    const auto optional_qk_norm = [&](NormGranularity granularity)
        -> std::optional<NormSpec> {
        if (!query_key_norm || !std::isfinite(*metadata.core.norm_epsilon) ||
            *metadata.core.norm_epsilon <= 0.0f) {
            return std::nullopt;
        }
        return NormSpec{
            *metadata.core.norm_epsilon,
            norm_weight_kind,
            granularity};
    };
    attention.query_norm = optional_qk_norm(query_norm_granularity);
    attention.key_norm = optional_qk_norm(key_norm_granularity);
    attention.pattern = FullCausalPattern{};
    attention.query_scale = 1.0f;
    /// Per-pattern positional encoding resolves through `value_for(layer)`,
    /// falling back to `global` when no nested `rope_parameters.<layer_type>`
    /// block exists, so single-theta checkpoints are unaffected.
    const std::optional<InferredPositionEncoding> inferred =
        metadata.attention.position_encoding.value_for(layer);
    if (!inferred.has_value()) {
        fail(ResolutionFailureKind::MissingRequiredMetadata,
             "positional encoding could not be resolved from checkpoint metadata for layer " +
                 std::to_string(layer));
    }
    std::visit([&](const auto& position) {
        using T = std::decay_t<decltype(position)>;
        if constexpr (std::is_same_v<T, NoPositionEncodingSpec>) {
            attention.position = NoPositionEncodingSpec{};
        } else if constexpr (std::is_same_v<T, InferredRopePosition>) {
            RopePositionSpec rope{position.theta, position.rotary_fraction, RopeScalingSpec{}};
            rope.pairing = position.pairing;
            if (!position.mrope_sections.empty()) {
                attention.position = MultiAxisRopeSpec{
                    rope,
                    {position.mrope_sections[0], position.mrope_sections[1],
                     position.mrope_sections[2]},
                    position.mrope_interleaved, 3};
            } else {
                attention.position = rope;
            }
        } else if constexpr (std::is_same_v<T, UnresolvedPositionEncoding>) {
            fail(ResolutionFailureKind::MissingRequiredMetadata,
                 "positional encoding could not be resolved from checkpoint metadata for layer " +
                     std::to_string(layer));
        } else {
            static_assert(always_false_v<T>, "unhandled inferred position encoding alternative");
        }
    }, *inferred);
    if (*metadata.attention.xsa_projection) {
        attention.output_transform = OrthogonalizeCurrentValueSpec{
            *metadata.attention.xsa_minimum_norm_squared};
    }
    return attention;
}

std::vector<std::string> query_norm_candidates(int layer) {
    const std::string index = std::to_string(layer);
    return {
        "blk." + index + ".attn_q_norm.weight",
        "model.layers." + index + ".self_attn.q_layernorm.weight",
        "model.layers." + index + ".self_attn.q_norm.weight",
        "model.language_model.layers." + index + ".self_attn.q_layernorm.weight",
        "model.language_model.layers." + index + ".self_attn.q_norm.weight",
        "model.layers." + index + ".global_attn.q_layernorm.weight",
        "model.layers." + index + ".global_attn.q_norm.weight",
        "model.language_model.layers." + index + ".global_attn.q_layernorm.weight",
        "model.language_model.layers." + index + ".global_attn.q_norm.weight",
        "layers." + index + ".self_attn.q_layernorm.weight",
        "layers." + index + ".self_attn.q_norm.weight",
    };
}

std::vector<std::string> key_norm_candidates(int layer) {
    const std::string index = std::to_string(layer);
    return {
        "blk." + index + ".attn_k_norm.weight",
        "model.layers." + index + ".self_attn.k_layernorm.weight",
        "model.layers." + index + ".self_attn.k_norm.weight",
        "model.language_model.layers." + index + ".self_attn.k_layernorm.weight",
        "model.language_model.layers." + index + ".self_attn.k_norm.weight",
        "model.layers." + index + ".global_attn.k_layernorm.weight",
        "model.layers." + index + ".global_attn.k_norm.weight",
        "model.language_model.layers." + index + ".global_attn.k_layernorm.weight",
        "model.language_model.layers." + index + ".global_attn.k_norm.weight",
        "layers." + index + ".self_attn.k_layernorm.weight",
        "layers." + index + ".self_attn.k_norm.weight",
    };
}

const TensorInventoryEntry* find_optional_unique(
    const TensorInventory& inventory,
    const std::vector<std::string>& candidates,
    TensorRole role,
    int layer) {
    const TensorInventoryEntry* found = nullptr;
    for (const std::string& candidate : candidates) {
        if (const auto* tensor = inventory.find(candidate)) {
            if (found != nullptr) {
                fail(
                    ResolutionFailureKind::AmbiguousTensorBinding,
                    "multiple query/key norm spellings are present for " +
                        std::string(tensor_role_name(role)) + " for layer " +
                        std::to_string(layer));
            }
            found = tensor;
        }
    }
    return found;
}

NormGranularity infer_qk_norm_granularity(
    const TensorInventoryEntry& tensor,
    TensorRole role,
    int layer,
    int per_head_width,
    int whole_width) {
    if (shape_is(tensor, {per_head_width})) {
        return NormGranularity::PerHead;
    }
    if (shape_is(tensor, {whole_width})) {
        return NormGranularity::WholeVector;
    }
    fail(
        ResolutionFailureKind::ShapeConstraintViolation,
        "tensor " + tensor.name + " has a shape inconsistent with " +
            std::string(tensor_role_name(role)) + " for layer " +
            std::to_string(layer) + "; expected per-head width " +
            std::to_string(per_head_width) + " or whole-projection width " +
            std::to_string(whole_width));
}

}
