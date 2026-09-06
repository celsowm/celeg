#include "../canonical_internal.hpp"
#include "../rules.hpp"
#include "../support.hpp"

#include <cmath>
#include <utility>

#include "detail.hpp"

namespace celeg::inference_detail {

/// Weightless per-head value RMSNorm for per-layer-input towers. Signature:
/// per-head `q_norm` and `k_norm` present, no `v_norm` weight tensor, alongside
/// a per-layer-input tower. This is the weakest of the structural claims --
/// a future checkpoint could ship the same grammar without the V norm, in which
/// case this would over-normalize -- so it is stated plainly here rather than
/// overstated. When the tower is absent (every currently-correct checkpoint)
/// nothing is touched. Generalizes to any checkpoint sharing this grammar,
/// without naming an architecture.
void apply_value_norm(CanonicalInferenceContext& context) {
    if (!context.facts.graph.per_layer_input.has_value()) return;
    if (!context.input.metadata.core.norm_epsilon.has_value()) return;
    const float eps = *context.input.metadata.core.norm_epsilon;
    for (int layer = 0; layer < context.layer_count; ++layer) {
        auto* attention = std::get_if<AttentionSpec>(
            &context.facts.graph.layers[static_cast<size_t>(layer)].mixer);
        if (attention == nullptr) continue;
        if (attention->uses_latent_state()) continue;
        if (!attention->query_norm.has_value() || !attention->key_norm.has_value()) {
            continue;
        }
        if (attention->query_norm->granularity != NormGranularity::PerHead ||
            attention->key_norm->granularity != NormGranularity::PerHead) {
            continue;
        }
        if (attention->value_norm.has_value()) continue;
        const int physical = context.physical_layer(layer);
        const std::string index = std::to_string(physical);
        bool has_v_norm_weight = false;
        for (const std::string& candidate :
             {std::string("blk.") + index + ".attn_v_norm.weight",
              std::string("model.layers.") + index + ".self_attn.v_norm.weight",
              std::string("model.layers.") + index + ".self_attn.v_layernorm.weight",
              std::string("model.language_model.layers.") + index +
                  ".self_attn.v_norm.weight",
              std::string("model.language_model.layers.") + index +
                  ".self_attn.v_layernorm.weight",
              std::string("layers.") + index + ".self_attn.v_norm.weight"}) {
            if (context.input.inventory.find(candidate) != nullptr) {
                has_v_norm_weight = true;
                break;
            }
        }
        if (has_v_norm_weight) continue;
        attention->value_norm =
            NormSpec{eps, NormWeightKind::None, NormGranularity::PerHead};
        context.facts.evidence.push_back(
            {EvidenceKind::Derived, "per_layer_input tower",
             "value_norm = weightless per-head derived from Q/K-norm + tower"});
    }
}

void apply_attention_output_scale(CanonicalInferenceContext& context) {
    const auto& m = context.input.metadata;
    auto& graph = context.facts.graph;
    auto& numerical_policy = context.facts.numerical_policy;

    /// Softmax scale 1.0 for per-layer-input towers: HF pins `scaling = 1.0`
    /// as a class constant alongside its per-head Q/K/V norms, absent from
    /// `config.json`. The tower celeg already detects is the evidence -- no
    /// tower checkpoint in the tree uses `1/sqrt(head_dim)`, and no non-tower
    /// checkpoint is touched by this branch, so existing models are
    /// bit-identical. Generalizes to any checkpoint sharing this grammar,
    /// without naming an architecture. Precedence: explicit
    /// `attention_multiplier` key wins, then this structural rule, then the
    /// `1/sqrt(head_dim)` default.
    const bool tower_implies_unit_scale =
        graph.per_layer_input.has_value() && !m.attention.attention_multiplier.has_value();
    if (tower_implies_unit_scale) {
        context.facts.evidence.push_back(
            {EvidenceKind::Derived, "per_layer_input tower",
             "attention_scale = 1.0 derived from per-layer-input tower"});
    }

    if (m.attention.attention_multiplier.has_value()) {
        numerical_policy.attention_multiplier =
            *m.attention.attention_multiplier;
    } else if (tower_implies_unit_scale) {
        numerical_policy.attention_multiplier = 1.0f;
    } else {
        int attention_head_dim = 0;
        for (int layer = 0; layer < context.layer_count; ++layer) {
            if (std::holds_alternative<AttentionSpec>(
                    graph.layers[static_cast<size_t>(layer)].mixer)) {
                attention_head_dim = std::get<AttentionSpec>(
                    graph.layers[static_cast<size_t>(layer)].mixer)
                                         .head_dim;
                break;
            }
        }
        numerical_policy.attention_multiplier =
            attention_head_dim > 0
                ? 1.0f /
                      std::sqrt(
                          static_cast<float>(attention_head_dim))
                : 1.0f;
    }

    /// Derive the query scale from each layer's own head_dim rather than the
    /// policy-wide multiplier: checkpoints that declare a per-layer head_dim
    /// schedule (a narrow width for sliding-attention layers and a wider one
    /// for the global/full-attention layers) need 1/sqrt(head_dim) computed
    /// per layer, otherwise the wider layers get their pre-softmax scores
    /// scaled by the narrower layers' factor. An explicit attention_multiplier
    /// in the checkpoint metadata still overrides the whole schedule.
    for (LayerSpec& semantic_layer : graph.layers) {
        if (auto* attention =
                std::get_if<AttentionSpec>(&semantic_layer.mixer);
            attention != nullptr && attention->query_heads > 0) {
            float layer_multiplier = 1.0f;
            if (m.attention.attention_multiplier.has_value()) {
                layer_multiplier = *m.attention.attention_multiplier;
            } else if (tower_implies_unit_scale) {
                /// Absolute 1.0 scale replaces the default; `query_scale` starts
                /// at 1.0 so `*=` here is a replacement, not a compound.
                layer_multiplier = 1.0f;
            } else {
                layer_multiplier = attention->head_dim > 0
                    ? 1.0f / std::sqrt(static_cast<float>(attention->head_dim))
                    : 1.0f;
            }
            attention->query_scale *= layer_multiplier;
        }
    }
}

}
