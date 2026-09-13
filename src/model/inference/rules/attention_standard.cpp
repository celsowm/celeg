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
namespace {

/// Recognizes the ordinary GQA attention grammar (q/k/v/o projections and
/// optional q/k norms) in any known spelling convention.
class StandardAttentionRule final : public ILayerInferenceRule {
public:
    std::string_view id() const override { return "standard_attention"; }
    int specificity() const override { return 2; }
    MixerFamily family() const override { return MixerFamily::Attention; }

    bool probe(const CanonicalInferenceContext& context, int layer)
        const override {
        const auto& inventory = context.input.inventory;
        const int physical_layer = context.physical_layer(layer);

        /// Factorized Gated Delta Net checkpoints such as Ling intentionally
        /// store q/k/v/o projections below an `.attention.` prefix.  Those
        /// projection names alone therefore do not establish standard
        /// attention.  Recurrent-only f_proj + q_conv1d are discriminators for
        /// the more specific factorized_gated_delta grammar; let that rule own
        /// the layer instead of manufacturing a cross-family ambiguity.
        const std::string recurrent_prefix =
            "model.layers." + std::to_string(physical_layer) + ".attention.";
        if (inventory.find(recurrent_prefix + "f_proj.weight") != nullptr &&
            inventory.find(recurrent_prefix + "q_conv1d.weight") != nullptr) {
            return false;
        }

        for (const std::string& candidate :
             attention_tensor_candidates(physical_layer, "q_proj.weight")) {
            if (inventory.find(candidate) != nullptr) return true;
        }
        return false;
    }

    void resolve(CanonicalInferenceContext& context, int layer)
        const override {
        const auto& input = context.input;
        const auto& m = input.metadata;
        auto& bindings = context.facts.bindings;
        const int physical_layer = context.physical_layer(layer);
        LayerSpec& semantic_layer =
            context.facts.graph.layers[static_cast<size_t>(layer)];

        const auto query_heads = m.attention.query_heads.value_for(layer);
        const auto key_value_heads = m.attention.key_value_heads.value_for(layer);
        const auto explicit_head_dim = m.attention.head_dim.value_for(layer);
        if (!query_heads.has_value() || !key_value_heads.has_value()) {
            fail(
                ResolutionFailureKind::MissingRequiredMetadata,
                "automatic resolution requires attention geometry for layer " +
                    std::to_string(layer));
        }
        if (*query_heads <= 0) {
            fail(
                ResolutionFailureKind::ConflictingInferenceFacts,
                "query_heads is non-positive for layer " +
                    std::to_string(layer));
        }

        const int head_dim =
            explicit_head_dim.value_or(*m.core.hidden_size / *query_heads);
        if (*key_value_heads <= 0 ||
            *query_heads % *key_value_heads != 0 ||
            head_dim <= 0 || head_dim % 2 != 0 ||
            *key_value_heads * head_dim > *m.core.hidden_size) {
            fail(
                ResolutionFailureKind::ConflictingInferenceFacts,
                "attention head geometry is not a valid GQA layout for layer " +
                    std::to_string(layer));
        }

        const int query_width = *query_heads * head_dim;
        const int key_value_width = *key_value_heads * head_dim;
        const TensorInventoryEntry* query_norm = find_optional_unique(
            input.inventory, query_norm_candidates(physical_layer),
            TensorRole::AttentionQueryNorm, layer);
        const TensorInventoryEntry* key_norm = find_optional_unique(
            input.inventory, key_norm_candidates(physical_layer),
            TensorRole::AttentionKeyNorm, layer);
        const bool metadata_qk_norm = *m.attention.query_key_norm;
        if (metadata_qk_norm && (query_norm == nullptr || key_norm == nullptr)) {
            fail(
                ResolutionFailureKind::MissingTensorRole,
                "query/key normalization is declared but its tensors are incomplete for layer " +
                    std::to_string(layer));
        }
        const bool has_query_norm = metadata_qk_norm || query_norm != nullptr;
        const bool has_key_norm = metadata_qk_norm || key_norm != nullptr;
        if (has_query_norm != has_key_norm) {
            fail(
                ResolutionFailureKind::ConflictingInferenceFacts,
                "query/key normalization evidence is incomplete for layer " +
                    std::to_string(layer));
        }

        const NormGranularity query_norm_granularity = query_norm
            ? infer_qk_norm_granularity(
                  *query_norm,
                  TensorRole::AttentionQueryNorm,
                  layer,
                  head_dim,
                  query_width)
            : NormGranularity::PerHead;
        const NormGranularity key_norm_granularity = key_norm
            ? infer_qk_norm_granularity(
                  *key_norm,
                  TensorRole::AttentionKeyNorm,
                  layer,
                  head_dim,
                  key_value_width)
            : NormGranularity::PerHead;

        AttentionSpec attention = make_attention(
            m,
            layer,
            *query_heads,
            *key_value_heads,
            head_dim,
            has_query_norm,
            query_norm_granularity,
            key_norm_granularity,
            context.facts.numerical_policy.norm_weight_kind);

        const auto q_candidates =
            attention_tensor_candidates(physical_layer, "q_proj.weight");
        const TensorInventoryEntry* query = nullptr;
        for (const std::string& name : q_candidates) {
            if (const auto* candidate = input.inventory.find(name)) {
                if (query != nullptr) {
                    fail(
                        ResolutionFailureKind::AmbiguousTensorBinding,
                        "multiple query projections are present for layer " +
                            std::to_string(layer));
                }
                query = candidate;
            }
        }
        if (query &&
            shape_is(
                *query,
                {2 * attention.query_width(), *m.core.hidden_size})) {
            attention.output_gate = SigmoidAttentionGateSpec{
                true,
                AttentionGateGranularity::ElementWise};
        }
        if (m.attention.output_gate.has_value()) {
            const bool stated = *m.attention.output_gate;
            const bool inferred = attention.output_gate.has_value();
            if (stated != inferred) {
                fail(
                    ResolutionFailureKind::ConflictingMetadata,
                    "stated attn_output_gate disagrees with the query-projection shape for layer " +
                        std::to_string(layer));
            }
        }

        semantic_layer.mixer = std::move(attention);
        if (!layer_has_feed_forward(context, layer)) {
            semantic_layer.feed_forward = std::monostate{};
        }

        const AttentionSpec& resolved =
            std::get<AttentionSpec>(semantic_layer.mixer);
        const int query_head_count = *m.attention.query_heads.value_for(layer);
        const int key_value_head_count = *m.attention.key_value_heads.value_for(layer);
        const int layer_head_dim = m.attention.head_dim.value_for(layer).value_or(
            *m.core.hidden_size / query_head_count);

        const auto* q = find_unique(
            input.inventory,
            attention_tensor_candidates(physical_layer, "q_proj.weight"),
            TensorRole::AttentionQuery,
            layer,
            {resolved.query_projection_width(), *m.core.hidden_size},
            {});
        add_binding(bindings, TensorRole::AttentionQuery, layer, *q, {});

        const auto* k = find_unique(
            input.inventory,
            attention_tensor_candidates(physical_layer, "k_proj.weight"),
            TensorRole::AttentionKey,
            layer,
            {key_value_head_count * layer_head_dim, *m.core.hidden_size},
            {});
        add_binding(bindings, TensorRole::AttentionKey, layer, *k, {});

        const auto* v = find_unique(
            input.inventory,
            attention_tensor_candidates(physical_layer, "v_proj.weight"),
            TensorRole::AttentionValue,
            layer,
            {key_value_head_count * layer_head_dim, *m.core.hidden_size},
            {});
        add_binding(bindings, TensorRole::AttentionValue, layer, *v, {});

        const auto* o = find_unique(
            input.inventory,
            attention_tensor_candidates(physical_layer, "o_proj.weight"),
            TensorRole::AttentionOutput,
            layer,
            {*m.core.hidden_size, query_head_count * layer_head_dim},
            {});
        add_binding(bindings, TensorRole::AttentionOutput, layer, *o, {});

        if (query_norm != nullptr) {
            add_binding(
                bindings,
                TensorRole::AttentionQueryNorm,
                layer,
                *query_norm,
                {});
        }
        if (key_norm != nullptr) {
            add_binding(
                bindings,
                TensorRole::AttentionKeyNorm,
                layer,
                *key_norm,
                {});
        }
    }
};

}

std::unique_ptr<ILayerInferenceRule> make_standard_attention_rule() {
    return std::make_unique<StandardAttentionRule>();
}

}
