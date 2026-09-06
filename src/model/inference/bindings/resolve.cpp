#include "../canonical_internal.hpp"
#include "../rules.hpp"
#include "../support.hpp"

#include <cmath>
#include <utility>

#include "detail.hpp"

namespace celeg::inference_detail {

/// Verifies a stated hybrid attention schedule (`layer_group_size`) against
/// the resolved mixers: the reference decoder makes every Nth layer (plus the
/// tail block) full attention and the rest linear attention, while celeg
/// resolves each layer from tensor grammar alone. A grammar-resolved model
/// that disagrees with its own schedule claim is misresolved, so the layers
/// fail loudly here instead of running with silently swapped mathematics.
void verify_hybrid_attention_schedule(CanonicalInferenceContext& context) {
    const std::optional<int> group_size =
        context.input.metadata.gated_delta.hybrid_group_size;
    if (!group_size.has_value() || *group_size <= 0) return;
    const int groups = *group_size;
    const int tail_start =
        context.layer_count / groups * groups;
    for (int layer = 0; layer < context.layer_count; ++layer) {
        const bool expect_attention =
            (layer + 1) % groups == 0 || layer >= tail_start;
        const MixerSpec& mixer =
            context.facts.graph.layers[static_cast<size_t>(layer)].mixer;
        const bool is_attention =
            std::holds_alternative<AttentionSpec>(mixer);
        const bool is_linear =
            std::holds_alternative<GatedDeltaNetSpec>(mixer);
        if (!is_attention && !is_linear) {
            fail(
                ResolutionFailureKind::ConflictingMetadata,
                "hybrid attention schedule does not cover mixer type for layer " +
                    std::to_string(layer));
        }
        if (is_attention != expect_attention) {
            fail(
                ResolutionFailureKind::ConflictingMetadata,
                "resolved mixer disagrees with hybrid attention schedule for layer " +
                    std::to_string(layer));
        }
    }
}

void resolve_canonical_layers(CanonicalInferenceContext& context) {
    auto& facts = context.facts;
    const auto rules = make_builtin_layer_inference_rules();

    bind_global_tensors(context);
    for (int layer = 0; layer < context.layer_count; ++layer) {
        const ILayerInferenceRule& rule =
            select_layer_inference_rule(rules, context, layer);
        rule.resolve(context, layer);
        validate_query_key_norm_consistency(context, layer);
        infer_and_bind_layer_norms(context, layer);
        resolve_layer_feed_forward(context, layer);
    }
    verify_hybrid_attention_schedule(context);

    bind_per_layer_input(context);
    apply_value_norm(context);
    apply_attention_output_scale(context);
    facts.graph.validate();
    facts.bindings = BindingSolver{}.solve(facts.bindings.values);
}

}
