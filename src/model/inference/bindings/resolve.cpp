#include "../canonical_internal.hpp"
#include "../rules.hpp"
#include "../support.hpp"

#include <cmath>
#include <utility>

#include "detail.hpp"

namespace celeg::inference_detail {

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

    apply_attention_output_scale(context);
    facts.graph.validate();
    facts.bindings = BindingSolver{}.solve(facts.bindings.values);
}

}
