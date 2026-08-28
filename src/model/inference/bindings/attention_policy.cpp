#include "../canonical_internal.hpp"
#include "../rules.hpp"
#include "../support.hpp"

#include <cmath>
#include <utility>

#include "detail.hpp"

namespace celeg::inference_detail {

void apply_attention_output_scale(CanonicalInferenceContext& context) {
    const auto& m = context.input.metadata;
    auto& graph = context.facts.graph;
    auto& numerical_policy = context.facts.numerical_policy;

    if (m.attention.attention_multiplier.has_value()) {
        numerical_policy.attention_multiplier =
            *m.attention.attention_multiplier;
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
            const float layer_multiplier =
                m.attention.attention_multiplier.has_value()
                    ? *m.attention.attention_multiplier
                    : (attention->head_dim > 0
                           ? 1.0f / std::sqrt(
                                        static_cast<float>(attention->head_dim))
                           : 1.0f);
            attention->query_scale *= layer_multiplier;
        }
    }
}

}
