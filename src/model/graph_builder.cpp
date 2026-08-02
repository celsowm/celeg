#include "celeg/model/graph_builder.hpp"

#include <utility>

namespace celeg {

void build_dense_transformer_graph(ResolvedModel& model) {
    const RuntimeTopology& t = model.topology;
    model.graph.embedding_multiplier = t.embedding_multiplier;
    model.graph.logits_divisor = t.logits_divisor;
    model.graph.final_norm.epsilon = t.norm_eps;
    model.graph.layers.clear();
    model.graph.layers.reserve(static_cast<size_t>(t.num_hidden_layers));
    for (int i = 0; i < t.num_hidden_layers; ++i) {
        LayerSpec layer;
        layer.operator_norm.epsilon = t.norm_eps;
        layer.feed_forward_norm.epsilon = t.norm_eps;
        layer.mixer = AttentionSpec{t.num_attention_heads, t.num_key_value_heads,
                                    t.head_dim, t.query_key_norm};
        layer.feed_forward = DenseFeedForwardSpec{t.intermediate};
        layer.residual.multiplier = t.residual_multiplier;
        model.graph.layers.push_back(std::move(layer));
    }
}

} // namespace celeg
