#include "celeg/model/graph_builder.hpp"

#include <utility>
#include <stdexcept>

namespace celeg {

void build_dense_transformer_graph(ResolvedModel& model) {
    const RuntimeTopology& t = model.topology;
    model.graph.embedding_multiplier = t.numerical_policy.embedding_multiplier;
    model.graph.logits_divisor = t.numerical_policy.logits_divisor;
    model.graph.final_norm.epsilon = t.numerical_policy.norm_eps;
    model.graph.final_logit_softcap = 0.0f;
    model.graph.norm_after_layers.clear();
    model.graph.layers.clear();
    model.graph.layers.reserve(static_cast<size_t>(t.num_hidden_layers));
    for (int i = 0; i < t.num_hidden_layers; ++i) {
        LayerSpec layer;
        layer.operator_norm.epsilon = t.numerical_policy.norm_eps;
        layer.feed_forward_norm.epsilon = t.numerical_policy.norm_eps;
        if (t.mixer_kinds[static_cast<size_t>(i)] == MixerKind::Attention) {
            if (i >= static_cast<int>(t.attention_layouts.size())) {
                throw std::runtime_error("attention layer has no resolved layout");
            }
            layer.mixer = t.attention_layout(i);
        } else {
            layer.mixer = ShortConvolutionSpec{t.conv_cache, t.conv_dim, false};
        }
        layer.feed_forward = DenseFeedForwardSpec{t.intermediate};
        layer.residual.multiplier = t.numerical_policy.residual_multiplier;
        model.graph.layers.push_back(std::move(layer));
    }
}

} // namespace celeg
