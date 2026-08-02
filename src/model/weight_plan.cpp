#include "celeg/model/weight_plan.hpp"

#include "celeg/model/weights/roles.hpp"

#include <utility>

namespace celeg {
namespace {

void add_request(ResolvedModel& model, TensorRequest request) {
    if (model.tensor_naming) {
        const auto names = model.tensor_naming->candidates(request);
        if (!names.empty()) {
            model.tensor_bindings.source_names.push_back(names.front());
        }
    }
    model.weight_plan.requests.push_back(std::move(request));
}

} // namespace

void build_dense_weight_plan(ResolvedModel& model) {
    const RuntimeTopology& t = model.topology;
    add_request(model, {TensorRole::TokenEmbedding, -1, -1,
                        {t.vocab_size, t.hidden}});
    add_request(model, {TensorRole::FinalNorm, -1, -1, {t.hidden}});
    add_request(model, {TensorRole::LanguageModelHead, -1, -1,
                        {t.vocab_size, t.hidden}});
    for (int layer = 0; layer < t.num_hidden_layers; ++layer) {
        add_request(model, {TensorRole::AttentionInputNorm, layer, -1,
                            {t.hidden}});
        const AttentionSpec& attention = t.attention_layout(layer);
        const int q_width = attention.query_heads * attention.head_dim;
        const int kv_width = attention.key_value_heads * attention.head_dim;
        add_request(model, {TensorRole::AttentionQuery, layer, -1,
                            {q_width, t.hidden}});
        add_request(model, {TensorRole::AttentionKey, layer, -1,
                            {kv_width, t.hidden}});
        add_request(model, {TensorRole::AttentionValue, layer, -1,
                            {kv_width, t.hidden}});
        add_request(model, {TensorRole::AttentionOutput, layer, -1,
                            {t.hidden, q_width}});
        add_request(model, {TensorRole::FfnInputNorm, layer, -1,
                            {t.hidden}});
        add_request(model, {TensorRole::FfnGate, layer, -1,
                            {t.intermediate, t.hidden}});
        add_request(model, {TensorRole::FfnUp, layer, -1,
                            {t.intermediate, t.hidden}});
        add_request(model, {TensorRole::FfnDown, layer, -1,
                            {t.hidden, t.intermediate}});
    }
}

} // namespace celeg
