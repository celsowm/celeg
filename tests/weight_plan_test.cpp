#include "celeg/model/weight_plan.hpp"

#include <iostream>
#include <stdexcept>

namespace {

class Naming final : public celeg::ITensorNamingPolicy {
public:
    std::vector<std::string> candidates(
        const celeg::TensorRequest&) const override {
        return {};
    }
};

} // namespace

int main() {
    try {
        celeg::ResolvedModel model;
        model.topology.exec.hidden = 8;
        model.topology.exec.intermediate = 16;
        model.topology.dims.vocab_size = 32;
        model.topology.exec.num_hidden_layers = 2;
        model.topology.exec.feed_forward_intermediates = {12, 20};
        model.topology.exec.attention_layouts.resize(2);
        for (auto& attention : model.topology.exec.attention_layouts) {
            attention.query_heads = 1;
            attention.key_value_heads = 1;
            attention.head_dim = 8;
        }
        model.graph.hidden = 8;
        model.graph.final_norm = {1.0e-5f, celeg::NormWeightKind::Scale};
        for (const int intermediate : {12, 20}) {
            celeg::LayerSpec layer;
            layer.operator_norm = {1.0e-5f, celeg::NormWeightKind::Scale};
            layer.feed_forward_norm = {1.0e-5f, celeg::NormWeightKind::Scale};
            layer.mixer = model.topology.exec.attention_layouts[model.graph.layers.size()];
            layer.feed_forward = celeg::DenseFeedForwardSpec{
                intermediate, celeg::ActivationKind::SwiGLU};
            model.graph.layers.push_back(std::move(layer));
        }

        Naming naming;
        celeg::build_weight_plan_from_graph(model, naming);
        const auto find = [&](celeg::TensorRole role, int layer) -> const celeg::TensorRequest& {
            for (const auto& request : model.weight_plan.requests) {
                if (request.role == role && request.layer == layer) return request;
            }
            throw std::runtime_error("weight request missing");
        };
        if (find(celeg::TensorRole::FfnGate, 0).expected_shape[0] != 12 ||
            find(celeg::TensorRole::FfnUp, 1).expected_shape[0] != 20 ||
            find(celeg::TensorRole::FfnDown, 1).expected_shape[1] != 20) {
            throw std::runtime_error("per-layer FFN widths were ignored");
        }
        std::cout << "weight_plan_test: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "weight_plan_test: " << error.what() << '\n';
        return 1;
    }
}
