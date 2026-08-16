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

}

int main() {
    try {
        celeg::ResolvedModel model;
        model.topology.dims.vocab_size = 32;
        model.graph.hidden = 8;
        model.graph.final_norm = {1.0e-5f, celeg::NormWeightKind::Scale};
        for (const int intermediate : {12, 20}) {
            celeg::LayerSpec layer;
            layer.operator_norm = {1.0e-5f, celeg::NormWeightKind::Scale};
            layer.feed_forward_norm = {1.0e-5f, celeg::NormWeightKind::Scale};
            celeg::AttentionSpec attention;
            attention.query_heads = 1;
            attention.key_value_heads = 1;
            attention.head_dim = 8;
            layer.mixer = attention;
            layer.feed_forward = celeg::DenseFeedForwardSpec{
                intermediate, celeg::ActivationKind::SwiGLU};
            model.graph.layers.push_back(std::move(layer));
        }

        model.topology = celeg::compose_runtime_topology(
            celeg::CheckpointDimensions{32, 0, {}, {}, 0}, model.graph);

        Naming naming;
        celeg::build_weight_plan_from_graph(model, naming, nullptr);
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
