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
        model.topology.hidden = 8;
        model.topology.intermediate = 16;
        model.topology.num_hidden_layers = 2;
        model.topology.feed_forward_intermediates = {12, 20};
        model.topology.attention_layouts.resize(2);
        for (auto& attention : model.topology.attention_layouts) {
            attention.query_heads = 1;
            attention.key_value_heads = 1;
            attention.head_dim = 8;
        }

        Naming naming;
        celeg::build_dense_weight_plan(model, naming);
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
