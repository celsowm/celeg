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
            layer.mixer_norm.before = celeg::NormSpec{
                1.0e-5f, celeg::NormWeightKind::Scale};
            layer.feed_forward_norm.before = celeg::NormSpec{
                1.0e-5f, celeg::NormWeightKind::Scale};
            celeg::AttentionSpec attention;
            attention.query_heads = 2;
            attention.key_value_heads = 2;
            attention.head_dim = 4;
            layer.mixer = attention;
            layer.feed_forward = celeg::DenseFeedForwardSpec{
                intermediate, celeg::ActivationKind::SwiGLU};
            model.graph.layers.push_back(std::move(layer));
        }
        auto& per_head_attention = std::get<celeg::AttentionSpec>(
            model.graph.layers[0].mixer);
        per_head_attention.query_norm = celeg::NormSpec{
            1.0e-5f, celeg::NormWeightKind::Scale,
            celeg::NormGranularity::PerHead};
        per_head_attention.key_norm = per_head_attention.query_norm;

        auto& whole_vector_attention = std::get<celeg::AttentionSpec>(
            model.graph.layers[1].mixer);
        whole_vector_attention.query_norm = celeg::NormSpec{
            1.0e-5f, celeg::NormWeightKind::Scale,
            celeg::NormGranularity::WholeVector};
        whole_vector_attention.key_norm = whole_vector_attention.query_norm;

        model.graph.layers[1].mixer_norm.before = std::nullopt;
        model.graph.layers[1].mixer_norm.after = celeg::NormSpec{
            2.0e-5f, celeg::NormWeightKind::Scale};
        model.graph.layers[1].feed_forward_norm.before = std::nullopt;
        model.graph.layers[1].feed_forward_norm.after = celeg::NormSpec{
            3.0e-5f, celeg::NormWeightKind::Scale};

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
        const auto has = [&](celeg::TensorRole role, int layer) {
            for (const auto& request : model.weight_plan.requests) {
                if (request.role == role && request.layer == layer) return true;
            }
            return false;
        };
        if (find(celeg::TensorRole::FfnGate, 0).expected_shape[0] != 12 ||
            find(celeg::TensorRole::FfnUp, 1).expected_shape[0] != 20 ||
            find(celeg::TensorRole::FfnDown, 1).expected_shape[1] != 20) {
            throw std::runtime_error("per-layer FFN widths were ignored");
        }
        if (find(celeg::TensorRole::AttentionQueryNorm, 0).expected_shape !=
                std::vector<int64_t>{4} ||
            find(celeg::TensorRole::AttentionKeyNorm, 0).expected_shape !=
                std::vector<int64_t>{4}) {
            throw std::runtime_error("per-head QK norm width was ignored");
        }
        if (find(celeg::TensorRole::AttentionQueryNorm, 1).expected_shape !=
                std::vector<int64_t>{8} ||
            find(celeg::TensorRole::AttentionKeyNorm, 1).expected_shape !=
                std::vector<int64_t>{8}) {
            throw std::runtime_error("whole-vector QK norm width was ignored");
        }
        if (!has(celeg::TensorRole::AttentionInputNorm, 0) ||
            !has(celeg::TensorRole::FfnInputNorm, 0) ||
            has(celeg::TensorRole::AttentionPostNorm, 0) ||
            has(celeg::TensorRole::FfnOutputNorm, 0)) {
            throw std::runtime_error("pre-norm layer emitted incorrect norm roles");
        }
        if (has(celeg::TensorRole::AttentionInputNorm, 1) ||
            has(celeg::TensorRole::FfnInputNorm, 1) ||
            !has(celeg::TensorRole::AttentionPostNorm, 1) ||
            !has(celeg::TensorRole::FfnOutputNorm, 1)) {
            throw std::runtime_error("post-norm layer emitted incorrect norm roles");
        }
        std::cout << "weight_plan_test: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "weight_plan_test: " << error.what() << '\n';
        return 1;
    }
}
