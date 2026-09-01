#include "celeg/backend/cpu/compiler.hpp"
#include "support/assertions.hpp"

#include <stdexcept>

namespace {

celeg::ResolvedModel base_model() {
    celeg::ResolvedModel model;
    model.provenance.identity = "cpu-attention-semantic-fixture";
    model.graph.hidden = 8;
    celeg::LayerSpec layer;
    celeg::AttentionSpec attention;
    attention.query_heads = 1;
    attention.key_value_heads = 1;
    attention.head_dim = 8;
    attention.position = celeg::RopePositionSpec{10000.0, 1.0, {}};
    layer.mixer = attention;
    layer.feed_forward = celeg::DenseFeedForwardSpec{
        16, celeg::ActivationKind::SwiGLU};
    model.graph.layers.push_back(std::move(layer));
    return model;
}

template <typename Mutator>
bool rejects(Mutator mutate) {
    celeg::ResolvedModel model = base_model();
    auto& attention = std::get<celeg::AttentionSpec>(model.graph.layers[0].mixer);
    mutate(attention);
    try {
        (void)celeg::CpuModelCompiler{}.compile(model);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}

int main() {
    CELEG_TEST_CHECK(!rejects([](auto&) {}));
    CELEG_TEST_CHECK(!rejects([](auto& attention) {
        celeg::SigmoidAttentionGateSpec gate;
        gate.granularity = celeg::AttentionGateGranularity::HeadWise;
        attention.output_gate = gate;
    }));
    CELEG_TEST_CHECK(rejects([](auto& attention) {
        celeg::SigmoidAttentionGateSpec gate;
        gate.packed_with_query = true;
        gate.granularity = celeg::AttentionGateGranularity::HeadWise;
        attention.output_gate = gate;
    }));
    CELEG_TEST_CHECK(rejects([](auto& attention) {
        auto& ordinary = std::get<celeg::OrdinaryKvStateSpec>(attention.state);
        ordinary.storage.key = celeg::StateScalarType::INT8;
    }));
    CELEG_TEST_CHECK(rejects([](auto& attention) {
        attention.state = celeg::LatentAttentionStateSpec{16, 2, 6, true};
        celeg::MultiAxisRopeSpec multi;
        multi.base = celeg::RopePositionSpec{10000.0, 1.0, {}};
        multi.sections = {2, 1, 1};
        multi.interleaved = true;
        multi.axes = 3;
        attention.position = multi;
    }));
    CELEG_TEST_CHECK(rejects([](auto& attention) {
        attention.state = celeg::LatentAttentionStateSpec{16, 2, 6, true};
        attention.output_gate = celeg::SigmoidAttentionGateSpec{};
    }));
    CELEG_TEST_CHECK(rejects([](auto& attention) {
        attention.state = celeg::LatentAttentionStateSpec{16, 2, 6, true};
        attention.output_transform = celeg::OrthogonalizeCurrentValueSpec{1.0e-6f};
    }));
    return 0;
}
