#include "celeg/backend/cpu/kv_topology.hpp"
#include "support/assertions.hpp"

#include <iostream>

int main() {
    celeg::ModelGraph graph;
    graph.hidden = 8;
    graph.layers.resize(2);
    for (auto& layer : graph.layers) {
        celeg::AttentionSpec attention;
        attention.query_heads = 2;
        attention.key_value_heads = 1;
        attention.head_dim = 4;
        attention.query_norm = std::optional<celeg::NormSpec>{
            celeg::NormSpec{1.0e-5f, celeg::NormWeightKind::Scale}};
        attention.key_norm = attention.query_norm;
        attention.position = celeg::RopePositionSpec{10000.0, 1.0, {}};
        layer.mixer = attention;
        layer.feed_forward = celeg::DenseFeedForwardSpec{16, celeg::ActivationKind::SwiGLU};
    }
    std::get<celeg::AttentionSpec>(graph.layers[0].mixer).kv_sharing =
        celeg::SharedKvPublisher{0};
    std::get<celeg::AttentionSpec>(graph.layers[1].mixer).kv_sharing =
        celeg::SharedKvConsumer{0};
    const celeg::RuntimeTopology shape = celeg::compose_runtime_topology({}, graph);

    celeg::CompiledModelProgram program;
    program.layers.resize(2);
    for (size_t index = 0; index < program.layers.size(); ++index) {
        celeg::CompiledOrdinaryKvStateLayout state_layout;
        state_layout.key_width = 4;
        state_layout.value_width = 4;
        program.layers[index].mixer = celeg::CompiledAttentionProgram{
            std::get<celeg::AttentionSpec>(graph.layers[index].mixer),
            state_layout};
    }

    const celeg::CpuKvTopology topology =
        celeg::build_cpu_kv_topology(shape.exec, program, celeg::CpuModelOptions{});
    CELEG_TEST_CHECK(topology.pools.size() == 1);
    CELEG_TEST_CHECK(topology.layer_to_pool == std::vector<int>({0, 0}));
    CELEG_TEST_CHECK(topology.layer_to_owner == std::vector<int>({0, 0}));

    celeg::ModelGraph latent_graph;
    latent_graph.hidden = 8;
    latent_graph.layers.resize(1);
    auto latent_attention = std::get<celeg::AttentionSpec>(graph.layers[0].mixer);
    latent_attention.kv_sharing = {};
    latent_attention.state =
        celeg::LatentAttentionStateSpec{16, 2, 6, true};
    latent_graph.layers[0].mixer = latent_attention;
    latent_graph.layers[0].feed_forward =
        celeg::DenseFeedForwardSpec{16, celeg::ActivationKind::SwiGLU};
    const celeg::RuntimeTopology latent_shape =
        celeg::compose_runtime_topology({}, latent_graph);

    celeg::CompiledLatentStateLayout latent_state_layout;
    latent_state_layout.latent_width = 16;
    latent_state_layout.rotary_width = 4;

    celeg::CompiledModelProgram latent_program;
    latent_program.layers.resize(1);
    latent_program.layers[0].mixer = celeg::CompiledAttentionProgram{
        latent_attention, latent_state_layout};

    const celeg::CpuKvTopology latent_topology =
        celeg::build_cpu_kv_topology(latent_shape.exec, latent_program,
                                     celeg::CpuModelOptions{});
    CELEG_TEST_CHECK(latent_topology.pools.size() == 1);
    CELEG_TEST_CHECK(latent_topology.pools[0]->layout().key_width == 0);
    CELEG_TEST_CHECK(latent_topology.pools[0]->layout().latent_width == 16);
    CELEG_TEST_CHECK(latent_topology.pools[0]->layout().rotary_width == 4);
    std::cout << "cpu_kv_topology_test: ok\n";
    return 0;
}
