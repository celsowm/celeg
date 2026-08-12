#include "celeg/backend/cpu/kv_topology.hpp"
#include "support/assertions.hpp"

#include <iostream>

int main() {
    celeg::RuntimeTopology shape;
    shape.exec.num_hidden_layers = 2;
    shape.exec.mixer_kinds = {celeg::MixerKind::Attention,
                         celeg::MixerKind::Attention};
    shape.exec.attention_layouts.resize(2);
    shape.exec.attention_layouts[0].query_heads = 2;
    shape.exec.attention_layouts[0].key_value_heads = 1;
    shape.exec.attention_layouts[0].head_dim = 4;
    shape.exec.attention_layouts[0].query_norm = {1.0e-5f, celeg::NormWeightKind::Scale};
    shape.exec.attention_layouts[0].key_norm = shape.exec.attention_layouts[0].query_norm;
    shape.exec.attention_layouts[0].kv_sharing = {0, true};
    shape.exec.attention_layouts[0].position = celeg::RopePositionSpec{10000.0, 1.0, {}};
    shape.exec.attention_layouts[1].query_heads = 2;
    shape.exec.attention_layouts[1].key_value_heads = 1;
    shape.exec.attention_layouts[1].head_dim = 4;
    shape.exec.attention_layouts[1].query_norm = {1.0e-5f, celeg::NormWeightKind::Scale};
    shape.exec.attention_layouts[1].key_norm = shape.exec.attention_layouts[1].query_norm;
    shape.exec.attention_layouts[1].kv_sharing = {0, false};
    shape.exec.attention_layouts[1].position = celeg::RopePositionSpec{10000.0, 1.0, {}};

    celeg::CompiledModelProgram program;
    program.layers.resize(2);
    for (size_t index = 0; index < program.layers.size(); ++index) {
        auto& layer = program.layers[index];
        layer.mixer = celeg::CompiledMixer::Attention;
        layer.attention = shape.exec.attention_layouts[index];
        layer.state_layout = celeg::CompiledAttentionStateLayout{};
        layer.state_layout->key_width = 4;
        layer.state_layout->value_width = 4;
        layer.state_layout->key_elements = 4;
        layer.state_layout->value_elements = 4;
        layer.state_layout->persistent_elements = 8;
    }

    const celeg::CpuKvTopology topology =
        celeg::build_cpu_kv_topology(shape.exec, program, celeg::CpuModelOptions{});
    CELEG_TEST_CHECK(topology.pools.size() == 1);
    CELEG_TEST_CHECK(topology.layer_to_pool == std::vector<int>({0, 0}));
    CELEG_TEST_CHECK(topology.layer_to_owner == std::vector<int>({0, 0}));

    celeg::RuntimeTopology latent_shape;
    latent_shape.exec.num_hidden_layers = 1;
    latent_shape.exec.mixer_kinds = {celeg::MixerKind::Attention};
    latent_shape.exec.attention_layouts = {shape.exec.attention_layouts[0]};
    latent_shape.exec.attention_layouts[0].kv_sharing = {};
    latent_shape.exec.attention_layouts[0].state =
        celeg::LatentAttentionStateSpec{16, 2, 6, true};
    celeg::CompiledModelProgram latent_program;
    latent_program.layers.resize(1);
    latent_program.layers[0].mixer = celeg::CompiledMixer::Attention;
    latent_program.layers[0].attention = latent_shape.exec.attention_layouts[0];
    latent_program.layers[0].state_layout = celeg::CompiledAttentionStateLayout{};
    latent_program.layers[0].state_layout->kind = celeg::CompiledStateLayoutKind::Latent;
    latent_program.layers[0].state_layout->latent_width = 16;
    latent_program.layers[0].state_layout->rotary_width = 4;
    latent_program.layers[0].state_layout->latent_elements = 16;
    latent_program.layers[0].state_layout->rotary_elements = 4;
    latent_program.layers[0].state_layout->persistent_elements = 20;
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
