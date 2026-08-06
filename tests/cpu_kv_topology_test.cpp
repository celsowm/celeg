#include "celeg/backend/cpu/kv_topology.hpp"
#include "support/assertions.hpp"

#include <iostream>

int main() {
    celeg::RuntimeTopology shape;
    shape.num_hidden_layers = 2;
    shape.mixer_kinds = {celeg::MixerKind::Attention,
                         celeg::MixerKind::Attention};
    shape.attention_layouts.resize(2);
    shape.attention_layouts[0] = celeg::AttentionSpec{
        2, 1, 4, true, celeg::AttentionMaskKind::Causal, 0, 10000.0, 1.0,
        {0, true}};
    shape.attention_layouts[1] = celeg::AttentionSpec{
        2, 1, 4, true, celeg::AttentionMaskKind::Causal, 0, 10000.0, 1.0,
        {0, false}};

    const celeg::CpuKvTopology topology =
        celeg::build_cpu_kv_topology(shape, celeg::CpuModelOptions{});
    CELEG_TEST_CHECK(topology.pools.size() == 1);
    CELEG_TEST_CHECK(topology.layer_to_pool == std::vector<int>({0, 0}));
    CELEG_TEST_CHECK(topology.layer_to_owner == std::vector<int>({0, 0}));
    std::cout << "cpu_kv_topology_test: ok\n";
    return 0;
}
