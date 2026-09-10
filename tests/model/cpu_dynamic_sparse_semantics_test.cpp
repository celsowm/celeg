#include "celeg/backend/cpu/attention_capabilities.hpp"
#include "celeg/backend/cpu/compiler.hpp"
#include "celeg/backend/cpu/paged_kv.hpp"
#include "support/assertions.hpp"

#include <stdexcept>
#include <utility>

namespace {

celeg::ResolvedModel dynamic_sparse_model() {
    celeg::ResolvedModel model;
    model.provenance.identity = "cpu-dynamic-sparse";
    model.graph.hidden = 8;
    celeg::LayerSpec layer;
    celeg::AttentionSpec attention;
    attention.query_heads = 1;
    attention.key_value_heads = 1;
    attention.head_dim = 8;
    attention.pattern = celeg::DynamicSparsePattern{2, 1};
    attention.position = celeg::RopePositionSpec{10000.0, 1.0, {}};
    layer.mixer = attention;
    layer.feed_forward = celeg::DenseFeedForwardSpec{
        16, celeg::ActivationKind::SwiGLU};
    model.graph.layers.push_back(std::move(layer));
    model.weight_plan.requests.push_back(
        {celeg::TensorRole::AttentionInputNorm, 0, -1, {}});
    return model;
}

bool compiler_accepts_ordinary_dynamic_sparse() {
    try {
        (void)celeg::CpuModelCompiler{}.compile(dynamic_sparse_model());
    } catch (const std::invalid_argument&) {
        return false;
    }
    return true;
}

bool compiler_rejects_latent_dynamic_sparse() {
    celeg::ResolvedModel model = dynamic_sparse_model();
    auto& attention = std::get<celeg::AttentionSpec>(model.graph.layers[0].mixer);
    celeg::LatentAttentionStateSpec latent;
    latent.latent_rank = 8;
    latent.nope_head_dim = 8;
    latent.rope_head_dim = 0;
    latent.decoupled_rope = false;
    attention.state = latent;
    try {
        (void)celeg::CpuModelCompiler{}.compile(model);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

bool low_level_rejects_invalid_geometry() {
    try {
        (void)celeg::CpuAttentionPattern::lower(
            celeg::DynamicSparsePattern{0, 1});
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}

int main() {
    CELEG_TEST_CHECK(celeg::cpu_attention_capabilities().dynamic_sparse);
    CELEG_TEST_CHECK(compiler_accepts_ordinary_dynamic_sparse());
    CELEG_TEST_CHECK(compiler_rejects_latent_dynamic_sparse());
    CELEG_TEST_CHECK(!celeg::CpuAttentionPattern::lower(
        celeg::DynamicSparsePattern{2, 1}).parallel_safe());
    CELEG_TEST_CHECK(low_level_rejects_invalid_geometry());
    return 0;
}
