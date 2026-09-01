#include "backend/cuda/model/attention_layer_support.hpp"
#include "support/assertions.hpp"

#include <cstdint>
#include <stdexcept>

namespace {

celeg::LinearWeight bf16_weight() {
    celeg::LinearWeight weight;
    weight.rows = 1;
    weight.cols = 1;
    weight.storage = celeg::Bf16LinearStorage{
        reinterpret_cast<const __nv_bfloat16*>(std::uintptr_t{1})};
    return weight;
}

celeg::AttentionLayer factorized_layer(const celeg::LinearWeight& weight) {
    celeg::AttentionLayer layer;
    layer.layout.query_heads = 2;
    layer.layout.key_value_heads = 1;
    layer.layout.head_dim = 4;
    layer.layout.position = celeg::NoPositionEncodingSpec{};

    celeg::LatentAttentionStateSpec latent;
    latent.latent_rank = 4;
    latent.nope_head_dim = 4;
    latent.rope_head_dim = 0;
    celeg::FactorizedLatentProjection factorized;
    factorized.query_rank = 4;
    factorized.value_head_dim = 4;
    latent.projection = factorized;
    layer.layout.state = latent;

    layer.latent_query_projection = &weight;
    layer.latent_query_expansion = &weight;
    layer.latent_key_projection = &weight;
    layer.latent_expansion = &weight;
    layer.out = &weight;
    layer.latent_query_norm = reinterpret_cast<const __nv_bfloat16*>(std::uintptr_t{1});
    layer.latent_key_norm = reinterpret_cast<const __nv_bfloat16*>(std::uintptr_t{1});
    return layer;
}

}

int main() {
    const celeg::LinearWeight weight = bf16_weight();

    celeg::AttentionLayer ungated = factorized_layer(weight);
    CELEG_TEST_CHECK(
        celeg::require_cuda_factorized_latent_bindings(ungated) != nullptr);

    celeg::AttentionLayer missing_gate = factorized_layer(weight);
    missing_gate.layout.output_gate = celeg::SigmoidAttentionGateSpec{};
    bool rejected_missing_gate = false;
    try {
        (void)celeg::require_cuda_factorized_latent_bindings(missing_gate);
    } catch (const std::logic_error&) {
        rejected_missing_gate = true;
    }
    CELEG_TEST_CHECK(rejected_missing_gate);

    celeg::AttentionLayer gated = factorized_layer(weight);
    gated.layout.output_gate = celeg::SigmoidAttentionGateSpec{};
    gated.gate = &weight;
    CELEG_TEST_CHECK(
        celeg::require_cuda_factorized_latent_bindings(gated) != nullptr);

    return 0;
}
