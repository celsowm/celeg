#include "../attention_layer_support.hpp"
#include "attention/policy.cpp"
#include "attention/latent_factorized.cpp"
#include "attention/latent_projected.cpp"
#include "attention/latent.cpp"
#include "attention/regular.cpp"

namespace celeg::prefill_detail {

void run_attention(
    CudaCompiledModel& model,
    AttentionLayer& attention,
    const CompiledLayerProgram& semantics,
    int layer_index,
    int rows) {
    const CudaAttentionOwner resolved_owner = resolve_cuda_attention_owner(
        attention, layer_index, model.resources_.layers_);
    AttentionLayer& owner = *resolved_owner.layer;
    if (attention.layout.uses_latent_state()) {
        run_latent_attention(
            model, attention, owner, semantics, rows);
        return;
    }

    run_regular_attention(
        model, attention, owner, semantics, rows);
}

}
