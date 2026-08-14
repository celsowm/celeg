#include "attention/policy.cpp"
#include "attention/latent_factorized.cpp"
#include "attention/latent_projected.cpp"
#include "attention/latent.cpp"
#include "attention/regular.cpp"

namespace celeg::prefill_detail {

AttentionLayer& resolve_attention_owner(
    CudaCompiledModel& model,
    AttentionLayer& attention) {
    if (attention.kv_owner_layer < 0) {
        return attention;
    }

    AttentionLayer* owner = as_attention(model.resources_.layers_.at(
        static_cast<size_t>(attention.kv_owner_layer)));
    if (!owner) {
        throw std::logic_error("CUDA shared KV owner is not attention");
    }
    return *owner;
}

void run_attention(
    CudaCompiledModel& model,
    AttentionLayer& attention,
    LayerCommon& common_layer,
    const CompiledLayerProgram& semantics,
    int rows) {
    AttentionLayer& owner = resolve_attention_owner(model, attention);
    if (attention.layout.uses_latent_state()) {
        run_latent_attention(
            model, attention, owner, common_layer, semantics, rows);
        return;
    }

    run_regular_attention(
        model, attention, owner, common_layer, semantics, rows);
}

} // namespace celeg::prefill_detail
