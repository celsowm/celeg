#include "../../attention_kv_store.hpp"
#include "../../attention_latent_dispatch.hpp"
#include "../../attention_layer_support.hpp"
#include "../../attention_projection.hpp"
#include "../../attention_qk_prepare.hpp"

namespace celeg::prefill_detail {

void run_factorized_latent_attention(
    CudaCompiledModel& model,
    AttentionLayer& attention,
    AttentionLayer& owner,
    const CompiledLayerProgram& semantics,
    int rows) {
    require_cuda_factorized_latent_bindings(attention);

    auto& workspace = model.workspace_;
    auto& prof = prefill_phase_profile();
    const AttentionSpec& layout = attention.layout;

    prof.begin(model.stream_.get());
    project_cuda_prefill_factorized_latent_attention_qkv(model, attention, rows);
    prof.end(PrefillPhase::QkvProj, model.stream_.get());

    prof.begin(model.stream_.get());
    prepare_cuda_factorized_latent_attention_qk({
        .layout = &layout,
        .query_rope = workspace.prefill_latent_query_rope_.data(),
        .key_rope = workspace.prefill_latent_key_rope_.data(),
        .norm_epsilon = model.resources_.program_.final_norm.epsilon,
        .rows = rows,
        .device_position = nullptr,
        .stream = model.stream_.get()});
    prof.end(PrefillPhase::RopeKv, model.stream_.get());

    prof.begin(model.stream_.get());
    store_cuda_latent_kv_prefill(model, attention, owner, rows);
    dispatch_cuda_latent_attention_prefill(model, attention, owner, rows);
    prof.end(PrefillPhase::Attention, model.stream_.get());

    prof.begin(model.stream_.get());
    project_cuda_prefill_factorized_latent_attention_output(
        model, attention, semantics, rows);
    prof.end(PrefillPhase::AttnOut, model.stream_.get());
}

}
