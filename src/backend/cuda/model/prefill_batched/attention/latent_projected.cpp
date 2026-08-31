#include "../../attention_projection.hpp"
#include "../../attention_qk_prepare.hpp"

namespace celeg::prefill_detail {

void run_projected_latent_attention(
    CudaCompiledModel& model,
    AttentionLayer& attention,
    AttentionLayer& owner,
    const CompiledLayerProgram& semantics,
    int rows) {
    auto& workspace = model.workspace_;
    auto& prof = prefill_phase_profile();
    const AttentionSpec& layout = attention.layout;
    const auto& latent = *layout.latent_state();

    prof.begin(model.stream_.get());
    project_cuda_prefill_latent_attention_qkv(model, attention, rows);
    prof.end(PrefillPhase::QkvProj, model.stream_.get());

    prof.begin(model.stream_.get());
    if (layout.rope_position() && attention.latent_key_rope &&
        latent.decoupled_rope && latent.rope_head_dim != 0) {
        prepare_cuda_prefill_latent_attention_qk({
            .layout = &layout,
            .query_rope = workspace.prefill_latent_query_rope_.data(),
            .key_rope = attention.latent_key
                ? workspace.prefill_latent_key_rope_.data() : nullptr,
            .norm_epsilon = model.resources_.program_.final_norm.epsilon,
            .rows = rows,
            .stream = model.stream_.get()});
    }
    prof.end(PrefillPhase::RopeKv, model.stream_.get());

    prof.begin(model.stream_.get());
    if (attention.latent_key) {
        launch_store_latent_prefill(
            workspace.prefill_latent_key_.data(),
            workspace.prefill_latent_value_.data(),
            attention.latent_key_rope && latent.decoupled_rope &&
                    latent.rope_head_dim != 0
                ? workspace.prefill_latent_key_rope_.data()
                : nullptr,
            owner.latent_key_cache_ptr(), owner.latent_value_cache_ptr(),
            owner.latent_key_rope_cache_ptr(), rows, latent.latent_rank,
            latent.decoupled_rope ? latent.rope_head_dim : 0,
            model.stream_.get());
    }
    launch_latent_attention_prefill({
        .query = {
            .content = workspace.prefill_latent_query_content_.data(),
            .rope = layout.latent_query_rope_width() != 0
                ? workspace.prefill_latent_query_rope_.data()
                : nullptr},
        .kv = {
            .keys = owner.latent_key_cache_ptr(),
            .values = owner.latent_value_cache_ptr(),
            .key_rope = owner.latent_key_rope_cache_ptr()},
        .out = workspace.prefill_op_output_.data(),
        .extent = {.rows = rows},
        .alibi_slopes = attention.alibi_slopes.data(),
        .geometry = {
            .query_heads = layout.query_heads,
            .latent_rank = latent.latent_rank,
            .rotary_width = latent.decoupled_rope ? latent.rope_head_dim : 0,
            .score_scale = layout.query_scale,
            .sliding_window = layout.sliding_window_size()},
        .stream = model.stream_.get()});
    prof.end(PrefillPhase::Attention, model.stream_.get());

    prof.begin(model.stream_.get());
    project_cuda_prefill_latent_attention_output(
        model, attention, semantics, rows);
    prof.end(PrefillPhase::AttnOut, model.stream_.get());
}

}
