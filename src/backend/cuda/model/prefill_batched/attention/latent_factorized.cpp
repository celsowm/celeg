#include "../../attention_kv_store.hpp"
#include "../../attention_latent_dispatch.hpp"
#include "../../attention_layer_support.hpp"
#include "../../attention_projection.hpp"
#include "../../residual_fusion.hpp"

namespace celeg::prefill_detail {

void run_factorized_latent_attention(
    CudaCompiledModel& model,
    AttentionLayer& attention,
    AttentionLayer& owner,
    const CompiledLayerProgram& semantics,
    int rows) {
    const __nv_bfloat16* latent_expansion =
        require_cuda_factorized_latent_bindings(attention);

    auto& workspace = model.workspace_;
    auto& prof = prefill_phase_profile();
    const AttentionSpec& layout = attention.layout;
    const auto& latent = *layout.latent_state();
    const auto& factorized = *latent.factorized_projection();
    const int hidden = model.resources_.program_.hidden;

    prof.begin(model.stream_.get());
    project_cuda_prefill_factorized_latent_attention_qkv(model, attention, rows);
    prof.end(PrefillPhase::QkvProj, model.stream_.get());

    prof.begin(model.stream_.get());
    if (const auto* rope = layout.rope_position()) {
        launch_qk_norm_rope_positions(
            workspace.prefill_latent_query_rope_.data(),
            workspace.prefill_latent_key_rope_.data(), nullptr, nullptr,
            rows, layout.query_heads, 1, latent.rope_head_dim, nullptr,
            static_cast<float>(rope->theta), 1.0f,
            model.resources_.program_.final_norm.epsilon, false,
            rope->pairing, lower_cuda_rope_scaling(*rope), model.stream_.get());
    }
    prof.end(PrefillPhase::RopeKv, model.stream_.get());

    prof.begin(model.stream_.get());
    store_cuda_latent_kv_prefill(model, attention, owner, rows);
    dispatch_cuda_latent_attention_prefill(model, attention, owner, rows);
    prof.end(PrefillPhase::Attention, model.stream_.get());

    prof.begin(model.stream_.get());
    launch_factorized_latent_value({
        .latent_output = workspace.prefill_op_output_.data(),
        .expansion = latent_expansion,
        .value_output = workspace.prefill_latent_decompressed_.data(),
        .rows = rows,
        .query_heads = layout.query_heads,
        .query_nope = latent.nope_head_dim,
        .value_dim = factorized.value_head_dim,
        .latent_rank = latent.latent_rank,
        .stream = model.stream_.get()});
    model.linear(
        workspace.prefill_normed_.data(), *attention.gate,
        workspace.prefill_attention_gate_.data(), rows, layout.output_gate_width(), hidden);
    if (layout.output_gate->granularity == AttentionGateGranularity::HeadWise) {
        launch_sigmoid_multiply_headwise(
            workspace.prefill_latent_decompressed_.data(),
            workspace.prefill_attention_gate_.data(), rows, layout.query_heads,
            factorized.value_head_dim, model.stream_.get());
    } else {
        launch_sigmoid_multiply(
            workspace.prefill_latent_decompressed_.data(),
            workspace.prefill_attention_gate_.data(),
            rows * layout.latent_output_width(), model.stream_.get());
    }
    const bool fuse_residual = cuda_can_fuse_mixer_residual(
        model.resources_.options().fused_residuals, semantics);
    model.linear(
        workspace.prefill_latent_decompressed_.data(), *attention.out,
        workspace.prefill_hidden_.data(), rows, hidden,
        layout.latent_output_width(), fuse_residual ? 1.0f : 0.0f);
    launch_scale(
        workspace.prefill_hidden_.data(), rows * hidden,
        semantics.residual.multiplier, model.stream_.get());
    prof.end(PrefillPhase::AttnOut, model.stream_.get());
}

}
