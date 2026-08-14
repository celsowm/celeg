namespace celeg::prefill_detail {

void require_factorized_latent_bindings(const AttentionLayer& attention) {
    if (!attention.latent_query_projection ||
        !attention.latent_query_expansion ||
        !attention.latent_query_norm ||
        !attention.latent_key_projection ||
        !attention.latent_key_norm ||
        !attention.latent_expansion ||
        !attention.gate ||
        !attention.out) {
        throw std::logic_error(
            "CUDA factorized latent prefill has incomplete weight bindings");
    }
}

void run_factorized_latent_attention(
    CudaCompiledModel& model,
    AttentionLayer& attention,
    AttentionLayer& owner,
    LayerCommon& common_layer,
    const CompiledLayerProgram& semantics,
    int rows) {
    require_factorized_latent_bindings(attention);

    auto& workspace = model.workspace_;
    auto& prof = prefill_phase_profile();
    const AttentionSpec& layout = attention.layout;
    const auto& latent = *layout.latent_state();
    const int hidden = model.resources_.program_.hidden;

    prof.begin(model.stream_.get());
    {
        auto native_fanout = model.native_fanout_scope(
            workspace.prefill_normed_.data(), rows, hidden);
        model.linear(
            workspace.prefill_normed_.data(),
            *attention.latent_query_projection,
            workspace.prefill_latent_projection_.data(),
            rows, latent.query_rank, hidden);
        launch_rmsnorm(
            workspace.prefill_latent_projection_.data(),
            attention.latent_query_norm,
            workspace.prefill_latent_projection_.data(),
            rows, latent.query_rank, latent.query_latent_norm.epsilon,
            model.stream_.get());
        model.linear(
            workspace.prefill_latent_projection_.data(),
            *attention.latent_query_expansion,
            workspace.prefill_qkv_.data(),
            rows,
            layout.query_heads * (latent.nope_head_dim + latent.rope_head_dim),
            latent.query_rank);
        launch_factorized_latent_query({
            .query_projection = workspace.prefill_qkv_.data(),
            .expansion = attention.latent_expansion->bf16,
            .query_content = workspace.prefill_latent_query_content_.data(),
            .rows = rows,
            .query_heads = layout.query_heads,
            .query_nope = latent.nope_head_dim,
            .query_rope_dim = latent.rope_head_dim,
            .latent_rank = latent.latent_rank,
            .stream = model.stream_.get()});
        launch_factorized_latent_rope({
            .query_projection = workspace.prefill_qkv_.data(),
            .query_rope = workspace.prefill_latent_query_rope_.data(),
            .rows = rows,
            .query_heads = layout.query_heads,
            .query_nope = latent.nope_head_dim,
            .query_rope_dim = latent.rope_head_dim,
            .stream = model.stream_.get()});
        model.linear(
            workspace.prefill_normed_.data(),
            *attention.latent_key_projection,
            workspace.prefill_qkv_.data(),
            rows, latent.latent_rank + latent.rope_head_dim, hidden);
        launch_rmsnorm(
            workspace.prefill_qkv_.data(), attention.latent_key_norm,
            workspace.prefill_latent_key_.data(),
            rows, latent.latent_rank, latent.key_latent_norm.epsilon,
            model.stream_.get());
        CELEG_CUDA(cudaMemcpyAsync(
            workspace.prefill_latent_value_.data(),
            workspace.prefill_latent_key_.data(),
            static_cast<size_t>(rows) * latent.latent_rank * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice, model.stream_.get()));
        CELEG_CUDA(cudaMemcpy2DAsync(
            workspace.prefill_latent_key_rope_.data(),
            static_cast<size_t>(latent.rope_head_dim) * sizeof(__nv_bfloat16),
            workspace.prefill_qkv_.data() + latent.latent_rank,
            static_cast<size_t>(latent.latent_rank + latent.rope_head_dim) *
                sizeof(__nv_bfloat16),
            static_cast<size_t>(latent.rope_head_dim) * sizeof(__nv_bfloat16),
            static_cast<size_t>(rows),
            cudaMemcpyDeviceToDevice, model.stream_.get()));
    }
    prof.end(PrefillPhase::QkvProj, model.stream_.get());

    prof.begin(model.stream_.get());
    if (const auto* rope = layout.rope_position()) {
        launch_qk_norm_rope_positions(
            workspace.prefill_latent_query_rope_.data(),
            workspace.prefill_latent_key_rope_.data(),
            nullptr, nullptr,
            rows, layout.query_heads, 1, latent.rope_head_dim, nullptr,
            static_cast<float>(rope->theta), 1.0f,
            layout.query_norm.epsilon, false,
            rope->pairing, lower_cuda_rope_scaling(*rope), model.stream_.get());
    }
    prof.end(PrefillPhase::RopeKv, model.stream_.get());

    prof.begin(model.stream_.get());
    launch_store_latent_prefill(
        workspace.prefill_latent_key_.data(),
        workspace.prefill_latent_value_.data(),
        workspace.prefill_latent_key_rope_.data(),
        owner.latent_key_cache.data(), owner.latent_value_cache.data(),
        owner.latent_key_rope_cache.data(),
        rows, latent.latent_rank, latent.rope_head_dim, model.stream_.get());
    launch_latent_attention_prefill({
        .query = {
            .content = workspace.prefill_latent_query_content_.data(),
            .rope = workspace.prefill_latent_query_rope_.data()},
        .kv = {
            .keys = owner.latent_key_cache.data(),
            .values = owner.latent_value_cache.data(),
            .key_rope = owner.latent_key_rope_cache.data()},
        .out = workspace.prefill_op_output_.data(),
        .extent = {.rows = rows},
        .alibi_slopes = attention.alibi_slopes.data(),
        .geometry = {
            .query_heads = layout.query_heads,
            .latent_rank = latent.latent_rank,
            .rotary_width = latent.rope_head_dim,
            .score_scale = layout.query_scale,
            .sliding_window = layout.sliding_window_size()},
        .stream = model.stream_.get()});
    prof.end(PrefillPhase::Attention, model.stream_.get());

    prof.begin(model.stream_.get());
    launch_factorized_latent_value({
        .latent_output = workspace.prefill_op_output_.data(),
        .expansion = attention.latent_expansion->bf16,
        .value_output = workspace.prefill_latent_decompressed_.data(),
        .rows = rows,
        .query_heads = layout.query_heads,
        .query_nope = latent.nope_head_dim,
        .value_dim = latent.value_head_dim,
        .latent_rank = latent.latent_rank,
        .stream = model.stream_.get()});
    model.linear(
        workspace.prefill_normed_.data(), *attention.gate,
        workspace.prefill_attention_gate_.data(),
        rows, layout.output_gate_width(), hidden);
    if (layout.output_gate.granularity == AttentionGateGranularity::HeadWise) {
        launch_sigmoid_multiply_headwise(
            workspace.prefill_latent_decompressed_.data(),
            workspace.prefill_attention_gate_.data(),
            rows, layout.query_heads, latent.value_head_dim, model.stream_.get());
    } else {
        launch_sigmoid_multiply(
            workspace.prefill_latent_decompressed_.data(),
            workspace.prefill_attention_gate_.data(),
            rows * layout.latent_output_width(), model.stream_.get());
    }
    model.linear(
        workspace.prefill_latent_decompressed_.data(), *attention.out,
        workspace.prefill_hidden_.data(),
        rows, hidden, layout.latent_output_width(),
        model.resources_.options_.fused_residuals && !common_layer.post_attention_norm
            ? 1.0f : 0.0f);
    launch_scale(
        workspace.prefill_hidden_.data(), rows * hidden,
        semantics.residual.multiplier, model.stream_.get());
    prof.end(PrefillPhase::AttnOut, model.stream_.get());
}

} // namespace celeg::prefill_detail
