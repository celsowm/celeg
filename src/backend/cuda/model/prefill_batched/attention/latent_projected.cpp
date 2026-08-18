namespace celeg::prefill_detail {

void require_projected_latent_bindings(const AttentionLayer& attention) {
    const AttentionSpec& layout = attention.layout;
    if (!attention.latent_query || !attention.out) {
        throw std::logic_error(
            "CUDA projected latent prefill has incomplete query/output bindings");
    }
    if (layout.latent_query_rope_width() != 0 && !attention.latent_query_rope) {
        throw std::logic_error(
            "CUDA projected latent prefill is missing the query RoPE projection");
    }
    if ((attention.latent_key == nullptr) != (attention.latent_value == nullptr)) {
        throw std::logic_error(
            "CUDA projected latent prefill must bind latent key/value together");
    }
}

void run_projected_latent_attention(
    CudaCompiledModel& model,
    AttentionLayer& attention,
    AttentionLayer& owner,
    LayerCommon& common_layer,
    const CompiledLayerProgram& semantics,
    int rows) {
    require_projected_latent_bindings(attention);

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
            workspace.prefill_normed_.data(), *attention.latent_query,
            workspace.prefill_latent_query_content_.data(),
            rows, layout.latent_query_content_width(), hidden);
        if (layout.latent_query_rope_width() != 0) {
            model.linear(
                workspace.prefill_normed_.data(), *attention.latent_query_rope,
                workspace.prefill_latent_query_rope_.data(),
                rows, layout.latent_query_rope_width(), hidden);
        }
        if (attention.latent_key) {
            model.linear(
                workspace.prefill_normed_.data(), *attention.latent_key,
                workspace.prefill_latent_key_.data(),
                rows, latent.latent_rank, hidden);
            model.linear(
                workspace.prefill_normed_.data(), *attention.latent_value,
                workspace.prefill_latent_value_.data(),
                rows, latent.latent_rank, hidden);
            if (attention.latent_key_rope && latent.decoupled_rope &&
                latent.rope_head_dim != 0) {
                model.linear(
                    workspace.prefill_normed_.data(), *attention.latent_key_rope,
                    workspace.prefill_latent_key_rope_.data(),
                    rows, latent.rope_head_dim, hidden);
            }
        }
    }
    prof.end(PrefillPhase::QkvProj, model.stream_.get());

    prof.begin(model.stream_.get());
    if (const auto* rope = layout.rope_position();
        rope && attention.latent_key_rope && latent.decoupled_rope &&
        latent.rope_head_dim != 0) {
        launch_dynamic_qk_norm_rope_prefill(
            workspace.prefill_latent_query_rope_.data(),
            attention.latent_key
                ? workspace.prefill_latent_key_rope_.data()
                : nullptr,
            nullptr, nullptr,
            rows, layout.query_heads, 1, latent.rope_head_dim,
            static_cast<float>(rope->theta), 1.0f,
            layout.query_norm->epsilon, false,
            lower_cuda_rope_scaling(*rope), model.stream_.get());
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
            owner.latent_key_rope_cache_ptr(),
            rows, latent.latent_rank,
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
    model.linear(
        workspace.prefill_op_output_.data(), *attention.out,
        workspace.prefill_hidden_.data(),
        rows, hidden, layout.latent_query_content_width(), 0.0f);
    prof.end(PrefillPhase::AttnOut, model.stream_.get());
}

}