namespace celeg::prefill_detail {

void require_regular_attention_bindings(const AttentionLayer& attention) {
    const AttentionSpec& layout = attention.layout;
    if (!attention.query || !attention.out) {
        throw std::logic_error(
            "CUDA prefill attention has incomplete query/output bindings");
    }
    if ((attention.key == nullptr) != (attention.value == nullptr)) {
        throw std::logic_error(
            "CUDA prefill attention must bind key/value together");
    }
    if (layout.output_gate.enabled() &&
        !layout.output_gate.packed_with_query &&
        !attention.gate) {
        throw std::logic_error(
            "CUDA prefill attention is missing its output gate binding");
    }
}

void run_regular_attention(
    CudaCompiledModel& model,
    AttentionLayer& attention,
    AttentionLayer& owner,
    LayerCommon& common_layer,
    const CompiledLayerProgram& semantics,
    int rows) {
    require_regular_attention_bindings(attention);

    auto& workspace = model.workspace_;
    auto& prof = prefill_phase_profile();
    const AttentionSpec& layout = attention.layout;
    const AttentionSpec& owner_layout = owner.layout;
    const int hidden = model.resources_.program_.hidden;
    const int query_projection_width = attention.query->rows;
    const bool output_gate = layout.output_gate.enabled();
    const bool gate_packed = output_gate && layout.output_gate.packed_with_query;

    prof.begin(model.stream_.get());
    {
        auto native_fanout = model.native_fanout_scope(
            workspace.prefill_normed_.data(), rows, hidden);
        model.linear(
            workspace.prefill_normed_.data(), *attention.query,
            gate_packed ? workspace.prefill_qkv_.data() : workspace.prefill_q_.data(),
            rows, query_projection_width, hidden);
        if (attention.key) {
            model.linear(
                workspace.prefill_normed_.data(), *attention.key,
                workspace.prefill_k_.data(),
                rows, layout.key_value_width(), hidden);
            model.linear(
                workspace.prefill_normed_.data(), *attention.value,
                workspace.prefill_v_.data(),
                rows, layout.key_value_width(), hidden);
        }
    }
    prof.end(PrefillPhase::QkvProj, model.stream_.get());

    prof.begin(model.stream_.get());
    if (output_gate) {
        if (gate_packed) {
            launch_extract_attention_output_gate(
                workspace.prefill_qkv_.data(),
                workspace.prefill_q_.data(),
                workspace.prefill_attention_gate_.data(),
                rows, layout.query_width(), model.stream_.get());
        } else {
            model.linear(
                workspace.prefill_normed_.data(), *attention.gate,
                workspace.prefill_attention_gate_.data(),
                rows, layout.query_width(), hidden);
        }
    }
    if (const auto* rope = layout.rope_position()) {
        launch_dynamic_qk_norm_rope_prefill(
            workspace.prefill_q_.data(),
            attention.key ? workspace.prefill_k_.data() : nullptr,
            attention.q_norm, attention.k_norm,
            rows, layout.query_heads, layout.key_value_heads, layout.head_dim,
            static_cast<float>(rope->theta),
            static_cast<float>(rope->rotary_fraction),
            layout.query_norm.epsilon,
            layout.has_query_key_norm(),
            lower_cuda_rope_scaling(*rope), model.stream_.get());
    } else if (layout.has_query_key_norm()) {
        launch_dynamic_qk_norm_rope_prefill(
            workspace.prefill_q_.data(),
            attention.key ? workspace.prefill_k_.data() : nullptr,
            attention.q_norm, attention.k_norm,
            rows, layout.query_heads, layout.key_value_heads, layout.head_dim,
            1.0f, 0.0f, layout.query_norm.epsilon, true,
            CudaRopeScaling{}, model.stream_.get());
    }
    launch_scale(
        workspace.prefill_q_.data(),
        static_cast<size_t>(rows) * layout.query_width(),
        layout.query_scale, model.stream_.get());
    prof.end(PrefillPhase::RopeKv, model.stream_.get());

    prof.begin(model.stream_.get());
    const AttentionCapability plan =
        select_attention_plan(model, attention, owner_layout, rows);
    store_and_attend(model, attention, owner, plan, rows);
    prof.end(PrefillPhase::Attention, model.stream_.get());

    if (output_gate) {
        launch_sigmoid_multiply(
            workspace.prefill_op_output_.data(),
            workspace.prefill_attention_gate_.data(),
            rows * layout.query_width(), model.stream_.get());
    }

    prof.begin(model.stream_.get());
    model.linear(
        workspace.prefill_op_output_.data(), *attention.out,
        workspace.prefill_hidden_.data(),
        rows, hidden, layout.query_width(),
        model.resources_.options_.fused_residuals && !common_layer.post_attention_norm
            ? 1.0f : 0.0f);
    launch_scale(
        workspace.prefill_hidden_.data(), rows * hidden,
        semantics.residual.multiplier, model.stream_.get());
    prof.end(PrefillPhase::AttnOut, model.stream_.get());
}

} // namespace celeg::prefill_detail
