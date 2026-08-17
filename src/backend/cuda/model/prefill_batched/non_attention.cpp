namespace celeg::prefill_detail {

void run_gated_delta(
    CudaCompiledModel& model,
    GatedDeltaNetLayer& gated_delta,
    const CompiledLayerProgram& semantics,
    int rows) {
    auto& workspace = model.workspace_;
    auto& prof = prefill_phase_profile();
    const GatedDeltaNetSpec& spec = gated_delta.spec;
    const int hidden = model.resources_.program_.hidden;
    const int qkv_width =
        2 * spec.key_heads * spec.key_head_dim +
        spec.value_heads * spec.value_head_dim;
    const int value_width = spec.value_heads * spec.value_head_dim;

    prof.begin(model.stream_.get());
    {
        auto native_fanout = model.native_fanout_scope(
            workspace.prefill_normed_.data(), rows, hidden);
        if (spec.factorized_projections) {
            model.linear(
                workspace.prefill_normed_.data(), *gated_delta.q,
                workspace.prefill_gated_delta_qkv_.data(),
                rows, spec.key_heads * spec.key_head_dim, hidden);
            model.linear(
                workspace.prefill_normed_.data(), *gated_delta.k,
                workspace.prefill_k_.data(),
                rows, spec.key_heads * spec.key_head_dim, hidden);
            model.linear(
                workspace.prefill_normed_.data(), *gated_delta.v,
                workspace.prefill_v_.data(), rows, value_width, hidden);
            launch_interleave_gated_delta_qkv(
                workspace.prefill_gated_delta_qkv_.data(),
                workspace.prefill_k_.data(),
                workspace.prefill_v_.data(),
                workspace.prefill_gated_delta_qkv_.data(),
                rows, spec.key_heads * spec.key_head_dim, value_width,
                model.stream_.get());
        } else {
            model.linear(
                workspace.prefill_normed_.data(), *gated_delta.qkv,
                workspace.prefill_gated_delta_qkv_.data(),
                rows, qkv_width, hidden);
        }
        model.linear(
            workspace.prefill_normed_.data(), *gated_delta.z,
            workspace.prefill_gated_delta_z_.data(),
            rows, value_width, hidden);
        model.linear(
            workspace.prefill_normed_.data(), *gated_delta.b,
            workspace.prefill_gated_delta_b_.data(),
            rows, spec.value_heads, hidden);
        model.linear(
            workspace.prefill_normed_.data(), *gated_delta.a,
            workspace.prefill_gated_delta_a_.data(),
            rows, spec.decay_width(), hidden);
    }
    prof.end(PrefillPhase::QkvProj, model.stream_.get());

    prof.begin(model.stream_.get());
    launch_gated_delta_net(
        workspace.prefill_gated_delta_qkv_.data(),
        workspace.prefill_gated_delta_z_.data(),
        workspace.prefill_gated_delta_b_.data(),
        workspace.prefill_gated_delta_a_.data(),
        gated_delta.conv_weight, gated_delta.dt_bias, gated_delta.a_log,
        gated_delta.norm, gated_delta.conv_state.data(),
        gated_delta.recurrent_state.data(),
        workspace.prefill_gated_delta_output_.data(),
        rows, spec.conv_kernel, spec.key_head_dim, spec.value_head_dim,
        spec.key_heads, spec.value_heads, semantics.operator_norm.epsilon,
        spec.vector_decay, spec.safe_decay, spec.decay_lower_bound,
        spec.sigmoid_output_gate, model.stream_.get());
    prof.end(PrefillPhase::Conv, model.stream_.get());

    prof.begin(model.stream_.get());
    model.linear(
        workspace.prefill_gated_delta_output_.data(), *gated_delta.out,
        workspace.prefill_hidden_.data(), rows, hidden, value_width);
    launch_scale(
        workspace.prefill_hidden_.data(), rows * hidden,
        semantics.residual.multiplier, model.stream_.get());
    prof.end(PrefillPhase::AttnOut, model.stream_.get());
}

void run_mamba2(
    CudaCompiledModel& model,
    Mamba2Layer& mamba,
    const CompiledLayerProgram& semantics,
    int rows) {
    auto& workspace = model.workspace_;
    const Mamba2Spec& spec = mamba.spec;
    const int hidden = model.resources_.program_.hidden;
    const int projection_width =
        2 * spec.intermediate_size +
        2 * spec.group_count * spec.state_size + spec.num_heads;

    model.linear(
        workspace.prefill_normed_.data(), *mamba.in,
        workspace.prefill_mamba_projected_.data(),
        rows, projection_width, hidden);
    launch_mamba2_prefill(
        workspace.prefill_mamba_projected_.data(),
        mamba.conv_weight, mamba.conv_bias, mamba.dt_bias, mamba.a_log, mamba.d,
        mamba.conv_state.data(), mamba.ssm_state.data(),
        workspace.prefill_mamba_inner_.data(),
        rows, spec.intermediate_size, spec.state_size, spec.num_heads,
        spec.head_dim, spec.group_count, spec.conv_kernel, model.stream_.get());
    launch_rmsnorm(
        workspace.prefill_mamba_inner_.data(), mamba.norm,
        workspace.prefill_mamba_inner_.data(),
        rows, spec.intermediate_size, semantics.operator_norm.epsilon,
        model.stream_.get());
    launch_multiply(
        workspace.prefill_mamba_inner_.data(),
        workspace.prefill_mamba_projected_.data(),
        rows * spec.intermediate_size, model.stream_.get());
    model.linear(
        workspace.prefill_mamba_inner_.data(), *mamba.out,
        workspace.prefill_hidden_.data(),
        rows, hidden, spec.intermediate_size);
}

void run_mlp_only(
    CudaCompiledModel& model,
    MlpOnlyLayer& mlp,
    int rows) {
    auto& workspace = model.workspace_;
    const int hidden = model.resources_.program_.hidden;
    const int intermediate = mlp.spec.intermediate_size;

    model.linear(
        workspace.prefill_normed_.data(), *mlp.up,
        workspace.prefill_gate_up_.data(),
        rows, intermediate, hidden);
    switch (mlp.spec.activation) {
    case ActivationKind::Relu2:
        launch_relu2(
            workspace.prefill_gate_up_.data(),
            workspace.prefill_activated_.data(),
            rows * intermediate, model.stream_.get());
        break;
    case ActivationKind::GeluTanh:
        launch_gelu_tanh(
            workspace.prefill_gate_up_.data(),
            workspace.prefill_activated_.data(),
            rows * intermediate, model.stream_.get());
        break;
    default:
        throw std::runtime_error(
            "CUDA prefill does not implement MLP-only activation");
    }
    model.linear(
        workspace.prefill_activated_.data(), *mlp.down,
        workspace.prefill_hidden_.data(),
        rows, hidden, intermediate);
}

void run_convolution(
    CudaCompiledModel& model,
    ConvolutionLayer& convolution,
    int rows) {
    auto& workspace = model.workspace_;
    auto& prof = prefill_phase_profile();
    const int hidden = model.resources_.program_.hidden;

    prof.begin(model.stream_.get());
    model.linear(
        workspace.prefill_normed_.data(), *convolution.conv_in,
        workspace.prefill_conv_projected_.data(),
        rows, 3 * hidden, hidden);
    launch_conv_prefill(
        workspace.prefill_conv_projected_.data(), convolution.conv_weight,
        convolution.conv_state.data(), workspace.prefill_op_output_.data(),
        rows, hidden, convolution.spec.cache_length, model.stream_.get());
    model.linear(
        workspace.prefill_op_output_.data(), *convolution.conv_out,
        workspace.prefill_hidden_.data(),
        rows, hidden, hidden,
        model.resources_.options_.fused_residuals ? 1.0f : 0.0f);
    prof.end(PrefillPhase::Conv, model.stream_.get());
}

}

