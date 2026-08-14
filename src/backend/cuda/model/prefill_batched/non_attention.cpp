namespace celeg {

void CudaCompiledModel::run_prefill_gated_delta(GatedDeltaNetLayer& gated_delta,
                                                const CompiledLayerProgram& semantics,
                                                int rows) {
    auto& prof = prefill_phase_profile();
    const GatedDeltaNetSpec& spec = gated_delta.spec;
    const int qkv_width = 2 * spec.key_heads * spec.key_head_dim +
        spec.value_heads * spec.value_head_dim;
    const int value_width = spec.value_heads * spec.value_head_dim;
    prof.begin(stream_.get());
    {
    auto native_fanout = native_fanout_scope(
        workspace_.prefill_normed_.data(), rows, resources_.program_.hidden);
    if (spec.factorized_projections) {
        linear(workspace_.prefill_normed_.data(), *gated_delta.q,
               workspace_.prefill_gated_delta_qkv_.data(), rows,
               spec.key_heads * spec.key_head_dim, resources_.program_.hidden);
        linear(workspace_.prefill_normed_.data(), *gated_delta.k,
               workspace_.prefill_k_.data(), rows,
               spec.key_heads * spec.key_head_dim, resources_.program_.hidden);
        linear(workspace_.prefill_normed_.data(), *gated_delta.v,
               workspace_.prefill_v_.data(), rows, value_width,
               resources_.program_.hidden);
        launch_interleave_gated_delta_qkv(
            workspace_.prefill_gated_delta_qkv_.data(), workspace_.prefill_k_.data(),
            workspace_.prefill_v_.data(), workspace_.prefill_gated_delta_qkv_.data(),
            rows, spec.key_heads * spec.key_head_dim, value_width, stream_.get());
    } else {
        linear(workspace_.prefill_normed_.data(), *gated_delta.qkv,
               workspace_.prefill_gated_delta_qkv_.data(), rows, qkv_width,
               resources_.program_.hidden);
    }
    linear(workspace_.prefill_normed_.data(), *gated_delta.z,
           workspace_.prefill_gated_delta_z_.data(), rows, value_width,
           resources_.program_.hidden);
    linear(workspace_.prefill_normed_.data(), *gated_delta.b,
           workspace_.prefill_gated_delta_b_.data(), rows, spec.value_heads,
           resources_.program_.hidden);
    linear(workspace_.prefill_normed_.data(), *gated_delta.a,
           workspace_.prefill_gated_delta_a_.data(), rows, spec.decay_width(),
           resources_.program_.hidden);
    }
    prof.end(PrefillPhase::QkvProj, stream_.get());
    prof.begin(stream_.get());
    launch_gated_delta_net(workspace_.prefill_gated_delta_qkv_.data(),
        workspace_.prefill_gated_delta_z_.data(),
        workspace_.prefill_gated_delta_b_.data(),
        workspace_.prefill_gated_delta_a_.data(), gated_delta.conv_weight,
        gated_delta.dt_bias, gated_delta.a_log, gated_delta.norm,
        gated_delta.conv_state.data(), gated_delta.recurrent_state.data(),
        workspace_.prefill_gated_delta_output_.data(), rows, spec.conv_kernel,
        spec.key_head_dim, spec.value_head_dim, spec.key_heads,
        spec.value_heads, semantics.operator_norm.epsilon,
        spec.vector_decay, spec.safe_decay, spec.decay_lower_bound,
        spec.sigmoid_output_gate, stream_.get());
    prof.end(PrefillPhase::Conv, stream_.get());
    prof.begin(stream_.get());
    linear(workspace_.prefill_gated_delta_output_.data(), *gated_delta.out,
           workspace_.prefill_hidden_.data(), rows, resources_.program_.hidden,
           value_width);
    launch_scale(workspace_.prefill_hidden_.data(), rows * resources_.program_.hidden,
                 semantics.residual.multiplier, stream_.get());
    prof.end(PrefillPhase::AttnOut, stream_.get());
}

void CudaCompiledModel::run_prefill_mamba2(Mamba2Layer& mamba,
                                           const CompiledLayerProgram& semantics,
                                           int rows) {
    const Mamba2Spec& spec = mamba.spec;
    const int projection_width = 2 * spec.intermediate_size +
        2 * spec.group_count * spec.state_size + spec.num_heads;
    linear(workspace_.prefill_normed_.data(), *mamba.in,
           workspace_.prefill_mamba_projected_.data(), rows, projection_width,
           resources_.program_.hidden);
    launch_mamba2_prefill(
        workspace_.prefill_mamba_projected_.data(), mamba.conv_weight,
        mamba.conv_bias, mamba.dt_bias, mamba.a_log, mamba.d,
        mamba.conv_state.data(), mamba.ssm_state.data(),
        workspace_.prefill_mamba_inner_.data(), rows, spec.intermediate_size,
        spec.state_size, spec.num_heads, spec.head_dim, spec.group_count,
        spec.conv_kernel, stream_.get());
    launch_rmsnorm(workspace_.prefill_mamba_inner_.data(), mamba.norm,
                   workspace_.prefill_mamba_inner_.data(), rows,
                   spec.intermediate_size, semantics.operator_norm.epsilon,
                   stream_.get());
    launch_multiply(workspace_.prefill_mamba_inner_.data(),
                    workspace_.prefill_mamba_projected_.data(),
                    rows * spec.intermediate_size, stream_.get());
    linear(workspace_.prefill_mamba_inner_.data(), *mamba.out,
           workspace_.prefill_hidden_.data(), rows, resources_.program_.hidden,
           spec.intermediate_size);
}

void CudaCompiledModel::run_prefill_mlp_only(MlpOnlyLayer& mlp, int rows) {
    const int intermediate = mlp.spec.intermediate_size;
    linear(workspace_.prefill_normed_.data(), *mlp.up,
           workspace_.prefill_gate_up_.data(), rows, intermediate,
           resources_.program_.hidden);
    switch (mlp.spec.activation) {
    case ActivationKind::Relu2:
        launch_relu2(workspace_.prefill_gate_up_.data(),
                     workspace_.prefill_activated_.data(),
                     rows * intermediate, stream_.get());
        break;
    case ActivationKind::GeluTanh:
        launch_gelu_tanh(workspace_.prefill_gate_up_.data(),
                         workspace_.prefill_activated_.data(),
                         rows * intermediate, stream_.get());
        break;
    default:
        throw std::runtime_error("CUDA prefill does not implement MLP-only activation");
    }
    linear(workspace_.prefill_activated_.data(), *mlp.down,
           workspace_.prefill_hidden_.data(), rows, resources_.program_.hidden,
           intermediate);
}

void CudaCompiledModel::run_prefill_convolution(ConvolutionLayer& convolution, int rows) {
    auto& prof = prefill_phase_profile();
    prof.begin(stream_.get());
    linear(workspace_.prefill_normed_.data(), *convolution.conv_in,
           workspace_.prefill_conv_projected_.data(), rows,
           3 * resources_.program_.hidden, resources_.program_.hidden);
    launch_conv_prefill(
        workspace_.prefill_conv_projected_.data(), convolution.conv_weight,
        convolution.conv_state.data(), workspace_.prefill_op_output_.data(),
        rows, resources_.program_.hidden, resources_.shape_.conv_cache,
        stream_.get());
    linear(workspace_.prefill_op_output_.data(), *convolution.conv_out,
           workspace_.prefill_hidden_.data(), rows, resources_.program_.hidden,
           resources_.program_.hidden, resources_.options_.fused_residuals ? 1.0f : 0.0f);
    prof.end(PrefillPhase::Conv, stream_.get());
}

} // namespace celeg
