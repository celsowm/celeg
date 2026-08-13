#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/phase_profile.hpp"

namespace celeg {

PhaseProfile& decode_phase_profile();

void CudaCompiledModel::enqueue_decode_non_attention_mixer(Layer& layer,
                                                            int layer_index) {
    const CompiledLayerProgram& semantics = resources_.program_.layers.at(
        static_cast<size_t>(layer_index));
    if (GatedDeltaNetLayer* gated_delta = as_gated_delta_net(layer)) {
        // Recurrent Gated Delta execution is neither attention nor the
        // feed-forward block below. Attribute the complete neutral mixer so
        // decode profiles cannot hide its cost in the uninstrumented gap.
        decode_phase_profile().begin(stream_.get());
        const GatedDeltaNetSpec& spec = gated_delta->spec;
        const int qkv_width = 2 * spec.key_heads * spec.key_head_dim +
            spec.value_heads * spec.value_head_dim;
        const int value_width = spec.value_heads * spec.value_head_dim;
        if (spec.factorized_projections) {
            linear(workspace_.normed_.data(), *gated_delta->q,
                   workspace_.gated_delta_qkv_.data(), 1, spec.key_heads * spec.key_head_dim,
                   resources_.program_.hidden);
            linear(workspace_.normed_.data(), *gated_delta->k,
                   workspace_.qkv_output_.data(), 1, spec.key_heads * spec.key_head_dim,
                   resources_.program_.hidden);
            linear(workspace_.normed_.data(), *gated_delta->v,
                   workspace_.gated_delta_output_.data(), 1, value_width,
                   resources_.program_.hidden);
            launch_interleave_gated_delta_qkv(
                workspace_.gated_delta_qkv_.data(), workspace_.qkv_output_.data(),
                workspace_.gated_delta_output_.data(), workspace_.gated_delta_qkv_.data(),
                1, spec.key_heads * spec.key_head_dim, value_width, stream_.get());
        } else {
            linear(workspace_.normed_.data(), *gated_delta->qkv,
                   workspace_.gated_delta_qkv_.data(), 1, qkv_width,
                   resources_.program_.hidden);
        }
        linear(workspace_.normed_.data(), *gated_delta->z,
               workspace_.gated_delta_z_.data(), 1, value_width,
               resources_.program_.hidden);
        linear(workspace_.normed_.data(), *gated_delta->b,
               workspace_.gated_delta_b_.data(), 1, spec.value_heads,
               resources_.program_.hidden);
        linear(workspace_.normed_.data(), *gated_delta->a,
               workspace_.gated_delta_a_.data(), 1, spec.decay_width(),
               resources_.program_.hidden);
        launch_gated_delta_net(workspace_.gated_delta_qkv_.data(),
            workspace_.gated_delta_z_.data(), workspace_.gated_delta_b_.data(),
            workspace_.gated_delta_a_.data(), gated_delta->conv_weight,
            gated_delta->dt_bias, gated_delta->a_log, gated_delta->norm,
            gated_delta->conv_state.data(), gated_delta->recurrent_state.data(),
            workspace_.gated_delta_output_.data(), 1, spec.conv_kernel,
            spec.key_head_dim, spec.value_head_dim, spec.key_heads,
            spec.value_heads, semantics.operator_norm.epsilon,
            spec.vector_decay, spec.safe_decay, spec.decay_lower_bound,
            spec.sigmoid_output_gate, stream_.get());
        linear(workspace_.gated_delta_output_.data(), *gated_delta->out,
               workspace_.hidden_.data(), 1, resources_.program_.hidden,
               value_width);
        decode_phase_profile().end(DecodePhase::Other, stream_.get());
        return;
    }
    if (Mamba2Layer* mamba = as_mamba2(layer)) {
        const Mamba2Spec& spec = mamba->spec;
        const int projection_width = 2 * spec.intermediate_size +
            2 * spec.group_count * spec.state_size + spec.num_heads;
        linear(workspace_.normed_.data(), *mamba->in,
               workspace_.mamba_projected_.data(), 1, projection_width,
               resources_.program_.hidden);
        launch_mamba2_step(workspace_.mamba_projected_.data(), mamba->conv_weight,
                           mamba->conv_bias, mamba->dt_bias, mamba->a_log, mamba->d,
                           mamba->conv_state.data(), mamba->ssm_state.data(),
                           workspace_.mamba_inner_.data(), spec.intermediate_size,
                           spec.state_size, spec.num_heads, spec.head_dim,
                           spec.group_count, spec.conv_kernel, stream_.get());
        launch_rmsnorm(workspace_.mamba_inner_.data(), mamba->norm,
                       workspace_.op_output_.data(), 1, spec.intermediate_size,
                       semantics.operator_norm.epsilon, stream_.get());
        launch_multiply(workspace_.op_output_.data(), workspace_.mamba_projected_.data(),
                        spec.intermediate_size, stream_.get());
        linear(workspace_.op_output_.data(), *mamba->out, workspace_.hidden_.data(),
               1, resources_.program_.hidden, spec.intermediate_size);
        return;
    }
    if (MlpOnlyLayer* mlp = as_mlp_only(layer)) {
        linear(workspace_.normed_.data(), *mlp->up, workspace_.gate_up_.data(),
               1, mlp->spec.intermediate_size, resources_.program_.hidden);
        launch_relu2(workspace_.gate_up_.data(), workspace_.activated_.data(),
                     mlp->spec.intermediate_size, stream_.get());
        linear(workspace_.activated_.data(), *mlp->down, workspace_.hidden_.data(),
               1, resources_.program_.hidden, mlp->spec.intermediate_size);
        return;
    }
    ConvolutionLayer& convolution = *as_convolution(layer);
    decode_phase_profile().begin(stream_.get());
    linear(workspace_.normed_.data(), *convolution.conv_in,
           workspace_.conv_projected_.data(), 1, 3 * resources_.program_.hidden,
           resources_.program_.hidden);
    launch_conv_decode_device(
        workspace_.conv_projected_.data(), convolution.conv_weight,
        convolution.conv_state.data(), workspace_.op_output_.data(),
        resources_.program_.hidden, resources_.shape_.conv_cache,
        position_device_.data(), stream_.get());
    linear(workspace_.op_output_.data(), *convolution.conv_out,
           workspace_.hidden_.data(), 1, resources_.program_.hidden,
           resources_.program_.hidden,
           resources_.options_.fused_residuals ? 1.0f : 0.0f);
    decode_phase_profile().end(DecodePhase::Conv, stream_.get());
}

} // namespace celeg
