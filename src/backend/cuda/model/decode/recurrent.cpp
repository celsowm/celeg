#include "detail/compiled_model.hpp"
#include "celeg/backend/cuda/attention_norm.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/paged_kv.hpp"
#include "celeg/backend/cuda/weight_layout.hpp"
#include "celeg/backend/cuda/moe.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace celeg {

void CudaCompiledModel::run_token_gated_delta(GatedDeltaNetLayer& gated_delta,
                                              const CompiledLayerProgram& semantics) {
    const GatedDeltaNetSpec& spec = gated_delta.spec;
    const int qkv_width = 2 * spec.key_heads * spec.key_head_dim +
        spec.value_heads * spec.value_head_dim;
    const int value_width = spec.value_heads * spec.value_head_dim;
    if (spec.factorized_projections) {
        linear(workspace_.normed_.data(), *gated_delta.q,
               workspace_.gated_delta_qkv_.data(), 1,
               spec.key_heads * spec.key_head_dim, resources_.program_.hidden);
        linear(workspace_.normed_.data(), *gated_delta.k,
               workspace_.qkv_output_.data(), 1,
               spec.key_heads * spec.key_head_dim, resources_.program_.hidden);
        linear(workspace_.normed_.data(), *gated_delta.v,
               workspace_.gated_delta_output_.data(), 1, value_width,
               resources_.program_.hidden);
        launch_interleave_gated_delta_qkv(
            workspace_.gated_delta_qkv_.data(), workspace_.qkv_output_.data(),
            workspace_.gated_delta_output_.data(), workspace_.gated_delta_qkv_.data(),
            1, spec.key_heads * spec.key_head_dim, value_width, stream_.get());
    } else {
        linear(workspace_.normed_.data(), *gated_delta.qkv,
               workspace_.gated_delta_qkv_.data(), 1, qkv_width,
               resources_.program_.hidden);
    }
    linear(workspace_.normed_.data(), *gated_delta.z,
           workspace_.gated_delta_z_.data(), 1, value_width, resources_.program_.hidden);
    linear(workspace_.normed_.data(), *gated_delta.b,
           workspace_.gated_delta_b_.data(), 1, spec.value_heads, resources_.program_.hidden);
    linear(workspace_.normed_.data(), *gated_delta.a,
           workspace_.gated_delta_a_.data(), 1, spec.decay_width(), resources_.program_.hidden);
    const float epsilon = semantics.mixer_norm.before
        ? semantics.mixer_norm.before->epsilon
        : resources_.program_.final_norm.epsilon;
    launch_gated_delta_net(workspace_.gated_delta_qkv_.data(),
        workspace_.gated_delta_z_.data(), workspace_.gated_delta_b_.data(),
        workspace_.gated_delta_a_.data(), gated_delta.conv_weight,
        gated_delta.dt_bias, gated_delta.a_log, gated_delta.norm,
        gated_delta.conv_state.data(), gated_delta.recurrent_state.data(),
        workspace_.gated_delta_output_.data(), 1, spec.conv_kernel,
        spec.key_head_dim, spec.value_head_dim, spec.key_heads,
        spec.value_heads, epsilon, spec.vector_decay, spec.safe_decay,
        spec.decay_lower_bound, spec.sigmoid_output_gate, stream_.get());
    linear(workspace_.gated_delta_output_.data(), *gated_delta.out,
           workspace_.hidden_.data(), 1, resources_.program_.hidden, value_width);
}

void CudaCompiledModel::run_token_mamba2(Mamba2Layer& mamba,
                                         const CompiledLayerProgram& semantics,
                                         const TokenKvPolicy&) {
    const Mamba2Spec& spec = mamba.spec;
    linear(workspace_.normed_.data(), *mamba.in,
           workspace_.mamba_projected_.data(), 1,
           2 * spec.intermediate_size + 2 * spec.group_count * spec.state_size +
               spec.num_heads, resources_.program_.hidden);
    launch_mamba2_step(workspace_.mamba_projected_.data(), mamba.conv_weight,
                       mamba.conv_bias, mamba.dt_bias, mamba.a_log, mamba.d,
                       mamba.conv_state.data(), mamba.ssm_state.data(),
                       workspace_.mamba_inner_.data(), spec.intermediate_size,
                       spec.state_size, spec.num_heads, spec.head_dim,
                       spec.group_count, spec.conv_kernel, stream_.get());
    const float epsilon = semantics.mixer_norm.before
        ? semantics.mixer_norm.before->epsilon
        : (semantics.mixer_norm.after
            ? semantics.mixer_norm.after->epsilon
            : resources_.program_.final_norm.epsilon);
    launch_rmsnorm(workspace_.mamba_inner_.data(), mamba.norm,
                   workspace_.op_output_.data(), 1, spec.intermediate_size,
                   epsilon, stream_.get());
    launch_multiply(workspace_.op_output_.data(), workspace_.mamba_projected_.data(),
                    spec.intermediate_size, stream_.get());
    linear(workspace_.op_output_.data(), *mamba.out, workspace_.hidden_.data(),
           1, resources_.program_.hidden, spec.intermediate_size);
}

}
