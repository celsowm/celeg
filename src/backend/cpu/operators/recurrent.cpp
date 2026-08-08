#include "recurrent.hpp"
#include "common.hpp"

#include <chrono>
#include <cmath>

namespace celeg {

namespace {
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
}

void execute_cpu_gated_delta_token(
    CpuCompiledModel& model, size_t layer,
    const CpuCompiledModel::GatedDeltaNetWeights& weights) {
    auto& workspace = model.workspace_;
    auto& shared = *model.shared;
    const GatedDeltaNetSpec& spec = weights.spec;
    shared.linear.gemv(weights.qkv, workspace.normed.data(),
                       workspace.gated_delta_qkv.data());
    shared.linear.gemv(weights.z, workspace.normed.data(),
                       workspace.gated_delta_z.data());
    shared.linear.gemv(weights.b, workspace.normed.data(),
                       workspace.gated_delta_b.data());
    shared.linear.gemv(weights.a, workspace.normed.data(),
                       workspace.gated_delta_a.data());
    auto& state = model.gated_delta_net_state(layer);
    cpu_gated_delta_net_decode(
        workspace.gated_delta_qkv.data(), workspace.gated_delta_z.data(),
        workspace.gated_delta_b.data(), workspace.gated_delta_a.data(),
        weights.conv_weight.data(), weights.dt_bias.data(), weights.a_log.data(),
        weights.norm.data(), state.conv.data(), state.recurrent.data(),
        workspace.gated_delta_output.data(), spec.conv_kernel, spec.key_head_dim,
        spec.value_head_dim, spec.key_heads, spec.value_heads,
        shared.shape.numerical_policy.norm_eps);
    shared.linear.gemv(weights.out, workspace.gated_delta_output.data(),
                       workspace.hidden.data());
}

void execute_cpu_mamba2_token(
    CpuCompiledModel& model, size_t layer,
    const CpuCompiledModel::Mamba2Weights& weights) {
    auto& workspace = model.workspace_;
    auto& shared = *model.shared;
    const auto& spec = shared.shape.mamba2_layouts.at(layer);
    const int inner = spec.intermediate_size;
    const int conv_dim = inner + 2 * spec.group_count * spec.state_size;
    shared.linear.gemv(weights.in, workspace.normed.data(),
                       workspace.mamba_projected.data());
    const float* z = workspace.mamba_projected.data();
    const float* xbc = z + inner;
    const float* dt_raw = xbc + conv_dim;
    auto& state = model.mamba2_state(layer);
    for (int channel = 0; channel < conv_dim; ++channel) {
        float* history = state.conv.data() +
            static_cast<size_t>(channel) * spec.conv_kernel;
        for (int tap = 0; tap + 1 < spec.conv_kernel; ++tap) {
            history[tap] = history[tap + 1];
        }
        history[spec.conv_kernel - 1] = xbc[channel];
        float value = weights.conv_bias[channel];
        for (int tap = 0; tap < spec.conv_kernel; ++tap) {
            value += history[tap] * weights.conv_weight[
                static_cast<size_t>(channel) * spec.conv_kernel + tap];
        }
        workspace.mamba_bcx[channel] = value / (1.0f + std::exp(-value));
    }
    const int group_size = spec.num_heads / spec.group_count;
    for (int head = 0; head < spec.num_heads; ++head) {
        const float dt = std::log1p(std::exp(dt_raw[head] + weights.dt_bias[head]));
        const float decay = std::exp(dt * -std::exp(weights.a_log[head]));
        const int group = head / group_size;
        for (int d = 0; d < spec.head_dim; ++d) {
            const int channel = head * spec.head_dim + d;
            const float x = workspace.mamba_bcx[channel];
            const float* b = workspace.mamba_bcx.data() + inner + group * spec.state_size;
            const float* c = b + spec.group_count * spec.state_size;
            float* s = state.ssm.data() + static_cast<size_t>(channel) * spec.state_size;
            float output = 0.0f;
            for (int n = 0; n < spec.state_size; ++n) {
                s[n] = decay * s[n] + dt * b[n] * x;
                output += s[n] * c[n];
            }
            workspace.mamba_inner[channel] = output + weights.d[head] * x;
        }
    }
    cpu_rmsnorm(workspace.mamba_inner.data(), weights.norm.data(),
                workspace.op_output.data(), inner,
                shared.shape.numerical_policy.norm_eps);
    for (int i = 0; i < inner; ++i) {
        const float gate = z[i];
        workspace.op_output[i] *= gate / (1.0f + std::exp(-gate));
    }
    shared.linear.gemv(weights.out, workspace.op_output.data(),
                       workspace.hidden.data());
    cpu_residual_add(workspace.hidden.data(), workspace.residual.data(),
                     shared.shape.hidden);
}

void execute_cpu_short_convolution_token(
    CpuCompiledModel& model, size_t layer,
    const CpuCompiledModel::ConvolutionWeights& weights) {
    auto& workspace = model.workspace_;
    auto& shared = *model.shared;
    auto& state = model.convolution_state(layer);
    shared.linear.gemv(weights.in, workspace.normed.data(),
                       workspace.conv_projected.data());
    cpu_conv_decode(workspace.conv_projected.data(), weights.weight_tap_major.data(),
                    state.state.data(), workspace.op_output.data(), shared.shape.hidden,
                    shared.shape.conv_cache, model.session_.position_value);
    shared.linear.gemv(weights.out, workspace.op_output.data(),
                       workspace.hidden.data());
}

void execute_cpu_gated_delta_chunk(
    CpuCompiledModel& model, size_t layer,
    const CpuCompiledModel::GatedDeltaNetWeights& weights,
    size_t rows, bool& normed_q8_ready) {
    auto& workspace = model.workspace_;
    auto& shared = *model.shared;
    CpuExecutionContext execution{shared, workspace, model.session_};
    const GatedDeltaNetSpec& spec = weights.spec;
    const size_t hidden = static_cast<size_t>(shared.shape.hidden);

    auto started = Clock::now();
    cpu_chunk_layer_gemm(execution, weights.qkv, workspace.chunk_normed.data(),
                         workspace.chunk_gated_delta_qkv.data(), rows, hidden,
                         normed_q8_ready);
    cpu_chunk_layer_gemm(execution, weights.z, workspace.chunk_normed.data(),
                         workspace.chunk_gated_delta_z.data(), rows, hidden,
                         normed_q8_ready);
    cpu_chunk_layer_gemm(execution, weights.b, workspace.chunk_normed.data(),
                         workspace.chunk_gated_delta_b.data(), rows, hidden,
                         normed_q8_ready);
    cpu_chunk_layer_gemm(execution, weights.a, workspace.chunk_normed.data(),
                         workspace.chunk_gated_delta_a.data(), rows, hidden,
                         normed_q8_ready);
    model.session_.prefill_profile.linear_ms += elapsed_ms(started);

    started = Clock::now();
    auto& state = model.gated_delta_net_state(layer);
    cpu_gated_delta_net_prefill(
        workspace.chunk_gated_delta_qkv.data(), workspace.chunk_gated_delta_z.data(),
        workspace.chunk_gated_delta_b.data(), workspace.chunk_gated_delta_a.data(),
        weights.conv_weight.data(), weights.dt_bias.data(), weights.a_log.data(),
        weights.norm.data(), state.conv.data(), state.recurrent.data(),
        workspace.chunk_gated_delta_output.data(), rows, spec.conv_kernel,
        spec.key_head_dim, spec.value_head_dim, spec.key_heads, spec.value_heads,
        shared.shape.numerical_policy.norm_eps);
    model.session_.prefill_profile.shortconv_ms += elapsed_ms(started);

    started = Clock::now();
    cpu_chunk_layer_gemm(execution, weights.out, workspace.chunk_gated_delta_output.data(),
                         workspace.chunk_hidden.data(), rows, hidden,
                         normed_q8_ready);
    model.session_.prefill_profile.linear_ms += elapsed_ms(started);
}

void execute_cpu_short_convolution_chunk(
    CpuCompiledModel& model, size_t layer,
    const CpuCompiledModel::ConvolutionWeights& weights,
    size_t rows, bool& normed_q8_ready) {
    auto& workspace = model.workspace_;
    auto& shared = *model.shared;
    CpuExecutionContext execution{shared, workspace, model.session_};
    const size_t hidden = static_cast<size_t>(shared.shape.hidden);

    auto started = Clock::now();
    cpu_chunk_layer_gemm(execution, weights.in, workspace.chunk_normed.data(),
                         workspace.chunk_conv.data(), rows, hidden,
                         normed_q8_ready);
    model.session_.prefill_profile.linear_ms += elapsed_ms(started);

    started = Clock::now();
    auto& state = model.convolution_state(layer);
    cpu_conv_prefill(workspace.chunk_conv.data(), weights.weight_tap_major.data(),
                     state.state.data(), workspace.chunk_op.data(), rows,
                     shared.shape.hidden, shared.shape.conv_cache,
                     model.session_.position_value, shared.pool);
    model.session_.prefill_profile.shortconv_ms += elapsed_ms(started);

    started = Clock::now();
    cpu_chunk_layer_gemm(execution, weights.out, workspace.chunk_op.data(),
                         workspace.chunk_hidden.data(), rows, hidden,
                         normed_q8_ready);
    model.session_.prefill_profile.linear_ms += elapsed_ms(started);
}

}
