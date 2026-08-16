#include "feed_forward.hpp"

#include "common.hpp"

#include <chrono>
#include <stdexcept>

namespace celeg {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

const CompiledDenseFeedForwardProgram& dense_program(
    const CpuCompiledModel::Shared& shared, size_t layer) {
    const auto* dense = std::get_if<CompiledDenseFeedForwardProgram>(
        &shared.program.layers.at(layer).feed_forward);
    if (!dense) {
        throw std::logic_error("CPU dense FFN execution received non-dense semantics");
    }
    return *dense;
}

int intermediate_size(const CpuCompiledModel::Shared& shared, size_t layer) {
    return dense_program(shared, layer).intermediate_size;
}

bool uses_gelu_tanh(const CpuCompiledModel::Shared& shared, size_t layer) {
    return dense_program(shared, layer).activation == ActivationKind::GeluTanh;
}

}

void execute_cpu_mlp_only_token(CpuExecutionContext& context, size_t layer,
                                const CpuCompiledModel::MlpOnlyWeights& weights) {
    auto& shared = context.shared;
    auto& workspace = context.workspace;
    const CompiledLayerProgram& semantics = shared.program.layers.at(layer);
    const auto* mlp = std::get_if<MlpBlockSpec>(&semantics.mixer);
    if (!mlp || mlp->activation != ActivationKind::Relu2) {
        throw std::logic_error("CPU MLP-only execution received unsupported semantics");
    }
    const int intermediate = mlp->intermediate_size;
    shared.linear.gemv(weights.mlp_up, workspace.normed.data(),
                       workspace.activated.data());
    cpu_relu2(workspace.activated.data(), workspace.activated.data(), intermediate);
    shared.linear.gemv(weights.w2, workspace.activated.data(),
                       workspace.hidden.data());
    cpu_residual_add(workspace.hidden.data(), workspace.residual.data(),
                     shared.program.hidden);
}

void execute_cpu_dense_feed_forward_token(
    CpuExecutionContext& context, size_t layer,
    const CpuCompiledModel::DenseFeedForwardWeights& weights) {
    auto& shared = context.shared;
    auto& workspace = context.workspace;
    const int intermediate = intermediate_size(shared, layer);
    const auto started = Clock::now();
    shared.linear.gemv(weights.w13, workspace.normed.data(), workspace.gate_up.data());
    if (uses_gelu_tanh(shared, layer)) {
        cpu_gated_gelu_tanh(workspace.gate_up.data(), workspace.activated.data(), intermediate);
    } else {
        cpu_swiglu(workspace.gate_up.data(), workspace.activated.data(), intermediate);
    }
    shared.linear.gemv(weights.w2, workspace.activated.data(), workspace.mlp_output.data());
    if (context.session.phase == SessionPhase::Prefilling) {
        context.session.prefill_profile.linear_ms += elapsed_ms(started);
    }
}

void execute_cpu_dense_feed_forward_chunk(
    CpuExecutionContext& context, size_t layer,
    const CpuCompiledModel::DenseFeedForwardWeights& weights,
    size_t rows, bool& normed_q8_ready) {
    auto& shared = context.shared;
    auto& workspace = context.workspace;
    const int intermediate = intermediate_size(shared, layer);
    const auto started = Clock::now();
    cpu_chunk_layer_gemm(context, weights.w13,
                         workspace.chunk_normed.data(), workspace.chunk_gate_up.data(),
                         rows, static_cast<size_t>(shared.program.hidden), normed_q8_ready);
    if (context.session.phase == SessionPhase::Prefilling) {
        context.session.prefill_profile.linear_ms += elapsed_ms(started);
    }
    cpu_parallel_rows(shared.pool, rows, [&](size_t row) {
        const float* gate_up = workspace.chunk_gate_up.data() +
            row * 2ULL * static_cast<size_t>(intermediate);
        float* activated = workspace.chunk_activated.data() +
            row * static_cast<size_t>(intermediate);
        if (uses_gelu_tanh(shared, layer)) {
            cpu_gated_gelu_tanh(gate_up, activated, intermediate);
        } else {
            cpu_swiglu(gate_up, activated, intermediate);
        }
    });
    const auto output_started = Clock::now();
    cpu_chunk_layer_gemm(context, weights.w2,
                         workspace.chunk_activated.data(), workspace.chunk_mlp.data(),
                         rows, static_cast<size_t>(shared.program.hidden), normed_q8_ready);
    if (context.session.phase == SessionPhase::Prefilling) {
        context.session.prefill_profile.linear_ms += elapsed_ms(output_started);
    }
}

}
