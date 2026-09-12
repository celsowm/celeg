#include "feed_forward.hpp"

#include "common.hpp"
#include "../kernels/math.hpp"
#include "celeg/backend/cpu/elementwise.hpp"

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

int parallel_intermediate_size(const CpuCompiledModel::Shared& shared, size_t layer) {
    return dense_program(shared, layer).parallel_intermediate_size;
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
}

void execute_cpu_dense_feed_forward_token(
    CpuExecutionContext& context, size_t layer,
    const CpuCompiledModel::DenseFeedForwardWeights& weights) {
    auto& shared = context.shared;
    auto& workspace = context.workspace;
    const CpuMathEngine& math = cpu_math_engine(shared.linear.isa());
    const int intermediate = intermediate_size(shared, layer);
    const auto started = Clock::now();
    shared.linear.gemv(weights.w13, workspace.normed.data(), workspace.gate_up.data());
    if (uses_gelu_tanh(shared, layer)) {
        cpu_gated_gelu_tanh(workspace.gate_up.data(), workspace.activated.data(), intermediate);
    } else {
        math.swiglu(workspace.gate_up.data(), workspace.activated.data(), intermediate);
    }
    shared.linear.gemv(weights.w2, workspace.activated.data(), workspace.mlp_output.data());
    const int parallel = parallel_intermediate_size(shared, layer);
    if (parallel > 0) {
        shared.linear.gemv(weights.parallel_w13, workspace.normed.data(),
                           workspace.gate_up.data());
        if (uses_gelu_tanh(shared, layer)) {
            cpu_gated_gelu_tanh(workspace.gate_up.data(), workspace.activated.data(), parallel);
        } else {
            math.swiglu(workspace.gate_up.data(), workspace.activated.data(), parallel);
        }
        shared.linear.gemv(weights.parallel_w2, workspace.activated.data(),
                           workspace.shared_output.data());
        cpu_residual_add(workspace.mlp_output.data(), workspace.shared_output.data(),
                         static_cast<size_t>(shared.program.hidden));
    }
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
    const CpuMathEngine& math = cpu_math_engine(shared.linear.isa());
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
            math.swiglu(gate_up, activated, intermediate);
        }
    });
    const auto output_started = Clock::now();
    cpu_chunk_layer_gemm(context, weights.w2,
                         workspace.chunk_activated.data(), workspace.chunk_mlp.data(),
                         rows, static_cast<size_t>(shared.program.hidden), normed_q8_ready);
    const int parallel = parallel_intermediate_size(shared, layer);
    if (parallel > 0) {
        cpu_chunk_layer_gemm(context, weights.parallel_w13,
                             workspace.chunk_normed.data(), workspace.chunk_gate_up.data(),
                             rows, static_cast<size_t>(shared.program.hidden), normed_q8_ready);
        cpu_parallel_rows(shared.pool, rows, [&](size_t row) {
            const float* gate_up = workspace.chunk_gate_up.data() +
                row * 2ULL * static_cast<size_t>(parallel);
            float* activated = workspace.chunk_activated.data() +
                row * static_cast<size_t>(parallel);
            if (uses_gelu_tanh(shared, layer)) {
                cpu_gated_gelu_tanh(gate_up, activated, parallel);
            } else {
                math.swiglu(gate_up, activated, parallel);
            }
        });
        cpu_chunk_layer_gemm(context, weights.parallel_w2,
                             workspace.chunk_activated.data(), workspace.shared_output.data(),
                             rows, static_cast<size_t>(shared.program.hidden), normed_q8_ready);
        cpu_parallel_rows(shared.pool, rows, [&](size_t row) {
            float* destination = workspace.chunk_mlp.data() +
                row * static_cast<size_t>(shared.program.hidden);
            const float* source = workspace.shared_output.data() +
                row * static_cast<size_t>(shared.program.hidden);
            cpu_residual_add(destination, source,
                             static_cast<size_t>(shared.program.hidden));
        });
    }
    if (context.session.phase == SessionPhase::Prefilling) {
        context.session.prefill_profile.linear_ms += elapsed_ms(output_started);
    }
}

}
