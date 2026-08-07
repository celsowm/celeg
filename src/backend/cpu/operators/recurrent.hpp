#pragma once

#include "../detail/model_internal.hpp"

namespace celeg {

void execute_cpu_gated_delta_token(CpuCompiledModel& model, size_t layer,
                                   const CpuCompiledModel::GatedDeltaNetWeights& weights);
void execute_cpu_mamba2_token(CpuCompiledModel& model, size_t layer,
                              const CpuCompiledModel::Mamba2Weights& weights);
void execute_cpu_short_convolution_token(CpuCompiledModel& model, size_t layer,
                                         const CpuCompiledModel::ConvolutionWeights& weights);
void execute_cpu_gated_delta_chunk(CpuCompiledModel& model, size_t layer,
                                   const CpuCompiledModel::GatedDeltaNetWeights& weights,
                                   size_t rows, bool& normed_q8_ready);
void execute_cpu_short_convolution_chunk(
    CpuCompiledModel& model, size_t layer,
    const CpuCompiledModel::ConvolutionWeights& weights,
    size_t rows, bool& normed_q8_ready);

}
