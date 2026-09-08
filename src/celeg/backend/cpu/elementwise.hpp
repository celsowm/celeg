#pragma once

#include <cstddef>

namespace celeg {

void cpu_residual_add(float* data, const float* residual, size_t count);
void cpu_swiglu(const float* gate_up, float* output, size_t count);
void cpu_gated_gelu_tanh(const float* gate_up, float* output, size_t count);
void cpu_relu2(const float* input, float* output, size_t count);
void cpu_gelu_tanh(float* data, size_t count);

}
