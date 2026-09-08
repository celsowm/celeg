#pragma once

#include <cstddef>

namespace celeg {

void cpu_rmsnorm(const float* input, const float* weight, float* output,
                 size_t width, float eps);
void cpu_rmsnorm_inplace(float* data, const float* weight,
                         size_t width, float eps);

}
