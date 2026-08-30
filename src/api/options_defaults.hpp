#pragma once

#include "celeg/api.h"

namespace celeg::api::defaults {

void generation(celeg_generation_options& options);
void cpu_model_config(celeg_cpu_model_config& options);
void cpu_engine_options(celeg_cpu_engine_options& options);
void cuda_model_options(celeg_cuda_model_options& options);
void cuda_engine_options(celeg_cuda_engine_options& options);
void metal_model_options(celeg_metal_model_options& options);
void metal_engine_options(celeg_metal_engine_options& options);

}
