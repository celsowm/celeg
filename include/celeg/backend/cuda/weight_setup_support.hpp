#pragma once

#include "celeg/backend/cuda/weight_layout.hpp"
#include "celeg/backend/cuda/runtime_types.hpp"
#include "celeg/model/weights/roles.hpp"

#include <span>
#include <string>
#include <vector>

namespace celeg {

struct CudaCompiledModel;

std::string cuda_layer_name(int layer, const std::string& suffix);
std::string cuda_tensor_name(std::span<const TensorRequest> requests,
                             TensorRole role, int layer = -1);
std::unique_ptr<IWeightLayout> make_cuda_embedding_layout(
    WeightMode mode, const LinearWeight& weight, const char* label);

void configure_cuda_expert_resources(CudaCompiledModel& model);

} // namespace celeg
