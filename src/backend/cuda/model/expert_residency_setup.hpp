#pragma once

#include "backend/cuda/weights_loader.hpp"
#include "detail/layer_state.hpp"

#include <memory>

namespace celeg {

struct CudaCompiledModel;
struct ResidencyController;

OffloadedExpertWeights install_cuda_expert_controller(
    CudaCompiledModel& model,
    int resource_layer,
    std::unique_ptr<ResidencyController> controller);

OffloadedExpertWeights bind_cuda_host_expert_residency(
    CudaCompiledModel& model,
    const IWeightRepository& repo,
    const MoeExpertTensorNames& expert_names,
    int resource_layer,
    int expert_count,
    int intermediate);

}
