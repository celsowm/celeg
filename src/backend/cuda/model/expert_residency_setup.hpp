#pragma once

#include "backend/cuda/weights_loader.hpp"
#include "backend/cuda/moe/offload.hpp"
#include "detail/layer_state.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace celeg {

struct CudaCompiledModel;
struct ResidencyController;

struct PreparedDiskExpertResidency {
    std::vector<ExpertLocation> catalog;
    std::unique_ptr<ResidencyController> controller;
};

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

PreparedDiskExpertResidency prepare_cuda_disk_expert_residency(
    CudaCompiledModel& model,
    const IWeightRepository& repo,
    const MoeExpertTensorNames& expert_names,
    int resource_layer,
    int expert_count,
    int intermediate);

void promote_cuda_disk_expert_payload(
    ResidencyController& controller,
    int slot,
    const ExpertLocation& location,
    const std::byte* payload);

}
