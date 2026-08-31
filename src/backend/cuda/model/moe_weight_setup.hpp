#pragma once

#include "backend/cuda/weights_loader.hpp"
#include "detail/feed_forward_weights.hpp"

#include <string>

namespace celeg {

struct CudaCompiledModel;
class IWeightRepository;
struct MoeLayerProgram;
struct LayerCommon;

MoeFfnWeights bind_cuda_moe_router_weight(
    CudaCompiledModel& model,
    const IWeightRepository& repo,
    const std::string& router_name,
    int resource_layer,
    int expert_count,
    const float* expert_bias = nullptr);

ResidentExpertWeights bind_cuda_resident_experts(
    CudaCompiledModel& model,
    const IWeightRepository& repo,
    const MoeExpertTensorNames& expert_names,
    int expert_count,
    int intermediate);

void bind_cuda_moe_feed_forward(CudaCompiledModel& model,
                                const IWeightRepository& repo,
                                const MoeLayerProgram& semantics,
                                int layer_index,
                                LayerCommon& common_layer);

}
