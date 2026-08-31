#pragma once

#include "backend/cuda/weights_loader.hpp"
#include "detail/feed_forward_weights.hpp"

#include <optional>
#include <string>

namespace celeg {

struct CudaCompiledModel;
class IWeightRepository;
struct MoeLayerProgram;
struct LayerCommon;

struct CudaSharedExpertNames {
    std::string synthetic_w13;
    std::string gate;
    std::string up;
    std::string down;
    std::optional<std::string> gate_weight;
};

MoeFfnWeights bind_cuda_moe_router_weight(
    CudaCompiledModel& model,
    const IWeightRepository& repo,
    const std::string& router_name,
    int resource_layer,
    int expert_count,
    const float* expert_bias = nullptr);

void bind_cuda_shared_expert(
    CudaCompiledModel& model,
    const IWeightRepository& repo,
    const CudaSharedExpertNames& names,
    int intermediate,
    MoeFfnWeights& weights);

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
