#pragma once

#include "detail/feed_forward_weights.hpp"

#include <string>

namespace celeg {

struct CudaCompiledModel;
class IWeightRepository;
struct CompiledLayerProgram;
struct LayerCommon;

struct CudaDenseFfnNames {
    std::string synthetic_w13;
    std::string gate;
    std::string up;
    std::string down;
    std::string synthetic_parallel_w13;
    std::string parallel_gate;
    std::string parallel_up;
    std::string parallel_down;
};

void prepare_cuda_layer_weight_resources(CudaCompiledModel& model);

DenseFfnWeights bind_cuda_dense_ffn(
    CudaCompiledModel& model,
    const IWeightRepository& repo,
    const CudaDenseFfnNames& names,
    int intermediate,
    int parallel_intermediate = 0);

LayerCommon bind_cuda_layer_common(CudaCompiledModel& model,
                                   const IWeightRepository& repo,
                                   const CompiledLayerProgram& semantics,
                                   int layer_index);

}
