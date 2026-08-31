#pragma once

#include <vector>

namespace celeg {

struct CudaCompiledModel;
class IWeightRepository;
struct CompiledLayerProgram;
struct LayerCommon;

bool bind_cuda_attention_layer(CudaCompiledModel& model,
                               const IWeightRepository& repo,
                               const CompiledLayerProgram& semantics,
                               int layer_index,
                               const LayerCommon& common_layer,
                               std::vector<int>& shared_owner);

}
