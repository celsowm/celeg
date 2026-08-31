#pragma once

namespace celeg {

struct CudaCompiledModel;
class IWeightRepository;
struct CompiledLayerProgram;
struct LayerCommon;

void bind_cuda_non_attention_layer(CudaCompiledModel& model,
                                   const IWeightRepository& repo,
                                   const CompiledLayerProgram& semantics,
                                   int layer_index,
                                   const LayerCommon& common_layer);

}
