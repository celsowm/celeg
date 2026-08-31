#pragma once

namespace celeg {

struct CudaCompiledModel;
class IWeightRepository;
struct CompiledLayerProgram;
struct LayerCommon;

void prepare_cuda_layer_weight_resources(CudaCompiledModel& model);

LayerCommon bind_cuda_layer_common(CudaCompiledModel& model,
                                   const IWeightRepository& repo,
                                   const CompiledLayerProgram& semantics,
                                   int layer_index);

}
