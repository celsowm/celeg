#pragma once

namespace celeg {

struct CudaCompiledModel;
class IWeightRepository;
struct MoeLayerProgram;
struct LayerCommon;

void bind_cuda_moe_feed_forward(CudaCompiledModel& model,
                                const IWeightRepository& repo,
                                const MoeLayerProgram& semantics,
                                int layer_index,
                                LayerCommon& common_layer);

}
