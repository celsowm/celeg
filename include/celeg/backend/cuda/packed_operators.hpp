#pragma once

#include "celeg/backend/cuda/gemm_dispatcher.hpp"
#include "celeg/backend/cuda/packed_session.hpp"
#include "celeg/backend/cuda/packed_workspace.hpp"
#include "celeg/detail/model/compiled_model.hpp"

#include <cstddef>
#include <vector>

namespace celeg {

struct PackedOperatorContext {
    PackedWorkspace& workspace;
    GemmDispatcher& gemm;
    const CudaExecutionPlan& plan;
    const ExecutionTopology& shape;
    const CompiledModelProgram& program;

    void linear(const __nv_bfloat16* input,
                const LinearWeight& weight,
                __nv_bfloat16* output,
                int rows,
                int output_width,
                int input_width,
                float beta = 0.0f) const;
};

class PackedConvolutionExecutor {
public:
    static void run(PackedOperatorContext& context,
                    const PackedSessionContext& reference,
                    const ConvolutionLayer& convolution,
                    int rows,
                    int layer_index,
                    int ragged_requests);
};

class PackedGatedDeltaNetExecutor {
public:
    static void run(PackedOperatorContext& context,
                    const PackedSessionContext& reference,
                    const GatedDeltaNetLayer& gated_delta,
                    int rows,
                    int layer_index,
                    const std::vector<PackedSessionContext>* batch_models,
                    const std::vector<PackedPrefillRow>* row_descriptors);
};

class PackedMamba2Executor {
public:
    static void run(PackedOperatorContext& context,
                    const PackedSessionContext& reference,
                    const Mamba2Layer& mamba,
                    int rows,
                    int layer_index,
                    const std::vector<PackedSessionContext>* batch_models,
                    const std::vector<PackedPrefillRow>* row_descriptors);
};

class PackedAttentionExecutor {
public:
    static void run(PackedOperatorContext& context,
                    const PackedSessionContext& reference,
                    const AttentionLayer& attention,
                    int rows,
                    int layer_index,
                    bool segmented_attention,
                    int segmented_chunks);
};

class PackedDenseFfnExecutor {
public:
    static void run(PackedOperatorContext& context,
                    const PackedSessionContext& reference,
                    const LayerCommon& common_layer,
                    int rows,
                    int layer_index);
};

class PackedMoeExecutor {
public:
    static void run(PackedOperatorContext& context,
                    const PackedSessionContext& reference,
                    const LayerCommon& common_layer,
                    int rows,
                    const std::vector<PackedSessionContext>* batch_models,
                    int layer_index);
};

} // namespace celeg
