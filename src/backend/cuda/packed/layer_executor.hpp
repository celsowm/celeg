#pragma once

#include "packed/gemm_runtime.hpp"
#include "packed/layer_program.hpp"
#include "packed/operators.hpp"
#include "packed/workspace.hpp"

#include <vector>

namespace celeg {

class PackedLayerExecutor {
public:
    PackedLayerExecutor(PackedWorkspace& workspace,
                        const PackedLayerProgram& layer_program,
                        const CudaExecutionPlan& plan,
                        PackedGemmRuntime& gemm)
        : workspace_(workspace), layer_program_(layer_program),
          plan_(plan), gemm_(gemm) {}

    void launch_embedding_rows(const PackedSessionContext& reference, int rows);
    void linear(const __nv_bfloat16* x,
                const LinearWeight& weight,
                __nv_bfloat16* y,
                int m,
                int n,
                int k_width,
                float beta = 0.0f);
    void run_transformer_layers(
        const PackedSessionContext& reference,
        int rows,
        bool segmented_attention,
        int segmented_chunks,
        const std::vector<PackedSessionContext>* batch_models = nullptr,
        int ragged_requests = 0,
        const std::vector<PackedPrefillRow>* row_descriptors = nullptr);

private:
    void run_attention_layer(const PackedSessionContext& reference,
                             const AttentionLayer& attention,
                             int rows,
                             int layer_index,
                             bool segmented_attention,
                             int segmented_chunks);
    void run_convolution_layer(const PackedSessionContext& reference,
                               const ConvolutionLayer& convolution,
                               int rows,
                               int layer_index,
                               int ragged_requests);
    void run_mlp_layer(const PackedSessionContext& reference,
                       const LayerCommon& common_layer,
                       int rows,
                       const std::vector<PackedSessionContext>* batch_models,
                       int layer_index);

    PackedWorkspace& workspace_;
    const PackedLayerProgram& layer_program_;
    const CudaExecutionPlan& plan_;
    PackedGemmRuntime& gemm_;
};

}
