#include "celeg/backend/cuda/packed/layer_executor.hpp"

#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/moe.hpp"
#include "celeg/detail/model/compiled_model.hpp"

#include <algorithm>
#include <stdexcept>

namespace celeg {

void PackedLayerExecutor::linear(const __nv_bfloat16* x,
                                 const LinearWeight& weight,
                                 __nv_bfloat16* y,
                                 int m,
                                 int n,
                                 int k_width,
                                 float beta) {
    if (weight.rows != n || weight.cols != k_width) {
        throw std::runtime_error("packed linear shape mismatch");
    }
    gemm_.dispatcher().linear(x, weight, y, m, n, k_width, beta, plan_);
}

void PackedLayerExecutor::launch_embedding_rows(
    const PackedSessionContext& reference, int rows) {
    const LinearStorage& storage = reference.embedding()->storage;
    if (const auto* int4 = std::get_if<Int4LinearStorage>(&storage)) {
        launch_embedding_int4_batch(
            workspace_.sampled.data(), rows, int4->data,
            int4->scales, workspace_.hidden.data(),
            workspace_.program_.hidden, workspace_.stream.get());
    } else if (const auto* int8 = std::get_if<Int8LinearStorage>(&storage)) {
        launch_embedding_int8_batch(
            workspace_.sampled.data(), rows, int8->data,
            int8->scales, workspace_.hidden.data(),
            workspace_.program_.hidden, workspace_.stream.get());
    } else {
        launch_embedding_batch(
            workspace_.sampled.data(), rows, std::get<Bf16LinearStorage>(storage).data,
            workspace_.hidden.data(), workspace_.program_.hidden,
            workspace_.stream.get());
    }
    launch_scale(workspace_.hidden.data(), rows * workspace_.program_.hidden,
                 reference.program().embedding_transform.multiplier,
                 workspace_.stream.get());
}

void PackedLayerExecutor::run_attention_layer(
    const PackedSessionContext& reference,
    const AttentionLayer& attention,
    int rows,
    int layer_index,
    bool segmented_attention,
    int segmented_chunks) {
    PackedOperatorContext context{
        workspace_, gemm_.dispatcher(), plan_, workspace_.shape_, workspace_.program_};
    PackedAttentionExecutor::run(context, reference, attention, rows,
                                 layer_index, segmented_attention,
                                 segmented_chunks);
}

void PackedLayerExecutor::run_convolution_layer(
    const PackedSessionContext& reference,
    const ConvolutionLayer& convolution,
    int rows,
    int layer_index,
    int ragged_requests) {
    PackedOperatorContext context{
        workspace_, gemm_.dispatcher(), plan_, workspace_.shape_, workspace_.program_};
    PackedConvolutionExecutor::run(context, reference, convolution, rows,
                                   layer_index, ragged_requests);
}

void PackedLayerExecutor::run_mlp_layer(
    const PackedSessionContext& reference,
    const LayerCommon& common_layer,
    int rows,
    const std::vector<PackedSessionContext>* batch_models,
    int layer_index) {
    PackedOperatorContext context{
        workspace_, gemm_.dispatcher(), plan_, workspace_.shape_, workspace_.program_};
    if (const MoeFfnWeights* moe = as_moe_ffn(common_layer.feed_forward)) {
        (void)moe;
        PackedMoeExecutor::run(context, reference, common_layer, rows,
                               batch_models, layer_index);
        return;
    }
    PackedDenseFfnExecutor::run(context, reference, common_layer, rows,
                                layer_index);
}

void PackedLayerExecutor::run_transformer_layers(
    const PackedSessionContext& reference,
    int rows,
    bool segmented_attention,
    int segmented_chunks,
    const std::vector<PackedSessionContext>* batch_models,
    int ragged_requests,
    const std::vector<PackedPrefillRow>* row_descriptors) {
    for (size_t layer_index = 0; layer_index < layer_program_.size();
         ++layer_index) {
        const CompiledLayerProgram& semantics = reference.program().layers.at(layer_index);
        const auto& layer = reference.layers()[layer_index];
        const auto& common_layer = common(layer);
        if (!reference.options().fused_residuals) {
            CELEG_CUDA(cudaMemcpyAsync(
                workspace_.residual.data(), workspace_.hidden.data(),
                static_cast<size_t>(rows) * workspace_.program_.hidden *
                    sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToDevice, workspace_.stream.get()));
        }
        launch_rmsnorm(workspace_.hidden.data(), common_layer.mixer_norm_before,
                       workspace_.normed.data(), rows, workspace_.program_.hidden,
                       semantics.mixer_norm.before->epsilon,
                       workspace_.stream.get());
        PackedOperatorContext context{
            workspace_, gemm_.dispatcher(), plan_, workspace_.shape_, workspace_.program_};
        const PackedLayerKind kind = layer_program_.kind(layer_index);
        visit_layer(layer,
          [&](const MlpOnlyLayer* mlp) {
            if (kind != PackedLayerKind::MlpOnly) {
                throw std::logic_error("packed layer program disagrees with MLP-only binding");
            }
            const int intermediate = mlp->spec.intermediate_size;
            if (intermediate <= 0 ||
                intermediate > static_cast<int>(workspace_.requirements_.maximum_ffn_intermediate)) {
                throw std::runtime_error("invalid packed MLP-only width at layer " +
                                         std::to_string(layer_index));
            }
            linear(workspace_.normed.data(), *mlp->up, workspace_.gate_up.data(), rows,
                   intermediate, workspace_.program_.hidden);
            switch (mlp->spec.activation) {
            case ActivationKind::Relu2:
                launch_relu2(workspace_.gate_up.data(), workspace_.activated.data(),
                             rows * intermediate, workspace_.stream.get());
                break;
            case ActivationKind::GeluTanh:
                launch_gelu_tanh(workspace_.gate_up.data(), workspace_.activated.data(),
                                 rows * intermediate, workspace_.stream.get());
                break;
            default:
                throw std::runtime_error("packed CUDA does not implement MLP-only activation");
            }
            linear(workspace_.activated.data(), *mlp->down, workspace_.hidden.data(), rows,
                   workspace_.program_.hidden, intermediate);
          },
          [&](const AttentionLayer* attention) {
            if (kind != PackedLayerKind::Attention) {
                throw std::logic_error("packed layer program disagrees with layer binding");
            }
            run_attention_layer(reference, *attention, rows,
                                static_cast<int>(layer_index),
                                segmented_attention, segmented_chunks);
          },
          [&](const GatedDeltaNetLayer* gated_delta) {
            if (kind != PackedLayerKind::GatedDeltaNet) {
                throw std::logic_error(
                    "packed layer program disagrees with GatedDeltaNet binding");
            }
            PackedGatedDeltaNetExecutor::run(
                context, reference, *gated_delta, rows,
                static_cast<int>(layer_index), batch_models, row_descriptors);
          },
          [&](const Mamba2Layer* mamba) {
            if (kind != PackedLayerKind::Mamba2) {
                throw std::logic_error(
                    "packed layer program disagrees with Mamba2 binding");
            }
            PackedMamba2Executor::run(
                context, reference, *mamba, rows,
                static_cast<int>(layer_index), batch_models, row_descriptors);
          },
          [&](const ConvolutionLayer* convolution) {
            if (kind != PackedLayerKind::ShortConvolution) {
                throw std::logic_error("packed layer program disagrees with layer binding");
            }
            run_convolution_layer(reference, *convolution, rows,
                                  static_cast<int>(layer_index), ragged_requests);
          });
        if (!reference.options().fused_residuals) {
            launch_residual_add(workspace_.hidden.data(), workspace_.residual.data(),
                                rows * workspace_.program_.hidden,
                                workspace_.stream.get());
        }
        if (layer_program_.kind(layer_index) != PackedLayerKind::Mamba2) {
            run_mlp_layer(reference, common_layer, rows, batch_models,
                          static_cast<int>(layer_index));
        }
        if (std::binary_search(reference.program().norm_after_layers.begin(),
                               reference.program().norm_after_layers.end(),
                               static_cast<int>(layer_index))) {
            launch_rmsnorm(workspace_.hidden.data(), reference.final_norm(),
                           workspace_.hidden.data(), rows,
                           workspace_.program_.hidden,
                           reference.program().final_norm.epsilon,
                           workspace_.stream.get());
        }
    }
}

}
