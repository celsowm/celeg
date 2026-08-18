#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/moe.hpp"

namespace celeg {

void CudaCompiledModel::run_mlp_decode(const LayerCommon& common_layer, int layer) {
    const CompiledLayerProgram& semantics = resources_.program_.layers.at(
        static_cast<size_t>(layer));
    if (const MoeFfnWeights* moe = as_moe_ffn(common_layer.feed_forward)) {
        (void)moe;
        run_mlp_moe_decode(common_layer, layer);
    } else {
        launch_rmsnorm(workspace_.hidden_.data(), common_layer.feed_forward_norm_before, workspace_.normed_.data(),
                       1, resources_.program_.hidden,
                       semantics.feed_forward_norm.before->epsilon,
                       stream_.get());
        const auto& dense_semantics =
            std::get<CompiledDenseFeedForwardProgram>(semantics.feed_forward);
        const int intermediate = dense_semantics.intermediate_size;
        if (resources_.options_.fused_projections) {
            linear(workspace_.normed_.data(), *as_dense_ffn(common_layer.feed_forward)->w13, workspace_.gate_up_.data(),
                   1, 2 * intermediate, resources_.program_.hidden);
        } else {
            const LinearWeight w1 =
                slice_rows(*as_dense_ffn(common_layer.feed_forward)->w13, 0, intermediate);
            const LinearWeight w3 = slice_rows(
                *as_dense_ffn(common_layer.feed_forward)->w13, intermediate, intermediate);
            linear(workspace_.normed_.data(), w1, workspace_.gate_up_.data(),
                   1, intermediate, resources_.program_.hidden);
            linear(workspace_.normed_.data(), w3, workspace_.gate_up_.data() + intermediate,
                   1, intermediate, resources_.program_.hidden);
        }
        const bool gelu_tanh = dense_semantics.activation == ActivationKind::GeluTanh;
        if (gelu_tanh) {
            launch_gated_gelu_tanh(workspace_.gate_up_.data(), workspace_.activated_.data(),
                                   intermediate, stream_.get());
        } else {
            launch_swiglu_fused(workspace_.gate_up_.data(), workspace_.activated_.data(),
                                intermediate, stream_.get());
        }
        const bool split_output = common_layer.feed_forward_norm_after != nullptr;
        if (resources_.options_.fused_residuals && !split_output) {
            linear(workspace_.activated_.data(), *as_dense_ffn(common_layer.feed_forward)->w2, workspace_.hidden_.data(),
                   1, resources_.program_.hidden, intermediate, 1.0f);
        } else {
            linear(workspace_.activated_.data(), *as_dense_ffn(common_layer.feed_forward)->w2, workspace_.mlp_output_.data(),
                   1, resources_.program_.hidden, intermediate);
            if (split_output) {
                launch_rmsnorm(workspace_.mlp_output_.data(), common_layer.feed_forward_norm_after,
                               workspace_.mlp_output_.data(), 1, resources_.program_.hidden,
                               semantics.feed_forward_norm.after->epsilon,
                               stream_.get());
            }
            launch_scale(workspace_.mlp_output_.data(), resources_.program_.hidden,
                         semantics.residual.multiplier,
                         stream_.get());
            launch_residual_add(workspace_.hidden_.data(), workspace_.mlp_output_.data(),
                                resources_.program_.hidden, stream_.get());
        }
    }
    run_per_layer_input_decode(common_layer, layer);
}

void CudaCompiledModel::run_mlp_prefill(const LayerCommon& common_layer, int rows,
                                     int layer) {
    const CompiledLayerProgram& semantics = resources_.program_.layers.at(
        static_cast<size_t>(layer));
    if (const MoeFfnWeights* moe = as_moe_ffn(common_layer.feed_forward)) {
        (void)moe;
        run_mlp_moe_prefill(common_layer, rows, layer);
    } else {
        const auto& dense_semantics =
            std::get<CompiledDenseFeedForwardProgram>(semantics.feed_forward);
        const int intermediate = dense_semantics.intermediate_size;
        const size_t matrix_elements = static_cast<size_t>(rows) * intermediate;
        launch_rmsnorm(workspace_.prefill_hidden_.data(), common_layer.feed_forward_norm_before,
                       workspace_.prefill_normed_.data(), rows, resources_.program_.hidden,
                       semantics.feed_forward_norm.before->epsilon,
                       stream_.get());
        if (resources_.options_.fused_projections) {
        linear(workspace_.prefill_normed_.data(), *as_dense_ffn(common_layer.feed_forward)->w13, workspace_.prefill_gate_up_.data(),
               rows, 2 * intermediate, resources_.program_.hidden);
        if (dense_semantics.activation == ActivationKind::GeluTanh) {
            launch_gated_gelu_tanh(workspace_.prefill_gate_up_.data(),
                                   workspace_.prefill_activated_.data(),
                                   static_cast<int>(matrix_elements), stream_.get());
        } else {
            launch_swiglu_interleaved(workspace_.prefill_gate_up_.data(),
                                      workspace_.prefill_activated_.data(), rows,
                                      intermediate, stream_.get());
        }
        } else {
        const LinearWeight w1 =
            slice_rows(*as_dense_ffn(common_layer.feed_forward)->w13, 0, intermediate);
        const LinearWeight w3 = slice_rows(
            *as_dense_ffn(common_layer.feed_forward)->w13, intermediate, intermediate);
        linear(workspace_.prefill_normed_.data(), w1, workspace_.prefill_gate_up_.data(),
               rows, intermediate, resources_.program_.hidden);
        linear(workspace_.prefill_normed_.data(), w3,
               workspace_.prefill_gate_up_.data() + matrix_elements,
               rows, intermediate, resources_.program_.hidden);
        if (dense_semantics.activation == ActivationKind::GeluTanh) {
            launch_gated_gelu_tanh(workspace_.prefill_gate_up_.data(), workspace_.prefill_activated_.data(),
                                   static_cast<int>(matrix_elements), stream_.get());
        } else {
            launch_swiglu_fused(workspace_.prefill_gate_up_.data(), workspace_.prefill_activated_.data(),
                                static_cast<int>(matrix_elements), stream_.get());
        }
        }
        const bool split_output = common_layer.feed_forward_norm_after != nullptr;
        if (resources_.options_.fused_residuals && !split_output) {
        linear(workspace_.prefill_activated_.data(), *as_dense_ffn(common_layer.feed_forward)->w2, workspace_.prefill_hidden_.data(),
               rows, resources_.program_.hidden, intermediate, 1.0f);
        } else {
        linear(workspace_.prefill_activated_.data(), *as_dense_ffn(common_layer.feed_forward)->w2, workspace_.prefill_mlp_output_.data(),
                rows, resources_.program_.hidden, intermediate);
        if (split_output) {
            launch_rmsnorm(workspace_.prefill_mlp_output_.data(), common_layer.feed_forward_norm_after,
                           workspace_.prefill_mlp_output_.data(), rows, resources_.program_.hidden,
                           semantics.feed_forward_norm.after->epsilon,
                           stream_.get());
        }
        launch_scale(workspace_.prefill_mlp_output_.data(), rows * resources_.program_.hidden,
                     semantics.residual.multiplier, stream_.get());
        launch_residual_add(workspace_.prefill_hidden_.data(), workspace_.prefill_mlp_output_.data(),
                            rows * resources_.program_.hidden, stream_.get());
        }
    }
    run_per_layer_input_prefill(common_layer, rows, layer);
}

}
