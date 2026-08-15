#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/paged_kv.hpp"
#include "celeg/backend/cuda/phase_profile.hpp"
#include "celeg/backend/cuda/weight_layout.hpp"
#include "celeg/backend/cuda/moe.hpp"
#include "celeg/backend/cuda/kernels/rope_pairing.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

// Batched CUDA prefill owns one public execution seam: prefill_batched().
// Everything below that seam is a private implementation collaborator rather
// than CudaCompiledModel API. This keeps mixer growth out of the model surface
// and makes latent-attention routing structurally exclusive.

#include "prefill_batched/non_attention.cpp"
#include "prefill_batched/attention.cpp"

namespace celeg::prefill_detail {

void run_mixer(
    CudaCompiledModel& model,
    Layer& layer,
    LayerCommon& common_layer,
    const CompiledLayerProgram& semantics,
    int rows) {
    visit_layer(
        layer,
        [&](GatedDeltaNetLayer* gated_delta) {
            run_gated_delta(model, *gated_delta, semantics, rows);
        },
        [&](Mamba2Layer* mamba) {
            run_mamba2(model, *mamba, semantics, rows);
        },
        [&](MlpOnlyLayer* mlp) {
            run_mlp_only(model, *mlp, rows);
        },
        [&](AttentionLayer* attention) {
            run_attention(model, *attention, common_layer, semantics, rows);
        },
        [&](ConvolutionLayer* convolution) {
            run_convolution(model, *convolution, rows);
        });
}

void run_layer(
    CudaCompiledModel& model,
    Layer& layer,
    int layer_index,
    int rows) {
    auto& workspace = model.workspace_;
    auto& prof = prefill_phase_profile();
    LayerCommon& common_layer = common(layer);
    const CompiledLayerProgram& semantics =
        model.resources_.program_.layers.at(static_cast<size_t>(layer_index));
    const int hidden = model.resources_.program_.hidden;

    if (!model.resources_.options_.fused_residuals ||
        common_layer.post_attention_norm) {
        CELEG_CUDA(cudaMemcpyAsync(
            workspace.prefill_residual_.data(),
            workspace.prefill_hidden_.data(),
            workspace.prefill_hidden_.bytes(),
            cudaMemcpyDeviceToDevice, model.stream_.get()));
    }

    prof.begin(model.stream_.get());
    launch_rmsnorm(
        workspace.prefill_hidden_.data(), common_layer.operator_norm,
        workspace.prefill_normed_.data(),
        rows, hidden, semantics.operator_norm.epsilon, model.stream_.get());
    prof.end(PrefillPhase::Norm, model.stream_.get());

    run_mixer(model, layer, common_layer, semantics, rows);

    if (common_layer.post_attention_norm) {
        launch_rmsnorm(
            workspace.prefill_hidden_.data(), common_layer.post_attention_norm,
            workspace.prefill_hidden_.data(),
            rows, hidden, semantics.post_attention_norm.epsilon,
            model.stream_.get());
    }

    if (!model.resources_.options_.fused_residuals ||
        common_layer.post_attention_norm ||
        std::holds_alternative<std::monostate>(semantics.feed_forward)) {
        prof.begin(model.stream_.get());
        launch_residual_add(
            workspace.prefill_hidden_.data(),
            workspace.prefill_residual_.data(),
            rows * hidden, model.stream_.get());
        prof.end(PrefillPhase::Other, model.stream_.get());
    }

    prof.begin(model.stream_.get());
    if (!std::holds_alternative<std::monostate>(semantics.feed_forward)) {
        model.run_mlp_prefill(common_layer, rows, layer_index);
    }
    if (std::binary_search(
            model.resources_.program_.norm_after_layers.begin(),
            model.resources_.program_.norm_after_layers.end(),
            layer_index)) {
        launch_rmsnorm(
            workspace.prefill_hidden_.data(), model.resources_.final_norm_,
            workspace.prefill_hidden_.data(),
            rows, hidden, model.resources_.program_.final_norm.epsilon,
            model.stream_.get());
    }
    prof.end(PrefillPhase::Mlp, model.stream_.get());
}

void run_layers(CudaCompiledModel& model, int rows) {
    int layer_index = 0;
    for (Layer& layer : model.resources_.layers_) {
        run_layer(model, layer, layer_index, rows);
        ++layer_index;
    }
}

void run_logits(CudaCompiledModel& model, int rows) {
    auto& workspace = model.workspace_;
    const int hidden = model.resources_.program_.hidden;
    const __nv_bfloat16* last_hidden =
        workspace.prefill_hidden_.data() +
        static_cast<size_t>(rows - 1) * hidden;

    launch_rmsnorm(
        last_hidden, model.resources_.final_norm_, workspace.normed_.data(),
        1, hidden, model.resources_.program_.final_norm.epsilon,
        model.stream_.get());
    model.linear(
        workspace.normed_.data(), *model.logits_weight(), workspace.logits_.data(),
        1, model.resources_.dims_.vocab_size, hidden);

    if (model.resources_.program_.logits_divisor != 1.0f ||
        model.resources_.program_.logits_multiplier != 1.0f) {
        launch_scale(
            workspace.logits_.data(), model.resources_.dims_.vocab_size,
            model.resources_.program_.logits_multiplier /
                model.resources_.program_.logits_divisor,
            model.stream_.get());
        if (model.resources_.program_.final_logit_softcap > 0.0f) {
            launch_tanh_softcap(
                workspace.logits_.data(), model.resources_.dims_.vocab_size,
                model.resources_.program_.final_logit_softcap,
                model.stream_.get());
        }
    }
}

} // namespace celeg::prefill_detail

namespace celeg {

void CudaCompiledModel::prefill_batched(const std::vector<int32_t>& tokens) {
    validate_token_ids(tokens);
    reset();

    const int rows = static_cast<int>(tokens.size());
    allocate_prefill_workspace(rows);
    auto& prof = prefill_phase_profile();
    prof.count_step();

    CELEG_CUDA(cudaMemcpyAsync(
        workspace_.prefill_tokens_.data(), tokens.data(),
        tokens.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream_.get()));

    prof.begin(stream_.get());
    launch_mark_seen_batch(
        workspace_.prefill_tokens_.data(), rows, sampling_.seen_tokens.data(),
        resources_.dims_.vocab_size, stream_.get());
    resources_.weight_layout_->embed_batch(
        workspace_.prefill_tokens_.data(), rows,
        workspace_.prefill_hidden_.data(),
        resources_.program_.hidden, stream_.get());
    launch_scale(
        workspace_.prefill_hidden_.data(),
        rows * resources_.program_.hidden,
        resources_.program_.embedding_transform.multiplier,
        stream_.get());
    if (resources_.program_.embedding_transform.post_norm) {
        launch_rmsnorm(
            workspace_.prefill_hidden_.data(), resources_.embedding_norm_,
            workspace_.prefill_hidden_.data(),
            rows, resources_.program_.hidden,
            resources_.program_.embedding_transform.post_norm->epsilon,
            stream_.get());
    }
    initialize_per_layer_input_batch(workspace_.prefill_tokens_.data(), rows);
    prof.end(PrefillPhase::Embed, stream_.get());

    prefill_detail::run_layers(*this, rows);

    prof.begin(stream_.get());
    prefill_detail::run_logits(*this, rows);
    prof.end(PrefillPhase::Logits, stream_.get());

    session_.position_ = rows;
    CELEG_CUDA(cudaMemcpyAsync(
        position_device_.data(), &session_.position_,
        sizeof(session_.position_), cudaMemcpyHostToDevice, stream_.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));

    prof.report();
    release_prefill_workspace();
    session_.phase_ = SessionPhase::Ready;
}

} // namespace celeg
