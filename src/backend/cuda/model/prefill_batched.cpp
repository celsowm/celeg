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

// Bulk batched-prefill orchestration.
//
// Execution details live in private source fragments by responsibility:
//   non_attention.cpp  recurrent/convolution/MLP-only mixers
//   attention.cpp      attention capability, KV access, regular + latent attention
//
// This translation unit owns only layer orchestration, final logits, and the
// session transition for a completed batched prefill.

#include "prefill_batched/non_attention.cpp"
#include "prefill_batched/attention.cpp"

namespace celeg {

void CudaCompiledModel::run_prefill_mixer(Layer& layer, LayerCommon& common_layer,
                                          const CompiledLayerProgram& semantics,
                                          int rows) {
    visit_layer(layer,
      [&](GatedDeltaNetLayer* gated_delta) {
        run_prefill_gated_delta(*gated_delta, semantics, rows);
      },
      [&](Mamba2Layer* mamba) {
        run_prefill_mamba2(*mamba, semantics, rows);
      },
      [&](MlpOnlyLayer* mlp) {
        run_prefill_mlp_only(*mlp, rows);
      },
      [&](AttentionLayer* attention) {
        run_prefill_attention(*attention, common_layer, semantics, rows);
      },
      [&](ConvolutionLayer* convolution) {
        run_prefill_convolution(*convolution, rows);
      });
}

void CudaCompiledModel::run_prefill_layer(Layer& layer, int layer_index, int rows) {
    auto& prof = prefill_phase_profile();
    LayerCommon& common_layer = common(layer);
    const CompiledLayerProgram& semantics =
        resources_.program_.layers.at(static_cast<size_t>(layer_index));
    if (!resources_.options_.fused_residuals || common_layer.post_attention_norm) {
        CELEG_CUDA(cudaMemcpyAsync(
            workspace_.prefill_residual_.data(), workspace_.prefill_hidden_.data(),
            workspace_.prefill_hidden_.bytes(), cudaMemcpyDeviceToDevice,
            stream_.get()));
    }
    prof.begin(stream_.get());
    launch_rmsnorm(workspace_.prefill_hidden_.data(), common_layer.operator_norm,
                   workspace_.prefill_normed_.data(), rows, resources_.program_.hidden,
                   semantics.operator_norm.epsilon, stream_.get());
    prof.end(PrefillPhase::Norm, stream_.get());

    run_prefill_mixer(layer, common_layer, semantics, rows);

    if (common_layer.post_attention_norm) {
        launch_rmsnorm(workspace_.prefill_hidden_.data(), common_layer.post_attention_norm,
                       workspace_.prefill_hidden_.data(), rows, resources_.program_.hidden,
                       semantics.post_attention_norm.epsilon, stream_.get());
    }
    if (!resources_.options_.fused_residuals || common_layer.post_attention_norm ||
        !semantics.execute_feed_forward) {
        prof.begin(stream_.get());
        launch_residual_add(workspace_.prefill_hidden_.data(), workspace_.prefill_residual_.data(),
                            rows * resources_.program_.hidden, stream_.get());
        prof.end(PrefillPhase::Other, stream_.get());
    }
    prof.begin(stream_.get());
    if (semantics.execute_feed_forward) run_mlp_prefill(common_layer, rows, layer_index);
    if (std::binary_search(resources_.program_.norm_after_layers.begin(),
                           resources_.program_.norm_after_layers.end(), layer_index)) {
        launch_rmsnorm(workspace_.prefill_hidden_.data(), resources_.final_norm_,
                       workspace_.prefill_hidden_.data(), rows, resources_.program_.hidden,
                       resources_.program_.final_norm.epsilon, stream_.get());
    }
    prof.end(PrefillPhase::Mlp, stream_.get());
}

void CudaCompiledModel::run_prefill_layers(int rows) {
    int layer_index = 0;
    for (Layer& layer : resources_.layers_) {
        run_prefill_layer(layer, layer_index, rows);
        ++layer_index;
    }
}

void CudaCompiledModel::run_prefill_logits(int rows) {
    const __nv_bfloat16* last_hidden = workspace_.prefill_hidden_.data() +
        static_cast<size_t>(rows - 1) * resources_.program_.hidden;
    launch_rmsnorm(last_hidden, resources_.final_norm_, workspace_.normed_.data(),
                   1, resources_.program_.hidden, resources_.program_.final_norm.epsilon,
                   stream_.get());
    linear(workspace_.normed_.data(), *logits_weight(), workspace_.logits_.data(),
           1, resources_.dims_.vocab_size, resources_.program_.hidden);
    if (resources_.program_.logits_divisor != 1.0f ||
        resources_.program_.logits_multiplier != 1.0f) {
        launch_scale(workspace_.logits_.data(), resources_.dims_.vocab_size,
                     resources_.program_.logits_multiplier /
                         resources_.program_.logits_divisor, stream_.get());
        if (resources_.program_.final_logit_softcap > 0.0f) {
            launch_tanh_softcap(workspace_.logits_.data(), resources_.dims_.vocab_size,
                                resources_.program_.final_logit_softcap, stream_.get());
        }
    }
}

void CudaCompiledModel::prefill_batched(const std::vector<int32_t>& tokens) {
    validate_token_ids(tokens);
    reset();
    const int rows = static_cast<int>(tokens.size());
    allocate_prefill_workspace(rows);
    auto& prof = prefill_phase_profile();
    prof.count_step();

    CELEG_CUDA(cudaMemcpyAsync(workspace_.prefill_tokens_.data(), tokens.data(),
                             tokens.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    prof.begin(stream_.get());
    launch_mark_seen_batch(workspace_.prefill_tokens_.data(), rows, sampling_.seen_tokens.data(),
                           resources_.dims_.vocab_size, stream_.get());
    resources_.weight_layout_->embed_batch(
        workspace_.prefill_tokens_.data(), rows, workspace_.prefill_hidden_.data(),
        resources_.program_.hidden, stream_.get());
    launch_scale(workspace_.prefill_hidden_.data(), rows * resources_.program_.hidden,
                 resources_.program_.embedding_transform.multiplier, stream_.get());
    if (resources_.program_.embedding_transform.post_norm) {
        launch_rmsnorm(workspace_.prefill_hidden_.data(), resources_.embedding_norm_,
                       workspace_.prefill_hidden_.data(), rows, resources_.program_.hidden,
                       resources_.program_.embedding_transform.post_norm->epsilon,
                       stream_.get());
    }
    initialize_per_layer_input_batch(workspace_.prefill_tokens_.data(), rows);
    prof.end(PrefillPhase::Embed, stream_.get());

    run_prefill_layers(rows);

    prof.begin(stream_.get());
    run_prefill_logits(rows);
    prof.end(PrefillPhase::Logits, stream_.get());

    session_.position_ = rows;
    CELEG_CUDA(cudaMemcpyAsync(position_device_.data(), &session_.position_,
                             sizeof(session_.position_), cudaMemcpyHostToDevice,
                             stream_.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));
    prof.report();
    release_prefill_workspace();
    session_.phase_ = SessionPhase::Ready;
}

} // namespace celeg
