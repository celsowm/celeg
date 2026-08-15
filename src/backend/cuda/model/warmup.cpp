#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/weight_layout.hpp"

#include <stdexcept>
namespace celeg {

void CudaCompiledModel::warmup_decode_gemms() {
    workspace_.hidden_.zero_async(stream_.get());
    workspace_.normed_.zero_async(stream_.get());
    workspace_.op_output_.zero_async(stream_.get());
    workspace_.qkv_output_.zero_async(stream_.get());
    workspace_.conv_projected_.zero_async(stream_.get());
    workspace_.gate_up_.zero_async(stream_.get());
    workspace_.activated_.zero_async(stream_.get());
    workspace_.logits_.zero_async(stream_.get());

    const AttentionLayer* attention_layer = nullptr;
    const ConvolutionLayer* convolution_layer = nullptr;
    for (const Layer& layer : resources_.layers_) {
        if (!attention_layer) attention_layer = as_attention(layer);
        if (!convolution_layer) convolution_layer = as_convolution(layer);
    }
    if (resources_.layers_.empty()) {
        throw std::runtime_error("compiled layer map is empty");
    }
    if (!attention_layer) {
        launch_rmsnorm(workspace_.hidden_.data(),
                       common(resources_.layers_.front()).operator_norm,
                       workspace_.normed_.data(), 1, resources_.program_.hidden,
                       resources_.program_.layers.front().operator_norm.epsilon,
                       stream_.get());
        enqueue_decode_non_attention_mixer(resources_.layers_.front(), 0);
        linear(workspace_.normed_.data(), *logits_weight(), workspace_.logits_.data(),
               1, resources_.dims_.vocab_size, resources_.program_.hidden);
        CELEG_CUDA(cudaStreamSynchronize(stream_.get()));
        return;
    }
    const LayerCommon& first_common = common(resources_.layers_.front());

    const AttentionSpec& layout = attention_layer->layout;
    if (layout.uses_latent_state()) {
        linear(workspace_.normed_.data(), *attention_layer->latent_query,
               workspace_.latent_query_content_.data(), 1,
               layout.latent_query_content_width(), resources_.program_.hidden);
        if (layout.latent_query_rope_width() != 0) {
            linear(workspace_.normed_.data(), *attention_layer->latent_query_rope,
                   workspace_.latent_query_rope_.data(), 1,
                   layout.latent_query_rope_width(), resources_.program_.hidden);
        }
        linear(workspace_.op_output_.data(), *attention_layer->out,
               workspace_.hidden_.data(), 1, resources_.program_.hidden,
               layout.latent_query_content_width(),
               resources_.options_.fused_residuals ? 1.0f : 0.0f);
    } else {
    const int query_projection_width = attention_layer->query->rows;
    linear(workspace_.normed_.data(), *attention_layer->query, workspace_.qkv_output_.data(),
           1, query_projection_width, resources_.program_.hidden);
    if (attention_layer->key && attention_layer->value) {
        linear(workspace_.normed_.data(), *attention_layer->key,
               workspace_.qkv_output_.data() + layout.query_width(),
               1, layout.key_value_width(), resources_.program_.hidden);
        linear(workspace_.normed_.data(), *attention_layer->value,
               workspace_.qkv_output_.data() + layout.query_width() + layout.key_value_width(),
               1, layout.key_value_width(), resources_.program_.hidden);
    }
    if (const DenseFfnWeights* dense = as_dense_ffn(first_common.feed_forward);
        dense && dense->w13 && dense->w2) {
        const int intermediate = dense->w2->cols;
        linear(workspace_.normed_.data(), *dense->w13, workspace_.gate_up_.data(),
               1, 2 * intermediate, resources_.program_.hidden);
    }

    linear(workspace_.op_output_.data(), *attention_layer->out, workspace_.hidden_.data(),
           1, resources_.program_.hidden, layout.query_width(),
           resources_.options_.fused_residuals ? 1.0f : 0.0f);
    }
    if (convolution_layer) {
        linear(workspace_.normed_.data(), *convolution_layer->conv_in, workspace_.conv_projected_.data(),
               1, 3 * resources_.program_.hidden, resources_.program_.hidden);
    }
    if (const DenseFfnWeights* dense = as_dense_ffn(first_common.feed_forward);
        dense && dense->w2) {
        linear(workspace_.activated_.data(), *dense->w2, workspace_.hidden_.data(),
               1, resources_.program_.hidden, dense->w2->cols,
               resources_.options_.fused_residuals ? 1.0f : 0.0f);
    }
    linear(workspace_.normed_.data(), *logits_weight(), workspace_.logits_.data(),
           1, resources_.dims_.vocab_size, resources_.program_.hidden);
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));
}

void CudaCompiledModel::warmup_prefill_attention_gemm() {
    constexpr int kRows = 2;
    const AttentionLayer* attention = nullptr;
    for (const Layer& layer : resources_.layers_) {
        if ((attention = as_attention(layer)) != nullptr) break;
    }
    if (!attention) return;
    const AttentionSpec& layout = attention->layout;
    if (layout.uses_latent_state()) return;
    DeviceBuffer<__nv_bfloat16> q(static_cast<size_t>(kRows) * layout.query_width());
    DeviceBuffer<__nv_bfloat16> k(static_cast<size_t>(kRows) * layout.key_value_width());
    DeviceBuffer<__nv_bfloat16> v(static_cast<size_t>(kRows) * layout.key_value_width());
    DeviceBuffer<__nv_bfloat16> out(static_cast<size_t>(kRows) * layout.query_width());
    DeviceBuffer<float> scores(
        static_cast<size_t>(layout.query_heads) * kRows * kRows);
    DeviceBuffer<__nv_bfloat16> probs(
        static_cast<size_t>(layout.query_heads) * kRows * kRows);
    q.zero_async(stream_.get());
    k.zero_async(stream_.get());
    v.zero_async(stream_.get());
    launch_gqa_prefill_gemm({
        .cublas = gemm_->cublas().get(),
        .query = q.data(),
        .kv = {.keys = k.data(), .values = v.data()},
        .out = out.data(),
        .scores_scratch = scores.data(),
        .probs_scratch = probs.data(),
        .geometry = {.q_heads = layout.query_heads,
                     .kv_heads = layout.key_value_heads,
                     .head_dim = layout.head_dim,
                     .sliding_window = layout.sliding_window_size()},
        .strides = {.q_width = layout.query_width(),
                    .kv_width = layout.key_value_width(),
                    .out_width = layout.query_width()},
        .rows = kRows,
        .stream = stream_.get()});
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));
}

}

