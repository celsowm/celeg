#include "lfm/detail/model/impl.hpp"
#include "lfm/backend/cuda/kernels/kernels.cuh"
#include "lfm/model/weights/layout.hpp"

#include <stdexcept>
namespace lfm {

void Model::Impl::warmup_decode_gemms() {
    hidden_.zero_async(stream_.get());
    normed_.zero_async(stream_.get());
    op_output_.zero_async(stream_.get());
    qkv_output_.zero_async(stream_.get());
    conv_projected_.zero_async(stream_.get());
    gate_up_.zero_async(stream_.get());
    activated_.zero_async(stream_.get());
    logits_.zero_async(stream_.get());

    const AttentionLayer* attention_layer = nullptr;
    const ConvolutionLayer* convolution_layer = nullptr;
    for (const Layer& layer : layers_) {
        if (!attention_layer) attention_layer = as_attention(layer);
        if (!convolution_layer) convolution_layer = as_convolution(layer);
    }
    if (!attention_layer || layers_.empty()) {
        throw std::runtime_error("compiled attention layer map is incomplete");
    }
    const LayerCommon& first_common = common(layers_.front());

    if (options_.fused_projections) {
        linear(normed_.data(), *attention_layer->qkv, qkv_output_.data(),
               1, shape_.qkv_width, shape_.hidden);
        linear(normed_.data(), *as_dense_ffn(first_common.feed_forward)->w13, gate_up_.data(),
               1, 2 * shape_.intermediate, shape_.hidden);
    } else {
        const LinearWeight q_weight =
            slice_rows(*attention_layer->qkv, 0, shape_.q_width);
        const LinearWeight k_weight = slice_rows(
            *attention_layer->qkv, shape_.q_width, shape_.kv_width);
        const LinearWeight v_weight = slice_rows(
            *attention_layer->qkv, shape_.q_width + shape_.kv_width,
            shape_.kv_width);
        linear(normed_.data(), q_weight, qkv_output_.data(),
               1, shape_.q_width, shape_.hidden);
        linear(normed_.data(), k_weight, qkv_output_.data() + shape_.q_width,
               1, shape_.kv_width, shape_.hidden);
        linear(normed_.data(), v_weight,
               qkv_output_.data() + shape_.q_width + shape_.kv_width,
               1, shape_.kv_width, shape_.hidden);

        const LinearWeight w1 =
            slice_rows(*as_dense_ffn(first_common.feed_forward)->w13, 0, shape_.intermediate);
        const LinearWeight w3 = slice_rows(
            *as_dense_ffn(first_common.feed_forward)->w13, shape_.intermediate,
            shape_.intermediate);
        linear(normed_.data(), w1, gate_up_.data(),
               1, shape_.intermediate, shape_.hidden);
        linear(normed_.data(), w3, gate_up_.data() + shape_.intermediate,
               1, shape_.intermediate, shape_.hidden);
    }

    linear(op_output_.data(), *attention_layer->out, hidden_.data(),
           1, shape_.hidden, shape_.hidden,
           options_.fused_residuals ? 1.0f : 0.0f);
    if (convolution_layer) {
        linear(normed_.data(), *convolution_layer->conv_in, conv_projected_.data(),
               1, 3 * shape_.hidden, shape_.hidden);
    }
    linear(activated_.data(), *as_dense_ffn(first_common.feed_forward)->w2, hidden_.data(),
           1, shape_.hidden, shape_.intermediate,
           options_.fused_residuals ? 1.0f : 0.0f);
    linear(normed_.data(), *logits_weight(), logits_.data(),
           1, shape_.vocab_size, shape_.hidden);
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
}

void Model::Impl::warmup_prefill_attention_gemm() {
    constexpr int kRows = 2;
    DeviceBuffer<__nv_bfloat16> q(static_cast<size_t>(kRows) * shape_.q_width);
    DeviceBuffer<__nv_bfloat16> k(static_cast<size_t>(kRows) * shape_.kv_width);
    DeviceBuffer<__nv_bfloat16> v(static_cast<size_t>(kRows) * shape_.kv_width);
    DeviceBuffer<__nv_bfloat16> out(static_cast<size_t>(kRows) * shape_.q_width);
    DeviceBuffer<float> scores(
        static_cast<size_t>(shape_.num_attention_heads) * kRows * kRows);
    DeviceBuffer<__nv_bfloat16> probs(
        static_cast<size_t>(shape_.num_attention_heads) * kRows * kRows);
    q.zero_async(stream_.get());
    k.zero_async(stream_.get());
    v.zero_async(stream_.get());
    launch_gqa_prefill_gemm(
        gemm_->cublas().get(), q.data(), k.data(), v.data(), out.data(),
        scores.data(), probs.data(), kRows, shape_.num_attention_heads,
        shape_.num_key_value_heads, shape_.head_dim, shape_.q_width,
        shape_.kv_width, shape_.q_width, stream_.get());
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
}

} // namespace lfm

