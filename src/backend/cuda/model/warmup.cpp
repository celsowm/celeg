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
    if (!attention_layer || resources_.layers_.empty()) {
        throw std::runtime_error("compiled attention layer map is incomplete");
    }
    const LayerCommon& first_common = common(resources_.layers_.front());

    if (resources_.options_.fused_projections) {
        linear(workspace_.normed_.data(), *attention_layer->qkv, workspace_.qkv_output_.data(),
               1, resources_.shape_.qkv_width, resources_.shape_.hidden);
        linear(workspace_.normed_.data(), *as_dense_ffn(first_common.feed_forward)->w13, workspace_.gate_up_.data(),
               1, 2 * resources_.shape_.intermediate, resources_.shape_.hidden);
    } else {
        const LinearWeight q_weight =
            slice_rows(*attention_layer->qkv, 0, resources_.shape_.q_width);
        const LinearWeight k_weight = slice_rows(
            *attention_layer->qkv, resources_.shape_.q_width, resources_.shape_.kv_width);
        const LinearWeight v_weight = slice_rows(
            *attention_layer->qkv, resources_.shape_.q_width + resources_.shape_.kv_width,
            resources_.shape_.kv_width);
        linear(workspace_.normed_.data(), q_weight, workspace_.qkv_output_.data(),
               1, resources_.shape_.q_width, resources_.shape_.hidden);
        linear(workspace_.normed_.data(), k_weight, workspace_.qkv_output_.data() + resources_.shape_.q_width,
               1, resources_.shape_.kv_width, resources_.shape_.hidden);
        linear(workspace_.normed_.data(), v_weight,
               workspace_.qkv_output_.data() + resources_.shape_.q_width + resources_.shape_.kv_width,
               1, resources_.shape_.kv_width, resources_.shape_.hidden);

        const LinearWeight w1 =
            slice_rows(*as_dense_ffn(first_common.feed_forward)->w13, 0, resources_.shape_.intermediate);
        const LinearWeight w3 = slice_rows(
            *as_dense_ffn(first_common.feed_forward)->w13, resources_.shape_.intermediate,
            resources_.shape_.intermediate);
        linear(workspace_.normed_.data(), w1, workspace_.gate_up_.data(),
               1, resources_.shape_.intermediate, resources_.shape_.hidden);
        linear(workspace_.normed_.data(), w3, workspace_.gate_up_.data() + resources_.shape_.intermediate,
               1, resources_.shape_.intermediate, resources_.shape_.hidden);
    }

    linear(workspace_.op_output_.data(), *attention_layer->out, workspace_.hidden_.data(),
           1, resources_.shape_.hidden, resources_.shape_.hidden,
           resources_.options_.fused_residuals ? 1.0f : 0.0f);
    if (convolution_layer) {
        linear(workspace_.normed_.data(), *convolution_layer->conv_in, workspace_.conv_projected_.data(),
               1, 3 * resources_.shape_.hidden, resources_.shape_.hidden);
    }
    linear(workspace_.activated_.data(), *as_dense_ffn(first_common.feed_forward)->w2, workspace_.hidden_.data(),
           1, resources_.shape_.hidden, resources_.shape_.intermediate,
           resources_.options_.fused_residuals ? 1.0f : 0.0f);
    linear(workspace_.normed_.data(), *logits_weight(), workspace_.logits_.data(),
           1, resources_.shape_.vocab_size, resources_.shape_.hidden);
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));
}

void CudaCompiledModel::warmup_prefill_attention_gemm() {
    constexpr int kRows = 2;
    DeviceBuffer<__nv_bfloat16> q(static_cast<size_t>(kRows) * resources_.shape_.q_width);
    DeviceBuffer<__nv_bfloat16> k(static_cast<size_t>(kRows) * resources_.shape_.kv_width);
    DeviceBuffer<__nv_bfloat16> v(static_cast<size_t>(kRows) * resources_.shape_.kv_width);
    DeviceBuffer<__nv_bfloat16> out(static_cast<size_t>(kRows) * resources_.shape_.q_width);
    DeviceBuffer<float> scores(
        static_cast<size_t>(resources_.shape_.num_attention_heads) * kRows * kRows);
    DeviceBuffer<__nv_bfloat16> probs(
        static_cast<size_t>(resources_.shape_.num_attention_heads) * kRows * kRows);
    q.zero_async(stream_.get());
    k.zero_async(stream_.get());
    v.zero_async(stream_.get());
    launch_gqa_prefill_gemm(
        gemm_->cublas().get(), q.data(), k.data(), v.data(), out.data(),
        scores.data(), probs.data(), kRows, resources_.shape_.num_attention_heads,
        resources_.shape_.num_key_value_heads, resources_.shape_.head_dim, resources_.shape_.q_width,
        resources_.shape_.kv_width, resources_.shape_.q_width, stream_.get());
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));
}

} // namespace celeg

