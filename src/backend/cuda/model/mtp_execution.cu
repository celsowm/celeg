#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/attention_norm.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/moe.hpp"

#include <cstring>
#include <stdexcept>

namespace celeg {

namespace {

void copy_async(void* destination, const void* source, size_t bytes,
                cudaStream_t stream) {
    if (bytes == 0) return;
    CELEG_CUDA(cudaMemcpyAsync(destination, source, bytes,
                               cudaMemcpyDeviceToDevice, stream));
}

}

void CudaCompiledModel::run_mtp_forward(int32_t token,
                                        const std::array<int32_t, 3>* rope_position) {
    if (!resources_.mtp_.available()) return;
    CELEG_CUDA(cudaMemcpyAsync(workspace_.mtp_token_.data(), &token,
                               sizeof(token), cudaMemcpyHostToDevice,
                               stream_.get()));
    CELEG_CUDA(cudaMemcpyAsync(position_device_.data(), &session_.position_,
                               sizeof(session_.position_), cudaMemcpyHostToDevice,
                               stream_.get()));
    if (rope_position) {
        CELEG_CUDA(cudaMemcpyAsync(mrope_position_device_.data(), rope_position->data(),
                                   sizeof(*rope_position), cudaMemcpyHostToDevice,
                                   stream_.get()));
    }
    run_mtp_forward_device(workspace_.mtp_token_.data());
}

void CudaCompiledModel::run_mtp_forward_device(const int32_t* token_device) {
    if (!resources_.mtp_.available()) return;
    if (resources_.mtp_.layer_count != 1) {
        throw std::runtime_error(
            "MTP currently requires exactly one auxiliary decoder layer");
    }
    const cudaStream_t stream = stream_.get();
    const int hidden = resources_.program_.hidden;
    const int vocab = resources_.dims_.vocab_size;
    const float eps = resources_.program_.final_norm.epsilon;
    CudaMtpResources& mtp = resources_.mtp_;

    copy_async(workspace_.mtp_base_hidden_.data(), workspace_.hidden_.data(),
               workspace_.hidden_.bytes(), stream);
    resources_.weight_layout_->embed_token_device(
        token_device, workspace_.mtp_embedding_.data(), hidden, stream);
    launch_rmsnorm(workspace_.mtp_embedding_.data(), mtp.pre_fc_norm_embedding,
                   workspace_.mtp_embedding_.data(), 1, hidden, eps, stream);
    launch_rmsnorm(workspace_.mtp_base_hidden_.data(), mtp.pre_fc_norm_hidden,
                   workspace_.mtp_hidden_norm_.data(), 1, hidden, eps, stream);
    copy_async(workspace_.mtp_fused_.data(), workspace_.mtp_embedding_.data(),
               static_cast<size_t>(hidden) * sizeof(__nv_bfloat16), stream);
    copy_async(workspace_.mtp_fused_.data() + hidden,
               workspace_.mtp_hidden_norm_.data(),
               static_cast<size_t>(hidden) * sizeof(__nv_bfloat16), stream);
    linear(workspace_.mtp_fused_.data(), *mtp.fc, workspace_.hidden_.data(),
           1, hidden, 2 * hidden);

    Layer& layer = mtp.layers.front();
    AttentionLayer* attention = as_attention(layer);
    if (!attention) throw std::logic_error("MTP decoder layer is not full attention");
    LayerCommon& common_layer = common(layer);
    copy_async(workspace_.residual_.data(), workspace_.hidden_.data(),
               workspace_.hidden_.bytes(), stream);
    launch_rmsnorm(workspace_.hidden_.data(), common_layer.mixer_norm_before,
                   workspace_.normed_.data(), 1, hidden, eps, stream);

    const AttentionSpec& layout = attention->layout;
    __nv_bfloat16* q = workspace_.qkv_output_.data();
    __nv_bfloat16* k = q + layout.query_projection_width();
    __nv_bfloat16* v = k + layout.key_value_width();
    {
        auto native_fanout = native_fanout_scope(workspace_.normed_.data(), 1, hidden);
        linear(workspace_.normed_.data(), *attention->query, q, 1,
               layout.query_projection_width(), hidden);
        linear(workspace_.normed_.data(), *attention->key, k, 1,
               layout.key_value_width(), hidden);
        linear(workspace_.normed_.data(), *attention->value, v, 1,
               layout.key_value_width(), hidden);
    }
    const auto* rope = layout.rope_position();
    if (!rope) throw std::logic_error("MTP attention requires positional encoding");
    const float qk_epsilon = layout.query_norm ? layout.query_norm->epsilon : eps;
    if (layout.has_query_key_norm()) {
        launch_attention_qk_norm(
            layout, q, k, attention->q_norm, attention->k_norm, 1, stream);
    }
    if (const auto* multi = layout.multi_axis_position()) {
        launch_dynamic_mrope_qk_norm_rope(
            q, k, nullptr, nullptr,
            layout.query_heads, layout.key_value_heads, layout.head_dim,
            mrope_position_device_.data(), multi->sections[0],
            multi->sections[1], multi->sections[2], multi->interleaved,
            static_cast<float>(rope->theta),
            static_cast<float>(rope->rotary_fraction), qk_epsilon,
            false, lower_cuda_rope_scaling(*rope), stream);
    } else {
        launch_dynamic_qk_norm_rope_device(
            q, k, nullptr, nullptr,
            layout.query_heads, layout.key_value_heads, layout.head_dim,
            position_device_.data(), static_cast<float>(rope->theta),
            static_cast<float>(rope->rotary_fraction), qk_epsilon,
            false, lower_cuda_rope_scaling(*rope), rope->pairing, stream);
    }
    launch_scale(q, layout.query_width(),
                 cuda_query_prescale(layout), stream);
    AttentionRequest attention_request;
    attention_request.kv_format = resources_.options_.kv_cache_mode;
    attention_request.operation = AttentionOperation::Decode;
    attention_request.layout = AttentionKvLayout::Contiguous;
    attention_request.position_source = AttentionPositionSource::DeviceCounter;
    attention_request.bias = attention->alibi_slopes.data()
        ? AttentionPositionBias::Alibi : AttentionPositionBias::None;
    attention_request.fast_attention = resources_.options_.fast_attention;
    attention_request.head_dim = layout.head_dim;
    const AttentionCapability attention_plan =
        require_attention_capability(attention_request);
    const bool int8_kv = attention_request.kv_format == KvCacheMode::Int8;
    if (int8_kv) {
        launch_store_kv_int8_device(
            k, v, attention->key_cache_int8_ptr(), attention->value_cache_int8_ptr(),
            attention->key_cache_scales_ptr(), attention->value_cache_scales_ptr(),
            position_device_.data(), layout.key_value_heads, layout.head_dim, stream);
    } else {
        launch_store_kv_device(k, v, attention->key_cache_bf16(),
                               attention->value_cache_bf16(), position_device_.data(),
                               layout.key_value_width(), stream);
    }
    const GqaGeometry attention_geometry{
        .q_heads = layout.query_heads,
        .kv_heads = layout.key_value_heads,
        .head_dim = layout.head_dim,
        .sliding_window = layout.sliding_window_size()};
    const AttentionExtent attention_extent{.position = position_device_.data()};
    const Bf16KvView bf16_kv{.keys = attention->key_cache_bf16(),
                             .values = attention->value_cache_bf16()};
    const Int8KvView int8_kv_view{
        .keys = attention->key_cache_int8_ptr(),
        .values = attention->value_cache_int8_ptr(),
        .key_scales = attention->key_cache_scales_ptr(),
        .value_scales = attention->value_cache_scales_ptr()};
    switch (attention_plan.algorithm) {
    case AttentionAlgorithm::Alibi:
        if (int8_kv) {
            launch_gqa_decode_alibi_int8_device({
                .query = q,
                .kv = int8_kv_view,
                .out = workspace_.op_output_.data(),
                .geometry = attention_geometry,
                .extent = attention_extent,
                .alibi_slopes = attention->alibi_slopes.data(),
                .stream = stream});
        } else {
            launch_gqa_decode_alibi_device({
                .query = q,
                .kv = bf16_kv,
                .out = workspace_.op_output_.data(),
                .geometry = attention_geometry,
                .extent = attention_extent,
                .alibi_slopes = attention->alibi_slopes.data(),
                .stream = stream});
        }
        break;
    case AttentionAlgorithm::Online:
        if (int8_kv) {
            launch_gqa_decode_online_int8_device({
                .query = q,
                .kv = int8_kv_view,
                .out = workspace_.op_output_.data(),
                .geometry = attention_geometry,
                .extent = attention_extent,
                .stream = stream});
        } else {
            launch_gqa_decode_online_device({
                .query = q,
                .kv = bf16_kv,
                .out = workspace_.op_output_.data(),
                .geometry = attention_geometry,
                .extent = attention_extent,
                .stream = stream});
        }
        break;
    case AttentionAlgorithm::Strict:
        if (int8_kv) {
            launch_gqa_decode_strict_int8_device({
                .query = q,
                .kv = int8_kv_view,
                .out = workspace_.op_output_.data(),
                .geometry = attention_geometry,
                .extent = attention_extent,
                .stream = stream});
        } else {
            launch_gqa_decode_strict_device({
                .query = q,
                .kv = bf16_kv,
                .out = workspace_.op_output_.data(),
                .geometry = attention_geometry,
                .extent = attention_extent,
                .stream = stream});
        }
        break;
    case AttentionAlgorithm::Segmented:
    case AttentionAlgorithm::Flash:
    case AttentionAlgorithm::Gemm:
        throw UnsupportedAttentionCapability(attention_plan);
    }
    if (layout.output_gate.has_value()) {
        launch_sigmoid_multiply(workspace_.op_output_.data(),
                                q + layout.query_width(),
                                layout.query_width(), stream);
    }
    linear(workspace_.op_output_.data(), *attention->out, workspace_.hidden_.data(),
           1, hidden, layout.query_width());
    launch_residual_add(workspace_.hidden_.data(), workspace_.residual_.data(),
                        hidden, stream);

    run_mlp_decode(common_layer,
                   resources_.shape_.num_hidden_layers );
    launch_rmsnorm(workspace_.hidden_.data(), mtp.norm, workspace_.normed_.data(),
                   1, hidden, eps, stream);
    linear(workspace_.normed_.data(), *mtp.logits, workspace_.mtp_logits_.data(),
           1, vocab, hidden);
    launch_scale(workspace_.mtp_logits_.data(), vocab,
                 resources_.program_.logits_multiplier /
                     resources_.program_.logits_divisor, stream);
    if (resources_.program_.final_logit_softcap > 0.0f) {
        launch_tanh_softcap(workspace_.mtp_logits_.data(), vocab,
                            resources_.program_.final_logit_softcap, stream);
    }
    copy_async(workspace_.hidden_.data(), workspace_.mtp_base_hidden_.data(),
               workspace_.hidden_.bytes(), stream);
    ++session_.metrics_.mtp_forward_tokens;
}

void CudaCompiledModel::finalize_mtp_verification() {
    if (!resources_.mtp_.available()) return;
    const cudaStream_t stream = stream_.get();
    launch_argmax_bf16(
        workspace_.mtp_logits_.data(), sampling_.seen_tokens.data(),
        resources_.dims_.vocab_size,
        session_.generation_.repetition_penalty,
        workspace_.mtp_candidate_.data(), stream);
    launch_argmax_bf16(
        workspace_.logits_.data(), sampling_.seen_tokens.data(),
        resources_.dims_.vocab_size,
        session_.generation_.repetition_penalty,
        workspace_.mtp_target_candidate_.data(), stream);
}

}