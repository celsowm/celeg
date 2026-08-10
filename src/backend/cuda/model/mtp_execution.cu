#include "celeg/detail/model/compiled_model.hpp"
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

} // namespace

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
            "Qwen MTP currently requires exactly one auxiliary decoder layer");
    }
    const cudaStream_t stream = stream_.get();
    const int hidden = resources_.shape_.hidden;
    const int vocab = resources_.shape_.vocab_size;
    const float eps = resources_.shape_.numerical_policy.norm_eps;
    CudaMtpResources& mtp = resources_.mtp_;

    // The target hidden state is an input to the predictor, not a mutable
    // intermediate of it. Preserve it while the auxiliary layer reuses the
    // normal single-token workspace.
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
    launch_rmsnorm(workspace_.hidden_.data(), common_layer.operator_norm,
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
    if (const auto* multi = layout.multi_axis_position()) {
        launch_dynamic_mrope_qk_norm_rope(
            q, k, attention->q_norm, attention->k_norm,
            layout.query_heads, layout.key_value_heads, layout.head_dim,
            mrope_position_device_.data(), multi->sections[0],
            multi->sections[1], multi->sections[2], multi->interleaved,
            static_cast<float>(rope->theta),
            static_cast<float>(rope->rotary_fraction), eps,
            layout.has_query_key_norm(), lower_cuda_rope_scaling(*rope), stream);
    } else {
        launch_dynamic_qk_norm_rope_device(
            q, k, attention->q_norm, attention->k_norm,
            layout.query_heads, layout.key_value_heads, layout.head_dim,
            position_device_.data(), static_cast<float>(rope->theta),
            static_cast<float>(rope->rotary_fraction), eps,
            layout.has_query_key_norm(), lower_cuda_rope_scaling(*rope), stream);
    }
    launch_scale(q, layout.query_width(), layout.query_scale, stream);
    if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
        launch_store_kv_int8_device(
            k, v, attention->key_cache_int8.data(), attention->value_cache_int8.data(),
            attention->key_cache_scales.data(), attention->value_cache_scales.data(),
            position_device_.data(), layout.key_value_heads, layout.head_dim, stream);
        if (attention->alibi_slopes.data()) {
            launch_gqa_decode_alibi_int8_device(
                q, attention->key_cache_int8.data(), attention->value_cache_int8.data(),
                attention->key_cache_scales.data(), attention->value_cache_scales.data(),
                workspace_.op_output_.data(), position_device_.data(),
                attention->alibi_slopes.data(), layout.query_heads,
                layout.key_value_heads, layout.head_dim, layout.sliding_window_size(), stream);
        } else if (resources_.options_.fast_attention) {
            launch_gqa_decode_online_int8_device(
                q, attention->key_cache_int8.data(), attention->value_cache_int8.data(),
                attention->key_cache_scales.data(), attention->value_cache_scales.data(),
                workspace_.op_output_.data(), position_device_.data(),
                layout.query_heads, layout.key_value_heads, layout.head_dim,
                layout.sliding_window_size(), stream);
        } else {
            launch_gqa_decode_strict_int8_device(
                q, attention->key_cache_int8.data(), attention->value_cache_int8.data(),
                attention->key_cache_scales.data(), attention->value_cache_scales.data(),
                workspace_.op_output_.data(), position_device_.data(),
                layout.query_heads, layout.key_value_heads, layout.head_dim,
                layout.sliding_window_size(), stream);
        }
    } else {
        launch_store_kv_device(k, v, attention->key_cache.data(),
                               attention->value_cache.data(), position_device_.data(),
                               layout.key_value_width(), stream);
        if (attention->alibi_slopes.data()) {
            launch_gqa_decode_alibi_device(
                q, attention->key_cache.data(), attention->value_cache.data(),
                workspace_.op_output_.data(), position_device_.data(),
                attention->alibi_slopes.data(), layout.query_heads,
                layout.key_value_heads, layout.head_dim, layout.sliding_window_size(), stream);
        } else if (resources_.options_.fast_attention) {
            launch_gqa_decode_online_device(
                q, attention->key_cache.data(), attention->value_cache.data(),
                workspace_.op_output_.data(), position_device_.data(),
                layout.query_heads, layout.key_value_heads, layout.head_dim,
                layout.sliding_window_size(), stream);
        } else {
            launch_gqa_decode_strict_device(
                q, attention->key_cache.data(), attention->value_cache.data(),
                workspace_.op_output_.data(), position_device_.data(),
                layout.query_heads, layout.key_value_heads, layout.head_dim,
                layout.sliding_window_size(), stream);
        }
    }
    if (layout.output_gate.enabled()) {
        launch_sigmoid_multiply(workspace_.op_output_.data(),
                                q + layout.query_width(),
                                layout.query_width(), stream);
    }
    linear(workspace_.op_output_.data(), *attention->out, workspace_.hidden_.data(),
           1, hidden, layout.query_width());
    launch_residual_add(workspace_.hidden_.data(), workspace_.residual_.data(),
                        hidden, stream);

    // The auxiliary layer uses the same generalized MoE executor as the
    // target model. Its synthetic layer index points at its own residency
    // controller/catalog entry, so disk-backed experts remain bounded too.
    run_mlp_decode(common_layer,
                   resources_.shape_.num_hidden_layers /* MTP layer 0 */);
    launch_rmsnorm(workspace_.hidden_.data(), mtp.norm, workspace_.normed_.data(),
                   1, hidden, eps, stream);
    linear(workspace_.normed_.data(), *mtp.logits, workspace_.mtp_logits_.data(),
           1, vocab, hidden);
    launch_scale(workspace_.mtp_logits_.data(), vocab,
                 resources_.shape_.numerical_policy.logits_multiplier /
                     resources_.shape_.numerical_policy.logits_divisor, stream);
    if (resources_.shape_.numerical_policy.final_logit_softcap > 0.0f) {
        launch_tanh_softcap(workspace_.mtp_logits_.data(), vocab,
                            resources_.shape_.numerical_policy.final_logit_softcap, stream);
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
        resources_.shape_.vocab_size,
        session_.generation_.repetition_penalty,
        workspace_.mtp_candidate_.data(), stream);
    launch_argmax_bf16(
        workspace_.logits_.data(), sampling_.seen_tokens.data(),
        resources_.shape_.vocab_size,
        session_.generation_.repetition_penalty,
        workspace_.mtp_target_candidate_.data(), stream);
}

} // namespace celeg
