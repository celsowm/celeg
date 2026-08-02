#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/paged_kv.hpp"
#include "celeg/backend/cuda/weight_layout.hpp"
#include "celeg/backend/cuda/moe.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace celeg {

void CudaCompiledModel::forward_token_host(int32_t token, bool compute_logits) {
    if (session_.position_ >= max_context_) {
        throw std::runtime_error("context limit reached");
    }
    resources_.weight_layout_->embed_token(
        token, workspace_.hidden_.data(), resources_.shape_.hidden, stream_.get());
    launch_scale(workspace_.hidden_.data(), resources_.shape_.hidden,
                 resources_.shape_.embedding_multiplier, stream_.get());

    int layer_idx = 0;
    for (Layer& layer : resources_.layers_) {
        LayerCommon& common_layer = common(layer);
        if (!resources_.options_.fused_residuals) {
            CELEG_CUDA(cudaMemcpyAsync(
                workspace_.residual_.data(), workspace_.hidden_.data(), workspace_.hidden_.bytes(),
                cudaMemcpyDeviceToDevice, stream_.get()));
        }
        launch_rmsnorm(workspace_.hidden_.data(), common_layer.operator_norm, workspace_.normed_.data(),
                       1, resources_.shape_.hidden, resources_.shape_.norm_eps,
                       stream_.get());
        if (AttentionLayer* attention = as_attention(layer)) {
            __nv_bfloat16* q = workspace_.qkv_output_.data();
            __nv_bfloat16* k = q + resources_.shape_.q_width;
            __nv_bfloat16* v = k + resources_.shape_.kv_width;
            if (resources_.options_.fused_projections) {
                linear(workspace_.normed_.data(), *attention->qkv, workspace_.qkv_output_.data(),
                       1, resources_.shape_.qkv_width, resources_.shape_.hidden);
            } else {
                const LinearWeight q_weight =
                    slice_rows(*attention->qkv, 0, resources_.shape_.q_width);
                const LinearWeight k_weight = slice_rows(
                    *attention->qkv, resources_.shape_.q_width, resources_.shape_.kv_width);
                const LinearWeight v_weight = slice_rows(
                    *attention->qkv, resources_.shape_.q_width + resources_.shape_.kv_width,
                    resources_.shape_.kv_width);
                linear(workspace_.normed_.data(), q_weight, q,
                       1, resources_.shape_.q_width, resources_.shape_.hidden);
                linear(workspace_.normed_.data(), k_weight, k,
                       1, resources_.shape_.kv_width, resources_.shape_.hidden);
                linear(workspace_.normed_.data(), v_weight, v,
                       1, resources_.shape_.kv_width, resources_.shape_.hidden);
            }
            if (!resources_.shape_.query_key_norm) {
                launch_rope_strict(
                    q, k, workspace_.rope_cos_.data(), workspace_.rope_sin_.data(),
                    resources_.shape_.num_attention_heads, resources_.shape_.num_key_value_heads,
                    resources_.shape_.head_dim, session_.position_, stream_.get());
                const float ratio = resources_.shape_.attention_multiplier /
                    (1.0f / std::sqrt(static_cast<float>(resources_.shape_.head_dim)));
                launch_scale(q, resources_.shape_.q_width, ratio, stream_.get());
            } else if (resources_.options_.fast_attention) {
                launch_qk_norm_rope_fast(
                    q, k, attention->q_norm, attention->k_norm,
                    workspace_.rope_cos_.data(), workspace_.rope_sin_.data(),
                    resources_.shape_.num_attention_heads, resources_.shape_.num_key_value_heads,
                    resources_.shape_.head_dim, session_.position_,
                    resources_.shape_.norm_eps, stream_.get());
            } else {
                launch_qk_norm_rope_strict(
                    q, k, attention->q_norm, attention->k_norm,
                    workspace_.rope_cos_.data(), workspace_.rope_sin_.data(),
                    resources_.shape_.num_attention_heads, resources_.shape_.num_key_value_heads,
                    resources_.shape_.head_dim, session_.position_,
                    resources_.shape_.norm_eps, stream_.get());
            }
            if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
                launch_store_kv_int8(
                    k, v, attention->key_cache_int8.data(),
                    attention->value_cache_int8.data(),
                    attention->key_cache_scales.data(),
                    attention->value_cache_scales.data(), session_.position_,
                    resources_.shape_.num_key_value_heads, resources_.shape_.head_dim, stream_.get());
                if (resources_.options_.fast_attention) {
                    launch_gqa_decode_online_int8(
                        q, attention->key_cache_int8.data(),
                        attention->value_cache_int8.data(),
                        attention->key_cache_scales.data(),
                        attention->value_cache_scales.data(), workspace_.op_output_.data(),
                        session_.position_ + 1, resources_.shape_.num_attention_heads,
                        resources_.shape_.num_key_value_heads, resources_.shape_.head_dim, stream_.get());
                } else {
                    launch_gqa_decode_strict_int8(
                        q, attention->key_cache_int8.data(),
                        attention->value_cache_int8.data(),
                        attention->key_cache_scales.data(),
                        attention->value_cache_scales.data(), workspace_.op_output_.data(),
                        session_.position_ + 1, resources_.shape_.num_attention_heads,
                        resources_.shape_.num_key_value_heads, resources_.shape_.head_dim, stream_.get());
                }
            } else {
                launch_store_kv(k, v, attention->key_cache.data(),
                                attention->value_cache.data(), session_.position_,
                                resources_.shape_.kv_width, stream_.get());
                if (resources_.options_.fast_attention) {
                    launch_gqa_decode_online(
                        q, attention->key_cache.data(), attention->value_cache.data(),
                        workspace_.op_output_.data(), session_.position_ + 1,
                        resources_.shape_.num_attention_heads, resources_.shape_.num_key_value_heads,
                        resources_.shape_.head_dim, stream_.get());
                } else {
                    launch_gqa_decode_strict(
                        q, attention->key_cache.data(), attention->value_cache.data(),
                        workspace_.op_output_.data(), session_.position_ + 1,
                        resources_.shape_.num_attention_heads, resources_.shape_.num_key_value_heads,
                        resources_.shape_.head_dim, stream_.get());
                }
            }
            linear(workspace_.op_output_.data(), *attention->out, workspace_.hidden_.data(),
                   1, resources_.shape_.hidden, resources_.shape_.hidden,
                   resources_.options_.fused_residuals ? 1.0f : 0.0f);
            launch_scale(workspace_.hidden_.data(), resources_.shape_.hidden, resources_.shape_.residual_multiplier,
                         stream_.get());
        } else {
            ConvolutionLayer& convolution = *as_convolution(layer);
            linear(workspace_.normed_.data(), *convolution.conv_in, workspace_.conv_projected_.data(),
                   1, 3 * resources_.shape_.hidden, resources_.shape_.hidden);
            launch_conv_decode(
                workspace_.conv_projected_.data(), convolution.conv_weight,
                convolution.conv_state.data(), workspace_.op_output_.data(),
                resources_.shape_.hidden, resources_.shape_.conv_cache, session_.position_,
                stream_.get());
            linear(workspace_.op_output_.data(), *convolution.conv_out, workspace_.hidden_.data(),
                   1, resources_.shape_.hidden, resources_.shape_.hidden,
                   resources_.options_.fused_residuals ? 1.0f : 0.0f);
        }
        if (!resources_.options_.fused_residuals) {
            launch_residual_add(workspace_.hidden_.data(), workspace_.residual_.data(),
                                resources_.shape_.hidden, stream_.get());
        }
        run_mlp_decode(common_layer, layer_idx);
        ++layer_idx;
    }
    if (compute_logits) {
        launch_rmsnorm(workspace_.hidden_.data(), resources_.final_norm_, workspace_.normed_.data(),
                        1, resources_.shape_.hidden, resources_.shape_.norm_eps,
                        stream_.get());
        linear(workspace_.normed_.data(), *logits_weight(), workspace_.logits_.data(),
                1, resources_.shape_.vocab_size, resources_.shape_.hidden);
        launch_scale(workspace_.logits_.data(), resources_.shape_.vocab_size,
                     1.0f / resources_.shape_.logits_divisor, stream_.get());
    }
    ++session_.position_;
}

void CudaCompiledModel::forward_token_paged_host(
    int32_t token, bool compute_logits, PhysicalPagedKvCache& paged_kv,
    const uint32_t* device_page_table, int page_table_stride) {
    if (session_.position_ >= max_context_) {
        throw std::runtime_error("context limit reached");
    }
    if (paged_kv.mode() != resources_.options_.kv_cache_mode) {
        throw std::invalid_argument("model and physical paged KV modes differ");
    }
    resources_.weight_layout_->embed_token(
        token, workspace_.hidden_.data(), resources_.shape_.hidden, stream_.get());
    launch_scale(workspace_.hidden_.data(), resources_.shape_.hidden,
                 resources_.shape_.embedding_multiplier, stream_.get());

    for (int layer_index = 0; layer_index < resources_.shape_.num_hidden_layers; ++layer_index) {
        Layer& layer = resources_.layers_[static_cast<size_t>(layer_index)];
        LayerCommon& common_layer = common(layer);
        if (!resources_.options_.fused_residuals) {
            CELEG_CUDA(cudaMemcpyAsync(workspace_.residual_.data(), workspace_.hidden_.data(), workspace_.hidden_.bytes(),
                                     cudaMemcpyDeviceToDevice, stream_.get()));
        }
        launch_rmsnorm(workspace_.hidden_.data(), common_layer.operator_norm, workspace_.normed_.data(),
                       1, resources_.shape_.hidden, resources_.shape_.norm_eps, stream_.get());
        if (AttentionLayer* attention = as_attention(layer)) {
            __nv_bfloat16* q = workspace_.qkv_output_.data();
            __nv_bfloat16* k = q + resources_.shape_.q_width;
            __nv_bfloat16* v = k + resources_.shape_.kv_width;
            if (resources_.options_.fused_projections) {
                linear(workspace_.normed_.data(), *attention->qkv, workspace_.qkv_output_.data(), 1,
                       resources_.shape_.qkv_width, resources_.shape_.hidden);
            } else {
                const LinearWeight q_weight =
                    slice_rows(*attention->qkv, 0, resources_.shape_.q_width);
                const LinearWeight k_weight = slice_rows(
                    *attention->qkv, resources_.shape_.q_width, resources_.shape_.kv_width);
                const LinearWeight v_weight = slice_rows(
                    *attention->qkv, resources_.shape_.q_width + resources_.shape_.kv_width,
                    resources_.shape_.kv_width);
                linear(workspace_.normed_.data(), q_weight, q, 1, resources_.shape_.q_width,
                       resources_.shape_.hidden);
                linear(workspace_.normed_.data(), k_weight, k, 1, resources_.shape_.kv_width,
                       resources_.shape_.hidden);
                linear(workspace_.normed_.data(), v_weight, v, 1, resources_.shape_.kv_width,
                       resources_.shape_.hidden);
            }
            if (!resources_.shape_.query_key_norm) {
                launch_rope_strict(
                    q, k, workspace_.rope_cos_.data(), workspace_.rope_sin_.data(),
                    resources_.shape_.num_attention_heads, resources_.shape_.num_key_value_heads,
                    resources_.shape_.head_dim, session_.position_, stream_.get());
                const float ratio = resources_.shape_.attention_multiplier /
                    (1.0f / std::sqrt(static_cast<float>(resources_.shape_.head_dim)));
                launch_scale(q, resources_.shape_.q_width, ratio, stream_.get());
            } else if (resources_.options_.fast_attention) {
                launch_qk_norm_rope_fast(
                    q, k, attention->q_norm, attention->k_norm, workspace_.rope_cos_.data(),
                    workspace_.rope_sin_.data(), resources_.shape_.num_attention_heads, resources_.shape_.num_key_value_heads,
                    resources_.shape_.head_dim, session_.position_, resources_.shape_.norm_eps,
                    stream_.get());
            } else {
                launch_qk_norm_rope_strict(
                    q, k, attention->q_norm, attention->k_norm, workspace_.rope_cos_.data(),
                    workspace_.rope_sin_.data(), resources_.shape_.num_attention_heads, resources_.shape_.num_key_value_heads,
                    resources_.shape_.head_dim, session_.position_, resources_.shape_.norm_eps,
                    stream_.get());
            }
            const int slot = paged_kv.attention_slot(layer_index);
            if (slot < 0) throw std::logic_error("attention layer has no page slot");
            if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
                launch_store_kv_int8_paged_batch(
                    k, v, paged_kv.key_int8(), paged_kv.value_int8(),
                    paged_kv.key_scales(), paged_kv.value_scales(),
                    device_page_table, page_table_stride, position_device_.data(),
                    1, slot, paged_kv.page_tokens(),
                    paged_kv.attention_layers(), resources_.shape_.num_key_value_heads,
                    resources_.shape_.head_dim, stream_.get());
                if (use_segmented_attention(session_.position_)) {
                    const int chunks = (session_.position_ + 1 +
                        resources_.options_.attention_chunk_tokens - 1) /
                        resources_.options_.attention_chunk_tokens;
                    launch_gqa_decode_int8_paged_segmented_batch(
                        q, paged_kv.key_int8(), paged_kv.value_int8(),
                        paged_kv.key_scales(), paged_kv.value_scales(),
                        device_page_table, page_table_stride, workspace_.op_output_.data(),
                        position_device_.data(), 1, slot,
                        paged_kv.page_tokens(),
                        paged_kv.attention_layers(),
                        resources_.shape_.num_attention_heads, resources_.shape_.num_key_value_heads,
                        resources_.shape_.head_dim, resources_.options_.attention_chunk_tokens,
                        chunks, workspace_.attention_partial_max_.data(),
                        workspace_.attention_partial_denom_.data(),
                        workspace_.attention_partial_accum_.data(), stream_.get());
                } else {
                    launch_gqa_decode_int8_paged_batch(
                        q, paged_kv.key_int8(), paged_kv.value_int8(),
                        paged_kv.key_scales(), paged_kv.value_scales(),
                        device_page_table, page_table_stride, workspace_.op_output_.data(),
                        position_device_.data(), 1, slot,
                        paged_kv.page_tokens(),
                        paged_kv.attention_layers(),
                        resources_.shape_.num_attention_heads, resources_.shape_.num_key_value_heads,
                        resources_.shape_.head_dim, resources_.options_.fast_attention,
                        stream_.get());
                }
            } else {
                launch_store_kv_paged_batch(
                    k, v, paged_kv.key_bf16(), paged_kv.value_bf16(),
                    device_page_table, page_table_stride, position_device_.data(),
                    1, slot, paged_kv.page_tokens(),
                    paged_kv.attention_layers(), resources_.shape_.num_key_value_heads,
                    resources_.shape_.head_dim, stream_.get());
                if (use_segmented_attention(session_.position_)) {
                    const int chunks = (session_.position_ + 1 +
                        resources_.options_.attention_chunk_tokens - 1) /
                        resources_.options_.attention_chunk_tokens;
                    launch_gqa_decode_paged_segmented_batch(
                        q, paged_kv.key_bf16(), paged_kv.value_bf16(),
                        device_page_table, page_table_stride,
                        workspace_.op_output_.data(), position_device_.data(), 1, slot,
                        paged_kv.page_tokens(),
                        paged_kv.attention_layers(),
                        resources_.shape_.num_attention_heads, resources_.shape_.num_key_value_heads,
                        resources_.shape_.head_dim, resources_.options_.attention_chunk_tokens,
                        chunks, workspace_.attention_partial_max_.data(),
                        workspace_.attention_partial_denom_.data(),
                        workspace_.attention_partial_accum_.data(), stream_.get());
                } else {
                    launch_gqa_decode_paged_batch(
                        q, paged_kv.key_bf16(), paged_kv.value_bf16(),
                        device_page_table, page_table_stride,
                        workspace_.op_output_.data(), position_device_.data(), 1, slot,
                        paged_kv.page_tokens(),
                        paged_kv.attention_layers(),
                        resources_.shape_.num_attention_heads, resources_.shape_.num_key_value_heads,
                        resources_.shape_.head_dim, resources_.options_.fast_attention,
                        stream_.get());
                }
            }
            linear(workspace_.op_output_.data(), *attention->out, workspace_.hidden_.data(), 1,
                   resources_.shape_.hidden, resources_.shape_.hidden,
                   resources_.options_.fused_residuals ? 1.0f : 0.0f);
            launch_scale(workspace_.hidden_.data(), resources_.shape_.hidden,
                         resources_.shape_.residual_multiplier, stream_.get());
        } else {
            ConvolutionLayer& convolution = *as_convolution(layer);
            linear(workspace_.normed_.data(), *convolution.conv_in, workspace_.conv_projected_.data(), 1,
                   3 * resources_.shape_.hidden, resources_.shape_.hidden);
            launch_conv_decode(workspace_.conv_projected_.data(), convolution.conv_weight,
                               convolution.conv_state.data(), workspace_.op_output_.data(),
                               resources_.shape_.hidden, resources_.shape_.conv_cache,
                               session_.position_, stream_.get());
            linear(workspace_.op_output_.data(), *convolution.conv_out, workspace_.hidden_.data(), 1,
                   resources_.shape_.hidden, resources_.shape_.hidden,
                   resources_.options_.fused_residuals ? 1.0f : 0.0f);
        }
        if (!resources_.options_.fused_residuals) {
            launch_residual_add(workspace_.hidden_.data(), workspace_.residual_.data(),
                                resources_.shape_.hidden, stream_.get());
        }
        run_mlp_decode(common_layer, layer_index);
    }
    if (compute_logits) {
        launch_rmsnorm(workspace_.hidden_.data(), resources_.final_norm_, workspace_.normed_.data(), 1,
                       resources_.shape_.hidden, resources_.shape_.norm_eps, stream_.get());
        linear(workspace_.normed_.data(), *logits_weight(), workspace_.logits_.data(), 1,
               resources_.shape_.vocab_size, resources_.shape_.hidden);
        launch_scale(workspace_.logits_.data(), resources_.shape_.vocab_size,
                     1.0f / resources_.shape_.logits_divisor, stream_.get());
    }
    ++session_.position_;
    CELEG_CUDA(cudaMemcpyAsync(position_device_.data(), &session_.position_, sizeof(session_.position_),
                             cudaMemcpyHostToDevice, stream_.get()));
}

} // namespace celeg

