#include "celeg/detail/model/impl.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/paged_kv.hpp"
#include "celeg/model/weights/layout.hpp"
#include "celeg/runtime/moe.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace celeg {

void Model::Impl::forward_token_host(int32_t token, bool compute_logits) {
    if (position_ >= max_context_) {
        throw std::runtime_error("context limit reached");
    }
    weight_layout_->embed_token(
        token, hidden_.data(), shape_.hidden, stream_.get());
    launch_scale(hidden_.data(), shape_.hidden,
                 shape_.embedding_multiplier, stream_.get());

    int layer_idx = 0;
    for (Layer& layer : layers_) {
        LayerCommon& common_layer = common(layer);
        if (!options_.fused_residuals) {
            CELEG_CUDA(cudaMemcpyAsync(
                residual_.data(), hidden_.data(), hidden_.bytes(),
                cudaMemcpyDeviceToDevice, stream_.get()));
        }
        launch_rmsnorm(hidden_.data(), common_layer.operator_norm, normed_.data(),
                       1, shape_.hidden, shape_.norm_eps,
                       stream_.get());
        if (AttentionLayer* attention = as_attention(layer)) {
            __nv_bfloat16* q = qkv_output_.data();
            __nv_bfloat16* k = q + shape_.q_width;
            __nv_bfloat16* v = k + shape_.kv_width;
            if (options_.fused_projections) {
                linear(normed_.data(), *attention->qkv, qkv_output_.data(),
                       1, shape_.qkv_width, shape_.hidden);
            } else {
                const LinearWeight q_weight =
                    slice_rows(*attention->qkv, 0, shape_.q_width);
                const LinearWeight k_weight = slice_rows(
                    *attention->qkv, shape_.q_width, shape_.kv_width);
                const LinearWeight v_weight = slice_rows(
                    *attention->qkv, shape_.q_width + shape_.kv_width,
                    shape_.kv_width);
                linear(normed_.data(), q_weight, q,
                       1, shape_.q_width, shape_.hidden);
                linear(normed_.data(), k_weight, k,
                       1, shape_.kv_width, shape_.hidden);
                linear(normed_.data(), v_weight, v,
                       1, shape_.kv_width, shape_.hidden);
            }
            if (!shape_.query_key_norm) {
                launch_rope_strict(
                    q, k, rope_cos_.data(), rope_sin_.data(),
                    shape_.num_attention_heads, shape_.num_key_value_heads,
                    shape_.head_dim, position_, stream_.get());
                const float ratio = shape_.attention_multiplier /
                    (1.0f / std::sqrt(static_cast<float>(shape_.head_dim)));
                launch_scale(q, shape_.q_width, ratio, stream_.get());
            } else if (options_.fast_attention) {
                launch_qk_norm_rope_fast(
                    q, k, attention->q_norm, attention->k_norm,
                    rope_cos_.data(), rope_sin_.data(),
                    shape_.num_attention_heads, shape_.num_key_value_heads,
                    shape_.head_dim, position_,
                    shape_.norm_eps, stream_.get());
            } else {
                launch_qk_norm_rope_strict(
                    q, k, attention->q_norm, attention->k_norm,
                    rope_cos_.data(), rope_sin_.data(),
                    shape_.num_attention_heads, shape_.num_key_value_heads,
                    shape_.head_dim, position_,
                    shape_.norm_eps, stream_.get());
            }
            if (options_.kv_cache_mode == KvCacheMode::Int8) {
                launch_store_kv_int8(
                    k, v, attention->key_cache_int8.data(),
                    attention->value_cache_int8.data(),
                    attention->key_cache_scales.data(),
                    attention->value_cache_scales.data(), position_,
                    shape_.num_key_value_heads, shape_.head_dim, stream_.get());
                if (options_.fast_attention) {
                    launch_gqa_decode_online_int8(
                        q, attention->key_cache_int8.data(),
                        attention->value_cache_int8.data(),
                        attention->key_cache_scales.data(),
                        attention->value_cache_scales.data(), op_output_.data(),
                        position_ + 1, shape_.num_attention_heads,
                        shape_.num_key_value_heads, shape_.head_dim, stream_.get());
                } else {
                    launch_gqa_decode_strict_int8(
                        q, attention->key_cache_int8.data(),
                        attention->value_cache_int8.data(),
                        attention->key_cache_scales.data(),
                        attention->value_cache_scales.data(), op_output_.data(),
                        position_ + 1, shape_.num_attention_heads,
                        shape_.num_key_value_heads, shape_.head_dim, stream_.get());
                }
            } else {
                launch_store_kv(k, v, attention->key_cache.data(),
                                attention->value_cache.data(), position_,
                                shape_.kv_width, stream_.get());
                if (options_.fast_attention) {
                    launch_gqa_decode_online(
                        q, attention->key_cache.data(), attention->value_cache.data(),
                        op_output_.data(), position_ + 1,
                        shape_.num_attention_heads, shape_.num_key_value_heads,
                        shape_.head_dim, stream_.get());
                } else {
                    launch_gqa_decode_strict(
                        q, attention->key_cache.data(), attention->value_cache.data(),
                        op_output_.data(), position_ + 1,
                        shape_.num_attention_heads, shape_.num_key_value_heads,
                        shape_.head_dim, stream_.get());
                }
            }
            linear(op_output_.data(), *attention->out, hidden_.data(),
                   1, shape_.hidden, shape_.hidden,
                   options_.fused_residuals ? 1.0f : 0.0f);
            launch_scale(hidden_.data(), shape_.hidden, shape_.residual_multiplier,
                         stream_.get());
        } else {
            ConvolutionLayer& convolution = *as_convolution(layer);
            linear(normed_.data(), *convolution.conv_in, conv_projected_.data(),
                   1, 3 * shape_.hidden, shape_.hidden);
            launch_conv_decode(
                conv_projected_.data(), convolution.conv_weight,
                convolution.conv_state.data(), op_output_.data(),
                shape_.hidden, shape_.conv_cache, position_,
                stream_.get());
            linear(op_output_.data(), *convolution.conv_out, hidden_.data(),
                   1, shape_.hidden, shape_.hidden,
                   options_.fused_residuals ? 1.0f : 0.0f);
        }
        if (!options_.fused_residuals) {
            launch_residual_add(hidden_.data(), residual_.data(),
                                shape_.hidden, stream_.get());
        }
        run_mlp_decode(common_layer, layer_idx);
        ++layer_idx;
    }
    if (compute_logits) {
        launch_rmsnorm(hidden_.data(), final_norm_, normed_.data(),
                        1, shape_.hidden, shape_.norm_eps,
                        stream_.get());
        linear(normed_.data(), *logits_weight(), logits_.data(),
                1, shape_.vocab_size, shape_.hidden);
        launch_scale(logits_.data(), shape_.vocab_size,
                     1.0f / shape_.logits_divisor, stream_.get());
    }
    ++position_;
}

void Model::Impl::forward_token_paged_host(
    int32_t token, bool compute_logits, PhysicalPagedKvCache& paged_kv,
    const uint32_t* device_page_table, int page_table_stride) {
    if (position_ >= max_context_) {
        throw std::runtime_error("context limit reached");
    }
    if (paged_kv.mode() != options_.kv_cache_mode) {
        throw std::invalid_argument("model and physical paged KV modes differ");
    }
    weight_layout_->embed_token(
        token, hidden_.data(), shape_.hidden, stream_.get());
    launch_scale(hidden_.data(), shape_.hidden,
                 shape_.embedding_multiplier, stream_.get());

    for (int layer_index = 0; layer_index < shape_.num_hidden_layers; ++layer_index) {
        Layer& layer = layers_[static_cast<size_t>(layer_index)];
        LayerCommon& common_layer = common(layer);
        if (!options_.fused_residuals) {
            CELEG_CUDA(cudaMemcpyAsync(residual_.data(), hidden_.data(), hidden_.bytes(),
                                     cudaMemcpyDeviceToDevice, stream_.get()));
        }
        launch_rmsnorm(hidden_.data(), common_layer.operator_norm, normed_.data(),
                       1, shape_.hidden, shape_.norm_eps, stream_.get());
        if (AttentionLayer* attention = as_attention(layer)) {
            __nv_bfloat16* q = qkv_output_.data();
            __nv_bfloat16* k = q + shape_.q_width;
            __nv_bfloat16* v = k + shape_.kv_width;
            if (options_.fused_projections) {
                linear(normed_.data(), *attention->qkv, qkv_output_.data(), 1,
                       shape_.qkv_width, shape_.hidden);
            } else {
                const LinearWeight q_weight =
                    slice_rows(*attention->qkv, 0, shape_.q_width);
                const LinearWeight k_weight = slice_rows(
                    *attention->qkv, shape_.q_width, shape_.kv_width);
                const LinearWeight v_weight = slice_rows(
                    *attention->qkv, shape_.q_width + shape_.kv_width,
                    shape_.kv_width);
                linear(normed_.data(), q_weight, q, 1, shape_.q_width,
                       shape_.hidden);
                linear(normed_.data(), k_weight, k, 1, shape_.kv_width,
                       shape_.hidden);
                linear(normed_.data(), v_weight, v, 1, shape_.kv_width,
                       shape_.hidden);
            }
            if (!shape_.query_key_norm) {
                launch_rope_strict(
                    q, k, rope_cos_.data(), rope_sin_.data(),
                    shape_.num_attention_heads, shape_.num_key_value_heads,
                    shape_.head_dim, position_, stream_.get());
                const float ratio = shape_.attention_multiplier /
                    (1.0f / std::sqrt(static_cast<float>(shape_.head_dim)));
                launch_scale(q, shape_.q_width, ratio, stream_.get());
            } else if (options_.fast_attention) {
                launch_qk_norm_rope_fast(
                    q, k, attention->q_norm, attention->k_norm, rope_cos_.data(),
                    rope_sin_.data(), shape_.num_attention_heads, shape_.num_key_value_heads,
                    shape_.head_dim, position_, shape_.norm_eps,
                    stream_.get());
            } else {
                launch_qk_norm_rope_strict(
                    q, k, attention->q_norm, attention->k_norm, rope_cos_.data(),
                    rope_sin_.data(), shape_.num_attention_heads, shape_.num_key_value_heads,
                    shape_.head_dim, position_, shape_.norm_eps,
                    stream_.get());
            }
            const int slot = paged_kv.attention_slot(layer_index);
            if (slot < 0) throw std::logic_error("attention layer has no page slot");
            if (options_.kv_cache_mode == KvCacheMode::Int8) {
                launch_store_kv_int8_paged_batch(
                    k, v, paged_kv.key_int8(), paged_kv.value_int8(),
                    paged_kv.key_scales(), paged_kv.value_scales(),
                    device_page_table, page_table_stride, position_device_.data(),
                    1, slot, paged_kv.page_tokens(),
                    paged_kv.attention_layers(), shape_.num_key_value_heads,
                    shape_.head_dim, stream_.get());
                if (use_segmented_attention(position_)) {
                    const int chunks = (position_ + 1 +
                        options_.attention_chunk_tokens - 1) /
                        options_.attention_chunk_tokens;
                    launch_gqa_decode_int8_paged_segmented_batch(
                        q, paged_kv.key_int8(), paged_kv.value_int8(),
                        paged_kv.key_scales(), paged_kv.value_scales(),
                        device_page_table, page_table_stride, op_output_.data(),
                        position_device_.data(), 1, slot,
                        paged_kv.page_tokens(),
                        paged_kv.attention_layers(),
                        shape_.num_attention_heads, shape_.num_key_value_heads,
                        shape_.head_dim, options_.attention_chunk_tokens,
                        chunks, attention_partial_max_.data(),
                        attention_partial_denom_.data(),
                        attention_partial_accum_.data(), stream_.get());
                } else {
                    launch_gqa_decode_int8_paged_batch(
                        q, paged_kv.key_int8(), paged_kv.value_int8(),
                        paged_kv.key_scales(), paged_kv.value_scales(),
                        device_page_table, page_table_stride, op_output_.data(),
                        position_device_.data(), 1, slot,
                        paged_kv.page_tokens(),
                        paged_kv.attention_layers(),
                        shape_.num_attention_heads, shape_.num_key_value_heads,
                        shape_.head_dim, options_.fast_attention,
                        stream_.get());
                }
            } else {
                launch_store_kv_paged_batch(
                    k, v, paged_kv.key_bf16(), paged_kv.value_bf16(),
                    device_page_table, page_table_stride, position_device_.data(),
                    1, slot, paged_kv.page_tokens(),
                    paged_kv.attention_layers(), shape_.num_key_value_heads,
                    shape_.head_dim, stream_.get());
                if (use_segmented_attention(position_)) {
                    const int chunks = (position_ + 1 +
                        options_.attention_chunk_tokens - 1) /
                        options_.attention_chunk_tokens;
                    launch_gqa_decode_paged_segmented_batch(
                        q, paged_kv.key_bf16(), paged_kv.value_bf16(),
                        device_page_table, page_table_stride,
                        op_output_.data(), position_device_.data(), 1, slot,
                        paged_kv.page_tokens(),
                        paged_kv.attention_layers(),
                        shape_.num_attention_heads, shape_.num_key_value_heads,
                        shape_.head_dim, options_.attention_chunk_tokens,
                        chunks, attention_partial_max_.data(),
                        attention_partial_denom_.data(),
                        attention_partial_accum_.data(), stream_.get());
                } else {
                    launch_gqa_decode_paged_batch(
                        q, paged_kv.key_bf16(), paged_kv.value_bf16(),
                        device_page_table, page_table_stride,
                        op_output_.data(), position_device_.data(), 1, slot,
                        paged_kv.page_tokens(),
                        paged_kv.attention_layers(),
                        shape_.num_attention_heads, shape_.num_key_value_heads,
                        shape_.head_dim, options_.fast_attention,
                        stream_.get());
                }
            }
            linear(op_output_.data(), *attention->out, hidden_.data(), 1,
                   shape_.hidden, shape_.hidden,
                   options_.fused_residuals ? 1.0f : 0.0f);
            launch_scale(hidden_.data(), shape_.hidden,
                         shape_.residual_multiplier, stream_.get());
        } else {
            ConvolutionLayer& convolution = *as_convolution(layer);
            linear(normed_.data(), *convolution.conv_in, conv_projected_.data(), 1,
                   3 * shape_.hidden, shape_.hidden);
            launch_conv_decode(conv_projected_.data(), convolution.conv_weight,
                               convolution.conv_state.data(), op_output_.data(),
                               shape_.hidden, shape_.conv_cache,
                               position_, stream_.get());
            linear(op_output_.data(), *convolution.conv_out, hidden_.data(), 1,
                   shape_.hidden, shape_.hidden,
                   options_.fused_residuals ? 1.0f : 0.0f);
        }
        if (!options_.fused_residuals) {
            launch_residual_add(hidden_.data(), residual_.data(),
                                shape_.hidden, stream_.get());
        }
        run_mlp_decode(common_layer, layer_index);
    }
    if (compute_logits) {
        launch_rmsnorm(hidden_.data(), final_norm_, normed_.data(), 1,
                       shape_.hidden, shape_.norm_eps, stream_.get());
        linear(normed_.data(), *logits_weight(), logits_.data(), 1,
               shape_.vocab_size, shape_.hidden);
        launch_scale(logits_.data(), shape_.vocab_size,
                     1.0f / shape_.logits_divisor, stream_.get());
    }
    ++position_;
    CELEG_CUDA(cudaMemcpyAsync(position_device_.data(), &position_, sizeof(position_),
                             cudaMemcpyHostToDevice, stream_.get()));
}

} // namespace celeg

