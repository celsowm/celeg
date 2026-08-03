#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/paged_kv.hpp"
#include "celeg/backend/cuda/weight_layout.hpp"
#include "celeg/backend/cuda/moe.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace celeg {

void CudaCompiledModel::forward_token_host(int32_t token, bool compute_logits,
                                           const float* raw_embedding) {
    if (session_.position_ >= max_context_) {
        throw std::runtime_error("context limit reached");
    }
    if (raw_embedding) {
        std::vector<__nv_bfloat16> converted(static_cast<size_t>(resources_.shape_.hidden));
        for (int index = 0; index < resources_.shape_.hidden; ++index) {
            converted[static_cast<size_t>(index)] = __float2bfloat16(raw_embedding[index]);
        }
        CELEG_CUDA(cudaMemcpyAsync(workspace_.hidden_.data(), converted.data(),
                                   converted.size() * sizeof(__nv_bfloat16),
                                   cudaMemcpyHostToDevice, stream_.get()));
        initialize_per_layer_input_host(0);
    } else {
        resources_.weight_layout_->embed_token(
            token, workspace_.hidden_.data(), resources_.shape_.hidden, stream_.get());
        launch_scale(workspace_.hidden_.data(), resources_.shape_.hidden,
                     resources_.shape_.embedding_multiplier, stream_.get());
        initialize_per_layer_input_host(token);
    }

    int layer_idx = 0;
    for (Layer& layer : resources_.layers_) {
        LayerCommon& common_layer = common(layer);
        if (!resources_.options_.fused_residuals || common_layer.post_attention_norm) {
            CELEG_CUDA(cudaMemcpyAsync(
                workspace_.residual_.data(), workspace_.hidden_.data(), workspace_.hidden_.bytes(),
                cudaMemcpyDeviceToDevice, stream_.get()));
        }
        launch_rmsnorm(workspace_.hidden_.data(), common_layer.operator_norm, workspace_.normed_.data(),
                       1, resources_.shape_.hidden, resources_.shape_.norm_eps,
                       stream_.get());
        if (AttentionLayer* attention = as_attention(layer)) {
            const AttentionSpec& layout = attention->layout;
            AttentionLayer* owner = attention;
            if (attention->kv_owner_layer >= 0) {
                owner = as_attention(resources_.layers_.at(
                    static_cast<size_t>(attention->kv_owner_layer)));
                if (!owner) throw std::logic_error("CUDA shared KV owner is not attention");
            }
            __nv_bfloat16* q = workspace_.qkv_output_.data();
            __nv_bfloat16* k = q + layout.query_width();
            __nv_bfloat16* v = k + layout.key_value_width();
            linear(workspace_.normed_.data(), *attention->query, q,
                   1, layout.query_width(), resources_.shape_.hidden);
            if (attention->key && attention->value) {
                linear(workspace_.normed_.data(), *attention->key, k,
                       1, layout.key_value_width(), resources_.shape_.hidden);
                linear(workspace_.normed_.data(), *attention->value, v,
                       1, layout.key_value_width(), resources_.shape_.hidden);
            }
            if (layout.positional_encoding == PositionalEncodingKind::Rope) {
                launch_dynamic_qk_norm_rope(
                    q, attention->key ? k : nullptr, attention->q_norm, attention->k_norm,
                    layout.query_heads, layout.key_value_heads, layout.head_dim,
                    session_.position_, static_cast<float>(layout.rope_theta),
                    static_cast<float>(layout.rotary_fraction), resources_.shape_.norm_eps,
                    layout.query_key_norm, stream_.get());
            }
            launch_scale(q, layout.query_width(), layout.query_scale, stream_.get());
            const AttentionSpec& owner_layout = owner->layout;
            if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
                launch_store_kv_int8(
                    k, v, owner->key_cache_int8.data(), owner->value_cache_int8.data(),
                    owner->key_cache_scales.data(), owner->value_cache_scales.data(),
                    session_.position_, owner_layout.key_value_heads, owner_layout.head_dim, stream_.get());
                if (resources_.options_.fast_attention) {
                    launch_gqa_decode_online_int8(
                        q, owner->key_cache_int8.data(), owner->value_cache_int8.data(),
                        owner->key_cache_scales.data(), owner->value_cache_scales.data(), workspace_.op_output_.data(),
                        session_.position_ + 1, layout.query_heads, owner_layout.key_value_heads,
                        owner_layout.head_dim, layout.sliding_window, stream_.get());
                } else {
                    launch_gqa_decode_strict_int8(
                        q, owner->key_cache_int8.data(), owner->value_cache_int8.data(),
                        owner->key_cache_scales.data(), owner->value_cache_scales.data(), workspace_.op_output_.data(),
                        session_.position_ + 1, layout.query_heads, owner_layout.key_value_heads,
                        owner_layout.head_dim, layout.sliding_window, stream_.get());
                }
            } else {
                launch_store_kv(k, v, owner->key_cache.data(), owner->value_cache.data(),
                                session_.position_, owner_layout.key_value_width(), stream_.get());
                if (resources_.options_.fast_attention) {
                    launch_gqa_decode_online(
                        q, owner->key_cache.data(), owner->value_cache.data(),
                        workspace_.op_output_.data(), session_.position_ + 1,
                        layout.query_heads, owner_layout.key_value_heads,
                        owner_layout.head_dim, layout.sliding_window, stream_.get());
                } else {
                    launch_gqa_decode_strict(
                        q, owner->key_cache.data(), owner->value_cache.data(),
                        workspace_.op_output_.data(), session_.position_ + 1,
                        layout.query_heads, owner_layout.key_value_heads,
                        owner_layout.head_dim, layout.sliding_window, stream_.get());
                }
            }
            linear(workspace_.op_output_.data(), *attention->out, workspace_.hidden_.data(),
                   1, resources_.shape_.hidden, layout.query_width(),
                   resources_.options_.fused_residuals && !common_layer.post_attention_norm ? 1.0f : 0.0f);
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
        if (common_layer.post_attention_norm) {
            launch_rmsnorm(workspace_.hidden_.data(), common_layer.post_attention_norm,
                           workspace_.hidden_.data(), 1, resources_.shape_.hidden,
                           resources_.shape_.norm_eps, stream_.get());
        }
        if (!resources_.options_.fused_residuals || common_layer.post_attention_norm) {
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
        if (resources_.shape_.final_logit_softcap > 0.0f) {
            launch_tanh_softcap(workspace_.logits_.data(), resources_.shape_.vocab_size,
                                resources_.shape_.final_logit_softcap, stream_.get());
        }
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
    initialize_per_layer_input_host(token);

    for (int layer_index = 0; layer_index < resources_.shape_.num_hidden_layers; ++layer_index) {
        Layer& layer = resources_.layers_[static_cast<size_t>(layer_index)];
        LayerCommon& common_layer = common(layer);
        if (!resources_.options_.fused_residuals || common_layer.post_attention_norm) {
            CELEG_CUDA(cudaMemcpyAsync(workspace_.residual_.data(), workspace_.hidden_.data(), workspace_.hidden_.bytes(),
                                     cudaMemcpyDeviceToDevice, stream_.get()));
        }
        launch_rmsnorm(workspace_.hidden_.data(), common_layer.operator_norm, workspace_.normed_.data(),
                       1, resources_.shape_.hidden, resources_.shape_.norm_eps, stream_.get());
        if (AttentionLayer* attention = as_attention(layer)) {
            const AttentionSpec& layout = attention->layout;
            __nv_bfloat16* q = workspace_.qkv_output_.data();
            __nv_bfloat16* k = q + layout.query_width();
            __nv_bfloat16* v = k + layout.key_value_width();
            linear(workspace_.normed_.data(), *attention->query, q, 1,
                   layout.query_width(), resources_.shape_.hidden);
            if (attention->key && attention->value) {
                linear(workspace_.normed_.data(), *attention->key, k, 1,
                       layout.key_value_width(), resources_.shape_.hidden);
                linear(workspace_.normed_.data(), *attention->value, v, 1,
                       layout.key_value_width(), resources_.shape_.hidden);
            }
            if (layout.positional_encoding == PositionalEncodingKind::Rope) {
                launch_dynamic_qk_norm_rope(
                    q, attention->key ? k : nullptr, attention->q_norm, attention->k_norm,
                    layout.query_heads, layout.key_value_heads, layout.head_dim,
                    session_.position_, static_cast<float>(layout.rope_theta),
                    static_cast<float>(layout.rotary_fraction), resources_.shape_.norm_eps,
                    layout.query_key_norm, stream_.get());
            }
            launch_scale(q, layout.query_width(), layout.query_scale, stream_.get());
            const int cache_model_layer = attention->kv_owner_layer >= 0
                ? attention->kv_owner_layer : layer_index;
            const int slot = paged_kv.attention_slot(cache_model_layer);
            if (slot < 0) throw std::logic_error("attention layer has no page slot");
            AttentionLayer* owner = attention->kv_owner_layer >= 0
                ? as_attention(resources_.layers_.at(static_cast<size_t>(cache_model_layer)))
                : attention;
            if (!owner) throw std::logic_error("CUDA shared KV owner is not attention");
            const AttentionSpec& owner_layout = owner->layout;
            if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
                launch_store_kv_int8_paged_batch(
                    k, v, paged_kv.key_int8(), paged_kv.value_int8(),
                    paged_kv.key_scales(), paged_kv.value_scales(),
                    device_page_table, page_table_stride, position_device_.data(),
                    1, slot, paged_kv.page_tokens(),
                    paged_kv.page_vector_elements(), paged_kv.layer_vector_offset(slot),
                    paged_kv.page_scale_elements(), paged_kv.layer_scale_offset(slot),
                    owner_layout.key_value_heads, owner_layout.head_dim, stream_.get());
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
                        paged_kv.page_vector_elements(), paged_kv.layer_vector_offset(slot),
                        paged_kv.page_scale_elements(), paged_kv.layer_scale_offset(slot),
                        layout.query_heads, owner_layout.key_value_heads,
                        owner_layout.head_dim, resources_.options_.attention_chunk_tokens,
                        chunks, layout.sliding_window, workspace_.attention_partial_max_.data(),
                        workspace_.attention_partial_denom_.data(),
                        workspace_.attention_partial_accum_.data(), stream_.get());
                } else {
                    launch_gqa_decode_int8_paged_batch(
                        q, paged_kv.key_int8(), paged_kv.value_int8(),
                        paged_kv.key_scales(), paged_kv.value_scales(),
                        device_page_table, page_table_stride, workspace_.op_output_.data(),
                        position_device_.data(), 1, slot,
                        paged_kv.page_tokens(),
                        paged_kv.page_vector_elements(), paged_kv.layer_vector_offset(slot),
                        paged_kv.page_scale_elements(), paged_kv.layer_scale_offset(slot),
                        layout.query_heads, owner_layout.key_value_heads,
                        owner_layout.head_dim, layout.sliding_window,
                        resources_.options_.fast_attention,
                        stream_.get());
                }
            } else {
                launch_store_kv_paged_batch(
                    k, v, paged_kv.key_bf16(), paged_kv.value_bf16(),
                    device_page_table, page_table_stride, position_device_.data(),
                    1, slot, paged_kv.page_tokens(),
                    paged_kv.page_vector_elements(), paged_kv.layer_vector_offset(slot),
                    owner_layout.key_value_heads, owner_layout.head_dim, stream_.get());
                if (use_segmented_attention(session_.position_)) {
                    const int chunks = (session_.position_ + 1 +
                        resources_.options_.attention_chunk_tokens - 1) /
                        resources_.options_.attention_chunk_tokens;
                    launch_gqa_decode_paged_segmented_batch(
                        q, paged_kv.key_bf16(), paged_kv.value_bf16(),
                        device_page_table, page_table_stride,
                        workspace_.op_output_.data(), position_device_.data(), 1, slot,
                        paged_kv.page_tokens(),
                        paged_kv.page_vector_elements(), paged_kv.layer_vector_offset(slot),
                        layout.query_heads, owner_layout.key_value_heads,
                        owner_layout.head_dim, resources_.options_.attention_chunk_tokens,
                        chunks, layout.sliding_window, workspace_.attention_partial_max_.data(),
                        workspace_.attention_partial_denom_.data(),
                        workspace_.attention_partial_accum_.data(), stream_.get());
                } else {
                    launch_gqa_decode_paged_batch(
                        q, paged_kv.key_bf16(), paged_kv.value_bf16(),
                        device_page_table, page_table_stride,
                        workspace_.op_output_.data(), position_device_.data(), 1, slot,
                        paged_kv.page_tokens(),
                        paged_kv.page_vector_elements(), paged_kv.layer_vector_offset(slot),
                        layout.query_heads, owner_layout.key_value_heads,
                        owner_layout.head_dim, layout.sliding_window,
                        resources_.options_.fast_attention,
                        stream_.get());
                }
            }
            linear(workspace_.op_output_.data(), *attention->out, workspace_.hidden_.data(), 1,
                   resources_.shape_.hidden, layout.query_width(),
                   resources_.options_.fused_residuals && !common_layer.post_attention_norm ? 1.0f : 0.0f);
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
        if (common_layer.post_attention_norm) {
            launch_rmsnorm(workspace_.hidden_.data(), common_layer.post_attention_norm,
                           workspace_.hidden_.data(), 1, resources_.shape_.hidden,
                           resources_.shape_.norm_eps, stream_.get());
        }
        if (!resources_.options_.fused_residuals || common_layer.post_attention_norm) {
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
        if (resources_.shape_.final_logit_softcap > 0.0f) {
            launch_tanh_softcap(workspace_.logits_.data(), resources_.shape_.vocab_size,
                                resources_.shape_.final_logit_softcap, stream_.get());
        }
    }
    ++session_.position_;
    CELEG_CUDA(cudaMemcpyAsync(position_device_.data(), &session_.position_, sizeof(session_.position_),
                             cudaMemcpyHostToDevice, stream_.get()));
}

} // namespace celeg

