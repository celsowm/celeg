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
                                           const float* raw_embedding,
                                           const std::array<int32_t, 3>* rope_position) {
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
        initialize_per_layer_input_host(resources_.shape_.token_policy.pad_token_id);
    } else {
        resources_.weight_layout_->embed_token(
            token, workspace_.hidden_.data(), resources_.shape_.hidden, stream_.get());
        launch_scale(workspace_.hidden_.data(), resources_.shape_.hidden,
                     resources_.shape_.numerical_policy.embedding_multiplier, stream_.get());
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
                       1, resources_.shape_.hidden, resources_.shape_.numerical_policy.norm_eps,
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
            const int query_projection_width = attention->query->rows;
            const bool query_gate = layout.query_gate ||
                query_projection_width == 2 * layout.query_width();
            __nv_bfloat16* k = q + query_projection_width;
            __nv_bfloat16* v = k + layout.key_value_width();
            {
            auto native_fanout = native_fanout_scope(
                workspace_.normed_.data(), 1, resources_.shape_.hidden);
            linear(workspace_.normed_.data(), *attention->query, q,
                   1, query_projection_width, resources_.shape_.hidden);
            if (attention->key && attention->value) {
                linear(workspace_.normed_.data(), *attention->key, k,
                       1, layout.key_value_width(), resources_.shape_.hidden);
                linear(workspace_.normed_.data(), *attention->value, v,
                       1, layout.key_value_width(), resources_.shape_.hidden);
            }
            }
            if (layout.positional_encoding == PositionalEncodingKind::Rope) {
                if (resources_.shape_.mrope_interleaved) {
                    const auto& position = rope_position ? *rope_position : session_.next_rope_position_;
                    CELEG_CUDA(cudaMemcpyAsync(mrope_position_device_.data(), position.data(),
                                               sizeof(position), cudaMemcpyHostToDevice, stream_.get()));
                    launch_dynamic_mrope_qk_norm_rope(
                        q, attention->key ? k : nullptr, attention->q_norm, attention->k_norm,
                        layout.query_heads, layout.key_value_heads, layout.head_dim,
                        mrope_position_device_.data(), resources_.shape_.mrope_section[0],
                        resources_.shape_.mrope_section[1], resources_.shape_.mrope_section[2],
                        true, static_cast<float>(layout.rope_theta),
                        static_cast<float>(layout.rotary_fraction),
                        resources_.shape_.numerical_policy.norm_eps, layout.query_key_norm,
                        stream_.get());
                } else {
                    launch_dynamic_qk_norm_rope(
                        q, attention->key ? k : nullptr, attention->q_norm, attention->k_norm,
                        layout.query_heads, layout.key_value_heads, layout.head_dim,
                        session_.position_, static_cast<float>(layout.rope_theta),
                        static_cast<float>(layout.rotary_fraction), resources_.shape_.numerical_policy.norm_eps,
                        layout.query_key_norm, stream_.get());
                }
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
            if (query_gate) {
                launch_sigmoid_multiply(workspace_.op_output_.data(),
                                        q + layout.query_width(),
                                        layout.query_width(), stream_.get());
            }
            linear(workspace_.op_output_.data(), *attention->out, workspace_.hidden_.data(),
                   1, resources_.shape_.hidden, layout.query_width(),
                   resources_.options_.fused_residuals && !common_layer.post_attention_norm &&
                       resources_.shape_.mamba2_layer_count == 0 ? 1.0f : 0.0f);
            launch_scale(workspace_.hidden_.data(), resources_.shape_.hidden, resources_.shape_.numerical_policy.residual_multiplier,
                         stream_.get());
        } else if (GatedDeltaNetLayer* gated_delta = as_gated_delta_net(layer)) {
            const GatedDeltaNetSpec& spec = gated_delta->spec;
            const int qkv_width = 2 * spec.key_heads * spec.key_head_dim +
                spec.value_heads * spec.value_head_dim;
            const int value_width = spec.value_heads * spec.value_head_dim;
            linear(workspace_.normed_.data(), *gated_delta->qkv,
                   workspace_.gated_delta_qkv_.data(), 1, qkv_width,
                   resources_.shape_.hidden);
            linear(workspace_.normed_.data(), *gated_delta->z,
                   workspace_.gated_delta_z_.data(), 1, value_width,
                   resources_.shape_.hidden);
            linear(workspace_.normed_.data(), *gated_delta->b,
                   workspace_.gated_delta_b_.data(), 1, spec.value_heads,
                   resources_.shape_.hidden);
            linear(workspace_.normed_.data(), *gated_delta->a,
                   workspace_.gated_delta_a_.data(), 1, spec.value_heads,
                   resources_.shape_.hidden);
            launch_gated_delta_net(workspace_.gated_delta_qkv_.data(),
                workspace_.gated_delta_z_.data(), workspace_.gated_delta_b_.data(),
                workspace_.gated_delta_a_.data(), gated_delta->conv_weight,
                gated_delta->dt_bias, gated_delta->a_log, gated_delta->norm,
                gated_delta->conv_state.data(), gated_delta->recurrent_state.data(),
                workspace_.gated_delta_output_.data(), 1, spec.conv_kernel,
                spec.key_head_dim, spec.value_head_dim, spec.key_heads,
                spec.value_heads, resources_.shape_.numerical_policy.norm_eps,
                stream_.get());
            linear(workspace_.gated_delta_output_.data(), *gated_delta->out,
                   workspace_.hidden_.data(), 1, resources_.shape_.hidden,
                   value_width);
        } else if (Mamba2Layer* mamba = as_mamba2(layer)) {
            const Mamba2Spec& spec = mamba->spec;
            linear(workspace_.normed_.data(), *mamba->in,
                   workspace_.mamba_projected_.data(), 1,
                   2 * spec.intermediate_size + 2 * spec.group_count * spec.state_size +
                       spec.num_heads, resources_.shape_.hidden);
            launch_mamba2_step(workspace_.mamba_projected_.data(), mamba->conv_weight,
                                mamba->conv_bias, mamba->dt_bias, mamba->a_log, mamba->d,
                                mamba->conv_state.data(), mamba->ssm_state.data(),
                                workspace_.mamba_inner_.data(), spec.intermediate_size,
                                spec.state_size, spec.num_heads, spec.head_dim,
                                spec.group_count, spec.conv_kernel, stream_.get());
            launch_rmsnorm(workspace_.mamba_inner_.data(), mamba->norm,
                           workspace_.op_output_.data(), 1, spec.intermediate_size,
                           resources_.shape_.numerical_policy.norm_eps, stream_.get());
            launch_multiply(workspace_.op_output_.data(), workspace_.mamba_projected_.data(),
                            spec.intermediate_size, stream_.get());
            linear(workspace_.op_output_.data(), *mamba->out, workspace_.hidden_.data(),
                   1, resources_.shape_.hidden, spec.intermediate_size);
        } else if (MlpOnlyLayer* mlp = as_mlp_only(layer)) {
            linear(workspace_.normed_.data(), *mlp->up, workspace_.gate_up_.data(),
                   1, mlp->spec.intermediate_size, resources_.shape_.hidden);
            launch_relu2(workspace_.gate_up_.data(), workspace_.activated_.data(),
                         mlp->spec.intermediate_size, stream_.get());
            linear(workspace_.activated_.data(), *mlp->down, workspace_.hidden_.data(),
                   1, resources_.shape_.hidden, mlp->spec.intermediate_size);
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
                           resources_.shape_.numerical_policy.norm_eps, stream_.get());
        }
        if (!resources_.options_.fused_residuals || common_layer.post_attention_norm ||
            resources_.shape_.mamba2_layer_count > 0) {
            launch_residual_add(workspace_.hidden_.data(), workspace_.residual_.data(),
                                resources_.shape_.hidden, stream_.get());
        }
        if (resources_.shape_.mamba2_layer_count == 0) run_mlp_decode(common_layer, layer_idx);
        ++layer_idx;
    }
    if (resources_.mtp_.available()) {
        run_mtp_forward(token, rope_position);
    }
    if (compute_logits) {
        launch_rmsnorm(workspace_.hidden_.data(), resources_.final_norm_, workspace_.normed_.data(),
                        1, resources_.shape_.hidden, resources_.shape_.numerical_policy.norm_eps,
                        stream_.get());
        linear(workspace_.normed_.data(), *logits_weight(), workspace_.logits_.data(),
                1, resources_.shape_.vocab_size, resources_.shape_.hidden);
    launch_scale(workspace_.logits_.data(), resources_.shape_.vocab_size,
                 1.0f / resources_.shape_.numerical_policy.logits_divisor, stream_.get());
        if (resources_.shape_.numerical_policy.final_logit_softcap > 0.0f) {
            launch_tanh_softcap(workspace_.logits_.data(), resources_.shape_.vocab_size,
                                resources_.shape_.numerical_policy.final_logit_softcap, stream_.get());
        }
        finalize_mtp_verification();
    }
    ++session_.position_;
    if (!rope_position) {
        for (int32_t& value : session_.next_rope_position_) ++value;
    }
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
    if (resources_.mtp_.available()) {
        throw std::runtime_error("MTP is incompatible with paged KV execution");
    }
    resources_.weight_layout_->embed_token(
        token, workspace_.hidden_.data(), resources_.shape_.hidden, stream_.get());
    launch_scale(workspace_.hidden_.data(), resources_.shape_.hidden,
                 resources_.shape_.numerical_policy.embedding_multiplier, stream_.get());
    initialize_per_layer_input_host(token);

    for (int layer_index = 0; layer_index < resources_.shape_.num_hidden_layers; ++layer_index) {
        Layer& layer = resources_.layers_[static_cast<size_t>(layer_index)];
        LayerCommon& common_layer = common(layer);
        if (!resources_.options_.fused_residuals || common_layer.post_attention_norm) {
            CELEG_CUDA(cudaMemcpyAsync(workspace_.residual_.data(), workspace_.hidden_.data(), workspace_.hidden_.bytes(),
                                     cudaMemcpyDeviceToDevice, stream_.get()));
        }
        launch_rmsnorm(workspace_.hidden_.data(), common_layer.operator_norm, workspace_.normed_.data(),
                       1, resources_.shape_.hidden, resources_.shape_.numerical_policy.norm_eps, stream_.get());
        if (AttentionLayer* attention = as_attention(layer)) {
            const AttentionSpec& layout = attention->layout;
            __nv_bfloat16* q = workspace_.qkv_output_.data();
            const int query_projection_width = attention->query->rows;
            const bool query_gate = layout.query_gate ||
                query_projection_width == 2 * layout.query_width();
            __nv_bfloat16* k = q + query_projection_width;
            __nv_bfloat16* v = k + layout.key_value_width();
            {
            auto native_fanout = native_fanout_scope(
                workspace_.normed_.data(), 1, resources_.shape_.hidden);
            linear(workspace_.normed_.data(), *attention->query, q, 1,
                   query_projection_width, resources_.shape_.hidden);
            if (attention->key && attention->value) {
                linear(workspace_.normed_.data(), *attention->key, k, 1,
                       layout.key_value_width(), resources_.shape_.hidden);
                linear(workspace_.normed_.data(), *attention->value, v, 1,
                       layout.key_value_width(), resources_.shape_.hidden);
            }
            }
            if (layout.positional_encoding == PositionalEncodingKind::Rope) {
                launch_dynamic_qk_norm_rope(
                    q, attention->key ? k : nullptr, attention->q_norm, attention->k_norm,
                    layout.query_heads, layout.key_value_heads, layout.head_dim,
                    session_.position_, static_cast<float>(layout.rope_theta),
                    static_cast<float>(layout.rotary_fraction), resources_.shape_.numerical_policy.norm_eps,
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
            if (query_gate) {
                launch_sigmoid_multiply(workspace_.op_output_.data(),
                                        q + layout.query_width(),
                                        layout.query_width(), stream_.get());
            }
            linear(workspace_.op_output_.data(), *attention->out, workspace_.hidden_.data(), 1,
                   resources_.shape_.hidden, layout.query_width(),
                   resources_.options_.fused_residuals && !common_layer.post_attention_norm &&
                       resources_.shape_.mamba2_layer_count == 0 ? 1.0f : 0.0f);
            launch_scale(workspace_.hidden_.data(), resources_.shape_.hidden,
                         resources_.shape_.numerical_policy.residual_multiplier, stream_.get());
        } else if (GatedDeltaNetLayer* gated_delta = as_gated_delta_net(layer)) {
            const GatedDeltaNetSpec& spec = gated_delta->spec;
            const int qkv_width = 2 * spec.key_heads * spec.key_head_dim +
                spec.value_heads * spec.value_head_dim;
            const int value_width = spec.value_heads * spec.value_head_dim;
            linear(workspace_.normed_.data(), *gated_delta->qkv,
                   workspace_.gated_delta_qkv_.data(), 1, qkv_width,
                   resources_.shape_.hidden);
            linear(workspace_.normed_.data(), *gated_delta->z,
                   workspace_.gated_delta_z_.data(), 1, value_width,
                   resources_.shape_.hidden);
            linear(workspace_.normed_.data(), *gated_delta->b,
                   workspace_.gated_delta_b_.data(), 1, spec.value_heads,
                   resources_.shape_.hidden);
            linear(workspace_.normed_.data(), *gated_delta->a,
                   workspace_.gated_delta_a_.data(), 1, spec.value_heads,
                   resources_.shape_.hidden);
            launch_gated_delta_net(workspace_.gated_delta_qkv_.data(),
                workspace_.gated_delta_z_.data(), workspace_.gated_delta_b_.data(),
                workspace_.gated_delta_a_.data(), gated_delta->conv_weight,
                gated_delta->dt_bias, gated_delta->a_log, gated_delta->norm,
                gated_delta->conv_state.data(), gated_delta->recurrent_state.data(),
                workspace_.gated_delta_output_.data(), 1, spec.conv_kernel,
                spec.key_head_dim, spec.value_head_dim, spec.key_heads,
                spec.value_heads, resources_.shape_.numerical_policy.norm_eps,
                stream_.get());
            linear(workspace_.gated_delta_output_.data(), *gated_delta->out,
                   workspace_.hidden_.data(), 1, resources_.shape_.hidden,
                   value_width);
        } else if (Mamba2Layer* mamba = as_mamba2(layer)) {
            const Mamba2Spec& spec = mamba->spec;
            linear(workspace_.normed_.data(), *mamba->in, workspace_.mamba_projected_.data(), 1,
                   2 * spec.intermediate_size + 2 * spec.group_count * spec.state_size +
                       spec.num_heads, resources_.shape_.hidden);
            launch_mamba2_step(workspace_.mamba_projected_.data(), mamba->conv_weight,
                               mamba->conv_bias, mamba->dt_bias, mamba->a_log, mamba->d,
                               mamba->conv_state.data(), mamba->ssm_state.data(),
                               workspace_.mamba_inner_.data(), spec.intermediate_size,
                               spec.state_size, spec.num_heads, spec.head_dim,
                               spec.group_count, spec.conv_kernel, stream_.get());
            launch_rmsnorm(workspace_.mamba_inner_.data(), mamba->norm,
                           workspace_.op_output_.data(), 1, spec.intermediate_size,
                           resources_.shape_.numerical_policy.norm_eps, stream_.get());
            launch_multiply(workspace_.op_output_.data(), workspace_.mamba_projected_.data(),
                            spec.intermediate_size, stream_.get());
            linear(workspace_.op_output_.data(), *mamba->out, workspace_.hidden_.data(), 1,
                   resources_.shape_.hidden, spec.intermediate_size);
        } else if (MlpOnlyLayer* mlp = as_mlp_only(layer)) {
            linear(workspace_.normed_.data(), *mlp->up, workspace_.gate_up_.data(), 1,
                   mlp->spec.intermediate_size, resources_.shape_.hidden);
            launch_relu2(workspace_.gate_up_.data(), workspace_.activated_.data(),
                         mlp->spec.intermediate_size, stream_.get());
            linear(workspace_.activated_.data(), *mlp->down, workspace_.hidden_.data(), 1,
                   resources_.shape_.hidden, mlp->spec.intermediate_size);
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
                           resources_.shape_.numerical_policy.norm_eps, stream_.get());
        }
        if (!resources_.options_.fused_residuals || common_layer.post_attention_norm ||
            resources_.shape_.mamba2_layer_count > 0) {
            launch_residual_add(workspace_.hidden_.data(), workspace_.residual_.data(),
                                resources_.shape_.hidden, stream_.get());
        }
        if (resources_.shape_.mamba2_layer_count == 0) run_mlp_decode(common_layer, layer_index);
    }
    if (compute_logits) {
        launch_rmsnorm(workspace_.hidden_.data(), resources_.final_norm_, workspace_.normed_.data(), 1,
                       resources_.shape_.hidden, resources_.shape_.numerical_policy.norm_eps, stream_.get());
        linear(workspace_.normed_.data(), *logits_weight(), workspace_.logits_.data(), 1,
               resources_.shape_.vocab_size, resources_.shape_.hidden);
        launch_scale(workspace_.logits_.data(), resources_.shape_.vocab_size,
                     1.0f / resources_.shape_.numerical_policy.logits_divisor, stream_.get());
        if (resources_.shape_.numerical_policy.final_logit_softcap > 0.0f) {
            launch_tanh_softcap(workspace_.logits_.data(), resources_.shape_.vocab_size,
                                resources_.shape_.numerical_policy.final_logit_softcap, stream_.get());
        }
    }
    ++session_.position_;
    CELEG_CUDA(cudaMemcpyAsync(position_device_.data(), &session_.position_, sizeof(session_.position_),
                             cudaMemcpyHostToDevice, stream_.get()));
}

} // namespace celeg

