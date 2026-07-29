#include "lfm/detail/model/impl.hpp"
#include "lfm/backend/cuda/kernels/kernels.cuh"
#include "lfm/backend/cuda/paged_kv.hpp"
#include "lfm/model/weights/layout.hpp"
#include "lfm/runtime/moe.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lfm {
void LfmModel::Impl::prefill_batched(const std::vector<int32_t>& tokens) {
    reset();
    const int rows = static_cast<int>(tokens.size());
    allocate_prefill_workspace(rows);

    LFM_CUDA(cudaMemcpyAsync(prefill_tokens_.data(), tokens.data(),
                             tokens.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    launch_mark_seen_batch(prefill_tokens_.data(), rows, seen_tokens_.data(),
                           shape_.vocab_size, stream_.get());
    weight_layout_->embed_batch(
        prefill_tokens_.data(), rows, prefill_hidden_.data(),
        shape_.hidden, stream_.get());

    int layer_idx = 0;
    for (Layer& layer : layers_) {
        LayerCommon& common_layer = common(layer);
        if (!options_.fused_residuals) {
            LFM_CUDA(cudaMemcpyAsync(
                prefill_residual_.data(), prefill_hidden_.data(),
                prefill_hidden_.bytes(), cudaMemcpyDeviceToDevice,
                stream_.get()));
        }
        launch_rmsnorm(prefill_hidden_.data(), common_layer.operator_norm,
                       prefill_normed_.data(), rows, shape_.hidden,
                       shape_.norm_eps, stream_.get());

        if (AttentionLayer* attention = as_attention(layer)) {
            const LinearWeight q_weight =
                slice_rows(*attention->qkv, 0, shape_.q_width);
            const LinearWeight k_weight = slice_rows(
                *attention->qkv, shape_.q_width, shape_.kv_width);
            const LinearWeight v_weight = slice_rows(
                *attention->qkv, shape_.q_width + shape_.kv_width,
                shape_.kv_width);
            linear(prefill_normed_.data(), q_weight, prefill_q_.data(),
                   rows, shape_.q_width, shape_.hidden);
            linear(prefill_normed_.data(), k_weight, prefill_k_.data(),
                   rows, shape_.kv_width, shape_.hidden);
            linear(prefill_normed_.data(), v_weight, prefill_v_.data(),
                   rows, shape_.kv_width, shape_.hidden);

            if (options_.fast_attention) {
                launch_qk_norm_rope_prefill_fast(
                    prefill_q_.data(), prefill_k_.data(),
                    attention->q_norm, attention->k_norm,
                    rope_cos_.data(), rope_sin_.data(), rows,
                    shape_.num_attention_heads, shape_.num_key_value_heads,
                    shape_.head_dim, shape_.norm_eps,
                    stream_.get());
            } else {
                launch_qk_norm_rope_prefill_strict(
                    prefill_q_.data(), prefill_k_.data(),
                    attention->q_norm, attention->k_norm,
                    rope_cos_.data(), rope_sin_.data(), rows,
                    shape_.num_attention_heads, shape_.num_key_value_heads,
                    shape_.head_dim, shape_.norm_eps,
                    stream_.get());
            }
            if (options_.kv_cache_mode == KvCacheMode::Int8) {
                launch_store_kv_int8_prefill(
                    prefill_k_.data(), prefill_v_.data(),
                    attention->key_cache_int8.data(), attention->value_cache_int8.data(),
                    attention->key_cache_scales.data(), attention->value_cache_scales.data(),
                    rows, shape_.num_key_value_heads, shape_.head_dim,
                    stream_.get());
                if (options_.fast_attention) {
                    launch_gqa_prefill_online_int8(
                        prefill_q_.data(), attention->key_cache_int8.data(),
                        attention->value_cache_int8.data(),
                        attention->key_cache_scales.data(),
                        attention->value_cache_scales.data(),
                        prefill_op_output_.data(), rows, shape_.num_attention_heads,
                        shape_.num_key_value_heads, shape_.head_dim, stream_.get());
                } else {
                    launch_gqa_prefill_strict_int8(
                        prefill_q_.data(), attention->key_cache_int8.data(),
                        attention->value_cache_int8.data(),
                        attention->key_cache_scales.data(),
                        attention->value_cache_scales.data(),
                        prefill_op_output_.data(), rows, shape_.num_attention_heads,
                        shape_.num_key_value_heads, shape_.head_dim, stream_.get());
                }
            } else {
                launch_store_kv_prefill(
                    prefill_k_.data(), prefill_v_.data(),
                    attention->key_cache.data(), attention->value_cache.data(),
                    rows, shape_.kv_width, stream_.get());
                if (options_.fast_attention) {
                    launch_gqa_prefill_online(
                        prefill_q_.data(), attention->key_cache.data(),
                        attention->value_cache.data(), prefill_op_output_.data(),
                        rows, shape_.num_attention_heads, shape_.num_key_value_heads,
                        shape_.head_dim, stream_.get());
                } else {
                    launch_gqa_prefill_strict(
                        prefill_q_.data(), attention->key_cache.data(),
                        attention->value_cache.data(), prefill_op_output_.data(),
                        rows, shape_.num_attention_heads, shape_.num_key_value_heads,
                        shape_.head_dim, stream_.get());
                }
            }
            linear(prefill_op_output_.data(), *attention->out,
                   prefill_hidden_.data(), rows, shape_.hidden,
                   shape_.hidden, options_.fused_residuals ? 1.0f : 0.0f);
        } else {
            ConvolutionLayer& convolution = *as_convolution(layer);
            linear(prefill_normed_.data(), *convolution.conv_in,
                   prefill_conv_projected_.data(), rows,
                   3 * shape_.hidden, shape_.hidden);
            launch_conv_prefill(
                prefill_conv_projected_.data(), convolution.conv_weight,
                convolution.conv_state.data(), prefill_op_output_.data(),
                rows, shape_.hidden, shape_.conv_cache,
                stream_.get());
            linear(prefill_op_output_.data(), *convolution.conv_out,
                   prefill_hidden_.data(), rows, shape_.hidden,
                   shape_.hidden, options_.fused_residuals ? 1.0f : 0.0f);
        }

        if (!options_.fused_residuals) {
            launch_residual_add(prefill_hidden_.data(), prefill_residual_.data(),
                                rows * shape_.hidden, stream_.get());
        }
        run_mlp_prefill(common_layer, rows, layer_idx);
        ++layer_idx;
    }

    const __nv_bfloat16* last_hidden = prefill_hidden_.data() +
        static_cast<size_t>(rows - 1) * shape_.hidden;
    launch_rmsnorm(last_hidden, final_norm_, normed_.data(),
                   1, shape_.hidden, shape_.norm_eps,
                   stream_.get());
    linear(normed_.data(), *logits_weight(), logits_.data(),
           1, shape_.vocab_size, shape_.hidden);

    position_ = rows;
    LFM_CUDA(cudaMemcpyAsync(position_device_.data(), &position_,
                             sizeof(position_), cudaMemcpyHostToDevice,
                             stream_.get()));
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    release_prefill_workspace();
    phase_ = SessionPhase::Ready;
}

void LfmModel::Impl::forward_token_host(int32_t token, bool compute_logits) {
    if (position_ >= max_context_) {
        throw std::runtime_error("context limit reached");
    }
    weight_layout_->embed_token(
        token, hidden_.data(), shape_.hidden, stream_.get());

    int layer_idx = 0;
    for (Layer& layer : layers_) {
        LayerCommon& common_layer = common(layer);
        if (!options_.fused_residuals) {
            LFM_CUDA(cudaMemcpyAsync(
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
            if (options_.fast_attention) {
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
    }
    ++position_;
}


void LfmModel::Impl::forward_token_paged_host(
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

    for (int layer_index = 0; layer_index < shape_.num_hidden_layers; ++layer_index) {
        Layer& layer = layers_[static_cast<size_t>(layer_index)];
        LayerCommon& common_layer = common(layer);
        if (!options_.fused_residuals) {
            LFM_CUDA(cudaMemcpyAsync(residual_.data(), hidden_.data(), hidden_.bytes(),
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
            if (options_.fast_attention) {
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
    }
    ++position_;
    LFM_CUDA(cudaMemcpyAsync(position_device_.data(), &position_, sizeof(position_),
                             cudaMemcpyHostToDevice, stream_.get()));
}

void LfmModel::Impl::prefill_chunk_paged(
    const std::vector<int32_t>& tokens, bool begin, bool finalize,
    PhysicalPagedKvCache& paged_kv,
    const std::vector<uint32_t>& page_table) {
    if (tokens.empty()) {
        throw std::invalid_argument("prefill_chunk_paged needs at least one token");
    }
    if (paged_kv.mode() != options_.kv_cache_mode) {
        throw std::invalid_argument("model and physical paged KV modes differ");
    }
    if (begin) {
        release_local_kv_cache();
        reset(false);
        metrics_ = {};
    } else if (position_ == 0) {
        throw std::runtime_error(
            "paged prefill continuation requires an initial chunk or prefix state");
    }
    if (position_ + static_cast<int>(tokens.size()) > max_context_) {
        throw std::invalid_argument("paged prefill chunks exceed max_context");
    }
    const size_t final_position =
        static_cast<size_t>(position_) + tokens.size();
    const size_t pages_needed =
        (final_position + static_cast<size_t>(paged_kv.page_tokens()) - 1) /
        static_cast<size_t>(paged_kv.page_tokens());
    if (page_table.size() < pages_needed ||
        page_table.size() > static_cast<size_t>(paged_kv.max_pages_per_request())) {
        throw std::invalid_argument("paged prefill page table has invalid length");
    }
    if (paged_page_table_.size() < page_table.size()) {
        paged_page_table_.reset(page_table.size());
    }
    LFM_CUDA(cudaMemcpyAsync(paged_page_table_.data(), page_table.data(),
                             page_table.size() * sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    if (paged_prefill_tokens_.size() < tokens.size()) {
        paged_prefill_tokens_.reset(tokens.size());
    }
    LFM_CUDA(cudaMemcpyAsync(paged_prefill_tokens_.data(), tokens.data(),
                             tokens.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    launch_mark_seen_batch(paged_prefill_tokens_.data(),
                           static_cast<int>(tokens.size()), seen_tokens_.data(),
                           shape_.vocab_size, stream_.get());
    phase_ = SessionPhase::Prefilling;
    const auto started = std::chrono::steady_clock::now();
    for (size_t i = 0; i < tokens.size(); ++i) {
        forward_token_paged_host(tokens[i], finalize && i + 1 == tokens.size(),
                                 paged_kv, paged_page_table_.data(),
                                 static_cast<int>(page_table.size()));
    }
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    const auto ended = std::chrono::steady_clock::now();
    metrics_.last_prefill_ms +=
        std::chrono::duration<double, std::milli>(ended - started).count();
    metrics_.prefill_tokens += tokens.size();
    if (finalize) {
        phase_ = SessionPhase::Ready;
        active_segmented_attention_ = use_segmented_attention(position_);
    }
}

void LfmModel::Impl::prefill_legacy(const std::vector<int32_t>& tokens) {
    reset();
    DeviceBuffer<int32_t> input(tokens.size());
    LFM_CUDA(cudaMemcpyAsync(input.data(), tokens.data(),
                             tokens.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    launch_mark_seen_batch(input.data(), static_cast<int>(tokens.size()),
                           seen_tokens_.data(), shape_.vocab_size,
                           stream_.get());
    for (size_t i = 0; i < tokens.size(); ++i) {
        forward_token_host(tokens[i], i + 1 == tokens.size());
    }
    LFM_CUDA(cudaMemcpyAsync(position_device_.data(), &position_,
                             sizeof(position_), cudaMemcpyHostToDevice,
                             stream_.get()));
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    phase_ = SessionPhase::Ready;
}

void LfmModel::Impl::prefill(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) {
        throw std::invalid_argument("prefill needs at least one token");
    }
    if (tokens.size() > static_cast<size_t>(max_context_)) {
        throw std::invalid_argument("prefill exceeds max_context");
    }
    const auto begin = std::chrono::steady_clock::now();
    if (options_.legacy_prefill) {
        prefill_legacy(tokens);
    } else {
        prefill_batched(tokens);
    }
    const auto end = std::chrono::steady_clock::now();
    metrics_.last_prefill_ms =
        std::chrono::duration<double, std::milli>(end - begin).count();
    metrics_.prefill_tokens = tokens.size();
    metrics_.cumulative_decode_ms = 0.0;
    metrics_.decoded_tokens = 0;
}

void LfmModel::Impl::prefill_chunk(const std::vector<int32_t>& tokens,
                              bool begin, bool finalize) {
    if (tokens.empty()) {
        throw std::invalid_argument("prefill_chunk needs at least one token");
    }
    if (begin) {
        reset();
        metrics_ = {};
    } else if (position_ == 0) {
        throw std::runtime_error(
            "prefill_chunk continuation requires an initial chunk");
    }
    if (phase_ == SessionPhase::Ready) {
        throw std::runtime_error("cannot append prefill after finalization");
    }
    if (position_ + static_cast<int>(tokens.size()) > max_context_) {
        throw std::invalid_argument("prefill chunks exceed max_context");
    }
    phase_ = SessionPhase::Prefilling;

    DeviceBuffer<int32_t> input(tokens.size());
    LFM_CUDA(cudaMemcpyAsync(input.data(), tokens.data(),
                             tokens.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    launch_mark_seen_batch(input.data(), static_cast<int>(tokens.size()),
                           seen_tokens_.data(), shape_.vocab_size,
                           stream_.get());
    const auto started = std::chrono::steady_clock::now();
    for (size_t i = 0; i < tokens.size(); ++i) {
        forward_token_host(tokens[i], finalize && i + 1 == tokens.size());
    }
    LFM_CUDA(cudaMemcpyAsync(position_device_.data(), &position_,
                             sizeof(position_), cudaMemcpyHostToDevice,
                             stream_.get()));
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    const auto ended = std::chrono::steady_clock::now();
    metrics_.last_prefill_ms +=
        std::chrono::duration<double, std::milli>(ended - started).count();
    metrics_.prefill_tokens += tokens.size();
    if (finalize) {
        phase_ = SessionPhase::Ready;
        active_segmented_attention_ = use_segmented_attention(position_);
    }
}

void LfmModel::Impl::set_generation_config(GenerationConfig generation) {
    generation.validate();
    if (phase_ == SessionPhase::DecodePending) {
        throw std::runtime_error(
            "cannot change generation configuration during decode");
    }
    generation_ = generation;
    decode_graph_.reset();
    segmented_decode_graph_.reset();
}

void LfmModel::Impl::enqueue_sampling() {
    const int effective_top_k = generation_.greedy() ? 1 : generation_.top_k;
    const float effective_temperature =
        generation_.temperature > 0.0f ? generation_.temperature : 1.0f;
    if (plan_.sampling_kernel() == SamplingKernelKind::Fused) {
        launch_fused_sample_topk(
            logits_.data(), seen_tokens_.data(), sampling_scores_.data(),
            topk_values_.data(), topk_indices_.data(), shape_.vocab_size,
            effective_temperature, generation_.repetition_penalty,
            effective_top_k,
            generation_.greedy() ? 1.0f : generation_.top_p,
            rng_state_.data(), sampled_device_.data(), stream_.get());
    } else {
        launch_prepare_sampling_scores(
            logits_.data(), seen_tokens_.data(), sampling_scores_.data(),
            shape_.vocab_size, effective_temperature,
            generation_.repetition_penalty, stream_.get());
        for (int rank = 0; rank < effective_top_k; ++rank) {
            launch_select_topk(sampling_scores_.data(), topk_values_.data(),
                               topk_indices_.data(), rank, shape_.vocab_size,
                               stream_.get());
        }
        launch_sample_topk(topk_values_.data(), topk_indices_.data(),
                           effective_top_k,
                           generation_.greedy() ? 1.0f : generation_.top_p,
                           rng_state_.data(), sampled_device_.data(),
                           stream_.get());
        launch_mark_seen(sampled_device_.data(), seen_tokens_.data(),
                         shape_.vocab_size, stream_.get());
    }
}

void LfmModel::Impl::enqueue_decode_forward() {
    weight_layout_->embed_token_device(
        sampled_device_.data(), hidden_.data(), shape_.hidden,
        stream_.get());

    int layer_idx = 0;
    for (Layer& layer : layers_) {
        LayerCommon& common_layer = common(layer);
        if (!options_.fused_residuals) {
            LFM_CUDA(cudaMemcpyAsync(
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
            if (options_.fast_attention) {
                launch_qk_norm_rope_fast_device(
                    q, k, attention->q_norm, attention->k_norm,
                    rope_cos_.data(), rope_sin_.data(),
                    shape_.num_attention_heads, shape_.num_key_value_heads,
                    shape_.head_dim, position_device_.data(),
                    shape_.norm_eps, stream_.get());
            } else {
                launch_qk_norm_rope_strict_device(
                    q, k, attention->q_norm, attention->k_norm,
                    rope_cos_.data(), rope_sin_.data(),
                    shape_.num_attention_heads, shape_.num_key_value_heads,
                    shape_.head_dim, position_device_.data(),
                    shape_.norm_eps, stream_.get());
            }
            if (options_.kv_cache_mode == KvCacheMode::Int8) {
                launch_store_kv_int8_device(
                    k, v, attention->key_cache_int8.data(),
                    attention->value_cache_int8.data(),
                    attention->key_cache_scales.data(),
                    attention->value_cache_scales.data(), position_device_.data(),
                    shape_.num_key_value_heads, shape_.head_dim, stream_.get());
                if (active_segmented_attention_) {
                    launch_gqa_decode_segmented_int8_device(
                        q, attention->key_cache_int8.data(),
                        attention->value_cache_int8.data(),
                        attention->key_cache_scales.data(),
                        attention->value_cache_scales.data(), op_output_.data(),
                        position_device_.data(), shape_.num_attention_heads,
                        shape_.num_key_value_heads, shape_.head_dim,
                        options_.attention_chunk_tokens, attention_chunks_,
                        attention_partial_max_.data(),
                        attention_partial_denom_.data(),
                        attention_partial_accum_.data(), stream_.get());
                } else if (options_.fast_attention) {
                    launch_gqa_decode_online_int8_device(
                        q, attention->key_cache_int8.data(),
                        attention->value_cache_int8.data(),
                        attention->key_cache_scales.data(),
                        attention->value_cache_scales.data(), op_output_.data(),
                        position_device_.data(), shape_.num_attention_heads,
                        shape_.num_key_value_heads, shape_.head_dim, stream_.get());
                } else {
                    launch_gqa_decode_strict_int8_device(
                        q, attention->key_cache_int8.data(),
                        attention->value_cache_int8.data(),
                        attention->key_cache_scales.data(),
                        attention->value_cache_scales.data(), op_output_.data(),
                        position_device_.data(), shape_.num_attention_heads,
                        shape_.num_key_value_heads, shape_.head_dim, stream_.get());
                }
            } else {
                launch_store_kv_device(
                    k, v, attention->key_cache.data(), attention->value_cache.data(),
                    position_device_.data(), shape_.kv_width, stream_.get());
                if (active_segmented_attention_) {
                    launch_gqa_decode_segmented_device(
                        q, attention->key_cache.data(), attention->value_cache.data(),
                        op_output_.data(), position_device_.data(),
                        shape_.num_attention_heads, shape_.num_key_value_heads,
                        shape_.head_dim, options_.attention_chunk_tokens,
                        attention_chunks_, attention_partial_max_.data(),
                        attention_partial_denom_.data(),
                        attention_partial_accum_.data(), stream_.get());
                } else if (options_.fast_attention) {
                    launch_gqa_decode_online_device(
                        q, attention->key_cache.data(), attention->value_cache.data(),
                        op_output_.data(), position_device_.data(),
                        shape_.num_attention_heads, shape_.num_key_value_heads,
                        shape_.head_dim, stream_.get());
                } else {
                    launch_gqa_decode_strict_device(
                        q, attention->key_cache.data(), attention->value_cache.data(),
                        op_output_.data(), position_device_.data(),
                        shape_.num_attention_heads, shape_.num_key_value_heads,
                        shape_.head_dim, stream_.get());
                }
            }
            linear(op_output_.data(), *attention->out, hidden_.data(),
                   1, shape_.hidden, shape_.hidden,
                   options_.fused_residuals ? 1.0f : 0.0f);
        } else {
            ConvolutionLayer& convolution = *as_convolution(layer);
            linear(normed_.data(), *convolution.conv_in, conv_projected_.data(),
                   1, 3 * shape_.hidden, shape_.hidden);
            launch_conv_decode_device(
                conv_projected_.data(), convolution.conv_weight,
                convolution.conv_state.data(), op_output_.data(),
                shape_.hidden, shape_.conv_cache,
                position_device_.data(), stream_.get());
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
    launch_rmsnorm(hidden_.data(), final_norm_, normed_.data(),
                    1, shape_.hidden, shape_.norm_eps,
                    stream_.get());
    linear(normed_.data(), *logits_weight(), logits_.data(),
            1, shape_.vocab_size, shape_.hidden);
}

void LfmModel::Impl::enqueue_decode_step() {
    enqueue_sampling();
    enqueue_decode_forward();
    launch_increment_position(position_device_.data(), stream_.get());
}

bool LfmModel::Impl::use_segmented_attention(int host_position) const {
    return plan_.segmented_attention(host_position);
}

CudaGraphExec& LfmModel::Impl::graph_for_attention(bool segmented) {
    return segmented ? segmented_decode_graph_ : decode_graph_;
}

void LfmModel::Impl::capture_decode_graph(bool segmented) {
    if (!options_.cuda_graph) return;
    CudaGraphExec& graph = graph_for_attention(segmented);
    if (graph.ready()) return;
    active_segmented_attention_ = segmented;
    graph.capture_begin(stream_.get());
    try {
        enqueue_decode_step();
        graph.capture_end(stream_.get());
    } catch (...) {
        graph.abort_capture(stream_.get());
        throw;
    }
}

int32_t LfmModel::Impl::decode() {
    decode_async_begin();
    return decode_async_finish();
}

void LfmModel::Impl::decode_async_begin() {
    if (!local_kv_cache_available_) {
        throw std::runtime_error(
            "lane decode is unavailable after transferring KV to the shared paged cache");
    }
    if (phase_ != SessionPhase::Ready) {
        throw std::runtime_error("decode requires a successful prefill");
    }
    if (phase_ == SessionPhase::DecodePending) {
        throw std::runtime_error("decode_async_begin called twice");
    }
    if (position_ >= max_context_) {
        throw std::runtime_error("context limit reached");
    }
    decode_async_begin_time_ = std::chrono::steady_clock::now();
    const bool segmented = use_segmented_attention(position_);
    active_segmented_attention_ = segmented;
    if (options_.cuda_graph) {
        CudaGraphExec& graph = graph_for_attention(segmented);
        if (!graph.ready()) capture_decode_graph(segmented);
        graph.launch(stream_.get());
    } else {
        enqueue_decode_step();
    }
    LFM_CUDA(cudaMemcpyAsync(sampled_host_.data(), sampled_device_.data(),
                             sizeof(int32_t), cudaMemcpyDeviceToHost,
                             stream_.get()));
    phase_ = SessionPhase::DecodePending;
}

int32_t LfmModel::Impl::decode_async_finish() {
    if (phase_ != SessionPhase::DecodePending) {
        throw std::runtime_error("decode_async_finish without begin");
    }
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    phase_ = SessionPhase::Ready;
    ++position_;
    const auto ended = std::chrono::steady_clock::now();
    metrics_.cumulative_decode_ms +=
        std::chrono::duration<double, std::milli>(
            ended - decode_async_begin_time_).count();
    ++metrics_.decoded_tokens;
    return sampled_host_.data()[0];
}

DecodeBenchmark LfmModel::Impl::benchmark_decode(int warmup_steps,
                                                int measured_steps) {
    if (phase_ != SessionPhase::Ready) {
        throw std::runtime_error("benchmark_decode requires a successful prefill");
    }
    if (warmup_steps < 0 || measured_steps <= 0) {
        throw std::invalid_argument(
            "benchmark steps require warmup >= 0 and measured > 0");
    }
    const int total_steps = warmup_steps + measured_steps;
    if (position_ + total_steps > max_context_) {
        throw std::runtime_error("decode benchmark exceeds context limit");
    }
    if (options_.cuda_graph) {
        const int final_position = position_ + total_steps - 1;
        const bool starts_segmented = use_segmented_attention(position_);
        const bool ends_segmented = use_segmented_attention(final_position);
        if (!starts_segmented || !ends_segmented) capture_decode_graph(false);
        if (starts_segmented || ends_segmented) capture_decode_graph(true);
    }

    int simulated_position = position_;
    auto launch_step = [&]() {
        const bool segmented = use_segmented_attention(simulated_position);
        active_segmented_attention_ = segmented;
        if (options_.cuda_graph) {
            CudaGraphExec& graph = graph_for_attention(segmented);
            if (!graph.ready()) capture_decode_graph(segmented);
            graph.launch(stream_.get());
        } else {
            enqueue_decode_step();
        }
        ++simulated_position;
    };
    for (int i = 0; i < warmup_steps; ++i) launch_step();
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));

    CudaEvent begin;
    CudaEvent end;
    begin.record(stream_.get());
    for (int i = 0; i < measured_steps; ++i) launch_step();
    end.record(stream_.get());
    end.synchronize();

    position_ += total_steps;
    DecodeBenchmark result;
    result.warmup_steps = warmup_steps;
    result.measured_steps = measured_steps;
    result.elapsed_ms = CudaEvent::elapsed_ms(begin, end);
    return result;
}

ModelMemoryStats LfmModel::Impl::memory_stats() const {
    ModelMemoryStats stats;
    stats.weights = weights_ ? weights_->memory_bytes() : 0;
    for (const Layer& layer : layers_) {
        if (const AttentionLayer* attention = as_attention(layer)) {
            stats.kv_cache += attention->key_cache.bytes() + attention->value_cache.bytes() +
                attention->key_cache_int8.bytes() + attention->value_cache_int8.bytes() +
                attention->key_cache_scales.bytes() + attention->value_cache_scales.bytes();
        } else {
            stats.conv_state += as_convolution(layer)->conv_state.bytes();
        }
    }
    stats.rope_tables = rope_cos_.bytes() + rope_sin_.bytes();
    stats.activations =
        hidden_.bytes() + residual_.bytes() + normed_.bytes() +
        op_output_.bytes() + qkv_output_.bytes() + conv_projected_.bytes() +
        gate_up_.bytes() + activated_.bytes() + mlp_output_.bytes() +
        logits_.bytes() + paged_page_table_.bytes() +
        paged_prefill_tokens_.bytes() + prefill_tokens_.bytes() +
        prefill_hidden_.bytes() +
        prefill_residual_.bytes() + prefill_normed_.bytes() +
        prefill_op_output_.bytes() + prefill_q_.bytes() + prefill_k_.bytes() +
        prefill_v_.bytes() + prefill_conv_projected_.bytes() +
        prefill_gate_up_.bytes() + prefill_activated_.bytes() +
        prefill_mlp_output_.bytes();
    stats.sampling =
        position_device_.bytes() + sampled_device_.bytes() +
        seen_tokens_.bytes() + sampling_scores_.bytes() +
        topk_values_.bytes() + topk_indices_.bytes() + rng_state_.bytes();
    stats.matmul_workspace = gemm_ ? gemm_->workspace_bytes() : 0;
    stats.attention_workspace = attention_partial_max_.bytes() +
        attention_partial_denom_.bytes() + attention_partial_accum_.bytes();
    return stats;
}

LfmDiagnostics::ExpertOffloadStats LfmModel::Impl::expert_offload_stats() const {
    LfmDiagnostics::ExpertOffloadStats stats;
    if (!expert_offload_plan_.enabled) {
        stats.hit_rate = -1.0;
        return stats;
    }
    stats.experts_per_layer = expert_offload_plan_.experts_per_layer;
    stats.host_experts_per_layer = expert_offload_plan_.host_experts_per_layer;
    uint64_t hits = 0, misses = 0;
    for (const auto& cache : expert_caches_) {
        if (cache) {
            hits += cache->hits();
            misses += cache->misses();
        }
    }
    stats.hits = hits;
    stats.misses = misses;
    const uint64_t total = hits + misses;
    stats.hit_rate = total == 0 ? 0.0 : static_cast<double>(hits) / total;
    return stats;
}

void LfmModel::Impl::release_local_kv_cache() {
    if (!local_kv_cache_available_) return;
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    for (Layer& layer : layers_) {
        AttentionLayer* attention = as_attention(layer);
        if (!attention) continue;
        attention->key_cache.reset(0);
        attention->value_cache.reset(0);
        attention->key_cache_int8.reset(0);
        attention->value_cache_int8.reset(0);
        attention->key_cache_scales.reset(0);
        attention->value_cache_scales.reset(0);
    }
    local_kv_cache_available_ = false;
    decode_graph_.reset();
    segmented_decode_graph_.reset();
}

std::vector<float> LfmModel::Impl::copy_logits() {
    if (phase_ != SessionPhase::Ready) {
        throw std::runtime_error("logits are unavailable before prefill");
    }
    std::vector<__nv_bfloat16> bf16_logits(shape_.vocab_size);
    LFM_CUDA(cudaMemcpyAsync(
        bf16_logits.data(), logits_.data(), logits_.bytes(),
        cudaMemcpyDeviceToHost, stream_.get()));
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));

    std::vector<float> result(shape_.vocab_size);
    for (int i = 0; i < shape_.vocab_size; ++i) {
        result[static_cast<size_t>(i)] =
            __bfloat162float(bf16_logits[static_cast<size_t>(i)]);
    }
    return result;
}



} // namespace lfm



