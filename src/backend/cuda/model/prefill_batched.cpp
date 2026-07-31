#include "celeg/detail/model/impl.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/paged_kv.hpp"
#include "celeg/backend/cuda/phase_profile.hpp"
#include "celeg/model/weights/layout.hpp"
#include "celeg/runtime/moe.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace celeg {

void Model::Impl::prefill_batched(const std::vector<int32_t>& tokens) {
    reset();
    const int rows = static_cast<int>(tokens.size());
    allocate_prefill_workspace(rows);
    auto& prof = prefill_phase_profile();
    prof.count_step();

    CELEG_CUDA(cudaMemcpyAsync(prefill_tokens_.data(), tokens.data(),
                             tokens.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    prof.begin(stream_.get());
    launch_mark_seen_batch(prefill_tokens_.data(), rows, seen_tokens_.data(),
                           shape_.vocab_size, stream_.get());
    weight_layout_->embed_batch(
        prefill_tokens_.data(), rows, prefill_hidden_.data(),
        shape_.hidden, stream_.get());
    launch_scale(prefill_hidden_.data(), rows * shape_.hidden,
                 shape_.embedding_multiplier, stream_.get());
    prof.end(PrefillPhase::Embed, stream_.get());

    int layer_idx = 0;
    for (Layer& layer : layers_) {
        LayerCommon& common_layer = common(layer);
        if (!options_.fused_residuals) {
            CELEG_CUDA(cudaMemcpyAsync(
                prefill_residual_.data(), prefill_hidden_.data(),
                prefill_hidden_.bytes(), cudaMemcpyDeviceToDevice,
                stream_.get()));
        }
        prof.begin(stream_.get());
        launch_rmsnorm(prefill_hidden_.data(), common_layer.operator_norm,
                       prefill_normed_.data(), rows, shape_.hidden,
                       shape_.norm_eps, stream_.get());
        prof.end(PrefillPhase::Norm, stream_.get());

        if (AttentionLayer* attention = as_attention(layer)) {
            prof.begin(stream_.get());
            if (options_.fused_projections) {
                linear(prefill_normed_.data(), *attention->qkv,
                       prefill_qkv_.data(), rows, shape_.qkv_width,
                       shape_.hidden);
                launch_split_qkv_rows(
                    prefill_qkv_.data(), prefill_q_.data(), prefill_k_.data(),
                    prefill_v_.data(), rows, shape_.q_width, shape_.kv_width,
                    stream_.get());
            } else {
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
            }
            prof.end(PrefillPhase::QkvProj, stream_.get());

            prof.begin(stream_.get());
            if (!shape_.query_key_norm) {
                launch_rope_prefill(
                    prefill_q_.data(), prefill_k_.data(), rope_cos_.data(),
                    rope_sin_.data(), rows, shape_.num_attention_heads,
                    shape_.num_key_value_heads, shape_.head_dim, stream_.get());
                const float ratio = shape_.attention_multiplier /
                    (1.0f / std::sqrt(static_cast<float>(shape_.head_dim)));
                launch_scale(prefill_q_.data(), rows * shape_.q_width,
                             ratio, stream_.get());
            } else if (options_.fast_attention) {
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
            prof.end(PrefillPhase::RopeKv, stream_.get());

            prof.begin(stream_.get());
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
                    static const bool use_flash = []{
                        const char* f = std::getenv("CELEG_FLASH_ATTN");
                        return f != nullptr && f[0] != '\0' && f[0] != '0';
                    }();
                    if (use_flash) {
                        launch_gqa_prefill_flash(
                            prefill_q_.data(),
                            attention->key_cache.data(), attention->value_cache.data(),
                            prefill_op_output_.data(), rows,
                            shape_.num_attention_heads, shape_.num_key_value_heads,
                            shape_.head_dim, shape_.q_width, shape_.kv_width,
                            shape_.q_width, stream_.get());
                    } else if (rows <= kMaxGemmAttentionRows) {
                        launch_gqa_prefill_gemm(
                            gemm_->cublas().get(), prefill_q_.data(),
                            attention->key_cache.data(), attention->value_cache.data(),
                            prefill_op_output_.data(), prefill_attn_scores_.data(),
                            prefill_attn_probs_.data(), rows,
                            shape_.num_attention_heads, shape_.num_key_value_heads,
                            shape_.head_dim, shape_.q_width, shape_.kv_width,
                            shape_.q_width, stream_.get());
                    } else {
                        const int chunks = (rows + kPrefillAttnChunkTokens - 1) /
                            kPrefillAttnChunkTokens;
                        launch_gqa_prefill_segmented(
                            prefill_q_.data(), attention->key_cache.data(),
                            attention->value_cache.data(), prefill_op_output_.data(),
                            rows, shape_.num_attention_heads, shape_.num_key_value_heads,
                            shape_.head_dim, kPrefillAttnChunkTokens, chunks,
                            prefill_attn_partial_max_.data(),
                            prefill_attn_partial_denom_.data(),
                            prefill_attn_partial_accum_.data(), stream_.get());
                    }
                } else {
                    launch_gqa_prefill_strict(
                        prefill_q_.data(), attention->key_cache.data(),
                        attention->value_cache.data(), prefill_op_output_.data(),
                        rows, shape_.num_attention_heads, shape_.num_key_value_heads,
                        shape_.head_dim, stream_.get());
                }
            }
            prof.end(PrefillPhase::Attention, stream_.get());

            prof.begin(stream_.get());
            linear(prefill_op_output_.data(), *attention->out,
                   prefill_hidden_.data(), rows, shape_.hidden,
                   shape_.hidden, options_.fused_residuals ? 1.0f : 0.0f);
            launch_scale(prefill_hidden_.data(), rows * shape_.hidden,
                         shape_.residual_multiplier, stream_.get());
            prof.end(PrefillPhase::AttnOut, stream_.get());
        } else {
            ConvolutionLayer& convolution = *as_convolution(layer);
            prof.begin(stream_.get());
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
            prof.end(PrefillPhase::Conv, stream_.get());
        }

        if (!options_.fused_residuals) {
            prof.begin(stream_.get());
            launch_residual_add(prefill_hidden_.data(), prefill_residual_.data(),
                                rows * shape_.hidden, stream_.get());
            prof.end(PrefillPhase::Other, stream_.get());
        }
        prof.begin(stream_.get());
        run_mlp_prefill(common_layer, rows, layer_idx);
        prof.end(PrefillPhase::Mlp, stream_.get());
        ++layer_idx;
    }

    prof.begin(stream_.get());
    const __nv_bfloat16* last_hidden = prefill_hidden_.data() +
        static_cast<size_t>(rows - 1) * shape_.hidden;
    launch_rmsnorm(last_hidden, final_norm_, normed_.data(),
                   1, shape_.hidden, shape_.norm_eps,
                   stream_.get());
    linear(normed_.data(), *logits_weight(), logits_.data(),
           1, shape_.vocab_size, shape_.hidden);
    if (shape_.logits_divisor != 1.0f) {
        launch_scale(logits_.data(), shape_.vocab_size,
                     1.0f / shape_.logits_divisor, stream_.get());
    }
    prof.end(PrefillPhase::Logits, stream_.get());

    position_ = rows;
    CELEG_CUDA(cudaMemcpyAsync(position_device_.data(), &position_,
                             sizeof(position_), cudaMemcpyHostToDevice,
                             stream_.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));
    release_prefill_workspace();
    phase_ = SessionPhase::Ready;
}

} // namespace celeg

