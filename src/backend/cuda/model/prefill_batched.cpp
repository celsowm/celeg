#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/paged_kv.hpp"
#include "celeg/backend/cuda/phase_profile.hpp"
#include "celeg/backend/cuda/weight_layout.hpp"
#include "celeg/backend/cuda/moe.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace celeg {

void CudaCompiledModel::prefill_batched(const std::vector<int32_t>& tokens) {
    reset();
    const int rows = static_cast<int>(tokens.size());
    allocate_prefill_workspace(rows);
    auto& prof = prefill_phase_profile();
    prof.count_step();

    CELEG_CUDA(cudaMemcpyAsync(workspace_.prefill_tokens_.data(), tokens.data(),
                             tokens.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    prof.begin(stream_.get());
    launch_mark_seen_batch(workspace_.prefill_tokens_.data(), rows, sampling_.seen_tokens.data(),
                           resources_.shape_.vocab_size, stream_.get());
    resources_.weight_layout_->embed_batch(
        workspace_.prefill_tokens_.data(), rows, workspace_.prefill_hidden_.data(),
        resources_.shape_.hidden, stream_.get());
    launch_scale(workspace_.prefill_hidden_.data(), rows * resources_.shape_.hidden,
                 resources_.shape_.embedding_multiplier, stream_.get());
    initialize_per_layer_input_batch(workspace_.prefill_tokens_.data(), rows);
    prof.end(PrefillPhase::Embed, stream_.get());

    int layer_idx = 0;
    for (Layer& layer : resources_.layers_) {
        LayerCommon& common_layer = common(layer);
        if (!resources_.options_.fused_residuals || common_layer.post_attention_norm) {
            CELEG_CUDA(cudaMemcpyAsync(
                workspace_.prefill_residual_.data(), workspace_.prefill_hidden_.data(),
                workspace_.prefill_hidden_.bytes(), cudaMemcpyDeviceToDevice,
                stream_.get()));
        }
        prof.begin(stream_.get());
        launch_rmsnorm(workspace_.prefill_hidden_.data(), common_layer.operator_norm,
                       workspace_.prefill_normed_.data(), rows, resources_.shape_.hidden,
                       resources_.shape_.norm_eps, stream_.get());
        prof.end(PrefillPhase::Norm, stream_.get());

        if (AttentionLayer* attention = as_attention(layer)) {
            const AttentionSpec& layout = attention->layout;
            AttentionLayer* owner = attention;
            if (attention->kv_owner_layer >= 0) {
                owner = as_attention(resources_.layers_.at(
                    static_cast<size_t>(attention->kv_owner_layer)));
                if (!owner) throw std::logic_error("CUDA shared KV owner is not attention");
            }
            const AttentionSpec& owner_layout = owner->layout;
            prof.begin(stream_.get());
            linear(workspace_.prefill_normed_.data(), *attention->query,
                   workspace_.prefill_q_.data(), rows, layout.query_width(),
                   resources_.shape_.hidden);
            if (attention->key && attention->value) {
                linear(workspace_.prefill_normed_.data(), *attention->key,
                       workspace_.prefill_k_.data(), rows, layout.key_value_width(),
                       resources_.shape_.hidden);
                linear(workspace_.prefill_normed_.data(), *attention->value,
                       workspace_.prefill_v_.data(), rows, layout.key_value_width(),
                       resources_.shape_.hidden);
            }
            prof.end(PrefillPhase::QkvProj, stream_.get());

            prof.begin(stream_.get());
            launch_dynamic_qk_norm_rope_prefill(
                workspace_.prefill_q_.data(), attention->key ? workspace_.prefill_k_.data() : nullptr,
                attention->q_norm, attention->k_norm, rows, layout.query_heads,
                layout.key_value_heads, layout.head_dim, static_cast<float>(layout.rope_theta),
                static_cast<float>(layout.rotary_fraction), resources_.shape_.norm_eps,
                layout.query_key_norm, stream_.get());
            prof.end(PrefillPhase::RopeKv, stream_.get());

            prof.begin(stream_.get());
            if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
                if (attention->key && attention->value) launch_store_kv_int8_prefill(
                    workspace_.prefill_k_.data(), workspace_.prefill_v_.data(),
                    owner->key_cache_int8.data(), owner->value_cache_int8.data(),
                    owner->key_cache_scales.data(), owner->value_cache_scales.data(),
                    rows, owner_layout.key_value_heads, owner_layout.head_dim,
                    stream_.get());
                if (resources_.options_.fast_attention) {
                    launch_gqa_prefill_online_int8(
                        workspace_.prefill_q_.data(), owner->key_cache_int8.data(), owner->value_cache_int8.data(),
                        owner->key_cache_scales.data(), owner->value_cache_scales.data(),
                        workspace_.prefill_op_output_.data(), rows, layout.query_heads,
                        owner_layout.key_value_heads, owner_layout.head_dim,
                        layout.sliding_window, stream_.get());
                } else {
                    launch_gqa_prefill_strict_int8(
                        workspace_.prefill_q_.data(), owner->key_cache_int8.data(), owner->value_cache_int8.data(),
                        owner->key_cache_scales.data(), owner->value_cache_scales.data(),
                        workspace_.prefill_op_output_.data(), rows, layout.query_heads,
                        owner_layout.key_value_heads, owner_layout.head_dim,
                        layout.sliding_window, stream_.get());
                }
            } else {
                if (attention->key && attention->value) launch_store_kv_prefill(
                    workspace_.prefill_k_.data(), workspace_.prefill_v_.data(),
                    owner->key_cache.data(), owner->value_cache.data(),
                    rows, owner_layout.key_value_width(), stream_.get());
                if (resources_.options_.fast_attention) {
                    static const bool use_flash = []{
                        const char* f = std::getenv("CELEG_FLASH_ATTN");
                        return f != nullptr && f[0] != '\0' && f[0] != '0';
                    }();
                    // The batched GEMM path is only valid for the narrow-head
                    // layouts it was tuned for.  With head_dim=128 its
                    // strided GQA batches can corrupt the KV state; use the
                    // tiled path automatically for wider heads so the next
                    // decode step sees a valid cache.  This is a kernel
                    // capability boundary, not an architecture dispatch.
                    const bool flash_supported = owner_layout.head_dim <= 128;
                    if ((use_flash && flash_supported) ||
                        (owner_layout.head_dim > 64 && flash_supported)) {
                        launch_gqa_prefill_flash(
                            workspace_.prefill_q_.data(),
                            owner->key_cache.data(), owner->value_cache.data(),
                            workspace_.prefill_op_output_.data(), rows,
                            layout.query_heads, owner_layout.key_value_heads,
                            owner_layout.head_dim, layout.query_width(), owner_layout.key_value_width(),
                            layout.query_width(), layout.sliding_window, stream_.get());
                    } else if (rows <= kMaxGemmAttentionRows) {
                        launch_gqa_prefill_gemm(
                            gemm_->cublas().get(), workspace_.prefill_q_.data(),
                            owner->key_cache.data(), owner->value_cache.data(),
                            workspace_.prefill_op_output_.data(), workspace_.prefill_attn_scores_.data(),
                            workspace_.prefill_attn_probs_.data(), rows,
                            layout.query_heads, owner_layout.key_value_heads,
                            owner_layout.head_dim, layout.query_width(), owner_layout.key_value_width(),
                            layout.query_width(), layout.sliding_window, stream_.get());
                    } else {
                        const int chunks = (rows + kPrefillAttnChunkTokens - 1) /
                            kPrefillAttnChunkTokens;
                        launch_gqa_prefill_segmented(
                            workspace_.prefill_q_.data(), owner->key_cache.data(),
                            owner->value_cache.data(), workspace_.prefill_op_output_.data(),
                            rows, layout.query_heads, owner_layout.key_value_heads,
                            owner_layout.head_dim, kPrefillAttnChunkTokens, chunks,
                            layout.sliding_window,
                            workspace_.prefill_attn_partial_max_.data(),
                            workspace_.prefill_attn_partial_denom_.data(),
                            workspace_.prefill_attn_partial_accum_.data(), stream_.get());
                    }
                } else {
                    launch_gqa_prefill_strict(
                        workspace_.prefill_q_.data(), owner->key_cache.data(),
                        owner->value_cache.data(), workspace_.prefill_op_output_.data(),
                        rows, layout.query_heads, owner_layout.key_value_heads,
                        owner_layout.head_dim, layout.sliding_window, stream_.get());
                }
            }
            prof.end(PrefillPhase::Attention, stream_.get());

            prof.begin(stream_.get());
            linear(workspace_.prefill_op_output_.data(), *attention->out,
                   workspace_.prefill_hidden_.data(), rows, resources_.shape_.hidden,
                   layout.query_width(),
                   resources_.options_.fused_residuals && !common_layer.post_attention_norm ? 1.0f : 0.0f);
            launch_scale(workspace_.prefill_hidden_.data(), rows * resources_.shape_.hidden,
                         resources_.shape_.residual_multiplier, stream_.get());
            prof.end(PrefillPhase::AttnOut, stream_.get());
        } else {
            ConvolutionLayer& convolution = *as_convolution(layer);
            prof.begin(stream_.get());
            linear(workspace_.prefill_normed_.data(), *convolution.conv_in,
                   workspace_.prefill_conv_projected_.data(), rows,
                   3 * resources_.shape_.hidden, resources_.shape_.hidden);
            launch_conv_prefill(
                workspace_.prefill_conv_projected_.data(), convolution.conv_weight,
                convolution.conv_state.data(), workspace_.prefill_op_output_.data(),
                rows, resources_.shape_.hidden, resources_.shape_.conv_cache,
                stream_.get());
            linear(workspace_.prefill_op_output_.data(), *convolution.conv_out,
                   workspace_.prefill_hidden_.data(), rows, resources_.shape_.hidden,
                   resources_.shape_.hidden, resources_.options_.fused_residuals ? 1.0f : 0.0f);
            prof.end(PrefillPhase::Conv, stream_.get());
        }

        if (common_layer.post_attention_norm) {
            launch_rmsnorm(workspace_.prefill_hidden_.data(), common_layer.post_attention_norm,
                           workspace_.prefill_hidden_.data(), rows, resources_.shape_.hidden,
                           resources_.shape_.norm_eps, stream_.get());
        }
        if (!resources_.options_.fused_residuals || common_layer.post_attention_norm) {
            prof.begin(stream_.get());
            launch_residual_add(workspace_.prefill_hidden_.data(), workspace_.prefill_residual_.data(),
                                rows * resources_.shape_.hidden, stream_.get());
            prof.end(PrefillPhase::Other, stream_.get());
        }
        prof.begin(stream_.get());
        run_mlp_prefill(common_layer, rows, layer_idx);
        prof.end(PrefillPhase::Mlp, stream_.get());
        ++layer_idx;
    }

    prof.begin(stream_.get());
    const __nv_bfloat16* last_hidden = workspace_.prefill_hidden_.data() +
        static_cast<size_t>(rows - 1) * resources_.shape_.hidden;
    launch_rmsnorm(last_hidden, resources_.final_norm_, workspace_.normed_.data(),
                   1, resources_.shape_.hidden, resources_.shape_.norm_eps,
                   stream_.get());
    linear(workspace_.normed_.data(), *logits_weight(), workspace_.logits_.data(),
           1, resources_.shape_.vocab_size, resources_.shape_.hidden);
    if (resources_.shape_.logits_divisor != 1.0f) {
    launch_scale(workspace_.logits_.data(), resources_.shape_.vocab_size,
                 1.0f / resources_.shape_.logits_divisor, stream_.get());
    if (resources_.shape_.final_logit_softcap > 0.0f) {
        launch_tanh_softcap(workspace_.logits_.data(), resources_.shape_.vocab_size,
                            resources_.shape_.final_logit_softcap, stream_.get());
    }
    }
    prof.end(PrefillPhase::Logits, stream_.get());

    session_.position_ = rows;
    CELEG_CUDA(cudaMemcpyAsync(position_device_.data(), &session_.position_,
                             sizeof(session_.position_), cudaMemcpyHostToDevice,
                             stream_.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));
    release_prefill_workspace();
    session_.phase_ = SessionPhase::Ready;
}

} // namespace celeg

