#include "celeg/detail/model/impl.hpp"
#include "celeg/backend/cuda/phase_profile.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/paged_kv.hpp"
#include "celeg/model/weights/layout.hpp"
#include "celeg/runtime/moe.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace celeg {
void Model::Impl::set_generation_config(GenerationConfig generation) {
    generation.validate();
    if (phase_ == SessionPhase::DecodePending) {
        throw std::runtime_error(
            "cannot change generation configuration during decode");
    }
    generation_ = generation;
    decode_graph_.reset();
    segmented_decode_graph_.reset();
}

PhaseProfile& decode_phase_profile() {
    static PhaseProfile instance;
    return instance;
}

void Model::Impl::enqueue_sampling() {
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

void Model::Impl::enqueue_decode_forward() {
    decode_phase_profile().begin(stream_.get());
    weight_layout_->embed_token_device(
        sampled_device_.data(), hidden_.data(), shape_.hidden,
        stream_.get());
    launch_scale(hidden_.data(), shape_.hidden, shape_.embedding_multiplier,
                 stream_.get());
    decode_phase_profile().end(DecodePhase::Embed, stream_.get());

    int layer_idx = 0;
    for (Layer& layer : layers_) {
        LayerCommon& common_layer = common(layer);
        if (!options_.fused_residuals) {
            CELEG_CUDA(cudaMemcpyAsync(
                residual_.data(), hidden_.data(), hidden_.bytes(),
                cudaMemcpyDeviceToDevice, stream_.get()));
        }
        decode_phase_profile().begin(stream_.get());
        launch_rmsnorm(hidden_.data(), common_layer.operator_norm, normed_.data(),
                       1, shape_.hidden, shape_.norm_eps,
                       stream_.get());
        decode_phase_profile().end(DecodePhase::Norm, stream_.get());
        if (AttentionLayer* attention = as_attention(layer)) {
            __nv_bfloat16* q = qkv_output_.data();
            __nv_bfloat16* k = q + shape_.q_width;
            __nv_bfloat16* v = k + shape_.kv_width;
            decode_phase_profile().begin(stream_.get());
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
            decode_phase_profile().end(DecodePhase::Projection, stream_.get());
            decode_phase_profile().begin(stream_.get());
            if (!shape_.query_key_norm) {
                launch_rope_strict_device(
                    q, k, rope_cos_.data(), rope_sin_.data(),
                    shape_.num_attention_heads, shape_.num_key_value_heads,
                    shape_.head_dim, position_device_.data(), stream_.get());
                const float ratio = shape_.attention_multiplier /
                    (1.0f / std::sqrt(static_cast<float>(shape_.head_dim)));
                launch_scale(q, shape_.q_width, ratio, stream_.get());
            } else if (options_.fast_attention) {
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
            decode_phase_profile().end(DecodePhase::RopeKv, stream_.get());
            decode_phase_profile().begin(stream_.get());
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
            decode_phase_profile().end(DecodePhase::Attention, stream_.get());
            decode_phase_profile().begin(stream_.get());
            linear(op_output_.data(), *attention->out, hidden_.data(),
                   1, shape_.hidden, shape_.hidden,
                   options_.fused_residuals ? 1.0f : 0.0f);
            launch_scale(hidden_.data(), shape_.hidden, shape_.residual_multiplier,
                         stream_.get());
            decode_phase_profile().end(DecodePhase::AttnOut, stream_.get());
        } else {
            ConvolutionLayer& convolution = *as_convolution(layer);
            decode_phase_profile().begin(stream_.get());
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
            decode_phase_profile().end(DecodePhase::Conv, stream_.get());
        }
        if (!options_.fused_residuals) {
            decode_phase_profile().begin(stream_.get());
            launch_residual_add(hidden_.data(), residual_.data(),
                                shape_.hidden, stream_.get());
            decode_phase_profile().end(DecodePhase::Other, stream_.get());
        }
        decode_phase_profile().begin(stream_.get());
        run_mlp_decode(common_layer, layer_idx);
        decode_phase_profile().end(DecodePhase::Mlp, stream_.get());
        ++layer_idx;
    }
    decode_phase_profile().begin(stream_.get());
    launch_rmsnorm(hidden_.data(), final_norm_, normed_.data(),
                    1, shape_.hidden, shape_.norm_eps,
                    stream_.get());
    linear(normed_.data(), *logits_weight(), logits_.data(),
            1, shape_.vocab_size, shape_.hidden);
    launch_scale(logits_.data(), shape_.vocab_size,
                 1.0f / shape_.logits_divisor, stream_.get());
    decode_phase_profile().end(DecodePhase::Logits, stream_.get());
}

void Model::Impl::enqueue_decode_step() {
    decode_phase_profile().begin(stream_.get());
    enqueue_sampling();
    decode_phase_profile().end(DecodePhase::Sampling, stream_.get());
    decode_phase_profile().count_step();
    enqueue_decode_forward();
    decode_phase_profile().begin(stream_.get());
    launch_increment_position(position_device_.data(), stream_.get());
    decode_phase_profile().end(DecodePhase::Other, stream_.get());
}

bool Model::Impl::use_segmented_attention(int host_position) const {
    return plan_.segmented_attention(host_position);
}

CudaGraphExec& Model::Impl::graph_for_attention(bool segmented) {
    return segmented ? segmented_decode_graph_ : decode_graph_;
}

void Model::Impl::capture_decode_graph(bool segmented) {
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

int32_t Model::Impl::decode() {
    decode_async_begin();
    return decode_async_finish();
}

void Model::Impl::decode_async_begin() {
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
    CELEG_CUDA(cudaMemcpyAsync(sampled_host_.data(), sampled_device_.data(),
                             sizeof(int32_t), cudaMemcpyDeviceToHost,
                             stream_.get()));
    phase_ = SessionPhase::DecodePending;
}

int32_t Model::Impl::decode_async_finish() {
    if (phase_ != SessionPhase::DecodePending) {
        throw std::runtime_error("decode_async_finish without begin");
    }
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));
    phase_ = SessionPhase::Ready;
    ++position_;
    const auto ended = std::chrono::steady_clock::now();
    metrics_.cumulative_decode_ms +=
        std::chrono::duration<double, std::milli>(
            ended - decode_async_begin_time_).count();
    ++metrics_.decoded_tokens;
    return sampled_host_.data()[0];
}

DecodeBenchmark Model::Impl::benchmark_decode(int warmup_steps,
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
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));

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

ModelMemoryStats Model::Impl::memory_stats() const {
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

ModelDiagnostics::ExpertOffloadStats Model::Impl::expert_offload_stats() const {
    ModelDiagnostics::ExpertOffloadStats stats;
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

void Model::Impl::release_local_kv_cache() {
    if (!local_kv_cache_available_) return;
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));
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

std::vector<float> Model::Impl::copy_logits() {
    if (phase_ != SessionPhase::Ready) {
        throw std::runtime_error("logits are unavailable before prefill");
    }
    std::vector<__nv_bfloat16> bf16_logits(shape_.vocab_size);
    CELEG_CUDA(cudaMemcpyAsync(
        bf16_logits.data(), logits_.data(), logits_.bytes(),
        cudaMemcpyDeviceToHost, stream_.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));

    std::vector<float> result(shape_.vocab_size);
    for (int i = 0; i < shape_.vocab_size; ++i) {
        result[static_cast<size_t>(i)] =
            __bfloat162float(bf16_logits[static_cast<size_t>(i)]);
    }
    return result;
}



} // namespace celeg


