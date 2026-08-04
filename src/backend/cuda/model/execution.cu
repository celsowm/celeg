#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/phase_profile.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/paged_kv.hpp"
#include "celeg/backend/cuda/weight_layout.hpp"
#include "celeg/backend/cuda/moe.hpp"
#include "celeg/backend/cuda/sampler.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace celeg {
void CudaCompiledModel::set_generation_config(GenerationConfig generation) {
    generation.validate();
    if (session_.phase_ == SessionPhase::DecodePending) {
        throw std::runtime_error(
            "cannot change generation configuration during decode");
    }
    session_.generation_ = generation;
    decode_graphs_.reset();
}

PhaseProfile& decode_phase_profile() {
    static PhaseProfile instance;
    return instance;
}

void CudaCompiledModel::enqueue_sampling() {
    CudaSampler::enqueue(
        workspace_.logits_, sampling_.seen_tokens, sampling_.sampling_scores, sampling_.topk_values, sampling_.topk_indices,
        resources_.shape_, session_.generation_, sampling_.rng_state,
        sampling_.sampled_device, stream_.get());
}

void CudaCompiledModel::enqueue_decode_forward() {
    decode_phase_profile().begin(stream_.get());
    resources_.weight_layout_->embed_token_device(
        sampling_.sampled_device.data(), workspace_.hidden_.data(), resources_.shape_.hidden,
        stream_.get());
    launch_scale(workspace_.hidden_.data(), resources_.shape_.hidden, resources_.shape_.numerical_policy.embedding_multiplier,
                 stream_.get());
    initialize_per_layer_input_device(sampling_.sampled_device.data());
    decode_phase_profile().end(DecodePhase::Embed, stream_.get());

    int layer_idx = 0;
    for (Layer& layer : resources_.layers_) {
        LayerCommon& common_layer = common(layer);
        if (!resources_.options_.fused_residuals) {
            CELEG_CUDA(cudaMemcpyAsync(
                workspace_.residual_.data(), workspace_.hidden_.data(), workspace_.hidden_.bytes(),
                cudaMemcpyDeviceToDevice, stream_.get()));
        }
        decode_phase_profile().begin(stream_.get());
        launch_rmsnorm(workspace_.hidden_.data(), common_layer.operator_norm, workspace_.normed_.data(),
                       1, resources_.shape_.hidden, resources_.shape_.numerical_policy.norm_eps,
                       stream_.get());
        decode_phase_profile().end(DecodePhase::Norm, stream_.get());
        if (AttentionLayer* attention = as_attention(layer)) {
            const AttentionSpec& layout = attention->layout;
            AttentionLayer* owner = attention;
            if (attention->kv_owner_layer >= 0) {
                owner = as_attention(resources_.layers_.at(
                    static_cast<size_t>(attention->kv_owner_layer)));
                if (!owner) throw std::logic_error("CUDA shared KV owner is not attention");
            }
            const AttentionSpec& owner_layout = owner->layout;
            __nv_bfloat16* q = workspace_.qkv_output_.data();
            __nv_bfloat16* k = q + layout.query_width();
            __nv_bfloat16* v = k + layout.key_value_width();
            decode_phase_profile().begin(stream_.get());
            {
            auto native_fanout = native_fanout_scope(
                workspace_.normed_.data(), 1, resources_.shape_.hidden);
            linear(workspace_.normed_.data(), *attention->query, q,
                   1, layout.query_width(), resources_.shape_.hidden);
            if (attention->key && attention->value) {
                linear(workspace_.normed_.data(), *attention->key, k,
                       1, layout.key_value_width(), resources_.shape_.hidden);
                linear(workspace_.normed_.data(), *attention->value, v,
                       1, layout.key_value_width(), resources_.shape_.hidden);
            }
            }
            decode_phase_profile().end(DecodePhase::Projection, stream_.get());
            decode_phase_profile().begin(stream_.get());
            if (layout.positional_encoding == PositionalEncodingKind::Rope) {
                launch_dynamic_qk_norm_rope_device(
                    q, attention->key ? k : nullptr, attention->q_norm, attention->k_norm,
                    layout.query_heads, layout.key_value_heads, layout.head_dim,
                    position_device_.data(), static_cast<float>(layout.rope_theta),
                    static_cast<float>(layout.rotary_fraction), resources_.shape_.numerical_policy.norm_eps,
                    layout.query_key_norm, stream_.get());
            }
            launch_scale(q, layout.query_width(), layout.query_scale, stream_.get());
            decode_phase_profile().end(DecodePhase::RopeKv, stream_.get());
            decode_phase_profile().begin(stream_.get());
            if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
                if (attention->key && attention->value) launch_store_kv_int8_device(
                    k, v, owner->key_cache_int8.data(), owner->value_cache_int8.data(),
                    owner->key_cache_scales.data(), owner->value_cache_scales.data(), position_device_.data(),
                    owner_layout.key_value_heads, owner_layout.head_dim, stream_.get());
                if (session_.active_segmented_attention_) {
                    launch_gqa_decode_segmented_int8_device(
                        q, owner->key_cache_int8.data(), owner->value_cache_int8.data(),
                        owner->key_cache_scales.data(), owner->value_cache_scales.data(), workspace_.op_output_.data(),
                        position_device_.data(), layout.query_heads,
                        owner_layout.key_value_heads, owner_layout.head_dim,
                        resources_.options_.attention_chunk_tokens, workspace_.attention_chunks_,
                        layout.sliding_window,
                        workspace_.attention_partial_max_.data(),
                        workspace_.attention_partial_denom_.data(),
                        workspace_.attention_partial_accum_.data(), stream_.get());
                } else if (resources_.options_.fast_attention) {
                    launch_gqa_decode_online_int8_device(
                        q, owner->key_cache_int8.data(), owner->value_cache_int8.data(),
                        owner->key_cache_scales.data(), owner->value_cache_scales.data(), workspace_.op_output_.data(),
                        position_device_.data(), layout.query_heads,
                        owner_layout.key_value_heads, owner_layout.head_dim,
                        layout.sliding_window, stream_.get());
                } else {
                    launch_gqa_decode_strict_int8_device(
                        q, owner->key_cache_int8.data(), owner->value_cache_int8.data(),
                        owner->key_cache_scales.data(), owner->value_cache_scales.data(), workspace_.op_output_.data(),
                        position_device_.data(), layout.query_heads,
                        owner_layout.key_value_heads, owner_layout.head_dim,
                        layout.sliding_window, stream_.get());
                }
            } else {
                if (attention->key && attention->value) launch_store_kv_device(
                    k, v, owner->key_cache.data(), owner->value_cache.data(),
                    position_device_.data(), owner_layout.key_value_width(), stream_.get());
                if (session_.active_segmented_attention_) {
                    launch_gqa_decode_segmented_device(
                        q, owner->key_cache.data(), owner->value_cache.data(),
                        workspace_.op_output_.data(), position_device_.data(),
                        layout.query_heads, owner_layout.key_value_heads,
                        owner_layout.head_dim, resources_.options_.attention_chunk_tokens,
                        workspace_.attention_chunks_, layout.sliding_window,
                        workspace_.attention_partial_max_.data(),
                        workspace_.attention_partial_denom_.data(),
                        workspace_.attention_partial_accum_.data(), stream_.get());
                } else if (resources_.options_.fast_attention) {
                    launch_gqa_decode_online_device(
                        q, owner->key_cache.data(), owner->value_cache.data(),
                        workspace_.op_output_.data(), position_device_.data(),
                        layout.query_heads, owner_layout.key_value_heads,
                        owner_layout.head_dim, layout.sliding_window, stream_.get());
                } else {
                    launch_gqa_decode_strict_device(
                        q, owner->key_cache.data(), owner->value_cache.data(),
                        workspace_.op_output_.data(), position_device_.data(),
                        layout.query_heads, owner_layout.key_value_heads,
                        owner_layout.head_dim, layout.sliding_window, stream_.get());
                }
            }
            decode_phase_profile().end(DecodePhase::Attention, stream_.get());
            decode_phase_profile().begin(stream_.get());
            linear(workspace_.op_output_.data(), *attention->out, workspace_.hidden_.data(),
                   1, resources_.shape_.hidden, layout.query_width(),
                   resources_.options_.fused_residuals && !common_layer.post_attention_norm ? 1.0f : 0.0f);
            launch_scale(workspace_.hidden_.data(), resources_.shape_.hidden, resources_.shape_.numerical_policy.residual_multiplier,
                         stream_.get());
            decode_phase_profile().end(DecodePhase::AttnOut, stream_.get());
        } else {
            ConvolutionLayer& convolution = *as_convolution(layer);
            decode_phase_profile().begin(stream_.get());
            linear(workspace_.normed_.data(), *convolution.conv_in, workspace_.conv_projected_.data(),
                   1, 3 * resources_.shape_.hidden, resources_.shape_.hidden);
            launch_conv_decode_device(
                workspace_.conv_projected_.data(), convolution.conv_weight,
                convolution.conv_state.data(), workspace_.op_output_.data(),
                resources_.shape_.hidden, resources_.shape_.conv_cache,
                position_device_.data(), stream_.get());
            linear(workspace_.op_output_.data(), *convolution.conv_out, workspace_.hidden_.data(),
                   1, resources_.shape_.hidden, resources_.shape_.hidden,
                   resources_.options_.fused_residuals ? 1.0f : 0.0f);
            decode_phase_profile().end(DecodePhase::Conv, stream_.get());
        }
        if (common_layer.post_attention_norm) {
            launch_rmsnorm(workspace_.hidden_.data(), common_layer.post_attention_norm,
                           workspace_.hidden_.data(), 1, resources_.shape_.hidden,
                           resources_.shape_.numerical_policy.norm_eps, stream_.get());
        }
        if (!resources_.options_.fused_residuals || common_layer.post_attention_norm) {
            decode_phase_profile().begin(stream_.get());
            launch_residual_add(workspace_.hidden_.data(), workspace_.residual_.data(),
                                resources_.shape_.hidden, stream_.get());
            decode_phase_profile().end(DecodePhase::Other, stream_.get());
        }
        decode_phase_profile().begin(stream_.get());
        run_mlp_decode(common_layer, layer_idx);
        decode_phase_profile().end(DecodePhase::Mlp, stream_.get());
        ++layer_idx;
    }
    decode_phase_profile().begin(stream_.get());
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
    decode_phase_profile().end(DecodePhase::Logits, stream_.get());
}

void CudaCompiledModel::enqueue_decode_step() {
    decode_phase_profile().begin(stream_.get());
    enqueue_sampling();
    decode_phase_profile().end(DecodePhase::Sampling, stream_.get());
    decode_phase_profile().count_step();
    enqueue_decode_forward();
    decode_phase_profile().begin(stream_.get());
    launch_increment_position(position_device_.data(), stream_.get());
    decode_phase_profile().end(DecodePhase::Other, stream_.get());
}

bool CudaCompiledModel::use_segmented_attention(int host_position) const {
    return resources_.plan_.segmented_attention(host_position);
}

CudaGraphExec& CudaCompiledModel::graph_for_attention(bool segmented) {
    return decode_graphs_.select(segmented);
}

void CudaCompiledModel::capture_decode_graph(bool segmented) {
    if (!resources_.options_.cuda_graph) return;
    CudaGraphExec& graph = graph_for_attention(segmented);
    if (graph.ready()) return;
    session_.active_segmented_attention_ = segmented;
    graph.capture_begin(stream_.get());
    try {
        enqueue_decode_step();
        graph.capture_end(stream_.get());
    } catch (...) {
        graph.abort_capture(stream_.get());
        throw;
    }
}

int32_t CudaCompiledModel::decode() {
    decode_async_begin();
    return decode_async_finish();
}

void CudaCompiledModel::decode_async_begin() {
    if (!local_kv_cache_available_) {
        throw std::runtime_error(
            "lane decode is unavailable after transferring KV to the shared paged cache");
    }
    if (session_.phase_ != SessionPhase::Ready) {
        throw std::runtime_error("decode requires a successful prefill");
    }
    if (session_.phase_ == SessionPhase::DecodePending) {
        throw std::runtime_error("decode_async_begin called twice");
    }
    if (session_.position_ >= max_context_) {
        throw std::runtime_error("context limit reached");
    }
    decode_async_begin_time_ = std::chrono::steady_clock::now();
    const bool segmented = use_segmented_attention(session_.position_);
    session_.active_segmented_attention_ = segmented;
    if (resources_.options_.cuda_graph) {
        CudaGraphExec& graph = graph_for_attention(segmented);
        if (!graph.ready()) capture_decode_graph(segmented);
        graph.launch(stream_.get());
    } else {
        enqueue_decode_step();
    }
    CELEG_CUDA(cudaMemcpyAsync(sampling_.sampled_host.data(), sampling_.sampled_device.data(),
                             sizeof(int32_t), cudaMemcpyDeviceToHost,
                             stream_.get()));
    session_.phase_ = SessionPhase::DecodePending;
}

int32_t CudaCompiledModel::decode_async_finish() {
    if (session_.phase_ != SessionPhase::DecodePending) {
        throw std::runtime_error("decode_async_finish without begin");
    }
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));
    session_.phase_ = SessionPhase::Ready;
    ++session_.position_;
    const auto ended = std::chrono::steady_clock::now();
    session_.metrics_.cumulative_decode_ms +=
        std::chrono::duration<double, std::milli>(
            ended - decode_async_begin_time_).count();
    ++session_.metrics_.decoded_tokens;
    return sampling_.sampled_host.data()[0];
}

DecodeBenchmark CudaCompiledModel::benchmark_decode(int warmup_steps,
                                                int measured_steps) {
    if (session_.phase_ != SessionPhase::Ready) {
        throw std::runtime_error("benchmark_decode requires a successful prefill");
    }
    if (warmup_steps < 0 || measured_steps <= 0) {
        throw std::invalid_argument(
            "benchmark steps require warmup >= 0 and measured > 0");
    }
    const int total_steps = warmup_steps + measured_steps;
    if (session_.position_ + total_steps > max_context_) {
        throw std::runtime_error("decode benchmark exceeds context limit");
    }
    if (resources_.options_.cuda_graph) {
        const int final_position = session_.position_ + total_steps - 1;
        const bool starts_segmented = use_segmented_attention(session_.position_);
        const bool ends_segmented = use_segmented_attention(final_position);
        if (!starts_segmented || !ends_segmented) capture_decode_graph(false);
        if (starts_segmented || ends_segmented) capture_decode_graph(true);
    }

    int simulated_position = session_.position_;
    auto launch_step = [&]() {
        const bool segmented = use_segmented_attention(simulated_position);
        session_.active_segmented_attention_ = segmented;
        if (resources_.options_.cuda_graph) {
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

    session_.position_ += total_steps;
    DecodeBenchmark result;
    result.warmup_steps = warmup_steps;
    result.measured_steps = measured_steps;
    result.elapsed_ms = CudaEvent::elapsed_ms(begin, end);
    return result;
}

ModelMemoryStats CudaCompiledModel::memory_stats() const {
    ModelMemoryStats stats;
    stats.weights = resources_.weights_ ? resources_.weights_->memory_bytes() : 0;
    for (const Layer& layer : resources_.layers_) {
        if (const AttentionLayer* attention = as_attention(layer)) {
            stats.kv_cache += attention->key_cache.bytes() + attention->value_cache.bytes() +
                attention->key_cache_int8.bytes() + attention->value_cache_int8.bytes() +
                attention->key_cache_scales.bytes() + attention->value_cache_scales.bytes();
        } else {
            stats.conv_state += as_convolution(layer)->conv_state.bytes();
        }
    }
    stats.activations =
        workspace_.hidden_.bytes() + workspace_.residual_.bytes() + workspace_.normed_.bytes() +
        workspace_.op_output_.bytes() + workspace_.qkv_output_.bytes() + workspace_.conv_projected_.bytes() +
        workspace_.gate_up_.bytes() + workspace_.activated_.bytes() + workspace_.mlp_output_.bytes() +
        workspace_.logits_.bytes() + workspace_.paged_page_table_.bytes() +
        workspace_.paged_prefill_tokens_.bytes() + workspace_.prefill_tokens_.bytes() +
        workspace_.prefill_hidden_.bytes() +
        workspace_.prefill_residual_.bytes() + workspace_.prefill_normed_.bytes() +
        workspace_.prefill_op_output_.bytes() + workspace_.prefill_q_.bytes() + workspace_.prefill_k_.bytes() +
        workspace_.prefill_v_.bytes() + workspace_.prefill_conv_projected_.bytes() +
        workspace_.prefill_gate_up_.bytes() + workspace_.prefill_activated_.bytes() +
        workspace_.prefill_mlp_output_.bytes();
    stats.sampling =
        position_device_.bytes() + sampling_.sampled_device.bytes() +
        sampling_.seen_tokens.bytes() + sampling_.sampling_scores.bytes() +
        sampling_.topk_values.bytes() + sampling_.topk_indices.bytes() + sampling_.rng_state.bytes();
    stats.matmul_workspace = gemm_ ? gemm_->workspace_bytes() : 0;
    stats.attention_workspace = workspace_.attention_partial_max_.bytes() +
        workspace_.attention_partial_denom_.bytes() + workspace_.attention_partial_accum_.bytes();
    return stats;
}

CudaModelDiagnostics::ExpertOffloadStats CudaCompiledModel::expert_offload_stats() const {
    CudaModelDiagnostics::ExpertOffloadStats stats;
    if (!workspace_.expert_offload_plan_.enabled) {
        stats.hit_rate = -1.0;
        return stats;
    }
    stats.experts_per_layer = workspace_.expert_offload_plan_.experts_per_layer;
    stats.host_experts_per_layer = workspace_.expert_offload_plan_.host_experts_per_layer;
    uint64_t hits = 0, misses = 0;
    for (const auto& cache : workspace_.expert_caches_) {
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

void CudaCompiledModel::release_local_kv_cache() {
    if (!local_kv_cache_available_) return;
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));
    for (Layer& layer : resources_.layers_) {
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
    ++storage_generation_;
    decode_graphs_.reset();
}

std::vector<float> CudaCompiledModel::copy_logits() {
    if (session_.phase_ != SessionPhase::Ready) {
        throw std::runtime_error("logits are unavailable before prefill");
    }
    std::vector<__nv_bfloat16> bf16_logits(resources_.shape_.vocab_size);
    CELEG_CUDA(cudaMemcpyAsync(
        bf16_logits.data(), workspace_.logits_.data(), workspace_.logits_.bytes(),
        cudaMemcpyDeviceToHost, stream_.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));

    std::vector<float> result(resources_.shape_.vocab_size);
    for (int i = 0; i < resources_.shape_.vocab_size; ++i) {
        result[static_cast<size_t>(i)] =
            __bfloat162float(bf16_logits[static_cast<size_t>(i)]);
    }
    return result;
}



} // namespace celeg


