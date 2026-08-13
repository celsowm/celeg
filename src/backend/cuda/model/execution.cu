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
    mtp_candidate_ready_ = false;
    decode_graphs_.reset();
}

PhaseProfile& decode_phase_profile() {
    static PhaseProfile instance;
    return instance;
}

void CudaCompiledModel::enqueue_sampling() {
    if (session_.generation_.forced_prefix &&
        session_.generation_.forced_prefix->position <
            session_.generation_.forced_prefix->tokens.size()) {
        const int32_t* token =
            session_.generation_.forced_prefix->tokens.data() +
            session_.generation_.forced_prefix->position;
        CELEG_CUDA(cudaMemcpyAsync(
            sampling_.sampled_device.data(), token, sizeof(*token),
            cudaMemcpyHostToDevice, stream_.get()));
        launch_mark_seen(sampling_.sampled_device.data(), sampling_.seen_tokens.data(),
                         resources_.dims_.vocab_size, stream_.get());
        return;
    }
    CudaSampler::enqueue(
        workspace_.logits_, sampling_.seen_tokens, sampling_.sampling_scores, sampling_.topk_values, sampling_.topk_indices,
        sampling_.partial_values, sampling_.partial_indices,
        resources_.shape_, resources_.dims_.vocab_size, session_.generation_, sampling_.rng_state,
        sampling_.sampled_device, stream_.get());
}

void CudaCompiledModel::enqueue_decode_forward() {
    decode_phase_profile().begin(stream_.get());
    resources_.weight_layout_->embed_token_device(
        sampling_.sampled_device.data(), workspace_.hidden_.data(), resources_.program_.hidden,
        stream_.get());
        launch_scale(workspace_.hidden_.data(), resources_.program_.hidden, resources_.program_.embedding_transform.multiplier,
                 stream_.get());
    initialize_per_layer_input_device(sampling_.sampled_device.data());
    decode_phase_profile().end(DecodePhase::Embed, stream_.get());

    int layer_idx = 0;
    for (Layer& layer : resources_.layers_) {
        LayerCommon& common_layer = common(layer);
        const CompiledLayerProgram& semantics = resources_.program_.layers.at(static_cast<size_t>(layer_idx));
        if (!resources_.options_.fused_residuals) {
            CELEG_CUDA(cudaMemcpyAsync(
                workspace_.residual_.data(), workspace_.hidden_.data(), workspace_.hidden_.bytes(),
                cudaMemcpyDeviceToDevice, stream_.get()));
        }
        decode_phase_profile().begin(stream_.get());
        launch_rmsnorm(workspace_.hidden_.data(), common_layer.operator_norm, workspace_.normed_.data(),
                       1, resources_.program_.hidden, semantics.operator_norm.epsilon,
                       stream_.get());
        decode_phase_profile().end(DecodePhase::Norm, stream_.get());
        if (as_attention(layer)) {
            enqueue_decode_attention(layer, common_layer, layer_idx);
        } else {
            enqueue_decode_non_attention_mixer(layer, layer_idx);
        }
        if (common_layer.post_attention_norm) {
            launch_rmsnorm(workspace_.hidden_.data(), common_layer.post_attention_norm,
                           workspace_.hidden_.data(), 1, resources_.program_.hidden,
                           semantics.post_attention_norm.epsilon, stream_.get());
        }
        if (!resources_.options_.fused_residuals || common_layer.post_attention_norm ||
            resources_.shape_.mamba2_layer_count > 0) {
            decode_phase_profile().begin(stream_.get());
            launch_residual_add(workspace_.hidden_.data(), workspace_.residual_.data(),
                                resources_.program_.hidden, stream_.get());
            decode_phase_profile().end(DecodePhase::Other, stream_.get());
        }
        decode_phase_profile().begin(stream_.get());
        if (resources_.shape_.mamba2_layer_count == 0) run_mlp_decode(common_layer, layer_idx);
        if (std::binary_search(resources_.program_.norm_after_layers.begin(),
                               resources_.program_.norm_after_layers.end(), layer_idx)) {
            launch_rmsnorm(workspace_.hidden_.data(), resources_.final_norm_,
                           workspace_.hidden_.data(), 1, resources_.program_.hidden,
                           resources_.program_.final_norm.epsilon, stream_.get());
        }
        decode_phase_profile().end(DecodePhase::Mlp, stream_.get());
        ++layer_idx;
    }
    if (resources_.mtp_.available()) {
        run_mtp_forward_device(sampling_.sampled_device.data());
    }
    decode_phase_profile().begin(stream_.get());
    launch_rmsnorm(workspace_.hidden_.data(), resources_.final_norm_, workspace_.normed_.data(),
                    1, resources_.program_.hidden, resources_.program_.final_norm.epsilon,
                    stream_.get());
    linear(workspace_.normed_.data(), *logits_weight(), workspace_.logits_.data(),
            1, resources_.dims_.vocab_size, resources_.program_.hidden);
    launch_scale(workspace_.logits_.data(), resources_.dims_.vocab_size,
                 resources_.program_.logits_multiplier /
                     resources_.program_.logits_divisor, stream_.get());
    if (resources_.program_.final_logit_softcap > 0.0f) {
            launch_tanh_softcap(workspace_.logits_.data(), resources_.dims_.vocab_size,
                            resources_.program_.final_logit_softcap, stream_.get());
    }
    finalize_mtp_verification();
    decode_phase_profile().end(DecodePhase::Logits, stream_.get());
}

void CudaCompiledModel::enqueue_decode_step() {
    decode_phase_profile().begin(stream_.get());
    mtp_candidate_used_ = resources_.mtp_.available() &&
        session_.generation_.greedy() && mtp_candidate_ready_ &&
        !(session_.generation_.forced_prefix &&
          session_.generation_.forced_prefix->position <
              session_.generation_.forced_prefix->tokens.size());
    if (mtp_candidate_used_) {
        CELEG_CUDA(cudaMemcpyAsync(
            sampling_.sampled_device.data(), workspace_.mtp_candidate_.data(),
            sizeof(int32_t), cudaMemcpyDeviceToDevice, stream_.get()));
        launch_mark_seen(sampling_.sampled_device.data(),
                         sampling_.seen_tokens.data(),
                         resources_.dims_.vocab_size, stream_.get());
        ++session_.metrics_.mtp_used_tokens;
    } else {
        enqueue_sampling();
    }
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
    const bool forcing_prefix = session_.generation_.forced_prefix &&
        session_.generation_.forced_prefix->position <
            session_.generation_.forced_prefix->tokens.size();
    if (resources_.options_.cuda_graph && !forcing_prefix) {
        CudaGraphExec& graph = graph_for_attention(segmented);
        if (!graph.ready()) capture_decode_graph(segmented);
        graph.launch(stream_.get());
    } else {
        enqueue_decode_step();
    }
    CELEG_CUDA(cudaMemcpyAsync(sampling_.sampled_host.data(), sampling_.sampled_device.data(),
                             sizeof(int32_t), cudaMemcpyDeviceToHost,
                             stream_.get()));
    if (resources_.mtp_.available()) {
        CELEG_CUDA(cudaMemcpyAsync(
            mtp_verification_host_.data(), workspace_.mtp_candidate_.data(),
            sizeof(int32_t), cudaMemcpyDeviceToHost, stream_.get()));
        CELEG_CUDA(cudaMemcpyAsync(
            mtp_verification_host_.data() + 1,
            workspace_.mtp_target_candidate_.data(), sizeof(int32_t),
            cudaMemcpyDeviceToHost, stream_.get()));
    }
    session_.phase_ = SessionPhase::DecodePending;
}

int32_t CudaCompiledModel::decode_async_finish() {
    if (session_.phase_ != SessionPhase::DecodePending) {
        throw std::runtime_error("decode_async_finish without begin");
    }
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));
    session_.phase_ = SessionPhase::Ready;
    ++session_.position_;
    if (session_.generation_.forced_prefix &&
        session_.generation_.forced_prefix->position <
            session_.generation_.forced_prefix->tokens.size()) {
        ++session_.generation_.forced_prefix->position;
    }
    if (resources_.mtp_.available() && session_.generation_.greedy()) {
        ++session_.metrics_.mtp_verified_tokens;
        if (mtp_verification_host_.data()[0] ==
            mtp_verification_host_.data()[1]) {
            ++session_.metrics_.mtp_accepted_tokens;
            mtp_candidate_ready_ = true;
        } else {
            ++session_.metrics_.mtp_rejected_tokens;
            mtp_candidate_ready_ = false;
        }
    } else {
        mtp_candidate_ready_ = false;
    }
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
        } else if (const ConvolutionLayer* convolution = as_convolution(layer)) {
            stats.conv_state += convolution->conv_state.bytes();
        } else if (const Mamba2Layer* mamba = as_mamba2(layer)) {
            stats.conv_state += mamba->conv_state.bytes() + mamba->ssm_state.bytes();
        }
    }
    stats.activations =
        workspace_.hidden_.bytes() + workspace_.residual_.bytes() + workspace_.normed_.bytes() +
        workspace_.op_output_.bytes() + workspace_.qkv_output_.bytes() + workspace_.conv_projected_.bytes() +
        workspace_.mamba_projected_.bytes() + workspace_.mamba_inner_.bytes() +
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
    std::vector<__nv_bfloat16> bf16_logits(resources_.dims_.vocab_size);
    CELEG_CUDA(cudaMemcpyAsync(
        bf16_logits.data(), workspace_.logits_.data(), workspace_.logits_.bytes(),
        cudaMemcpyDeviceToHost, stream_.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));

    std::vector<float> result(resources_.dims_.vocab_size);
    for (int i = 0; i < resources_.dims_.vocab_size; ++i) {
        result[static_cast<size_t>(i)] =
            __bfloat162float(bf16_logits[static_cast<size_t>(i)]);
    }
    return result;
}



} // namespace celeg


