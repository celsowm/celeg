#pragma once

#include "celeg/backend/cuda/model.hpp"
#include "celeg/backend/cuda/utils.cuh"
#include "celeg/checkpoint/formats/safetensors.hpp"
#include "celeg/backend/cuda/execution_plan.hpp"
#include "celeg/model/resolved.hpp"
#include "celeg/backend/cuda/session_store.hpp"
#include "celeg/backend/cuda/weight_layout.hpp"
#include "celeg/backend/cuda/weights_loader.hpp"
#include "celeg/backend/cuda/gemm_dispatcher.hpp"
#include "celeg/backend/cuda/sampling_state.hpp"
#include "celeg/backend/cuda/decode_graphs.hpp"
#include "celeg/backend/cuda/workspace.hpp"
#include "celeg/backend/cuda/packed_session.hpp"
#include "celeg/backend/cuda/moe/offload.hpp"
#include "celeg/backend/cuda/moe/expert_residency.hpp"
#include "celeg/detail/model/types.hpp"
#include "celeg/detail/model/resources.hpp"
#include "celeg/model/session.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace celeg {

namespace detail {
struct ModelBootstrap;
}

struct CudaCompiledModel {
    // Chunk size for launch_gqa_prefill_segmented (batched, chunked causal
    // prefill attention). Smaller than the long-context decode default
    // (attention_chunk_tokens, 256) because prefill parallelism comes from
    // chunk *count*, not chunk size: more, shorter chunks -> more concurrent
    // blocks and a shorter serial critical path per block.
    static constexpr int kPrefillAttnChunkTokens = 64;

    // Row-count ceiling for launch_gqa_prefill_gemm's dense
    // O(q_heads*rows^2) score/probability scratch. A single prefill_batched
    // call above this falls back to launch_gqa_prefill_segmented
    // (O(rows) scratch) instead of allocating an unbounded scratch buffer.
    // 2048 rows keeps the scratch under ~1GB even at 32 attention heads.
    static constexpr int kMaxGemmAttentionRows = 2048;

    CudaModelResources resources_;
    std::shared_ptr<const RuntimeContext> runtime_;
    SessionState session_;

    CudaCompiledModel(const std::string& model_path,
         int max_context,
         CudaModelOptions options,
         GenerationConfig generation,
         std::shared_ptr<const RuntimeContext> runtime = nullptr);
    ~CudaCompiledModel();

    // Thin wrapper around gemm_->linear(..., plan_) so call sites in the
    // forward pass don't need to thread plan_ through every call. New
    // GEMM backends are added by extending GemmDispatcher (OCP).
    void linear(const __nv_bfloat16* x, const LinearWeight& weight,
                __nv_bfloat16* y, int m, int n, int k, float beta = 0.0f) {
        gemm_->linear(x, weight, y, m, n, k, beta, resources_.plan_);
    }

    GemmDispatcher::NativeFanoutScope native_fanout_scope(
        const __nv_bfloat16* x, int m, int k) {
        return GemmDispatcher::NativeFanoutScope(
            resources_.plan_.options().weight_mode == WeightMode::NativeGguf
                ? gemm_.get() : nullptr, x, m, k);
    }

    void warmup_decode_gemms();
    void warmup_prefill_attention_gemm();

    void configure_model(const detail::ModelBootstrap& bootstrap);
    void allocate_celeg_resources();
    void load_checkpoint_weights(const std::string& model_path,
                                 const detail::ModelBootstrap& bootstrap);

    void reset(bool allocate_local_kv = true);
    void prefill(const std::vector<int32_t>& tokens);
    void prefill(const std::vector<int32_t>& tokens,
                 const PromptEmbedding& embeddings);
    void prefill_chunk(const std::vector<int32_t>& tokens,
                       bool begin, bool finalize);
    void prefill_chunk_paged(const std::vector<int32_t>& tokens,
                             bool begin, bool finalize,
                             PhysicalPagedKvCache& paged_kv,
                             const std::vector<uint32_t>& page_table);
    void validate_token_ids(const std::vector<int32_t>& tokens) const;
    int32_t decode();
    void decode_async_begin();
    int32_t decode_async_finish();
    void set_generation_config(GenerationConfig generation);
    std::vector<float> copy_logits();
    DecodeBenchmark benchmark_decode(int warmup_steps, int measured_steps);
    ModelMemoryStats memory_stats() const;
    CudaModelDiagnostics::ExpertOffloadStats expert_offload_stats() const;
    RuntimeMetrics runtime_metrics() const { return session_.metrics_; }
    void clear_runtime_metrics() { session_.metrics_ = {}; }
    std::string execution_plan_description() const { return resources_.plan_.description(); }
    void save_session(const std::string& path);
    void load_session(const std::string& path);
    PrefixState export_prefix_state() const;
    void restore_prefix_state(const PrefixState& state);
    SessionStore::SessionState make_session_state();
    void release_local_kv_cache();
    bool local_kv_cache_available() const { return local_kv_cache_available_; }
    SessionPhase phase() const { return session_.phase_; }
    void set_phase(SessionPhase value) { session_.phase_ = value; }
    bool ready_for_decode() const { return session_.phase_ == SessionPhase::Ready; }
    bool decode_pending() const { return session_.phase_ == SessionPhase::DecodePending; }
    int position() const { return session_.position_; }
    void set_position(int value) { session_.position_ = value; }
    int max_context() const { return max_context_; }
    bool active_segmented_attention() const { return session_.active_segmented_attention_; }
    void set_active_segmented_attention(bool value) { session_.active_segmented_attention_ = value; }
    const CudaModelOptions& options() const { return resources_.options_; }
    const GenerationConfig& generation() const { return session_.generation_; }
    const RuntimeTopology& shape() const { return resources_.shape_; }
    const std::shared_ptr<SharedModelWeights>& weights() const { return resources_.weights_; }
    DeviceBuffer<__nv_bfloat16>& logits() { return workspace_.logits_; }
    DeviceBuffer<uint8_t>& seen_tokens() { return sampling_.seen_tokens; }
    DeviceBuffer<uint64_t>& rng_state() { return sampling_.rng_state; }
    DeviceBuffer<int32_t>& sampled_device() { return sampling_.sampled_device; }
    DeviceBuffer<int32_t>& position_device() { return position_device_; }
    PinnedBuffer<int32_t>& sampled_host() { return sampling_.sampled_host; }
    std::vector<Layer>& layers() { return resources_.layers_; }
    const std::vector<Layer>& layers() const { return resources_.layers_; }
    RuntimeMetrics& metrics() { return session_.metrics_; }
    int32_t sampled_host_value() const { return sampling_.sampled_host.data()[0]; }
    void set_sampled_host_value(int32_t value) { sampling_.sampled_host.data()[0] = value; }
    IWeightLayout& weight_layout() { return *resources_.weight_layout_; }
    const LinearWeight* embedding() const { return resources_.embedding_; }
    // Weight used for the final logits projection. When the checkpoint ships a
    // separate (untied) lm_head, that is returned; otherwise the tied
    // embed_tokens table is shared.
    const LinearWeight* logits_weight() const {
        return resources_.lm_head_ ? resources_.lm_head_ : resources_.embedding_;
    }
    bool tied_lm_head() const { return resources_.lm_head_ == nullptr; }
    const __nv_bfloat16* final_norm() const { return resources_.final_norm_; }
    bool cuda_graph_ready() const {
        return decode_graphs_.ready();
    }

    void prefill_batched(const std::vector<int32_t>& tokens);
    void allocate_prefill_workspace(int rows);
    void release_prefill_workspace();
    void run_mlp_decode(const LayerCommon& common_layer, int layer);
    void run_mlp_prefill(const LayerCommon& common_layer, int rows, int layer);
    void run_per_layer_input_decode(const LayerCommon& common_layer, int layer);
    void run_per_layer_input_prefill(const LayerCommon& common_layer, int rows, int layer);
    void initialize_per_layer_input_device(const int32_t* token);
    void initialize_per_layer_input_host(int32_t token);
    void initialize_per_layer_input_batch(const int32_t* tokens, int rows);
    void run_mlp_moe_decode(const LayerCommon& common_layer, int layer);
    void run_mlp_moe_prefill(const LayerCommon& common_layer, int rows, int layer);

    // When offload is enabled, ensures every expert selected for `layer` (read
    // from `sel_dev`, `rows*K` ints) is GPU-resident before the FFN kernel runs,
    // promoting cold experts from host on workspace_.expert_transfer_stream_ and ordering
    // the streams via the two events. No-op when the layer has no cache.
    void ensure_moe_experts_resident(int layer, const int* sel_dev, int rows,
                                     cudaStream_t compute_stream,
                                     const float* route_scores_dev = nullptr);
    void ensure_moe_experts_resident_packed(
        int layer, const int* sel_dev, int rows, cudaStream_t stream,
        const float* route_scores_dev);
    void forward_token_host(int32_t token, bool compute_logits,
                            const float* raw_embedding = nullptr);
    void forward_token_paged_host(int32_t token, bool compute_logits,
                                  PhysicalPagedKvCache& paged_kv,
                                  const uint32_t* device_page_table,
                                  int page_table_stride);

    void enqueue_sampling();
    void enqueue_decode_forward();
    void enqueue_decode_step();
    void capture_decode_graph(bool segmented);
    bool use_segmented_attention(int host_position) const;
    CudaGraphExec& graph_for_attention(bool segmented);

    PackedSessionContext packed_session_context();
    static bool packed_segmented_attention_callback(const void* owner,
                                                    int host_position);
    static void packed_expert_residency_callback(
        void* owner, int layer, const int* sel_dev, int rows,
        cudaStream_t stream, const float* route_scores_dev);

    CudaStream stream_;
    std::unique_ptr<GemmDispatcher> gemm_;
    CudaDecodeGraphs decode_graphs_;
    CudaWorkspace workspace_;
    int max_context_;
    bool local_kv_cache_available_ = true;
    uint64_t storage_generation_ = 0;
    std::chrono::steady_clock::time_point decode_async_begin_time_{};
    DeviceBuffer<int32_t> position_device_{1};
    CudaSamplingState sampling_;
};

} // namespace celeg
