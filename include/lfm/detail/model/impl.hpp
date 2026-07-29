#pragma once

#include "lfm/model/model.hpp"
#include "lfm/backend/cuda/utils.cuh"
#include "lfm/checkpoint/formats/safetensors.hpp"
#include "lfm/model/execution/plan.hpp"
#include "lfm/model/config/shape.hpp"
#include "lfm/model/config/variant.hpp"
#include "lfm/backend/cuda/session_store.hpp"
#include "lfm/model/weights/layout.hpp"
#include "lfm/model/weights/loader.hpp"
#include "lfm/backend/cuda/gemm_dispatcher.hpp"
#include "lfm/backend/cuda/packed_session.hpp"
#include "lfm/runtime/moe/offload.hpp"
#include "lfm/runtime/moe/expert_residency.hpp"
#include "lfm/detail/model/types.hpp"

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

namespace lfm {

struct LfmModel::Impl : public IPackedSession {
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

    Impl(const std::string& model_path,
         int max_context,
         ModelOptions options,
         GenerationConfig generation);
    ~Impl();

    // Thin wrapper around gemm_->linear(..., plan_) so call sites in the
    // forward pass don't need to thread plan_ through every call. New
    // GEMM backends are added by extending GemmDispatcher (OCP).
    void linear(const __nv_bfloat16* x, const LinearWeight& weight,
                __nv_bfloat16* y, int m, int n, int k, float beta = 0.0f) {
        gemm_->linear(x, weight, y, m, n, k, beta, plan_);
    }

    void initialize_rope_tables();
    void warmup_decode_gemms();
    void warmup_prefill_attention_gemm();

    void reset(bool allocate_local_kv = true);
    void prefill(const std::vector<int32_t>& tokens);
    void prefill_chunk(const std::vector<int32_t>& tokens,
                       bool begin, bool finalize);
    void prefill_chunk_paged(const std::vector<int32_t>& tokens,
                             bool begin, bool finalize,
                             PhysicalPagedKvCache& paged_kv,
                             const std::vector<uint32_t>& page_table);
    int32_t decode();
    void decode_async_begin();
    int32_t decode_async_finish();
    void set_generation_config(GenerationConfig generation);
    std::vector<float> copy_logits();
    DecodeBenchmark benchmark_decode(int warmup_steps, int measured_steps);
    ModelMemoryStats memory_stats() const;
    LfmDiagnostics::ExpertOffloadStats expert_offload_stats() const;
    RuntimeMetrics runtime_metrics() const { return metrics_; }
    void clear_runtime_metrics() { metrics_ = {}; }
    void save_session(const std::string& path);
    void load_session(const std::string& path);
    PrefixState export_prefix_state() const;
    void restore_prefix_state(const PrefixState& state);
    SessionStore::SessionState make_session_state();
    void release_local_kv_cache();
    bool local_kv_cache_available() const { return local_kv_cache_available_; }
    SessionPhase phase() const { return phase_; }
    void set_phase(SessionPhase value) { phase_ = value; }
    bool ready_for_decode() const { return phase_ == SessionPhase::Ready; }
    bool decode_pending() const { return phase_ == SessionPhase::DecodePending; }
    int position() const { return position_; }
    void set_position(int value) { position_ = value; }
    int max_context() const { return max_context_; }
    bool active_segmented_attention() const { return active_segmented_attention_; }
    void set_active_segmented_attention(bool value) { active_segmented_attention_ = value; }
    const ModelOptions& options() const { return options_; }
    const GenerationConfig& generation() const { return generation_; }
    const ModelShape& shape() const { return shape_; }
    const std::shared_ptr<SharedModelWeights>& weights() const { return weights_; }
    DeviceBuffer<__nv_bfloat16>& logits() { return logits_; }
    DeviceBuffer<uint8_t>& seen_tokens() { return seen_tokens_; }
    DeviceBuffer<uint64_t>& rng_state() { return rng_state_; }
    DeviceBuffer<int32_t>& sampled_device() { return sampled_device_; }
    DeviceBuffer<int32_t>& position_device() { return position_device_; }
    PinnedBuffer<int32_t>& sampled_host() { return sampled_host_; }
    std::vector<Layer>& layers() { return layers_; }
    const std::vector<Layer>& layers() const { return layers_; }
    RuntimeMetrics& metrics() { return metrics_; }
    int32_t sampled_host_value() const { return sampled_host_.data()[0]; }
    void set_sampled_host_value(int32_t value) { sampled_host_.data()[0] = value; }
    IWeightLayout& weight_layout() { return *weight_layout_; }
    const LinearWeight* embedding() const override { return embedding_; }
    // Weight used for the final logits projection. When the checkpoint ships a
    // separate (untied) lm_head, that is returned; otherwise the tied
    // embed_tokens table is shared.
    const LinearWeight* logits_weight() const override {
        return lm_head_ ? lm_head_ : embedding_;
    }
    bool tied_lm_head() const { return lm_head_ == nullptr; }
    const __nv_bfloat16* final_norm() const { return final_norm_; }
    const __nv_bfloat16* rope_cos() const { return rope_cos_.data(); }
    const __nv_bfloat16* rope_sin() const { return rope_sin_.data(); }
    bool cuda_graph_ready() const {
        return decode_graph_.ready() || segmented_decode_graph_.ready();
    }

    void prefill_batched(const std::vector<int32_t>& tokens);
    void prefill_legacy(const std::vector<int32_t>& tokens);
    void allocate_prefill_workspace(int rows);
    void release_prefill_workspace();
    void run_mlp_decode(const LayerCommon& common_layer, int layer);
    void run_mlp_prefill(const LayerCommon& common_layer, int rows, int layer);
    void run_mlp_moe_decode(const LayerCommon& common_layer, int layer);
    void run_mlp_moe_prefill(const LayerCommon& common_layer, int rows, int layer);

    // When offload is enabled, ensures every expert selected for `layer` (read
    // from `sel_dev`, `rows*K` ints) is GPU-resident before the FFN kernel runs,
    // promoting cold experts from host on expert_transfer_stream_ and ordering
    // the streams via the two events. No-op when the layer has no cache.
    void ensure_moe_experts_resident(int layer, const int* sel_dev, int rows,
                                     cudaStream_t compute_stream,
                                     const float* route_scores_dev = nullptr);
    void ensure_moe_experts_resident_packed(
        int layer, const int* sel_dev, int rows, cudaStream_t stream,
        const float* route_scores_dev) override;
    void forward_token_host(int32_t token, bool compute_logits);
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

    ModelShape shape_;
    const IModelVariant* variant_ = nullptr;
    ExecutionPlan plan_;
    ModelOptions options_;
    GenerationConfig generation_;
    CudaStream stream_;
    std::unique_ptr<GemmDispatcher> gemm_;
    CudaGraphExec decode_graph_;
    CudaGraphExec segmented_decode_graph_;
    int max_context_;
    int position_ = 0;
    SessionPhase phase_ = SessionPhase::Empty;
    bool active_segmented_attention_ = false;
    bool local_kv_cache_available_ = true;
    std::chrono::steady_clock::time_point decode_async_begin_time_{};
    RuntimeMetrics metrics_;

    std::shared_ptr<SharedModelWeights> weights_;
    std::unique_ptr<WeightLoader> weight_loader_;
    std::vector<Layer> layers_;
    const LinearWeight* embedding_ = nullptr;
    // Separate LM head weight when the checkpoint uses an untied head
    // (tie_word_embeddings == false and a model.lm_head.weight tensor exists).
    // Null when the head is tied to the embedding table.
    const LinearWeight* lm_head_ = nullptr;
    const __nv_bfloat16* final_norm_ = nullptr;
    std::unique_ptr<IWeightLayout> weight_layout_;

    DeviceBuffer<int32_t> position_device_{1};
    DeviceBuffer<int32_t> sampled_device_{1};
    PinnedBuffer<int32_t> sampled_host_{1};
    DeviceBuffer<uint8_t> seen_tokens_;
    DeviceBuffer<float> sampling_scores_;
    DeviceBuffer<float> topk_values_{static_cast<size_t>(kMaxTopK)};
    DeviceBuffer<int32_t> topk_indices_{static_cast<size_t>(kMaxTopK)};
    DeviceBuffer<uint64_t> rng_state_{1};

    DeviceBuffer<__nv_bfloat16> hidden_;
    DeviceBuffer<__nv_bfloat16> residual_;
    DeviceBuffer<__nv_bfloat16> normed_;
    DeviceBuffer<__nv_bfloat16> op_output_;
    DeviceBuffer<__nv_bfloat16> qkv_output_;
    DeviceBuffer<__nv_bfloat16> conv_projected_;
    DeviceBuffer<__nv_bfloat16> gate_up_;
    DeviceBuffer<__nv_bfloat16> activated_;
    DeviceBuffer<__nv_bfloat16> mlp_output_;
    DeviceBuffer<__nv_bfloat16> logits_;
    DeviceBuffer<__nv_bfloat16> rope_cos_;
    DeviceBuffer<__nv_bfloat16> rope_sin_;
    DeviceBuffer<float> attention_partial_max_;
    DeviceBuffer<float> attention_partial_denom_;
    DeviceBuffer<float> attention_partial_accum_;
    int attention_chunks_ = 0;

    DeviceBuffer<uint32_t> paged_page_table_;
    DeviceBuffer<int32_t> paged_prefill_tokens_;

    DeviceBuffer<int32_t> prefill_tokens_;
    DeviceBuffer<__nv_bfloat16> prefill_hidden_;
    DeviceBuffer<__nv_bfloat16> prefill_residual_;
    DeviceBuffer<__nv_bfloat16> prefill_normed_;
    DeviceBuffer<__nv_bfloat16> prefill_op_output_;
    DeviceBuffer<__nv_bfloat16> prefill_qkv_;
    DeviceBuffer<__nv_bfloat16> prefill_q_;
    DeviceBuffer<__nv_bfloat16> prefill_k_;
    DeviceBuffer<__nv_bfloat16> prefill_v_;
    DeviceBuffer<__nv_bfloat16> prefill_conv_projected_;
    DeviceBuffer<__nv_bfloat16> prefill_gate_up_;
    DeviceBuffer<__nv_bfloat16> prefill_activated_;
    DeviceBuffer<__nv_bfloat16> prefill_mlp_output_;

    // Chunked prefill attention scratch (launch_gqa_prefill_segmented): one
    // (max, denom) pair and one head_dim-wide accumulator per
    // (row, head, chunk). See prefill.cu's use of kPrefillAttnChunkTokens.
    DeviceBuffer<float> prefill_attn_partial_max_;
    DeviceBuffer<float> prefill_attn_partial_denom_;
    DeviceBuffer<float> prefill_attn_partial_accum_;

    // Batched-GEMM prefill attention scratch (launch_gqa_prefill_gemm):
    // dense [q_heads, rows, rows] raw scores (FP32, pre-softmax) and
    // [q_heads, rows, rows] causal-softmax probabilities (BF16).
    DeviceBuffer<float> prefill_attn_scores_;
    DeviceBuffer<__nv_bfloat16> prefill_attn_probs_;

    // ---- MoE feed-forward scratch (decode path, single token) ----
    DeviceBuffer<float> moe_hidden_float_;    // [hidden]
    DeviceBuffer<int> moe_sel_;               // [experts_per_token]
    DeviceBuffer<float> moe_routing_w_;       // [experts_per_token]
    DeviceBuffer<float> moe_router_scratch_;  // [num_experts]
    DeviceBuffer<float> moe_output_accum_;    // [hidden] FP32 expert accumulator
    DeviceBuffer<__nv_bfloat16> moe_output_;  // [hidden]
    DeviceBuffer<__nv_bfloat16> moe_gu_scratch_;  // [K * 2*moe_inter]
    DeviceBuffer<__nv_bfloat16> moe_act_scratch_; // [K * moe_inter]

    // ---- MoE feed-forward scratch (prefill path, rows tokens) ----
    DeviceBuffer<float> moe_pf_hidden_float_;
    DeviceBuffer<int> moe_pf_sel_;
    DeviceBuffer<int> moe_pf_sel_masked_;
    DeviceBuffer<std::uint8_t> expert_active_dev_;
    DeviceBuffer<float> moe_pf_routing_w_;
    DeviceBuffer<float> moe_pf_router_scratch_;
    DeviceBuffer<float> moe_pf_output_accum_;  // [rows * hidden] FP32 accumulator
    DeviceBuffer<__nv_bfloat16> moe_pf_output_;
    DeviceBuffer<__nv_bfloat16> moe_pf_gu_scratch_;
    DeviceBuffer<__nv_bfloat16> moe_pf_act_scratch_;

    // Per-MoE-layer device float copy of the router weight, produced at load.
    // Indexed by model layer index; non-MoE layers hold an empty buffer.
    std::vector<DeviceBuffer<float>> moe_router_float_;

    // ---- MoE expert offload (Tier 1 GPU cache + Tier 2 host store) ----
    // Resolved offload layout for this session; disabled unless the plan
    // enables it (see ExpertOffloadOptions in ModelOptions).
    ExpertOffloadPlan expert_offload_plan_;
    // Host-backed BF16 expert bytes (pinned+mapped). One store for all layers.
    HostExpertStore host_expert_store_;
    // Per-model-layer GPU expert cache. Empty unique_ptr for non-offloaded /
    // non-MoE layers. Indexed by model layer index.
    std::vector<ExpertLayerCache*> expert_caches_;
    std::vector<std::vector<ExpertLocation>> expert_catalog_;
    // Dedicated stream for host->device expert promotions so transfers overlap
    // compute. Created only when offload is enabled.
    std::unique_ptr<CudaStream> expert_transfer_stream_;
    // Ordering events between the compute stream (runs the FFN kernel) and the
    // transfer stream (promotes cold experts): the transfer stream must not
    // clobber a slot still being read by the prior FFN, and the FFN must not
    // start until the needed experts are resident.
    CudaEvent ffn_done_event_;
    CudaEvent promote_done_event_;
    // Recorded on the transfer stream after speculative prefetch completes; the
    // next token's residency resolution waits on it so prefetched experts are
    // guaranteed resident before that token's FFN.
    CudaEvent prefetch_done_event_;
    // Recorded on the compute stream right after the router kernel writes the
    // expert selection; the transfer stream waits on it before reading the
    // selection, so residency resolution never synchronizes the compute stream.
    CudaEvent router_done_event_;
    // Pinned host staging for asynchronously reading the router's expert
    // selection on the transfer stream (offload path only). Pinned so the DtoH
    // copy can run on the transfer stream without stalling the compute stream.
    int* moe_sel_host_ = nullptr;
    size_t moe_sel_host_cap_ = 0;
    // Pinned host staging for asynchronously reading the router's per-expert
    // scores on the transfer stream so speculative prefetch can rank experts by
    // actual routing likelihood rather than stale recency.
    float* moe_route_scores_host_ = nullptr;
    size_t moe_route_scores_cap_ = 0;
    // Reusable host buffers for speculative prefetch (avoids per-call alloc).
    std::vector<int> prefetch_idx_;
    std::vector<int> prefetch_ranked_;
    std::vector<float> prefetch_scores_;
    // Reusable host buffer for cold-expert list from device-side residency check.
    std::vector<int> cold_expert_host_;
    std::vector<float> cold_scores_host_;
};

} // namespace lfm
