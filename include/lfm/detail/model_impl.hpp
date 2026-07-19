#pragma once

#include "lfm/model.hpp"
#include "lfm/cuda_utils.cuh"
#include "lfm/safetensors.hpp"
#include "lfm/execution_plan.hpp"
#include "lfm/model_shape.hpp"
#include "lfm/model_variant.hpp"
#include "lfm/session_store.hpp"
#include "lfm/weight_layout.hpp"
#include "lfm/weight_loader.hpp"
#include "lfm/gemm_dispatcher.hpp"
#include "lfm/packed_session.hpp"
#include "lfm/detail/model_types.hpp"

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
    Impl(const std::string& safetensors_path,
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
    const LinearWeight* embedding() const { return embedding_; }
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
    void run_mlp_decode(const LayerCommon& common_layer);
    void run_mlp_prefill(const LayerCommon& common_layer, int rows);
    void run_mlp_moe_decode(const LayerCommon& common_layer);
    void run_mlp_moe_prefill(const LayerCommon& common_layer, int rows);
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
    DeviceBuffer<__nv_bfloat16> prefill_q_;
    DeviceBuffer<__nv_bfloat16> prefill_k_;
    DeviceBuffer<__nv_bfloat16> prefill_v_;
    DeviceBuffer<__nv_bfloat16> prefill_conv_projected_;
    DeviceBuffer<__nv_bfloat16> prefill_gate_up_;
    DeviceBuffer<__nv_bfloat16> prefill_activated_;
    DeviceBuffer<__nv_bfloat16> prefill_mlp_output_;

    // ---- MoE feed-forward scratch (decode path, single token) ----
    DeviceBuffer<float> moe_hidden_float_;    // [hidden]
    DeviceBuffer<int> moe_sel_;               // [experts_per_token]
    DeviceBuffer<float> moe_routing_w_;       // [experts_per_token]
    DeviceBuffer<float> moe_router_scratch_;  // [num_experts]
    DeviceBuffer<__nv_bfloat16> moe_output_;  // [hidden]
    DeviceBuffer<__nv_bfloat16> moe_gu_scratch_;  // [K * 2*moe_inter]
    DeviceBuffer<__nv_bfloat16> moe_act_scratch_; // [K * moe_inter]

    // ---- MoE feed-forward scratch (prefill path, rows tokens) ----
    DeviceBuffer<float> moe_pf_hidden_float_;
    DeviceBuffer<int> moe_pf_sel_;
    DeviceBuffer<float> moe_pf_routing_w_;
    DeviceBuffer<float> moe_pf_router_scratch_;
    DeviceBuffer<__nv_bfloat16> moe_pf_output_;
    DeviceBuffer<__nv_bfloat16> moe_pf_gu_scratch_;
    DeviceBuffer<__nv_bfloat16> moe_pf_act_scratch_;

    // Per-MoE-layer device float copy of the router weight, produced at load.
    // Indexed by model layer index; non-MoE layers hold an empty buffer.
    std::vector<DeviceBuffer<float>> moe_router_float_;
};

} // namespace lfm
