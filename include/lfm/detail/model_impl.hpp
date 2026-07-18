#pragma once

#include "lfm/model.hpp"
#include "lfm/cuda_utils.cuh"
#include "lfm/safetensors.hpp"
#include "lfm/execution_plan.hpp"

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

struct LfmModel::Impl {
    Impl(const std::string& safetensors_path,
         int max_context,
         ModelOptions options,
         GenerationConfig generation);
    ~Impl();

    struct MatmulKey {
        int m = 0;
        int n = 0;
        int k = 0;

        bool operator==(const MatmulKey& other) const {
            return m == other.m && n == other.n && k == other.k;
        }
    };

    struct MatmulKeyHash {
        size_t operator()(const MatmulKey& key) const {
            size_t value = static_cast<size_t>(key.m);
            value = value * 1315423911u + static_cast<size_t>(key.n);
            value = value * 2654435761u + static_cast<size_t>(key.k);
            return value;
        }
    };

    struct LtPlan;

    enum class LinearStorageKind : uint8_t {
        Bf16,
        Int8,
        Int4,
    };

    struct LinearWeight {
        LinearStorageKind kind = LinearStorageKind::Bf16;
        const __nv_bfloat16* bf16 = nullptr;
        const int8_t* int8 = nullptr;
        const uint8_t* int4 = nullptr;
        const float* scales = nullptr;
        int rows = 0;
        int cols = 0;

        bool quantized() const { return kind != LinearStorageKind::Bf16; }
        bool int4_quantized() const { return kind == LinearStorageKind::Int4; }
        bool int8_quantized() const { return kind == LinearStorageKind::Int8; }
        void validate_storage() const;
    };

    struct DeviceWeight {
        DeviceBuffer<__nv_bfloat16> bf16_storage;
        DeviceBuffer<int8_t> int8_storage;
        DeviceBuffer<uint8_t> int4_storage;
        DeviceBuffer<float> scales_storage;
        std::vector<int64_t> shape;
        LinearWeight linear;
    };

    using WeightMap = std::unordered_map<std::string, DeviceWeight>;

    struct SharedModelWeights {
        std::mutex mutex;
        WeightMap tensors;

        size_t memory_bytes() const;
    };

    struct LayerCommon {
        const __nv_bfloat16* operator_norm = nullptr;
        const __nv_bfloat16* ffn_norm = nullptr;
        const LinearWeight* w13 = nullptr;
        const LinearWeight* w2 = nullptr;
    };

    struct AttentionLayer {
        LayerCommon common;
        const LinearWeight* qkv = nullptr;
        const LinearWeight* out = nullptr;
        const __nv_bfloat16* q_norm = nullptr;
        const __nv_bfloat16* k_norm = nullptr;
        DeviceBuffer<__nv_bfloat16> key_cache;
        DeviceBuffer<__nv_bfloat16> value_cache;
        DeviceBuffer<int8_t> key_cache_int8;
        DeviceBuffer<int8_t> value_cache_int8;
        DeviceBuffer<float> key_cache_scales;
        DeviceBuffer<float> value_cache_scales;
    };

    struct ConvolutionLayer {
        LayerCommon common;
        const LinearWeight* conv_in = nullptr;
        const __nv_bfloat16* conv_weight = nullptr;
        const LinearWeight* conv_out = nullptr;
        DeviceBuffer<__nv_bfloat16> conv_state;
    };

    using Layer = std::variant<AttentionLayer, ConvolutionLayer>;

    static LayerCommon& common(Layer& layer) {
        return std::visit([](auto& value) -> LayerCommon& { return value.common; }, layer);
    }
    static const LayerCommon& common(const Layer& layer) {
        return std::visit([](const auto& value) -> const LayerCommon& { return value.common; }, layer);
    }
    static AttentionLayer* as_attention(Layer& layer) {
        return std::get_if<AttentionLayer>(&layer);
    }
    static const AttentionLayer* as_attention(const Layer& layer) {
        return std::get_if<AttentionLayer>(&layer);
    }
    static ConvolutionLayer* as_convolution(Layer& layer) {
        return std::get_if<ConvolutionLayer>(&layer);
    }
    static const ConvolutionLayer* as_convolution(const Layer& layer) {
        return std::get_if<ConvolutionLayer>(&layer);
    }

    const __nv_bfloat16* load_weight(const SafeTensorFile& file,
                                     const std::string& name,
                                     std::vector<int64_t> expected = {});
    const LinearWeight* load_linear_weight(
        const SafeTensorFile& file, const std::string& name,
        std::vector<int64_t> expected);
    const LinearWeight* load_concat_linear_weight(
        const SafeTensorFile& file,
        const std::string& synthetic_name,
        const std::vector<std::pair<std::string, std::vector<int64_t>>>& parts);
    LinearWeight slice_rows(const LinearWeight& weight,
                            int row_offset, int rows) const;

    void linear(const __nv_bfloat16* x, const LinearWeight& weight,
                __nv_bfloat16* y, int m, int n, int k, float beta = 0.0f);
    void linear_cublas(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                       __nv_bfloat16* y, int m, int n, int k, float beta);
    void linear_cublaslt(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                         __nv_bfloat16* y, int m, int n, int k, float beta);
    LtPlan& get_or_create_lt_plan(const __nv_bfloat16* x,
                                  const __nv_bfloat16* weight,
                                  int m, int n, int k);
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
    void release_local_kv_cache();
    bool local_kv_cache_available() const { return local_kv_cache_available_; }
    SessionPhase phase() const { return phase_; }
    bool ready_for_decode() const { return phase_ == SessionPhase::Ready; }
    bool decode_pending() const { return phase_ == SessionPhase::DecodePending; }
    int position() const { return position_; }
    bool cuda_graph_ready() const {
        return decode_graph_.ready() || segmented_decode_graph_.ready();
    }

    void prefill_batched(const std::vector<int32_t>& tokens);
    void prefill_legacy(const std::vector<int32_t>& tokens);
    void allocate_prefill_workspace(int rows);
    void release_prefill_workspace();
    void run_mlp_decode(const LayerCommon& common_layer);
    void run_mlp_prefill(const LayerCommon& common_layer, int rows);
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

    ExecutionPlan plan_;
    ModelOptions options_;
    GenerationConfig generation_;
    CudaStream stream_;
    CublasHandle cublas_;
    CublasLtHandle cublas_lt_;
    DeviceBuffer<std::byte> lt_workspace_;
    std::unordered_map<MatmulKey, std::unique_ptr<LtPlan>, MatmulKeyHash> lt_plans_;
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
    std::vector<Layer> layers_;
    const LinearWeight* embedding_ = nullptr;
    const __nv_bfloat16* final_norm_ = nullptr;

    DeviceBuffer<int32_t> position_device_{1};
    DeviceBuffer<int32_t> sampled_device_{1};
    PinnedBuffer<int32_t> sampled_host_{1};
    DeviceBuffer<uint8_t> seen_tokens_{LfmConfig::vocab};
    DeviceBuffer<float> sampling_scores_{LfmConfig::vocab};
    DeviceBuffer<float> topk_values_{LfmConfig::max_top_k};
    DeviceBuffer<int32_t> topk_indices_{LfmConfig::max_top_k};
    DeviceBuffer<uint64_t> rng_state_{1};

    DeviceBuffer<__nv_bfloat16> hidden_{LfmConfig::hidden};
    DeviceBuffer<__nv_bfloat16> residual_{LfmConfig::hidden};
    DeviceBuffer<__nv_bfloat16> normed_{LfmConfig::hidden};
    DeviceBuffer<__nv_bfloat16> op_output_{LfmConfig::hidden};
    DeviceBuffer<__nv_bfloat16> qkv_output_{LfmConfig::qkv_width};
    DeviceBuffer<__nv_bfloat16> conv_projected_{3 * LfmConfig::hidden};
    DeviceBuffer<__nv_bfloat16> gate_up_{2 * LfmConfig::intermediate};
    DeviceBuffer<__nv_bfloat16> activated_{LfmConfig::intermediate};
    DeviceBuffer<__nv_bfloat16> mlp_output_{LfmConfig::hidden};
    DeviceBuffer<__nv_bfloat16> logits_{LfmConfig::vocab};
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
};

} // namespace lfm
