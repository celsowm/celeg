#pragma once

#include "lfm/backend/cpu/kernels.hpp"
#include "lfm/backend/cpu/model.hpp"
#include "lfm/backend/cpu/paged_kv.hpp"
#include "lfm/backend/cpu/prefix_cache.hpp"
#include "lfm/backend/cpu/quantization.hpp"
#include "lfm/backend/cpu/thread_pool.hpp"
#include "lfm/model/config/shape.hpp"
#include "lfm/model/config/variant.hpp"
#include "lfm/model/weights/roles.hpp"

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace lfm {

// Capacity-managed activation storage shared by all CPU execution modes.
// The executor chooses the row count; buffers are retained between calls.
struct CpuExecutionWorkspace {
    void ensure(size_t rows, const ModelShape& shape) {
        hidden.resize(rows * shape.hidden);
        residual.resize(rows * shape.hidden);
        normed.resize(rows * shape.hidden);
        op_output.resize(rows * shape.hidden);
        qkv.resize(rows * shape.qkv_width);
        conv_projected.resize(rows * 3ULL * shape.hidden);
        gate_up.resize(rows * 2ULL * shape.intermediate);
        activated.resize(rows * shape.intermediate);
        mlp_output.resize(rows * shape.hidden);
    }

    std::vector<float> hidden, residual, normed, op_output, qkv;
    std::vector<float> conv_projected, gate_up, activated, mlp_output;
    std::vector<float> logits;
    std::vector<float> chunk_hidden, chunk_residual, chunk_normed, chunk_op;
    std::vector<float> chunk_qkv, chunk_conv, chunk_gate_up;
    std::vector<float> chunk_activated, chunk_mlp;
    std::vector<float> final_normed, final_logits;
    std::vector<size_t> terminal_rows;
    std::vector<float> moe_router_logits, moe_router_probs;
    std::vector<std::pair<float, int>> moe_router_scored;
    std::vector<int> moe_selected;
    std::vector<float> moe_weights;
    std::vector<size_t> moe_group_offsets, moe_group_cursor;
    std::vector<size_t> moe_route_order;
    std::vector<int> moe_route_rows, moe_route_experts;
    std::vector<float> moe_route_weights;
    std::vector<float> moe_gathered_normed, moe_gathered_gate_up;
    std::vector<float> moe_gathered_activated, moe_gathered_output;
    std::vector<CpuGroupedGemmJob> moe_gemm_jobs;
};

struct CpuModel::Impl : CpuExecutionWorkspace {
    struct BatchScratch;
    struct CommonWeights {
        std::vector<float> operator_norm;
        std::vector<float> ffn_norm;
        CpuLinearWeight w13;
        CpuLinearWeight w2;
    };
    struct AttentionWeights {
        CommonWeights common;
        CpuLinearWeight qkv;
        CpuLinearWeight out;
        std::vector<float> q_norm;
        std::vector<float> k_norm;
    };
    struct ConvolutionWeights {
        CommonWeights common;
        CpuLinearWeight in;
        // [tap][channel], so a tap is contiguous for SIMD/vector kernels.
        std::vector<float> weight_tap_major;
        CpuLinearWeight out;
    };
    struct MoeWeights {
        CommonWeights common;
        // Every MoE layer still has an LFM operator (attention or
        // short-convolution) before its routed FFN.
        std::variant<AttentionWeights, ConvolutionWeights> operator_layer;
        std::vector<float> router;            // [num_experts * hidden]
        std::vector<float> router_bias;       // [num_experts] (empty if unused)
        std::vector<CpuLinearWeight> expert_w13;  // [num_experts]
        std::vector<CpuLinearWeight> expert_w2;   // [num_experts]
        int num_experts = 0;
        int experts_per_token = 0;
        bool normalize_topk = false;
        bool use_expert_bias = false;
        float routed_scaling_factor = 1.0f;
    };
    using WeightLayer = std::variant<AttentionWeights, ConvolutionWeights, MoeWeights>;

    struct Shared {
        Shared(const std::string& path, int context, CpuModelOptions requested);

        static CpuIsa resolve_isa(CpuIsa requested);
        void prepare_pack_path();
        void load_weights();
        CommonWeights load_common(class IWeightRepository* repository,
                                  class CpuPackReader* reader,
                                  class CpuPackWriter* writer,
                                  int layer);
        CpuLinearWeight load_matrix(class IWeightRepository* repository,
                                    class CpuPackReader* reader,
                                    class CpuPackWriter* writer,
                                    const std::string& name,
                                    const std::vector<int64_t>& expected);
        CpuLinearWeight load_concat(
            class IWeightRepository* repository,
            class CpuPackReader* reader,
            class CpuPackWriter* writer,
            const std::string& synthetic,
            const std::vector<std::pair<std::string, std::vector<int64_t>>>& parts);
        std::vector<float> load_vector(class IWeightRepository* repository,
                                       class CpuPackReader* reader,
                                       class CpuPackWriter* writer,
                                       const std::string& name,
                                       const std::vector<int64_t>& expected);
        size_t weights_memory_bytes() const;

        std::string model_path;
        bool is_gguf = false;
        std::shared_ptr<IWeightRepository> repository;
        int max_context = 0;
        CpuModelOptions options;
        CpuCapabilities capabilities;
        CpuThreadPool pool;
        CpuLinearEngine linear;
        size_t group_size = 32;
        std::filesystem::path pack_file;
        std::string source_id;
        bool loaded_pack = false;
        ModelShape shape;
        const IModelVariant* variant = nullptr;
        const ITensorNamingPolicy* tensor_naming = nullptr;
        bool tie_word_embeddings = true;
        CpuLinearWeight embedding;
        CpuLinearWeight lm_head;
        std::vector<float> final_norm;
        std::vector<WeightLayer> layers;
        std::vector<std::shared_ptr<CpuKvPagePool>> kv_pools;
        std::vector<int> layer_to_kv_pool;
    };

    struct AttentionState {
        size_t pool_index = 0;
        std::vector<CpuKvPageId> pages;
        size_t token_count = 0;
    };
    struct ConvolutionState {
        std::vector<float> state;
    };
    using LayerState = std::variant<AttentionState, ConvolutionState>;

    Impl(std::shared_ptr<Shared> shared_weights,
         GenerationConfig generation_config,
         int preferred_numa_node = -1);
    ~Impl();

    void allocate_state();
    void allocate_activations();
    void reset();
    void forward_token(int32_t token, bool compute_logits);
    void forward_chunk(std::span<const int32_t> tokens, bool compute_logits);
    static void forward_batch(std::span<Impl* const> sessions,
                              std::span<const int32_t> tokens,
                              std::span<const uint8_t> compute_logits);
    int32_t sample();
    void set_generation(GenerationConfig config);
    CpuModelMemoryStats memory_stats() const;

    const CommonWeights& common_weights(size_t layer) const;
    static const AttentionWeights* attention_operator(const WeightLayer& layer);
    static const ConvolutionWeights* convolution_operator(const WeightLayer& layer);
    AttentionState& attention_state(size_t layer);
    const AttentionState& attention_state(size_t layer) const;
    ConvolutionState& convolution_state(size_t layer);
    const ConvolutionState& convolution_state(size_t layer) const;

    void store_kv(AttentionState& state, int position,
                  const float* key, const float* value);
    void run_attention(const AttentionState& state, const float* q,
                       float* output, int sequence_length) const;
    void release_attention_pages(AttentionState& state) noexcept;

    CpuPrefixSnapshot export_prefix_snapshot() const;
    void restore_prefix_snapshot(CpuPrefixSnapshot snapshot,
                                 bool ready_for_decode);

    std::shared_ptr<Shared> shared;
    GenerationConfig generation;
    std::vector<LayerState> states;

    std::vector<uint8_t> seen;

    int position_value = 0;
    int preferred_numa_node = -1;
    uint64_t attention_parallel_calls = 0;
    SessionPhase phase = SessionPhase::Empty;
    uint64_t rng_state = 1;
    RuntimeMetrics metrics;
    CpuPrefillProfile prefill_profile;
};

} // namespace lfm
