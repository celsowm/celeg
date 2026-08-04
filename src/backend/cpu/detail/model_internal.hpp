#pragma once

#include "celeg/backend/cpu/kernels.hpp"
#include "celeg/backend/cpu/model.hpp"
#include "celeg/backend/cpu/paged_kv.hpp"
#include "celeg/backend/cpu/prefix_cache.hpp"
#include "celeg/backend/cpu/quantization.hpp"
#include "celeg/backend/cpu/thread_pool.hpp"
#include "celeg/model/resolved.hpp"
#include "celeg/model/program.hpp"
#include "celeg/model/weights/roles.hpp"
#include "expert_cache.hpp"

#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace celeg {

struct CpuWorkspace {
    void ensure(size_t rows, const RuntimeTopology& shape) {
        hidden.resize(rows * shape.hidden);
        residual.resize(rows * shape.hidden);
        normed.resize(rows * shape.hidden);
        op_output.resize(rows * static_cast<size_t>(shape.maximum_attention_query_heads()) *
                         static_cast<size_t>(shape.maximum_attention_head_dim()));
        qkv.resize(rows * static_cast<size_t>(shape.maximum_attention_projection_width()));
        conv_projected.resize(rows * 3ULL * shape.hidden);
        gate_up.resize(rows * 2ULL * shape.max_feed_forward_intermediate);
        activated.resize(rows * shape.max_feed_forward_intermediate);
        mlp_output.resize(rows * shape.hidden);
    }

    std::vector<float> hidden, residual, normed, op_output, qkv;
    std::vector<float> per_layer_input, per_layer_context, per_layer_gate;
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

struct CpuCompiledModel {
    struct BatchScratch;
    struct CommonWeights {
        std::vector<float> operator_norm;
        std::vector<float> post_attention_norm;
        std::vector<float> pre_feed_forward_norm;
        std::vector<float> post_feed_forward_norm;
        std::vector<float> per_layer_input_norm;
        std::vector<float> ffn_norm;
        CpuLinearWeight w13;
        CpuLinearWeight w2;
        CpuLinearWeight per_layer_input_gate;
        CpuLinearWeight per_layer_projection;
        float layer_scalar = 1.0f;
    };
    struct AttentionWeights {
        CommonWeights common;
        CpuLinearWeight q;
        CpuLinearWeight k;
        CpuLinearWeight v;
        CpuLinearWeight out;
        std::vector<float> q_norm;
        std::vector<float> k_norm;
    };
    struct ConvolutionWeights {
        CommonWeights common;
        CpuLinearWeight in;
        std::vector<float> weight_tap_major;
        CpuLinearWeight out;
    };
    struct MoeWeights {
        CommonWeights common;
        std::variant<AttentionWeights, ConvolutionWeights> operator_layer;
        std::vector<float> router;
        std::vector<float> router_bias;
        std::vector<CpuLinearWeight> expert_w13;
        std::vector<CpuLinearWeight> expert_w2;
        int layer_index = -1;
        int num_experts = 0;
        int experts_per_token = 0;
        bool normalize_topk = false;
        bool use_expert_bias = false;
        bool disk_cached = false;
        float routed_scaling_factor = 1.0f;
    };
    using WeightLayer = std::variant<AttentionWeights, ConvolutionWeights, MoeWeights>;

    struct CpuWeightStore {
        CpuLinearWeight embedding;
        CpuLinearWeight lm_head;
        CpuLinearWeight per_layer_embedding;
        CpuLinearWeight per_layer_context_projection;
        std::vector<float> per_layer_projection_norm;
        std::vector<float> final_norm;
        std::vector<WeightLayer> layers;
    };

    struct Shared {
        Shared(const std::string& path, int context, CpuModelOptions requested,
               std::shared_ptr<const RuntimeContext> runtime);

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
        std::shared_ptr<const CpuExpertWeights> acquire_expert(int layer,
                                                               int expert);
        size_t weights_memory_bytes() const;

        std::string model_path;
        std::shared_ptr<const RuntimeContext> runtime;
        bool native_checkpoint = false;
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
        RuntimeTopology shape;
        CompiledModelProgram program;
        std::string model_identity;
        std::vector<TensorRequest> weight_requests;
        bool tie_word_embeddings = true;
        float final_logit_softcap = 0.0f;
        CpuWeightStore weight_store;
        std::unique_ptr<CpuPackReader> expert_pack_reader;
        std::unique_ptr<CpuExpertCache> expert_cache;
        mutable std::mutex expert_pack_mutex;
        std::vector<std::shared_ptr<CpuKvPagePool>> kv_pools;
        std::vector<int> layer_to_kv_pool;
        std::vector<int> layer_to_kv_owner;
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

    struct CpuSessionState {
        explicit CpuSessionState(GenerationConfig config)
            : generation(std::move(config)) {}

        GenerationConfig generation;
        std::vector<LayerState> states;
        std::vector<uint8_t> seen;
        int position_value = 0;
        uint64_t attention_parallel_calls = 0;
        SessionPhase phase = SessionPhase::Empty;
        uint64_t rng_state = 1;
        RuntimeMetrics metrics;
        CpuPrefillProfile prefill_profile;
    };

    CpuCompiledModel(std::shared_ptr<Shared> shared_weights,
                     GenerationConfig generation_config,
                     int preferred_numa_node = -1);
    ~CpuCompiledModel();

    void allocate_state();
    void allocate_activations();
    void reset();
    void forward_token(int32_t token, bool compute_logits,
                       const PromptEmbedding* embeddings = nullptr);
    void forward_chunk(std::span<const int32_t> tokens, bool compute_logits);
    static void forward_batch(std::span<CpuCompiledModel* const> sessions,
                              std::span<const int32_t> tokens,
                              std::span<const uint8_t> compute_logits);
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
    void run_attention(const AttentionState& state, const AttentionSpec& attention,
                       const float* q, float* output, int sequence_length) const;
    void release_attention_pages(AttentionState& state) noexcept;

    CpuPrefixSnapshot export_prefix_snapshot() const;
    void restore_prefix_snapshot(CpuPrefixSnapshot snapshot,
                                 bool ready_for_decode);

    std::shared_ptr<Shared> shared;
    CpuWorkspace workspace_;
    CpuSessionState session_;
    int preferred_numa_node = -1;
};

} // namespace celeg
