#pragma once

#include "celeg/backend/cpu/kernels.hpp"
#include "celeg/backend/cpu/kv_topology.hpp"
#include "celeg/backend/cpu/model.hpp"
#include "celeg/backend/cpu/paged_kv.hpp"
#include "celeg/backend/cpu/prefix_cache.hpp"
#include "celeg/backend/cpu/quantization.hpp"
#include "celeg/backend/cpu/thread_pool.hpp"
#include "celeg/model/resolved.hpp"
#include "celeg/model/program.hpp"
#include "celeg/model/weights/roles.hpp"
#include "expert_cache.hpp"
#include "expert_backing.hpp"

#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include <unordered_map>

namespace celeg {

struct CpuWorkspacePlan {
    size_t hidden = 0;
    size_t attention_output = 0;
    size_t attention_projection = 0;
    size_t latent_state = 0;
    size_t latent_rotary = 0;
    size_t latent_key_rotary = 0;
    size_t gated_delta_qkv = 0;
    size_t gated_delta_output = 0;
    size_t gated_delta_gate = 0;
    size_t mamba_projection = 0;
    size_t mamba_conv = 0;
    size_t mamba_inner = 0;
    size_t feed_forward = 0;
    size_t q8_blocks = 0;

    static CpuWorkspacePlan from_topology(const RuntimeTopology& shape) {
        CpuWorkspacePlan plan;
        plan.hidden = static_cast<size_t>(shape.hidden);
        plan.attention_output = static_cast<size_t>(shape.maximum_attention_query_heads()) *
            static_cast<size_t>(shape.maximum_attention_head_dim());
        plan.attention_output = std::max(plan.attention_output,
            std::max(static_cast<size_t>(shape.mamba2_intermediate),
                     static_cast<size_t>(shape.max_gated_delta_net_output_width())));
        plan.attention_projection = static_cast<size_t>(
            shape.maximum_attention_projection_width());
        for (const AttentionSpec& attention : shape.attention_layouts) {
            plan.attention_projection = std::max(plan.attention_projection,
                static_cast<size_t>(attention.latent_query_width()));
            plan.attention_output = std::max(plan.attention_output,
                static_cast<size_t>(attention.latent_query_content_width()));
            if (const auto* latent = attention.latent_state()) {
                plan.latent_state = std::max(plan.latent_state,
                    static_cast<size_t>(latent->latent_rank));
                plan.latent_rotary = std::max(plan.latent_rotary,
                    static_cast<size_t>(attention.latent_query_rope_width()));
                plan.latent_key_rotary = std::max(plan.latent_key_rotary,
                    static_cast<size_t>(latent->rope_head_dim));
            }
        }
        for (const GatedDeltaNetSpec& spec : shape.gated_delta_net_layouts) {
            plan.gated_delta_qkv = std::max(plan.gated_delta_qkv,
                static_cast<size_t>(2 * spec.key_heads * spec.key_head_dim +
                                    spec.value_heads * spec.value_head_dim));
            plan.gated_delta_output = std::max(plan.gated_delta_output,
                static_cast<size_t>(spec.value_heads * spec.value_head_dim));
            plan.gated_delta_gate = std::max(plan.gated_delta_gate,
                static_cast<size_t>(spec.value_heads));
        }
        for (const Mamba2Spec& spec : shape.mamba2_layouts) {
            plan.mamba_projection = std::max(plan.mamba_projection,
                2ULL * static_cast<size_t>(spec.intermediate_size) +
                2ULL * static_cast<size_t>(spec.group_count) * static_cast<size_t>(spec.state_size) +
                static_cast<size_t>(spec.num_heads));
            plan.mamba_conv = std::max(plan.mamba_conv,
                static_cast<size_t>(spec.intermediate_size) +
                2ULL * static_cast<size_t>(spec.group_count) * static_cast<size_t>(spec.state_size));
        }
        plan.mamba_inner = static_cast<size_t>(shape.mamba2_intermediate);
        plan.feed_forward = static_cast<size_t>(shape.max_feed_forward_intermediate);
        plan.q8_blocks = static_cast<size_t>(shape.hidden) / 256;
        return plan;
    }
};

struct CpuWorkspace {
    void ensure(size_t rows, const CpuWorkspacePlan& plan) {
        hidden.resize(rows * plan.hidden);
        residual.resize(rows * plan.hidden);
        normed.resize(rows * plan.hidden);
        op_output.resize(rows * plan.attention_output);
        attention_gate.resize(rows * plan.attention_output);
        qkv.resize(rows * plan.attention_projection);
        latent_key.resize(rows * plan.latent_state);
        latent_value.resize(rows * plan.latent_state);
        latent_rope.resize(rows * plan.latent_rotary);
        latent_key_rope.resize(rows * plan.latent_key_rotary);
        gated_delta_qkv.resize(rows * plan.gated_delta_qkv);
        gated_delta_z.resize(rows * plan.gated_delta_output);
        gated_delta_b.resize(rows * plan.gated_delta_gate);
        gated_delta_a.resize(rows * plan.gated_delta_gate);
        gated_delta_output.resize(rows * plan.gated_delta_output);
        mamba_projected.resize(rows * plan.mamba_projection);
        mamba_bcx.resize(rows * plan.mamba_conv);
        mamba_inner.resize(rows * plan.mamba_inner);
        conv_projected.resize(rows * 3ULL * plan.hidden);
        gate_up.resize(rows * 2ULL * plan.feed_forward);
        activated.resize(rows * plan.feed_forward);
        mlp_output.resize(rows * plan.hidden);
        shared_output.resize(rows * plan.hidden);
        shared_gate.resize(rows);
        chunk_q8.resize(rows * plan.q8_blocks);
    }

    void ensure_chunk(size_t rows, const CpuWorkspacePlan& plan) {
        chunk_hidden.resize(rows * plan.hidden);
        chunk_residual.resize(rows * plan.hidden);
        chunk_normed.resize(rows * plan.hidden);
        chunk_op.resize(rows * plan.attention_output);
        chunk_qkv.resize(rows * plan.attention_projection);
        chunk_latent_key.resize(rows * plan.latent_state);
        chunk_latent_value.resize(rows * plan.latent_state);
        chunk_latent_rope.resize(rows * plan.latent_rotary);
        chunk_latent_key_rope.resize(rows * plan.latent_key_rotary);
        chunk_attention_gate.resize(rows * plan.attention_output);
        chunk_gated_delta_qkv.resize(rows * plan.gated_delta_qkv);
        chunk_gated_delta_z.resize(rows * plan.gated_delta_output);
        chunk_gated_delta_b.resize(rows * plan.gated_delta_gate);
        chunk_gated_delta_a.resize(rows * plan.gated_delta_gate);
        chunk_gated_delta_output.resize(rows * plan.gated_delta_output);
        chunk_conv.resize(rows * 3ULL * plan.hidden);
        chunk_gate_up.resize(rows * 2ULL * plan.feed_forward);
        chunk_activated.resize(rows * plan.feed_forward);
        chunk_mlp.resize(rows * plan.hidden);
        shared_output.resize(rows * plan.hidden);
        shared_gate.resize(rows);
        chunk_q8.resize(rows * plan.q8_blocks);
        final_normed.resize(plan.hidden);
        terminal_rows.clear();
    }

    std::vector<float> hidden, residual, normed, op_output, attention_gate, qkv;
    std::vector<float> latent_key, latent_value, latent_rope, latent_key_rope;
    std::vector<float> per_layer_input, per_layer_context, per_layer_gate;
    std::vector<float> conv_projected, gate_up, activated, mlp_output;
    std::vector<float> shared_output, shared_gate;
    std::vector<float> mamba_projected, mamba_bcx, mamba_inner;
    std::vector<float> gated_delta_qkv, gated_delta_z, gated_delta_b, gated_delta_a;
    std::vector<float> gated_delta_output;
    std::vector<float> logits;
    std::vector<float> chunk_hidden, chunk_residual, chunk_normed, chunk_op;
    std::vector<float> chunk_qkv, chunk_conv, chunk_gate_up;
    std::vector<float> chunk_latent_key, chunk_latent_value, chunk_latent_rope;
    std::vector<float> chunk_latent_key_rope;
    std::vector<float> chunk_attention_gate;
    std::vector<float> chunk_gated_delta_qkv, chunk_gated_delta_z;
    std::vector<float> chunk_gated_delta_b, chunk_gated_delta_a, chunk_gated_delta_output;
    std::vector<float> chunk_activated, chunk_mlp;
    std::vector<CpuQ8KBlock> chunk_q8;
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
    std::vector<std::shared_ptr<const CpuExpertWeights>> moe_cached_experts;
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
        CpuLinearWeight mlp_up;
        CpuLinearWeight per_layer_input_gate;
        CpuLinearWeight per_layer_projection;
        float layer_scalar = 1.0f;
    };
    struct AttentionWeights {
        CommonWeights common;
        CpuLinearWeight q;
        CpuLinearWeight latent_q_rope;
        CpuLinearWeight k;
        CpuLinearWeight v;
        CpuLinearWeight gate;
        CpuLinearWeight latent_k_rope;
        CpuLinearWeight out;
        std::vector<float> q_norm;
        std::vector<float> k_norm;
        std::vector<float> relative_bias;
    };
    struct ConvolutionWeights {
        CommonWeights common;
        CpuLinearWeight in;
        std::vector<float> weight_tap_major;
        CpuLinearWeight out;
    };
    struct Mamba2Weights {
        CommonWeights common;
        CpuLinearWeight in;
        std::vector<float> conv_weight;
        std::vector<float> conv_bias;
        std::vector<float> dt_bias;
        std::vector<float> a_log;
        std::vector<float> d;
        std::vector<float> norm;
        CpuLinearWeight out;
    };
    struct GatedDeltaNetWeights {
        CommonWeights common;
        GatedDeltaNetSpec spec;
        CpuLinearWeight qkv;
        CpuLinearWeight z;
        CpuLinearWeight b;
        CpuLinearWeight a;
        std::vector<float> conv_weight;
        std::vector<float> dt_bias;
        std::vector<float> a_log;
        std::vector<float> norm;
        CpuLinearWeight out;
    };
    struct MlpOnlyWeights {
        CommonWeights common;
    };
    struct MoeWeights {
        CommonWeights common;
        std::variant<AttentionWeights, ConvolutionWeights, GatedDeltaNetWeights,
                     Mamba2Weights, MlpOnlyWeights> operator_layer;
        std::vector<float> router;
        std::vector<float> router_bias;
        std::vector<CpuLinearWeight> expert_w13;
        std::vector<CpuLinearWeight> expert_w2;
        CpuLinearWeight shared_w13;
        CpuLinearWeight shared_w2;
        CpuLinearWeight shared_gate;
        int layer_index = -1;
        int num_experts = 0;
        int experts_per_token = 0;
        bool normalize_topk = false;
        bool use_expert_bias = false;
        bool disk_cached = false;
        float routed_scaling_factor = 1.0f;
    };
    using WeightLayer = std::variant<AttentionWeights, ConvolutionWeights,
                                     GatedDeltaNetWeights, Mamba2Weights,
                                     MlpOnlyWeights, MoeWeights>;

    struct CpuWeightStore {
        CpuLinearWeight embedding;
        CpuLinearWeight lm_head;
        CpuLinearWeight per_layer_embedding;
        CpuLinearWeight per_layer_context_projection;
        std::vector<float> embedding_norm;
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
        bool compressed_checkpoint = false;
        std::shared_ptr<IWeightRepository> repository;
        int max_context = 0;
        CpuModelOptions options;
        CpuCapabilities capabilities;
        CpuThreadPool pool;
        CpuLinearEngine linear;
        CpuWorkspacePlan workspace_plan;
        size_t group_size = 32;
        std::filesystem::path pack_file;
        std::string source_id;
        bool loaded_pack = false;
        RuntimeTopology shape;
        CompiledModelProgram program;
        std::string model_identity;
        std::vector<TensorRequest> weight_requests;
        std::unordered_map<std::string, CpuLinearWeight> compressed_linear_cache;
        bool tie_word_embeddings = true;
        float final_logit_softcap = 0.0f;
        CpuWeightStore weight_store;
        std::unique_ptr<CpuExpertBackingStore> expert_backing_store;
        mutable std::mutex expert_pack_mutex;
        std::vector<std::shared_ptr<CpuKvPagePool>> kv_pools;
        std::vector<int> layer_to_kv_pool;
        std::vector<int> layer_to_kv_owner;
        std::unordered_map<int, std::shared_ptr<const CpuExternalAttentionMemory>>
            external_attention_memory;
    };

    struct AttentionState {
        size_t pool_index = 0;
        std::vector<CpuKvPageId> pages;
        size_t token_count = 0;
    };
    struct ConvolutionState {
        std::vector<float> state;
    };
    struct Mamba2State {
        std::vector<float> conv;
        std::vector<float> ssm;
    };
    struct GatedDeltaNetState {
        std::vector<float> conv;
        std::vector<float> recurrent;
    };
    using LayerState = std::variant<AttentionState, ConvolutionState,
                                    GatedDeltaNetState, Mamba2State>;

    struct CpuSessionState {
        explicit CpuSessionState(GenerationConfig config)
            : generation(std::move(config)) {}

        GenerationConfig generation;
        std::vector<LayerState> states;
        std::vector<uint8_t> seen;
        int position_value = 0;
        std::array<int32_t, 3> next_rope_position{0, 0, 0};
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
    void forward_chunk(std::span<const int32_t> tokens, bool compute_logits,
                       const PromptEmbedding* embeddings = nullptr);
    static void forward_batch(std::span<CpuCompiledModel* const> sessions,
                              std::span<const int32_t> tokens,
                              std::span<const uint8_t> compute_logits);
    void set_generation(GenerationConfig config);
    CpuModelMemoryStats memory_stats() const;

    const CommonWeights& common_weights(size_t layer) const;
    static const AttentionWeights* attention_operator(const WeightLayer& layer);
    static const ConvolutionWeights* convolution_operator(const WeightLayer& layer);
    static const GatedDeltaNetWeights* gated_delta_net_operator(const WeightLayer& layer);
    static const Mamba2Weights* mamba2_operator(const WeightLayer& layer);
    AttentionState& attention_state(size_t layer);
    const AttentionState& attention_state(size_t layer) const;
    ConvolutionState& convolution_state(size_t layer);
    const ConvolutionState& convolution_state(size_t layer) const;
    Mamba2State& mamba2_state(size_t layer);
    const Mamba2State& mamba2_state(size_t layer) const;
    GatedDeltaNetState& gated_delta_net_state(size_t layer);
    const GatedDeltaNetState& gated_delta_net_state(size_t layer) const;

    void store_kv(AttentionState& state, int position,
                  const float* key, const float* value);
    void store_latent(AttentionState& state, int position,
                      const float* key, const float* value,
                      const float* rotary);
    void run_attention(const AttentionState& state, const AttentionSpec& attention,
                       const float* q, float* output, int sequence_length,
                       std::span<const float> relative_bias = {}) const;
    void run_latent_attention(const AttentionState& state, const AttentionSpec& attention,
                              const float* query_content, const float* query_rope,
                              float* output, int sequence_length,
                              int query_position = -1,
                              std::span<const float> relative_bias = {}) const;
    void set_external_attention_memory(
        int slot, std::shared_ptr<const CpuExternalAttentionMemory> memory);
    void run_external_attention(const AttentionSpec& attention,
                                const CpuExternalAttentionMemory& memory,
                                const float* q, float* output,
                                std::span<const float> relative_bias = {}) const;
    void release_attention_pages(AttentionState& state) noexcept;

    CpuPrefixSnapshot export_prefix_snapshot() const;
    void restore_prefix_snapshot(CpuPrefixSnapshot snapshot,
                                 bool ready_for_decode);

    std::shared_ptr<Shared> shared;
    CpuWorkspace workspace_;
    CpuSessionState session_;
    int preferred_numa_node = -1;
};

// Focused view used by token/chunk operators.  Operators that only need
// linear execution, scratch storage, and session profiling no longer receive
// the owning model object and cannot accidentally reach unrelated state.
struct CpuExecutionContext {
    CpuCompiledModel::Shared& shared;
    CpuWorkspace& workspace;
    CpuCompiledModel::CpuSessionState& session;
};

void validate_cpu_packed_batch(std::span<CpuCompiledModel* const> sessions);
void execute_cpu_packed_batch(std::span<CpuCompiledModel* const> sessions,
                              std::span<const int32_t> tokens,
                              std::span<const uint8_t> compute_logits);

} // namespace celeg
