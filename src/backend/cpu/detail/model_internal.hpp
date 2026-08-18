#pragma once

#include "celeg/backend/cpu/kernels.hpp"
#include "celeg/backend/cpu/kv_topology.hpp"
#include "celeg/backend/cpu/model.hpp"
#include "celeg/backend/cpu/paged_kv.hpp"
#include "celeg/backend/cpu/prefix_cache.hpp"
#include "celeg/backend/cpu/quantization.hpp"
#include "celeg/backend/cpu/thread_pool.hpp"
#include "celeg/detail/exhaustive_visit.hpp"
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
    size_t latent_projection = 0;
    size_t latent_output = 0;
    size_t gated_delta_qkv = 0;
    size_t gated_delta_output = 0;
    size_t gated_delta_gate = 0;
    size_t mamba_projection = 0;
    size_t mamba_conv = 0;
    size_t mamba_inner = 0;
    size_t feed_forward = 0;
    size_t q8_blocks = 0;
    int conv_cache = 0;

    static CpuWorkspacePlan from_program(const CompiledModelProgram& program) {
        CpuWorkspacePlan plan;
        plan.hidden = static_cast<size_t>(program.hidden);
        plan.q8_blocks = static_cast<size_t>(program.hidden) / 256;
        for (const CompiledLayerProgram& layer : program.layers) {
            std::visit([&](const auto& mixer) {
                using Mixer = std::decay_t<decltype(mixer)>;
                if constexpr (std::is_same_v<Mixer, CompiledAttentionProgram>) {
                    const AttentionSpec& attn = mixer.semantics;
                    plan.attention_output = std::max(
                        plan.attention_output,
                        static_cast<size_t>(attn.query_heads) * static_cast<size_t>(attn.head_dim));
                    plan.attention_projection = std::max(
                        plan.attention_projection,
                        static_cast<size_t>(attn.projection_width()));
                    plan.attention_output = std::max(
                        plan.attention_output,
                        static_cast<size_t>(attn.uses_latent_state() ? attn.latent_query_content_width()
                                                                     : attn.query_width()));
                    if (const auto* latent = attn.latent_state()) {
                        plan.latent_state = std::max(
                            plan.latent_state, static_cast<size_t>(latent->latent_rank));
                        plan.latent_rotary = std::max(
                            plan.latent_rotary, static_cast<size_t>(attn.latent_query_rope_width()));
                        plan.latent_key_rotary = std::max(
                            plan.latent_key_rotary,
                            static_cast<size_t>(latent->decoupled_rope ? latent->rope_head_dim : 0));
                        plan.latent_output = std::max(
                            plan.latent_output, static_cast<size_t>(attn.latent_output_width()));
                        plan.latent_projection = std::max(
                            plan.latent_projection,
                            static_cast<size_t>(attn.latent_query_projection_width()));
                        plan.attention_projection = std::max(
                            plan.attention_projection, static_cast<size_t>(attn.latent_query_rope_width()));
                        plan.attention_projection = std::max(
                            plan.attention_projection,
                            static_cast<size_t>(attn.latent_query_projection_width()));
                        plan.attention_output = std::max(
                            plan.attention_output, static_cast<size_t>(attn.latent_output_width()));
                    }
            } else if constexpr (std::is_same_v<Mixer, ShortConvolutionSpec>) {
                plan.conv_cache = std::max(plan.conv_cache, mixer.cache_length);
            } else if constexpr (std::is_same_v<Mixer, GatedDeltaNetSpec>) {
                    plan.gated_delta_qkv = std::max(
                        plan.gated_delta_qkv, static_cast<size_t>(mixer.qkv_width()));
                    plan.gated_delta_output = std::max(
                        plan.gated_delta_output, static_cast<size_t>(mixer.value_width()));
                    plan.gated_delta_gate = std::max(
                        plan.gated_delta_gate,
                        static_cast<size_t>(std::max(mixer.value_heads, mixer.decay_width())));
                    plan.attention_output = std::max(
                        plan.attention_output, static_cast<size_t>(mixer.value_width()));
                } else if constexpr (std::is_same_v<Mixer, Mamba2Spec>) {
                    plan.mamba_inner = std::max(
                        plan.mamba_inner, static_cast<size_t>(mixer.intermediate_size));
                    plan.mamba_projection = std::max(
                        plan.mamba_projection,
                        static_cast<size_t>(2 * mixer.intermediate_size +
                                            2 * mixer.group_count * mixer.state_size + mixer.num_heads));
                    plan.mamba_conv = std::max(
                        plan.mamba_conv,
                        static_cast<size_t>(mixer.intermediate_size +
                                            2 * mixer.group_count * mixer.state_size));
                    plan.attention_output = std::max(
                        plan.attention_output, static_cast<size_t>(mixer.intermediate_size));
                } else if constexpr (std::is_same_v<Mixer, MlpBlockSpec>) {
                    plan.feed_forward = std::max(
                        plan.feed_forward, static_cast<size_t>(mixer.intermediate_size));
                }
            }, layer.mixer);

            std::visit([&](const auto& ff) {
                using FF = std::decay_t<decltype(ff)>;
                if constexpr (std::is_same_v<FF, std::monostate>) {
                    return;
                } else if constexpr (std::is_same_v<FF, CompiledDenseFeedForwardProgram>) {
                    plan.feed_forward = std::max(
                        plan.feed_forward, static_cast<size_t>(ff.intermediate_size));
                } else if constexpr (std::is_same_v<FF, MoeLayerProgram>) {
                    plan.feed_forward = std::max(
                        plan.feed_forward, static_cast<size_t>(ff.routed.mlp.intermediate_size));
                }
            }, layer.feed_forward);
        }
        return plan;
    }
};

struct CpuCommonWorkspace {
    std::vector<float> hidden;
    std::vector<float> residual;
    std::vector<float> normed;
    std::vector<float> mlp_output;
    std::vector<float> shared_output;
    std::vector<float> shared_gate;
    std::vector<float> logits;
    std::vector<float> per_layer_input;
    std::vector<float> per_layer_context;
    std::vector<float> per_layer_gate;
    std::vector<CpuQ8KBlock> chunk_q8;
    std::vector<float> final_normed;
    std::vector<float> final_logits;
    std::vector<size_t> terminal_rows;

    void ensure(size_t rows, const CpuWorkspacePlan& plan) {
        hidden.resize(rows * plan.hidden);
        residual.resize(rows * plan.hidden);
        normed.resize(rows * plan.hidden);
        mlp_output.resize(rows * plan.hidden);
        shared_output.resize(rows * plan.hidden);
        shared_gate.resize(rows);
        chunk_q8.resize(rows * plan.q8_blocks);
        final_normed.resize(plan.hidden);
    }
};

struct CpuAttentionWorkspace {
    std::vector<float> op_output;
    std::vector<float> attention_gate;
    std::vector<float> qkv;
    std::vector<float> latent_key;
    std::vector<float> latent_value;
    std::vector<float> latent_rope;
    std::vector<float> latent_key_rope;
    std::vector<float> latent_decompressed;
    std::vector<float> latent_projection;

    void ensure(size_t rows, const CpuWorkspacePlan& plan) {
        op_output.resize(rows * plan.attention_output);
        attention_gate.resize(rows * plan.attention_output);
        qkv.resize(rows * plan.attention_projection);
        latent_key.resize(rows * plan.latent_state);
        latent_value.resize(rows * plan.latent_state);
        latent_rope.resize(rows * plan.latent_rotary);
        latent_key_rope.resize(rows * plan.latent_key_rotary);
        latent_decompressed.resize(rows * plan.latent_output);
        latent_projection.resize(rows * plan.latent_projection);
    }
};

struct CpuRecurrentWorkspace {
    std::vector<float> conv_projected;
    std::vector<float> gated_delta_qkv;
    std::vector<float> gated_delta_z;
    std::vector<float> gated_delta_b;
    std::vector<float> gated_delta_a;
    std::vector<float> gated_delta_output;
    std::vector<float> mamba_projected;
    std::vector<float> mamba_bcx;
    std::vector<float> mamba_inner;

    void ensure(size_t rows, const CpuWorkspacePlan& plan) {
        conv_projected.resize(rows * 3ULL * plan.hidden);
        gated_delta_qkv.resize(rows * plan.gated_delta_qkv);
        gated_delta_z.resize(rows * plan.gated_delta_output);
        gated_delta_b.resize(rows * plan.gated_delta_gate);
        gated_delta_a.resize(rows * plan.gated_delta_gate);
        gated_delta_output.resize(rows * plan.gated_delta_output);
        mamba_projected.resize(rows * plan.mamba_projection);
        mamba_bcx.resize(rows * plan.mamba_conv);
        mamba_inner.resize(rows * plan.mamba_inner);
    }
};

struct CpuFeedForwardWorkspace {
    std::vector<float> gate_up;
    std::vector<float> activated;
    std::vector<float> moe_router_logits;
    std::vector<float> moe_router_probs;
    std::vector<std::pair<float, int>> moe_router_scored;
    std::vector<int> moe_selected;
    std::vector<float> moe_weights;
    std::vector<size_t> moe_group_offsets;
    std::vector<size_t> moe_group_cursor;
    std::vector<size_t> moe_route_order;
    std::vector<int> moe_route_rows;
    std::vector<int> moe_route_experts;
    std::vector<float> moe_route_weights;
    std::vector<float> moe_gathered_normed;
    std::vector<float> moe_gathered_gate_up;
    std::vector<float> moe_gathered_activated;
    std::vector<float> moe_gathered_output;
    std::vector<CpuGroupedGemmJob> moe_gemm_jobs;
    std::vector<std::shared_ptr<const CpuExpertWeights>> moe_cached_experts;

    void ensure(size_t rows, const CpuWorkspacePlan& plan) {
        gate_up.resize(rows * 2ULL * plan.feed_forward);
        activated.resize(rows * plan.feed_forward);
    }
};

struct CpuWorkspace {
    CpuCommonWorkspace common;
    CpuAttentionWorkspace attention;
    CpuRecurrentWorkspace recurrent;
    CpuFeedForwardWorkspace feed_forward;

    std::vector<float>& hidden = common.hidden;
    std::vector<float>& residual = common.residual;
    std::vector<float>& normed = common.normed;
    std::vector<float>& mlp_output = common.mlp_output;
    std::vector<float>& shared_output = common.shared_output;
    std::vector<float>& shared_gate = common.shared_gate;
    std::vector<float>& logits = common.logits;
    std::vector<float>& per_layer_input = common.per_layer_input;
    std::vector<float>& per_layer_context = common.per_layer_context;
    std::vector<float>& per_layer_gate = common.per_layer_gate;
    std::vector<CpuQ8KBlock>& chunk_q8 = common.chunk_q8;
    std::vector<float>& final_normed = common.final_normed;
    std::vector<float>& final_logits = common.final_logits;
    std::vector<size_t>& terminal_rows = common.terminal_rows;

    std::vector<float>& op_output = attention.op_output;
    std::vector<float>& attention_gate = attention.attention_gate;
    std::vector<float>& qkv = attention.qkv;
    std::vector<float>& latent_key = attention.latent_key;
    std::vector<float>& latent_value = attention.latent_value;
    std::vector<float>& latent_rope = attention.latent_rope;
    std::vector<float>& latent_key_rope = attention.latent_key_rope;
    std::vector<float>& latent_decompressed = attention.latent_decompressed;
    std::vector<float>& latent_projection = attention.latent_projection;

    std::vector<float>& conv_projected = recurrent.conv_projected;
    std::vector<float>& gated_delta_qkv = recurrent.gated_delta_qkv;
    std::vector<float>& gated_delta_z = recurrent.gated_delta_z;
    std::vector<float>& gated_delta_b = recurrent.gated_delta_b;
    std::vector<float>& gated_delta_a = recurrent.gated_delta_a;
    std::vector<float>& gated_delta_output = recurrent.gated_delta_output;
    std::vector<float>& mamba_projected = recurrent.mamba_projected;
    std::vector<float>& mamba_bcx = recurrent.mamba_bcx;
    std::vector<float>& mamba_inner = recurrent.mamba_inner;

    std::vector<float>& gate_up = feed_forward.gate_up;
    std::vector<float>& activated = feed_forward.activated;
    std::vector<float>& moe_router_logits = feed_forward.moe_router_logits;
    std::vector<float>& moe_router_probs = feed_forward.moe_router_probs;
    std::vector<std::pair<float, int>>& moe_router_scored = feed_forward.moe_router_scored;
    std::vector<int>& moe_selected = feed_forward.moe_selected;
    std::vector<float>& moe_weights = feed_forward.moe_weights;
    std::vector<size_t>& moe_group_offsets = feed_forward.moe_group_offsets;
    std::vector<size_t>& moe_group_cursor = feed_forward.moe_group_cursor;
    std::vector<size_t>& moe_route_order = feed_forward.moe_route_order;
    std::vector<int>& moe_route_rows = feed_forward.moe_route_rows;
    std::vector<int>& moe_route_experts = feed_forward.moe_route_experts;
    std::vector<float>& moe_route_weights = feed_forward.moe_route_weights;
    std::vector<float>& moe_gathered_normed = feed_forward.moe_gathered_normed;
    std::vector<float>& moe_gathered_gate_up = feed_forward.moe_gathered_gate_up;
    std::vector<float>& moe_gathered_activated = feed_forward.moe_gathered_activated;
    std::vector<float>& moe_gathered_output = feed_forward.moe_gathered_output;
    std::vector<CpuGroupedGemmJob>& moe_gemm_jobs = feed_forward.moe_gemm_jobs;
    std::vector<std::shared_ptr<const CpuExpertWeights>>& moe_cached_experts =
        feed_forward.moe_cached_experts;

    // Legacy chunk alias references for non-overlapping chunk forward paths
    std::vector<float>& chunk_hidden = common.hidden;
    std::vector<float>& chunk_residual = common.residual;
    std::vector<float>& chunk_normed = common.normed;
    std::vector<float>& chunk_op = attention.op_output;
    std::vector<float>& chunk_qkv = attention.qkv;
    std::vector<float>& chunk_conv = recurrent.conv_projected;
    std::vector<float>& chunk_gate_up = feed_forward.gate_up;
    std::vector<float>& chunk_latent_key = attention.latent_key;
    std::vector<float>& chunk_latent_value = attention.latent_value;
    std::vector<float>& chunk_latent_rope = attention.latent_rope;
    std::vector<float>& chunk_latent_key_rope = attention.latent_key_rope;
    std::vector<float>& chunk_latent_decompressed = attention.latent_decompressed;
    std::vector<float>& chunk_latent_projection = attention.latent_projection;
    std::vector<float>& chunk_attention_gate = attention.attention_gate;
    std::vector<float>& chunk_gated_delta_qkv = recurrent.gated_delta_qkv;
    std::vector<float>& chunk_gated_delta_z = recurrent.gated_delta_z;
    std::vector<float>& chunk_gated_delta_b = recurrent.gated_delta_b;
    std::vector<float>& chunk_gated_delta_a = recurrent.gated_delta_a;
    std::vector<float>& chunk_gated_delta_output = recurrent.gated_delta_output;
    std::vector<float>& chunk_activated = feed_forward.activated;
    std::vector<float>& chunk_mlp = common.mlp_output;

    void ensure(size_t rows, const CpuWorkspacePlan& plan) {
        common.ensure(rows, plan);
        attention.ensure(rows, plan);
        recurrent.ensure(rows, plan);
        feed_forward.ensure(rows, plan);
    }

    void ensure_chunk(size_t rows, const CpuWorkspacePlan& plan) {
        ensure(rows, plan);
        terminal_rows.clear();
    }
};

struct CpuCompiledModel {
    struct BatchScratch;
    struct CommonWeights {
        CommonWeights() : layer_scalar(1.0f) {}

        std::vector<float> mixer_norm_before;
        std::vector<float> mixer_norm_after;
        std::vector<float> feed_forward_norm_before;
        std::vector<float> feed_forward_norm_after;
        std::vector<float> per_layer_input_norm;
        float layer_scalar;
    };
    struct AttentionWeights {
        CpuLinearWeight q;
        CpuLinearWeight latent_q_rope;
        CpuLinearWeight latent_q_projection;
        CpuLinearWeight latent_q_expansion;
        CpuLinearWeight latent_k_projection;
        CpuLinearWeight latent_expansion;
        CpuLinearWeight k;
        CpuLinearWeight v;
        CpuLinearWeight gate;
        CpuLinearWeight latent_k_rope;
        CpuLinearWeight out;
        std::vector<float> q_norm;
        std::vector<float> k_norm;
        std::vector<float> latent_q_norm;
        std::vector<float> latent_k_norm;
        std::vector<float> relative_bias;
    };
    struct ConvolutionWeights {
        ShortConvolutionSpec spec;
        CpuLinearWeight in;
        std::vector<float> weight_tap_major;
        CpuLinearWeight out;
    };
    struct Mamba2Weights {
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
        GatedDeltaNetSpec spec;
        CpuLinearWeight qkv;
        CpuLinearWeight q;
        CpuLinearWeight k;
        CpuLinearWeight v;
        CpuLinearWeight z;
        CpuLinearWeight b;
        CpuLinearWeight a;
        std::vector<float> conv_weight;
        std::vector<float> q_conv_weight;
        std::vector<float> k_conv_weight;
        std::vector<float> v_conv_weight;
        std::vector<float> dt_bias;
        std::vector<float> a_log;
        std::vector<float> norm;
        CpuLinearWeight out;
    };
    struct MlpOnlyWeights {
        CpuLinearWeight mlp_up;
        CpuLinearWeight w2;
    };
    struct DenseFeedForwardWeights {
        CpuLinearWeight w13;
        CpuLinearWeight w2;
        CpuLinearWeight per_layer_input_gate;
        CpuLinearWeight per_layer_projection;
    };
    struct MoeWeights {
        std::vector<float> router;
        std::vector<float> router_bias;
        std::vector<CpuLinearWeight> expert_w13;
        std::vector<CpuLinearWeight> expert_w2;
        CpuLinearWeight shared_w13;
        CpuLinearWeight shared_w2;
        CpuLinearWeight shared_gate;
        bool has_shared_gate = false;
        int layer_index = -1;
        int num_experts = 0;
        int experts_per_token = 0;
        bool normalize_topk = false;
        bool use_expert_bias = false;
        bool disk_cached = false;
        float routed_scaling_factor = 1.0f;
    };
    using CpuMixerWeights = std::variant<AttentionWeights, ConvolutionWeights,
                                         GatedDeltaNetWeights, Mamba2Weights,
                                         MlpOnlyWeights>;
    using CpuFeedForwardWeights = std::variant<std::monostate,
                                               DenseFeedForwardWeights, MoeWeights>;

    struct CpuLayerWeights {
        CommonWeights common;
        CpuMixerWeights mixer;
        CpuFeedForwardWeights feed_forward;
    };

    struct CpuWeightStore {
        CpuLinearWeight embedding;
        CpuLinearWeight lm_head;
        CpuLinearWeight per_layer_embedding;
        CpuLinearWeight per_layer_context_projection;
        std::vector<float> embedding_norm;
        std::vector<float> per_layer_projection_norm;
        std::vector<float> final_norm;
        std::vector<CpuLayerWeights> layers;
    };

    /// Owns the resolved checkpoint/format binding for a compiled model: which
    /// checkpoint flavor was loaded, its source identity, and the resolved pack
    /// cache path. This is one fact with one owner, kept separate from the model's
    /// runtime/compute state in `Shared`.
    struct CpuCheckpointBinding {
        bool native_checkpoint = false;
        bool compressed_checkpoint = false;
        std::string source_id;
        std::filesystem::path pack_file;
        bool loaded_pack = false;
    };

    struct Shared {
        Shared(const std::string& path, int context, CpuModelOptions requested,
               std::shared_ptr<const RuntimeContext> runtime);

        void prepare_pack_path();
        void load_weights();
        CommonWeights load_common(class IWeightRepository* repository,
                                  class CpuPackReader* reader,
                                  class CpuPackWriter* writer,
                                  int layer);
        DenseFeedForwardWeights load_dense_feed_forward(
            class IWeightRepository* repository,
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
        std::shared_ptr<IWeightRepository> repository;
        CpuCheckpointBinding checkpoint;
        int max_context = 0;
        CpuModelOptions options;
        CpuCapabilities capabilities;
        CpuThreadPool pool;
        CpuLinearEngine linear;
        CpuWorkspacePlan workspace_plan;
        size_t group_size = 32;
        RuntimeTopology runtime_topology;
        ExecutionTopology& shape;
        CheckpointDimensions& dims;
        CompiledModelProgram program;
        std::string model_identity;
        std::vector<TensorRequest> weight_requests;
        std::unordered_map<std::string, CpuLinearWeight> compressed_linear_cache;
        bool tie_word_embeddings = true;
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

template <typename MixerRef, typename... Handlers>
decltype(auto) visit_operator_weights(MixerRef&& mixer, Handlers&&... handlers) {
    ExhaustiveHandlers<std::decay_t<Handlers>...> dispatch{
        std::forward<Handlers>(handlers)...};
    return std::visit(
        [&dispatch](auto&& alternative) -> decltype(auto) {
            return dispatch(&alternative);
        },
        std::forward<MixerRef>(mixer));
}

struct CpuExecutionContext {
    CpuCompiledModel::Shared& shared;
    CpuWorkspace& workspace;
    CpuCompiledModel::CpuSessionState& session;
};

struct CpuRecurrentStateView {
    std::vector<CpuCompiledModel::LayerState>& states;

    explicit CpuRecurrentStateView(CpuCompiledModel::CpuSessionState& session)
        : states(session.states) {}

    CpuCompiledModel::ConvolutionState& convolution(size_t layer) const {
        return std::get<CpuCompiledModel::ConvolutionState>(states.at(layer));
    }
    CpuCompiledModel::Mamba2State& mamba2(size_t layer) const {
        return std::get<CpuCompiledModel::Mamba2State>(states.at(layer));
    }
    CpuCompiledModel::GatedDeltaNetState& gated_delta_net(size_t layer) const {
        return std::get<CpuCompiledModel::GatedDeltaNetState>(states.at(layer));
    }
};

class CpuAttentionStateView {
public:
    explicit CpuAttentionStateView(CpuCompiledModel& owner) : owner_(owner) {}

    CpuCompiledModel::AttentionState& state(size_t layer) const {
        return owner_.attention_state(layer);
    }
    void store_kv(CpuCompiledModel::AttentionState& state, int position,
                  const float* key, const float* value) const {
        owner_.store_kv(state, position, key, value);
    }
    void store_latent(CpuCompiledModel::AttentionState& state, int position,
                      const float* key, const float* value,
                      const float* rotary) const {
        owner_.store_latent(state, position, key, value, rotary);
    }
    void run_attention(const CpuCompiledModel::AttentionState& state,
                       const AttentionSpec& attention, const float* q,
                       float* output, int sequence_length,
                       std::span<const float> relative_bias = {}) const {
        owner_.run_attention(state, attention, q, output, sequence_length,
                             relative_bias);
    }
    void run_latent_attention(const CpuCompiledModel::AttentionState& state,
                              const AttentionSpec& attention,
                              const float* query_content, const float* query_rope,
                              float* output, int sequence_length,
                              int query_position = -1,
                              std::span<const float> relative_bias = {}) const {
        owner_.run_latent_attention(state, attention, query_content, query_rope,
                                    output, sequence_length, query_position,
                                    relative_bias);
    }
    void run_external_attention(const AttentionSpec& attention,
                                const CpuExternalAttentionMemory& memory,
                                const float* q, float* output,
                                std::span<const float> relative_bias = {}) const {
        owner_.run_external_attention(attention, memory, q, output, relative_bias);
    }

private:
    CpuCompiledModel& owner_;
};

void validate_cpu_packed_batch(std::span<CpuCompiledModel* const> sessions);
void execute_cpu_packed_batch(std::span<CpuCompiledModel* const> sessions,
                              std::span<const int32_t> tokens,
                              std::span<const uint8_t> compute_logits);

}
