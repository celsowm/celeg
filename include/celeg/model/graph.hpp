#pragma once

#include "celeg/model/definition.hpp"
#include "celeg/model/norm.hpp"
#include "celeg/model/weights/roles.hpp"

#include <cstdint>
#include <cstddef>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace celeg {

enum class MixerKind : uint8_t {
    Attention,
    ShortConvolution,
    GatedDeltaNet,
    Mamba2,
    MlpOnly,
};
enum class FeedForwardKind : uint8_t { Dense, MixtureOfExperts };

enum class ActivationKind : uint8_t {
    SwiGLU,
    GeluTanh,
    Relu2,
};

struct FullCausalPattern {};

struct SlidingWindowPattern {
    int window = 0;
};

struct BidirectionalPattern {};

struct PrefixLmPattern {
    int prefix_length = 0;
};

struct BlockSparsePattern {
    int block_size = 0;
    int local_blocks = 0;
    int global_blocks = 0;
};

struct DynamicSparsePattern {
    int block_size = 0;
    int max_selected_blocks = 0;
};

using AttentionPatternSpec = std::variant<FullCausalPattern, SlidingWindowPattern,
                                         BidirectionalPattern, PrefixLmPattern,
                                         BlockSparsePattern, DynamicSparsePattern>;

// A KV group identifies the full-context stream published by an attention
// owner and consumed by later layers. -1 means that the layer owns its normal
// per-layer KV cache.
struct KvSharingSpec {
    int group = -1;
    bool publishes = false;

    bool shared() const { return group >= 0; }
};

struct OrdinaryKvStateSpec {
    bool quantizable = true;
};

struct NoAttentionBiasSpec {};

struct AlibiBiasSpec {
    std::vector<float> slopes;

    void validate(int query_heads) const;
};

struct RelativePositionBiasSpec {
    int bucket_count = 0;
    int max_distance = 0;
    bool bidirectional = false;

    void validate() const;
};

using AttentionBiasSpec = std::variant<NoAttentionBiasSpec, AlibiBiasSpec,
                                       RelativePositionBiasSpec>;

// Output transforms are semantic post-attention operations applied before the
// output projection. They deliberately do not encode a model-family name.
struct NoAttentionOutputTransformSpec {};

struct OrthogonalizeCurrentValueSpec {
    float minimum_norm_squared = 1.0e-6f;
};

using AttentionOutputTransformSpec = std::variant<
    NoAttentionOutputTransformSpec, OrthogonalizeCurrentValueSpec>;

// Latent state is a semantic representation.  It is lowered independently
// from ordinary K/V state so compressed attention does not inherit an
// equal-width K/V page contract.
struct LatentAttentionStateSpec {
    int latent_rank = 0;
    int rope_head_dim = 0;
    int nope_head_dim = 0;
    bool decoupled_rope = false;
};

using AttentionStateSpec = std::variant<OrdinaryKvStateSpec,
                                        LatentAttentionStateSpec>;

enum class StateScalarType : uint8_t {
    FP32,
    FP16,
    BF16,
    FP8,
    INT8,
    INT4,
};

enum class StateQuantizationGranularity : uint8_t {
    PerTensor,
    PerHead,
    PerToken,
    PerBlock,
};

struct AttentionStateStorageSpec {
    StateScalarType key = StateScalarType::BF16;
    StateScalarType value = StateScalarType::BF16;
    StateScalarType latent = StateScalarType::BF16;
    StateScalarType rotary = StateScalarType::BF16;
    StateScalarType recurrent = StateScalarType::FP32;
    StateQuantizationGranularity granularity = StateQuantizationGranularity::PerTensor;
    bool paged = true;

    void validate(const AttentionStateSpec& state) const;
};

enum class AttentionSourceKind : uint8_t {
    CurrentSequence,
    ExternalMemory,
};

struct AttentionSourceSpec {
    AttentionSourceKind query = AttentionSourceKind::CurrentSequence;
    AttentionSourceKind key_value = AttentionSourceKind::CurrentSequence;
    int memory_slot = -1;

    bool self_attention() const {
        return query == AttentionSourceKind::CurrentSequence &&
               key_value == AttentionSourceKind::CurrentSequence;
    }
};

enum class AttentionGateKind : uint8_t {
    None,
    Sigmoid,
};

struct AttentionOutputGateSpec {
    AttentionGateKind kind = AttentionGateKind::None;
    // Physical binding detail only: the semantic gate is still a separate
    // operation, but some checkpoints store its projection beside Q.
    bool packed_with_query = false;

    bool enabled() const { return kind != AttentionGateKind::None; }
};

struct AttentionSpec {
    int query_heads = 0;
    int key_value_heads = 0;
    int head_dim = 0;
    NormSpec query_norm;
    NormSpec key_norm;
    AttentionPatternSpec pattern = FullCausalPattern{};
    KvSharingSpec kv_sharing;
    float query_scale = 1.0f;
    AttentionOutputGateSpec output_gate;
    PositionSpec position = RopePositionSpec{};
    AttentionBiasSpec bias = NoAttentionBiasSpec{};
    AttentionStateSpec state = OrdinaryKvStateSpec{};
    AttentionStateStorageSpec state_storage;
    AttentionSourceSpec sources;
    AttentionOutputTransformSpec output_transform = NoAttentionOutputTransformSpec{};

    int query_width() const { return query_heads * head_dim; }
    int query_projection_width() const {
        return output_gate.enabled() && output_gate.packed_with_query
            ? query_width() * 2 : query_width();
    }
    int key_value_width() const { return key_value_heads * head_dim; }
    int projection_width() const {
        if (uses_latent_state()) {
            return latent_query_width() + latent_state_projection_width();
        }
        return query_projection_width() + 2 * key_value_width();
    }
    const LatentAttentionStateSpec* latent_state() const {
        return std::get_if<LatentAttentionStateSpec>(&state);
    }
    int latent_query_content_width() const {
        const auto* latent = latent_state();
        return latent ? query_heads * latent->latent_rank : 0;
    }
    int latent_query_rope_width() const {
        const auto* latent = latent_state();
        return latent && latent->decoupled_rope ? query_heads * latent->rope_head_dim : 0;
    }
    int latent_query_width() const {
        return latent_query_content_width() + latent_query_rope_width();
    }
    int latent_state_projection_width() const {
        const auto* latent = latent_state();
        return latent ? 2 * latent->latent_rank + latent->rope_head_dim : 0;
    }
    int rotary_pairs() const {
        const auto rotary_dimension = [this](const auto& position) {
            using T = std::decay_t<decltype(position)>;
            if constexpr (std::is_same_v<T, RopePositionSpec>) {
                return static_cast<int>(static_cast<double>(head_dim) *
                                        position.rotary_fraction) / 2;
            } else if constexpr (std::is_same_v<T, MultiAxisRopeSpec>) {
                return static_cast<int>(static_cast<double>(head_dim) *
                                        position.base.rotary_fraction) / 2;
            } else {
                return 0;
            }
        };
        return std::visit(rotary_dimension, position);
    }
    const RopePositionSpec* rope_position() const {
        if (const auto* rope = std::get_if<RopePositionSpec>(&position)) return rope;
        if (const auto* multi = std::get_if<MultiAxisRopeSpec>(&position)) return &multi->base;
        return nullptr;
    }
    const MultiAxisRopeSpec* multi_axis_position() const {
        return std::get_if<MultiAxisRopeSpec>(&position);
    }
    bool uses_latent_state() const {
        return std::holds_alternative<LatentAttentionStateSpec>(state);
    }
    bool uses_external_memory() const { return !sources.self_attention(); }
    bool has_query_key_norm() const {
        return query_norm.enabled() || key_norm.enabled();
    }
    bool has_causal_pattern() const {
        return std::holds_alternative<FullCausalPattern>(pattern) ||
               std::holds_alternative<SlidingWindowPattern>(pattern);
    }
    bool has_sparse_pattern() const {
        return std::holds_alternative<BlockSparsePattern>(pattern) ||
               std::holds_alternative<DynamicSparsePattern>(pattern);
    }
    int sliding_window_size() const {
        if (const auto* sliding = std::get_if<SlidingWindowPattern>(&pattern)) {
            return sliding->window;
        }
        return 0;
    }
};

struct ShortConvolutionSpec {
    int cache_length = 0;
    int channels = 0;
    bool bias = false;
};

// Recurrent linear attention used by hybrid decoder families.  The fields are
// execution semantics; tensor spelling remains in the architecture module.
struct GatedDeltaNetSpec {
    int conv_kernel = 0;
    int key_head_dim = 0;
    int value_head_dim = 0;
    int key_heads = 0;
    int value_heads = 0;
};

struct Mamba2Spec {
    int conv_kernel = 0;
    int intermediate_size = 0;
    int state_size = 0;
    int time_step_rank = 0;
    int num_heads = 0;
    int head_dim = 0;
    int group_count = 0;
    int chunk_size = 0;
    bool conv_bias = false;
    bool projection_bias = false;
};

struct MlpBlockSpec {
    int intermediate_size = 0;
    ActivationKind activation = ActivationKind::Relu2;
};

struct DenseFeedForwardSpec {
    int intermediate_size = 0;
    ActivationKind activation = ActivationKind::SwiGLU;
};

struct PerLayerInputSpec {
    int input_size = 0;
    ActivationKind activation = ActivationKind::GeluTanh;
    bool enabled = false;
};

struct MixtureOfExpertsSpec {
    int intermediate_size = 0;
    int num_experts = 0;
    int experts_per_token = 0;
    bool normalize_topk = false;
    bool use_expert_bias = false;
    float routed_scaling_factor = 1.0f;
    // These fields describe semantics, not a checkpoint layout.  Families
    // with ordinary top-K routing leave the grouped/shared fields at their
    // defaults; the compiled program still records those defaults explicitly.
    int routing_group_count = 0;
    int routing_experts_per_group = 0;
    bool has_shared_expert = false;
    int shared_intermediate_size = 0;
    bool shared_before_routed = false;
    bool router_softmax = false;
};

struct ResidualSpec {
    float multiplier = 1.0f;
};

struct LayerSpec {
    NormSpec operator_norm;
    NormSpec post_attention_norm;
    NormSpec pre_feed_forward_norm;
    NormSpec post_feed_forward_norm;
    NormSpec per_layer_input_norm;
    std::variant<AttentionSpec, ShortConvolutionSpec, GatedDeltaNetSpec,
                 Mamba2Spec, MlpBlockSpec> mixer;
    NormSpec feed_forward_norm;
    std::variant<DenseFeedForwardSpec, MixtureOfExpertsSpec> feed_forward;
    ResidualSpec residual;
    PerLayerInputSpec per_layer_input;
    float layer_scalar = 1.0f;
    // Some hybrid schedules use a mixer-only layer.  This is an explicit
    // semantic property resolved by the architecture, never inferred from
    // the presence of a particular mixer type elsewhere in the model.
    bool execute_feed_forward = true;

    MixerKind mixer_kind() const {
        if (std::holds_alternative<AttentionSpec>(mixer)) return MixerKind::Attention;
        if (std::holds_alternative<ShortConvolutionSpec>(mixer)) return MixerKind::ShortConvolution;
        if (std::holds_alternative<GatedDeltaNetSpec>(mixer)) return MixerKind::GatedDeltaNet;
        if (std::holds_alternative<Mamba2Spec>(mixer)) return MixerKind::Mamba2;
        return MixerKind::MlpOnly;
    }
    FeedForwardKind feed_forward_kind() const {
        return std::holds_alternative<DenseFeedForwardSpec>(feed_forward)
            ? FeedForwardKind::Dense : FeedForwardKind::MixtureOfExperts;
    }
};

struct ModelGraph {
    std::vector<LayerSpec> layers;
    NormSpec final_norm;
    // Intermediate normalization boundaries are semantic graph edges.  The
    // runtime shape may cache a derived copy for allocation, but compilation
    // must consume this graph-owned schedule.
    std::vector<int> norm_after_layers;
    struct EmbeddingTransformSpec {
        std::optional<NormSpec> post_norm;
        float multiplier = 1.0f;

        void validate() const;
    };
    EmbeddingTransformSpec embedding_transform;
    float logits_divisor = 1.0f;
    float final_logit_softcap = 0.0f;

    void validate() const;

    bool has_moe() const {
        for (const LayerSpec& layer : layers) {
            if (layer.feed_forward_kind() == FeedForwardKind::MixtureOfExperts) return true;
        }
        return false;
    }
};

struct WeightPlan {
    std::vector<TensorRequest> requests;
};

struct ModelCapabilities {
    bool supports_cpu = false;
    bool supports_cuda = false;
    bool supports_expert_offload = false;
    bool tied_embeddings = false;
};

} // namespace celeg
