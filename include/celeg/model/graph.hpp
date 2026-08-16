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

struct PrivateKv {};

struct SharedKvPublisher {
    int group = 0;
};

struct SharedKvConsumer {
    int group = 0;
};

using KvSharingSpec = std::variant<PrivateKv, SharedKvPublisher, SharedKvConsumer>;

inline bool kv_sharing_shared(const KvSharingSpec& spec) {
    return !std::holds_alternative<PrivateKv>(spec);
}

inline bool kv_sharing_publishes(const KvSharingSpec& spec) {
    return std::holds_alternative<SharedKvPublisher>(spec);
}

inline int kv_sharing_group(const KvSharingSpec& spec) {
    if (const auto* publisher = std::get_if<SharedKvPublisher>(&spec)) return publisher->group;
    if (const auto* consumer = std::get_if<SharedKvConsumer>(&spec)) return consumer->group;
    return -1;
}

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

struct OrdinaryKvStorageSpec {
    StateScalarType key = StateScalarType::BF16;
    StateScalarType value = StateScalarType::BF16;
    StateQuantizationGranularity granularity = StateQuantizationGranularity::PerTensor;
    bool paged = true;

    void validate() const;
};

struct LatentStorageSpec {
    StateScalarType latent = StateScalarType::BF16;
    StateScalarType rotary = StateScalarType::BF16;
    StateQuantizationGranularity granularity = StateQuantizationGranularity::PerTensor;
    bool paged = true;

    void validate() const;
};

struct OrdinaryKvStateSpec {
    bool quantizable = true;
    OrdinaryKvStorageSpec storage;
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

struct NoAttentionOutputTransformSpec {};

struct OrthogonalizeCurrentValueSpec {
    float minimum_norm_squared = 1.0e-6f;
};

using AttentionOutputTransformSpec = std::variant<
    NoAttentionOutputTransformSpec, OrthogonalizeCurrentValueSpec>;

struct DirectLatentProjection {};

struct FactorizedLatentProjection {
    int query_rank = 0;
    int value_head_dim = 0;
    NormSpec query_latent_norm;
    NormSpec key_latent_norm;
};

using LatentProjectionSpec = std::variant<DirectLatentProjection, FactorizedLatentProjection>;

struct LatentAttentionStateSpec {
    int latent_rank = 0;
    int rope_head_dim = 0;
    int nope_head_dim = 0;
    bool decoupled_rope = false;
    LatentProjectionSpec projection = DirectLatentProjection{};
    LatentStorageSpec storage;

    bool factorized() const {
        return std::holds_alternative<FactorizedLatentProjection>(projection);
    }
    const FactorizedLatentProjection* factorized_projection() const {
        return std::get_if<FactorizedLatentProjection>(&projection);
    }
    void validate() const;
};

using AttentionStateSpec = std::variant<OrdinaryKvStateSpec,
                                        LatentAttentionStateSpec>;

struct CurrentSequenceSource {};

struct ExternalMemorySource {
    int slot = 0;
};

using AttentionKeyValueSource = std::variant<CurrentSequenceSource, ExternalMemorySource>;

enum class AttentionGateGranularity : uint8_t {
    OutputWise,
    HeadWise,
    ElementWise,
};

struct SigmoidAttentionGateSpec {
    bool packed_with_query = false;
    AttentionGateGranularity granularity = AttentionGateGranularity::OutputWise;
};

using AttentionOutputGateSpec = std::optional<SigmoidAttentionGateSpec>;

struct AttentionSpec {
    int query_heads = 0;
    int key_value_heads = 0;
    int head_dim = 0;
    std::optional<NormSpec> query_norm;
    std::optional<NormSpec> key_norm;
    AttentionPatternSpec pattern = FullCausalPattern{};
    KvSharingSpec kv_sharing;
    float query_scale = 1.0f;
    AttentionOutputGateSpec output_gate;
    PositionSpec position = RopePositionSpec{};
    AttentionBiasSpec bias = NoAttentionBiasSpec{};
    AttentionStateSpec state = OrdinaryKvStateSpec{};
    AttentionKeyValueSource key_value_source = CurrentSequenceSource{};
    AttentionOutputTransformSpec output_transform = NoAttentionOutputTransformSpec{};

    int query_width() const { return query_heads * head_dim; }
    int output_gate_width() const {
        if (!output_gate.has_value() || output_gate->packed_with_query) return 0;
        return output_gate->granularity == AttentionGateGranularity::HeadWise
            ? query_heads : query_width();
    }
    int query_projection_width() const {
        return output_gate.has_value() && output_gate->packed_with_query
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
    StateQuantizationGranularity state_storage_granularity() const {
        return std::visit([](const auto& s) { return s.storage.granularity; }, state);
    }
    bool state_storage_paged() const {
        return std::visit([](const auto& s) { return s.storage.paged; }, state);
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
    bool uses_external_memory() const {
        return std::holds_alternative<ExternalMemorySource>(key_value_source);
    }
    int external_memory_slot() const {
        const auto* external = std::get_if<ExternalMemorySource>(&key_value_source);
        return external ? external->slot : -1;
    }
    bool has_query_key_norm() const {
        return query_norm.has_value() || key_norm.has_value();
    }
    int latent_query_projection_width() const {
        const auto* latent = latent_state();
        return latent && latent->factorized()
            ? query_heads * (latent->nope_head_dim + latent->rope_head_dim)
            : latent_query_width();
    }
    int latent_output_width() const {
        const auto* latent = latent_state();
        return latent && latent->factorized()
            ? query_heads * latent->factorized_projection()->value_head_dim
            : latent_query_content_width();
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

struct GatedDeltaNetSpec {
    int conv_kernel = 0;
    int key_head_dim = 0;
    int value_head_dim = 0;
    int key_heads = 0;
    int value_heads = 0;
    bool vector_decay = false;
    bool safe_decay = false;
    float decay_lower_bound = -5.0f;
    bool sigmoid_output_gate = false;
    bool factorized_projections = false;
    bool a_log_needs_exp = false;

    int qkv_width() const {
        return 2 * key_heads * key_head_dim + value_heads * value_head_dim;
    }
    int value_width() const { return value_heads * value_head_dim; }
    int decay_width() const { return vector_decay ? key_heads * key_head_dim : value_heads; }
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
    // GGUF conversion bakes A = -exp(A_log) into the stored tensor, so the CPU
    // kernel must use it as-is; raw safetensors checkpoints still store the
    // untransformed A_log parameter and need -exp() applied at inference time.
    bool a_log_needs_exp = true;
};

struct MlpBlockSpec {
    int intermediate_size = 0;
    ActivationKind activation = ActivationKind::Relu2;
};

struct DenseFeedForwardSpec {
    int intermediate_size = 0;
    ActivationKind activation = ActivationKind::SwiGLU;
};

struct PerLayerInputPolicy {
    int input_size = 0;
    ActivationKind activation = ActivationKind::GeluTanh;
    NormSpec norm;
};

enum class MoeCombineOrder : uint8_t {
    RoutedThenShared,
    SharedThenRouted,
};

struct MoeTopKSelectionSpec {};

struct MoeGroupedTopKSelectionSpec {
    int group_count = 0;
    int experts_per_group = 0;
    int groups_per_token = 0;
    int group_score_top_k = 0;
};

using MoeSelectionSpec = std::variant<
    MoeTopKSelectionSpec,
    MoeGroupedTopKSelectionSpec>;

struct SharedExpertSpec {
    int intermediate_size = 0;
    MoeCombineOrder combine_order = MoeCombineOrder::RoutedThenShared;
};

struct MixtureOfExpertsSpec {
    int intermediate_size = 0;
    int num_experts = 0;
    int experts_per_token = 0;
    bool normalize_topk = false;
    bool use_expert_bias = false;
    float routed_scaling_factor = 1.0f;
    MoeSelectionSpec selection = MoeTopKSelectionSpec{};
    std::optional<SharedExpertSpec> shared;
    bool router_softmax = false;
};

struct ResidualSpec {
    float multiplier = 1.0f;
};

using MixerSpec = std::variant<
    AttentionSpec,
    ShortConvolutionSpec,
    GatedDeltaNetSpec,
    Mamba2Spec,
    MlpBlockSpec>;

using FeedForwardSpec = std::variant<
    std::monostate,
    DenseFeedForwardSpec,
    MixtureOfExpertsSpec>;

struct LayerSpec {
    NormSpec operator_norm;
    std::optional<NormSpec> post_attention_norm;
    std::optional<NormSpec> pre_feed_forward_norm;
    std::optional<NormSpec> post_feed_forward_norm;
    MixerSpec mixer;
    std::optional<NormSpec> feed_forward_norm;
    FeedForwardSpec feed_forward;
    ResidualSpec residual;
    float layer_scalar = 1.0f;
};

struct ModelGraph {
    int hidden = 0;
    std::vector<LayerSpec> layers;
    NormSpec final_norm;
    std::vector<int> norm_after_layers;
    std::optional<PerLayerInputPolicy> per_layer_input;
    struct EmbeddingTransformSpec {
        std::optional<NormSpec> post_norm;
        float multiplier = 1.0f;

        void validate() const;
    };
    EmbeddingTransformSpec embedding_transform;
    float logits_divisor = 1.0f;
    float logits_multiplier = 1.0f;
    float final_logit_softcap = 0.0f;
    bool tied_embeddings = false;

    void validate() const;
    std::string fingerprint() const;

    bool has_moe() const {
        for (const LayerSpec& layer : layers) {
            if (std::holds_alternative<MixtureOfExpertsSpec>(layer.feed_forward)) {
                return true;
            }
        }
        return false;
    }
};

struct WeightPlan {
    std::vector<TensorRequest> requests;
};

}
