#pragma once

#include "celeg/model/definition.hpp"
#include "celeg/model/weights/roles.hpp"

#include <cstdint>
#include <optional>
#include <string>
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

enum class AttentionMaskKind : uint8_t {
    Causal,
    SlidingCausal,
};

// A KV group identifies the full-context stream published by an attention
// owner and consumed by later layers. -1 means that the layer owns its normal
// per-layer KV cache.
struct KvSharingSpec {
    int group = -1;
    bool publishes = false;

    bool shared() const { return group >= 0; }
};

struct NormSpec {
    float epsilon = 0.0f;
};

struct AttentionSpec {
    int query_heads = 0;
    int key_value_heads = 0;
    int head_dim = 0;
    bool query_key_norm = true;
    AttentionMaskKind mask = AttentionMaskKind::Causal;
    int sliding_window = 0;
    double rope_theta = 0.0;
    double rotary_fraction = 1.0;
    KvSharingSpec kv_sharing;
    float query_scale = 1.0f;
    PositionalEncodingKind positional_encoding = PositionalEncodingKind::Rope;

    int query_width() const { return query_heads * head_dim; }
    int key_value_width() const { return key_value_heads * head_dim; }
    int projection_width() const { return query_width() + 2 * key_value_width(); }
    int rotary_pairs() const {
        if (positional_encoding == PositionalEncodingKind::None) return 0;
        return static_cast<int>(static_cast<double>(head_dim) * rotary_fraction) / 2;
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
    float embedding_multiplier = 1.0f;
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
