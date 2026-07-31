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
    FullAttention = Attention,
    Convolution = ShortConvolution,
};
using LayerType = MixerKind;
enum class FeedForwardKind : uint8_t { Dense, MixtureOfExperts };

struct NormSpec {
    float epsilon = 0.0f;
};

struct AttentionSpec {
    int query_heads = 0;
    int key_value_heads = 0;
    int head_dim = 0;
    bool query_key_norm = true;
};

struct ShortConvolutionSpec {
    int cache_length = 0;
    int channels = 0;
    bool bias = false;
};

struct DenseFeedForwardSpec {
    int intermediate_size = 0;
};

struct MixtureOfExpertsSpec {
    int intermediate_size = 0;
    int num_experts = 0;
    int experts_per_token = 0;
    bool normalize_topk = false;
    bool use_expert_bias = false;
    float routed_scaling_factor = 1.0f;
};

struct ResidualSpec {
    float multiplier = 1.0f;
};

struct LayerSpec {
    NormSpec operator_norm;
    std::variant<AttentionSpec, ShortConvolutionSpec> mixer;
    NormSpec feed_forward_norm;
    std::variant<DenseFeedForwardSpec, MixtureOfExpertsSpec> feed_forward;
    ResidualSpec residual;

    MixerKind mixer_kind() const {
        return std::holds_alternative<AttentionSpec>(mixer)
            ? MixerKind::Attention : MixerKind::ShortConvolution;
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

struct TensorBindings {
    std::vector<std::string> source_names;
};

struct ModelCapabilities {
    bool supports_cpu = false;
    bool supports_cuda = false;
    bool supports_expert_offload = false;
    bool tied_embeddings = false;
};

} // namespace celeg
