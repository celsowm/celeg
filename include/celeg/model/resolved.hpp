#pragma once

#include "celeg/model/graph.hpp"
#include "celeg/model/profiles.hpp"

#include <memory>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace celeg {

// Runtime-only derived topology. It is generic execution data; architecture
// identity and checkpoint matching are deliberately absent.
struct RuntimeTopology {
    int hidden = 0;
    int intermediate = 0;
    int max_feed_forward_intermediate = 0;
    int num_hidden_layers = 0;
    int vocab_size = 0;
    int conv_cache = 0;
    int conv_dim = 0;
    int max_position_embeddings = 0;
    int bos_token_id = -1;
    std::vector<int> eos_token_ids;
    int pad_token_id = -1;
    float norm_eps = 0.0f;
    std::string rope_type = "default";
    float embedding_multiplier = 1.0f;
    float attention_multiplier = 0.0f;
    float residual_multiplier = 1.0f;
    float logits_divisor = 1.0f;
    float final_logit_softcap = 0.0f;
    std::vector<MixerKind> mixer_kinds;
    std::vector<FeedForwardKind> feed_forward_kinds;
    std::vector<int> attention_slot_for_layer;
    std::vector<int> layer_for_attention_slot;
    int attention_layer_count = 0;
    int conv_layer_count = 0;
    int dense_intermediate = 0;
    int moe_intermediate = 0;
    int num_dense_layers = 0;
    int num_experts = 0;
    int experts_per_token = 0;
    bool normalize_topk = false;
    bool use_expert_bias = false;
    float routed_scaling_factor = 1.0f;
    std::vector<AttentionSpec> attention_layouts;
    std::vector<int> feed_forward_intermediates;
    std::vector<ActivationKind> feed_forward_activations;
    int shared_kv_group_count = 0;
    bool has_split_attention_norms = false;
    bool has_per_layer_input = false;
    int per_layer_input_size = 0;

    int maximum_attention_projection_width() const {
        int maximum = 0;
        for (const AttentionSpec& layout : attention_layouts) {
            maximum = std::max(maximum, layout.projection_width());
        }
        return maximum;
    }
    int maximum_attention_query_heads() const {
        int maximum = 0;
        for (const AttentionSpec& layout : attention_layouts) {
            maximum = std::max(maximum, layout.query_heads);
        }
        return maximum;
    }
    int maximum_attention_head_dim() const {
        int maximum = 0;
        for (const AttentionSpec& layout : attention_layouts) {
            maximum = std::max(maximum, layout.head_dim);
        }
        return maximum;
    }

    bool layer_uses_moe(int layer) const {
        return layer >= 0 && layer < static_cast<int>(feed_forward_kinds.size()) &&
               feed_forward_kinds[static_cast<size_t>(layer)] == FeedForwardKind::MixtureOfExperts;
    }
    std::string fingerprint() const;
    std::string summary() const;
    void validate() const;

    const AttentionSpec& attention_layout(int layer) const {
        if (layer < 0 || layer >= static_cast<int>(attention_layouts.size())) {
            throw std::out_of_range("attention layer is out of range");
        }
        return attention_layouts[static_cast<size_t>(layer)];
    }
};

struct ResolvedModel {
    ModelGraph graph;
    RuntimeTopology topology;
    WeightPlan weight_plan;
    ModelCapabilities capabilities;
    std::string architecture_id;
    std::string source_format;
    CheckpointProfile profile;
    std::string checkpoint_profile_id;
    std::string chat_profile_id;
    std::string identity;

    const RuntimeTopology& shape() const { return topology; }
};

} // namespace celeg
