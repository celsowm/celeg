#pragma once

#include "celeg/checkpoint/metadata.hpp"
#include "celeg/model/graph.hpp"
#include "celeg/model/profiles.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace celeg {

// Runtime-only derived topology. It is generic execution data; architecture
// identity and checkpoint matching are deliberately absent.
struct RuntimeTopology {
    int hidden = 0;
    int intermediate = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int vocab_size = 0;
    int conv_cache = 0;
    int conv_dim = 0;
    int max_position_embeddings = 0;
    int bos_token_id = -1;
    int eos_token_id = -1;
    int pad_token_id = -1;
    float norm_eps = 0.0f;
    float rope_theta = 0.0f;
    std::string rope_type = "default";
    float embedding_multiplier = 1.0f;
    float attention_multiplier = 0.0f;
    float residual_multiplier = 1.0f;
    float logits_divisor = 1.0f;
    bool query_key_norm = true;
    std::vector<MixerKind> mixer_kinds;
    std::vector<MixerKind> layer_types;
    std::vector<int> attention_slot_for_layer;
    std::vector<int> layer_for_attention_slot;
    int attention_layer_count = 0;
    int conv_layer_count = 0;
    int q_width = 0;
    int kv_width = 0;
    int qkv_width = 0;
    int rope_pairs = 0;
    int dense_intermediate = 0;
    int moe_intermediate = 0;
    int num_dense_layers = 0;
    int num_experts = 0;
    int experts_per_token = 0;
    bool normalize_topk = false;
    bool use_expert_bias = false;
    float routed_scaling_factor = 1.0f;

    bool layer_uses_moe(int layer) const {
        return layer >= 0 && layer < static_cast<int>(mixer_kinds.size()) &&
               num_experts > 0 && layer >= num_dense_layers;
    }
    std::string fingerprint() const;
    std::string summary() const;
    void validate() const;
};

struct ResolvedModel {
    ModelDefinition definition;
    ModelGraph graph;
    RuntimeTopology topology;
    WeightPlan weight_plan;
    TensorBindings tensor_bindings;
    ModelCapabilities capabilities;
    std::shared_ptr<const ITensorNamingPolicy> tensor_naming;
    std::string architecture_id;
    CheckpointProfile profile;
    std::string checkpoint_profile_id;
    std::string chat_profile_id;
    std::string identity;
    bool is_gguf = false;

    const RuntimeTopology& shape() const { return topology; }
};

} // namespace celeg
