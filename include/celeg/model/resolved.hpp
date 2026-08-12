#pragma once

#include "celeg/model/graph.hpp"
#include "celeg/model/profiles.hpp"

#include <memory>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace celeg {

struct TokenPolicy {
    int bos_token_id = -1;
    std::vector<int> eos_token_ids;
    int pad_token_id = -1;

    void validate() const;
};

struct NumericalPolicy {
    float norm_eps = 0.0f;
    float post_norm_eps = 0.0f;
    float embedding_multiplier = 1.0f;
    float attention_multiplier = 0.0f;
    float residual_multiplier = 1.0f;
    float logits_multiplier = 1.0f;
    float logits_divisor = 1.0f;
    float final_logit_softcap = 0.0f;

    void validate() const;
};

struct CheckpointDimensions {
    int vocab_size = 0;
    int max_position_embeddings = 0;
    std::vector<int> checkpoint_layer_for_layer;
    TokenPolicy token_policy;
    int mtp_num_hidden_layers = 0;

    void validate() const;
};

// Runtime-only execution cache derived from the final semantic graph. It has
// no checkpoint/import ownership and cannot be constructed by a frontend.
class ExecutionTopology {
public:
    int hidden = 0;
    int intermediate = 0;
    int max_feed_forward_intermediate = 0;
    int num_hidden_layers = 0;
    int conv_cache = 0;
    int conv_dim = 0;
    std::vector<int> attention_slot_for_layer;
    std::vector<int> layer_for_attention_slot;
    int attention_layer_count = 0;
    int conv_layer_count = 0;
    int gated_delta_net_layer_count = 0;
    int mamba2_layer_count = 0;
    int mlp_only_layer_count = 0;
    int mamba2_intermediate = 0;
    int maximum_mamba_projection_width_value = 0;
    int maximum_mamba_conv_width_value = 0;
    int maximum_attention_projection_width_value = 0;
    int maximum_attention_query_heads_value = 0;
    int maximum_attention_head_dim_value = 0;
    int maximum_attention_output_width_value = 0;
    int maximum_attention_latent_rank_value = 0;
    int maximum_attention_latent_rope_width_value = 0;
    int maximum_attention_latent_query_rope_width_value = 0;
    int maximum_attention_latent_output_width_value = 0;
    int maximum_attention_latent_projection_width_value = 0;
    int maximum_gated_delta_net_qkv_width_value = 0;
    int maximum_gated_delta_net_output_width_value = 0;
    int maximum_gated_delta_net_gate_width_value = 0;
    int dense_intermediate = 0;
    int moe_intermediate = 0;
    int shared_expert_intermediate = 0;
    int num_dense_layers = 0;
    int num_experts = 0;
    int experts_per_token = 0;
    bool normalize_topk = false;
    bool moe_router_softmax = false;
    bool use_expert_bias = false;
    float routed_scaling_factor = 1.0f;
    int moe_routing_group_count = 0;
    int moe_routing_experts_per_group = 0;
    int moe_routing_groups_per_token = 0;
    int moe_routing_group_score_top_k = 0;

    int maximum_attention_projection_width() const {
        return maximum_attention_projection_width_value;
    }
    int maximum_attention_query_heads() const {
        return maximum_attention_query_heads_value;
    }
    int maximum_attention_head_dim() const {
        return maximum_attention_head_dim_value;
    }

    int maximum_mamba_projection_width() const {
        return maximum_mamba_projection_width_value;
    }
    int maximum_mamba_conv_width() const {
        return maximum_mamba_conv_width_value;
    }
    int maximum_attention_output_width() const {
        return maximum_attention_output_width_value;
    }
    int maximum_attention_latent_rank() const {
        return maximum_attention_latent_rank_value;
    }
    int maximum_attention_latent_rope_width() const {
        return maximum_attention_latent_rope_width_value;
    }
    int maximum_attention_latent_query_rope_width() const {
        return maximum_attention_latent_query_rope_width_value;
    }
    int maximum_attention_latent_output_width() const {
        return maximum_attention_latent_output_width_value;
    }
    int maximum_attention_latent_projection_width() const {
        return maximum_attention_latent_projection_width_value;
    }
    int max_gated_delta_net_qkv_width() const {
        return maximum_gated_delta_net_qkv_width_value;
    }
    int max_gated_delta_net_output_width() const {
        return maximum_gated_delta_net_output_width_value;
    }
    int max_gated_delta_net_gate_width() const {
        return maximum_gated_delta_net_gate_width_value;
    }

    std::string fingerprint() const;
    std::string summary() const;
    void validate() const;

    static ExecutionTopology derive(const ModelGraph& graph);

private:
    friend struct RuntimeTopology;
    ExecutionTopology() = default;
};

// Checkpoint/import dimensions are kept separate from the derived execution
// cache. Runtime code must explicitly choose the boundary it consumes.
struct RuntimeTopology {
    RuntimeTopology() : exec() {}
    CheckpointDimensions dims;
    ExecutionTopology exec;

    std::string fingerprint() const;
    std::string summary() const;
    void validate() const;
};

struct ModelProvenance {
    std::string architecture_id;
    std::string source_format;
    CheckpointProfile profile;
    std::string checkpoint_profile_id;
    std::string chat_template_id;
    std::string identity;
};

struct ResolvedModel {
    ModelGraph graph;
    RuntimeTopology topology;
    WeightPlan weight_plan;
    ModelCapabilities capabilities;
    ModelProvenance provenance;

    const ExecutionTopology& shape() const { return topology.exec; }
    void validate() const;
};

// Compose import facts with the allocation cache derived from the final graph.
// The only construction path for the derived cache is ExecutionTopology::derive.
RuntimeTopology compose_runtime_topology(CheckpointDimensions checkpoint,
                                         const ModelGraph& graph);

} // namespace celeg
