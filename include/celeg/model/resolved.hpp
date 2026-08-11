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

// Runtime-only derived topology. It is generic execution data; architecture
// identity and checkpoint matching are deliberately absent. Token and
// numerical behavior are explicit policies rather than loose topology fields.
struct RuntimeTopology {
    int hidden = 0;
    int intermediate = 0;
    int max_feed_forward_intermediate = 0;
    int num_hidden_layers = 0;
    int vocab_size = 0;
    int conv_cache = 0;
    int conv_dim = 0;
    int max_position_embeddings = 0;
    // Maps each executable layer to its source checkpoint block.  An empty
    // table means the identity mapping.
    std::vector<int> checkpoint_layer_for_layer;
    TokenPolicy token_policy;
    NumericalPolicy numerical_policy;
    std::vector<MixerKind> mixer_kinds;
    std::vector<FeedForwardKind> feed_forward_kinds;
    std::vector<int> attention_slot_for_layer;
    std::vector<int> layer_for_attention_slot;
    int attention_layer_count = 0;
    int conv_layer_count = 0;
    int gated_delta_net_layer_count = 0;
    int mamba2_layer_count = 0;
    int mlp_only_layer_count = 0;
    int mamba2_intermediate = 0;
    std::vector<GatedDeltaNetSpec> gated_delta_net_layouts;
    std::vector<Mamba2Spec> mamba2_layouts;
    std::vector<MlpBlockSpec> mlp_only_layouts;
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
    std::vector<AttentionSpec> attention_layouts;
    std::vector<int> feed_forward_intermediates;
    std::vector<ActivationKind> feed_forward_activations;
    int shared_kv_group_count = 0;
    bool has_split_attention_norms = false;
    bool has_per_layer_input = false;
    int per_layer_input_size = 0;
    // Auxiliary multi-token predictor depth. Kept after the established
    // execution fields so older backend translation units cannot misread the
    // existing topology while a build is being relinked.
    int mtp_num_hidden_layers = 0;

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
    int maximum_attention_output_width() const {
        int maximum = 0;
        for (const AttentionSpec& layout : attention_layouts) {
            maximum = std::max(maximum, layout.uses_latent_state()
                ? layout.latent_query_content_width() : layout.query_width());
        }
        return maximum;
    }
    int maximum_attention_latent_rank() const {
        int maximum = 0;
        for (const AttentionSpec& layout : attention_layouts) {
            if (const auto* latent = layout.latent_state()) {
                maximum = std::max(maximum, latent->latent_rank);
            }
        }
        return maximum;
    }
    int maximum_attention_latent_rope_width() const {
        int maximum = 0;
        for (const AttentionSpec& layout : attention_layouts) {
            if (const auto* latent = layout.latent_state()) {
                maximum = std::max(maximum, latent->decoupled_rope
                    ? latent->rope_head_dim : 0);
            }
        }
        return maximum;
    }
    int maximum_attention_latent_query_rope_width() const {
        int maximum = 0;
        for (const AttentionSpec& layout : attention_layouts) {
            maximum = std::max(maximum, layout.latent_query_rope_width());
        }
        return maximum;
    }
    int max_gated_delta_net_qkv_width() const {
        int maximum = 0;
        for (const GatedDeltaNetSpec& layout : gated_delta_net_layouts) {
            maximum = std::max(maximum, 2 * layout.key_heads * layout.key_head_dim +
                layout.value_heads * layout.value_head_dim);
        }
        return maximum;
    }
    int max_gated_delta_net_output_width() const {
        int maximum = 0;
        for (const GatedDeltaNetSpec& layout : gated_delta_net_layouts) {
            maximum = std::max(maximum, layout.value_heads * layout.value_head_dim);
        }
        return maximum;
    }
    int max_gated_delta_net_gate_width() const {
        int maximum = 0;
        for (const GatedDeltaNetSpec& layout : gated_delta_net_layouts) {
            maximum = std::max(maximum, layout.value_heads);
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

    const RuntimeTopology& shape() const { return topology; }
    void validate() const;
};

// Rebuild allocation-oriented shape tables from the canonical graph after
// import.  Importers may use an initial shape while constructing the graph,
// but execution-facing schedules have one semantic owner: ModelGraph.
void derive_runtime_topology_from_graph(RuntimeTopology& topology,
                                        const ModelGraph& graph);

} // namespace celeg
