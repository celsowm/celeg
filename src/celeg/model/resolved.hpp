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
    /// Convention for every structurally-bound RMSNorm weight (input/post-attention
    /// layernorm, q/k-norm, final norm) in this checkpoint: `Scale` multiplies the
    /// normalized activations directly by the stored weight (the common Llama-style
    /// convention); `OnePlusScale` multiplies by `1 + weight` instead (the checkpoint
    /// stores a zero-centered offset -- e.g. Qwen3.5's `Qwen3_5RMSNorm`). Detected once
    /// from checkpoint tensor structure in `global_facts.cpp`, not per-model name.
    NormWeightKind norm_weight_kind = NormWeightKind::Scale;
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

class ExecutionTopology {
public:
    int num_hidden_layers = 0;
    std::vector<int> attention_slot_for_layer;
    std::vector<int> layer_for_attention_slot;
    int attention_layer_count = 0;
    int conv_layer_count = 0;
    int gated_delta_net_layer_count = 0;
    int mamba2_layer_count = 0;
    int mlp_only_layer_count = 0;

    std::string summary() const;
    void validate() const;

    static ExecutionTopology derive(const ModelGraph& graph);

private:
    friend struct RuntimeTopology;
    ExecutionTopology() = default;
};

struct RuntimeTopology {
    RuntimeTopology() : exec() {}
    CheckpointDimensions dims;
    ExecutionTopology exec;

    std::string summary() const;
    void validate() const;
};

struct ModelProvenance {
    std::string architecture_id;
    std::string source_format;
    CheckpointProfile profile;
    std::string checkpoint_profile_id;
    std::string identity;
};

struct ResolvedModel {
    ModelGraph graph;
    RuntimeTopology topology;
    WeightPlan weight_plan;
    ModelProvenance provenance;

    const ExecutionTopology& shape() const { return topology.exec; }
    void validate() const;
};

RuntimeTopology compose_runtime_topology(CheckpointDimensions checkpoint,
                                         const ModelGraph& graph);

}
