#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace lfm {

enum class LayerType {
    Convolution,
    FullAttention,
};

// Discriminator between the dense LFM2 feed-forward architecture and the
// LFM2 MoE (mixture-of-experts) architecture. The two share the same
// attention/convolution operators but differ in their per-layer FFN.
enum class ArchitectureKind {
    DenseLfm2,
    MoeLfm2,
};

// Explicit MoE topology for the LFM2 MoE architecture. Populated only when
// ArchitectureKind is MoeLfm2.
struct MoeConfig {
    int intermediate_size = 0;     // dense FFN intermediate size (first layers)
    int moe_intermediate_size = 0; // expert FFN intermediate size
    int num_dense_layers = 0;      // layers using the dense FFN path
    int num_experts = 0;
    int experts_per_token = 0;
    bool normalize_topk = false;   // norm_topk_prob
    bool use_expert_bias = false;
    float routed_scaling_factor = 1.0f;
};

struct ModelConfig {
    std::string model_type;
    std::string dtype;
    int hidden_size = 0;
    int intermediate_size = 0;
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
    bool conv_bias = false;
    bool tie_word_embeddings = false;
    bool use_pos_enc = false;
    std::string rope_type;
    // Best-effort hint identifying the source checkpoint, captured from the
    // optional top-level "_name" field in config.json (HuggingFace stores the
    // repo id there, e.g. "LiquidAI/LFM2.5-1.2B-Thinking"). Variants use this
    // to disambiguate checkpoints that share an identical topology.
    std::string repo_hint;
    std::vector<LayerType> layer_types;

    // Architecture discriminator + optional MoE topology.
    ArchitectureKind architecture = ArchitectureKind::DenseLfm2;
    std::optional<MoeConfig> moe;

    static ModelConfig load(const std::string& path);
    void validate() const;
    std::string summary() const;
};

} // namespace lfm
