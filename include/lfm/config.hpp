#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lfm {

enum class LayerType {
    Convolution,
    FullAttention,
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

    static ModelConfig load(const std::string& path);
    void validate() const;
    std::string summary() const;
};

} // namespace lfm
