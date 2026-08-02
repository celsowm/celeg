#pragma once

#include "celeg/checkpoint/metadata.hpp"

#include <string>

namespace celeg::detail {

struct Lfm2Metadata {
    bool gguf = false;
    bool moe = false;
    std::string tensor_prefix;
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
    float norm_eps = 1.0e-5f;
    float rope_theta = 1.0e6f;
    int bos_token_id = 1;
    int eos_token_id = 7;
    int pad_token_id = 0;
    int moe_intermediate = 0;
    int num_dense_layers = 0;
    int num_experts = 0;
    int experts_per_token = 0;
    bool normalize_topk = true;
    bool use_expert_bias = false;
    float routed_scaling_factor = 1.0f;
};

Lfm2Metadata decode_lfm2_metadata(const CheckpointMetadata& metadata);

} // namespace celeg::detail
