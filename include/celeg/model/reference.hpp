#pragma once

#include "celeg/model/graph.hpp"

#include <cstdint>
#include <vector>

namespace celeg::reference {

uint16_t float_to_bf16(float value);
float bf16_to_float(uint16_t value);
float round_bf16(float value);

std::vector<float> rmsnorm_bf16(const std::vector<float>& x,
                                const std::vector<float>& weight,
                                float eps);
void rope_bf16_inplace(std::vector<float>& vector,
                       const std::vector<float>& cos_half,
                       const std::vector<float>& sin_half);
std::vector<float> gqa_decode_strict_bf16(
    const std::vector<float>& q,
    const std::vector<float>& key_cache,
    const std::vector<float>& value_cache,
    int seq_len,
    int q_heads,
    int kv_heads,
    int head_dim);

// Correctness-first latent-KV attention reference.  The persistent key/value
// state is represented by one latent vector per token; query-side content
// projections and the optional decoupled rotary component are already
// resolved by the caller.  This keeps the reference independent of a
// checkpoint naming convention or a particular latent-attention family.
struct LatentAttentionReferenceSpec {
    int query_heads = 0;
    int latent_rank = 0;
    int rope_head_dim = 0;
    float scale = 1.0f;
    AttentionPatternSpec pattern = FullCausalPattern{};
};

std::vector<float> latent_attention_decode_bf16(
    const LatentAttentionReferenceSpec& spec,
    const std::vector<float>& query_content,
    const std::vector<float>& query_rope,
    const std::vector<float>& latent_keys,
    const std::vector<float>& latent_values,
    const std::vector<float>& key_rope,
    int seq_len,
    int query_position);
std::vector<float> conv_decode_bf16(
    const std::vector<float>& projected_bcx,
    const std::vector<float>& conv_weight,
    std::vector<float>& state,
    int hidden,
    int cache_length,
    int position);

} // namespace celeg::reference
