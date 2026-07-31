#pragma once

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
std::vector<float> conv_decode_bf16(
    const std::vector<float>& projected_bcx,
    const std::vector<float>& conv_weight,
    std::vector<float>& state,
    int hidden,
    int cache_length,
    int position);

} // namespace celeg::reference
