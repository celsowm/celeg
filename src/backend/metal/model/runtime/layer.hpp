#pragma once

#import <Metal/Metal.h>

#include "celeg/model/program.hpp"
#include "linear.hpp"

#include <optional>
#include <vector>

namespace celeg {

/**
 * @brief Per-layer Metal state: mixer, caches, norms and FFN weights.
 *
 * Each `MetalLayer` owns exactly one mixer family (attention,
 * short-convolution, GatedDelta or Mamba2) plus the shared FFN/MoE
 * weights. Only the active mixer fields are populated.
 */
struct MetalLayer {
    enum class MixerKind : uint8_t {
        Attention,
        ShortConvolution,
        GatedDelta,
        Mamba2,
    };

    struct Expert {
        std::string gate_name;
        std::string up_name;
        std::string down_name;
    };

    struct Moe {
        RouterProgram router;
        std::vector<float> router_weight;
        std::vector<float> router_bias;
        std::vector<Expert> experts;
    };

    id<MTLBuffer> operator_norm = nil;
    id<MTLBuffer> ffn_norm = nil;
    MixerKind mixer_kind = MixerKind::Attention;
    int cache_length = 0;
    int page_tokens = 16;
    MetalLinear mixer_in;
    MetalLinear mixer_out;
    id<MTLBuffer> convolution_taps = nil;
    id<MTLBuffer> recurrent_conv_weight = nil;
    id<MTLBuffer> recurrent_conv_bias = nil;
    id<MTLBuffer> recurrent_dt_bias = nil;
    id<MTLBuffer> recurrent_a_log = nil;
    id<MTLBuffer> recurrent_d = nil;
    id<MTLBuffer> recurrent_norm = nil;
    id<MTLBuffer> recurrent_conv_state = nil;
    id<MTLBuffer> recurrent_state = nil;
    int recurrent_conv_kernel = 0;
    int recurrent_key_head_dim = 0;
    int recurrent_value_head_dim = 0;
    int recurrent_key_heads = 0;
    int recurrent_value_heads = 0;
    int recurrent_inner = 0;
    int recurrent_state_size = 0;
    int recurrent_group_count = 0;
    bool recurrent_vector_decay = false;
    bool recurrent_safe_decay = false;
    float recurrent_decay_lower_bound = -5.0f;
    bool recurrent_sigmoid_output_gate = false;
    bool recurrent_a_log_needs_exp = true;
    MetalLinear recurrent_in;
    MetalLinear recurrent_qkv;
    MetalLinear recurrent_q;
    MetalLinear recurrent_k;
    MetalLinear recurrent_v;
    MetalLinear recurrent_z_weight;
    MetalLinear recurrent_b;
    MetalLinear recurrent_a;
    MetalLinear recurrent_out;
    MetalLinear query;
    MetalLinear key;
    MetalLinear value;
    MetalLinear attention_gate;
    MetalLinear attention_out;
    id<MTLBuffer> query_norm = nil;
    id<MTLBuffer> key_norm = nil;
    id<MTLBuffer> key_cache = nil;
    id<MTLBuffer> value_cache = nil;
    id<MTLBuffer> alibi_slopes = nil;
    id<MTLBuffer> relative_bias = nil;
    int kv_owner_layer = -1;
    int query_heads = 0;
    int key_value_heads = 0;
    int head_dim = 0;
    float query_scale = 1.0f;
    float rope_theta = 10000.0f;
    float query_norm_epsilon = 1.0e-5f;
    float key_norm_epsilon = 1.0e-5f;
    MetalLinear ffn_gate;
    MetalLinear ffn_up;
    MetalLinear ffn_down;
    int intermediate = 0;
    std::optional<Moe> moe;
};

}
