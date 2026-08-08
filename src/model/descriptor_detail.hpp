#pragma once

#include "celeg/model/descriptor.hpp"
#include "celeg/model/graph_builder.hpp"
#include "celeg/model/weight_plan.hpp"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace celeg::descriptor_detail {

struct Field {
    std::string json;
    std::vector<std::string> json_alternatives;
    std::string gguf;
    double fallback = 0.0;
    std::string fallback_expression;
};

struct BindingPattern {
    TensorRole role;
    std::vector<std::string> candidates;
};

struct AttentionVariant {
    std::optional<Field> query_heads;
    std::optional<Field> key_value_heads;
    std::optional<Field> head_dim;
    std::optional<Field> rope_theta;
    std::optional<Field> rotary_fraction;
    std::optional<Field> sliding_window;
};

struct ProbeCondition {
    std::string key;
    std::string json;
    std::string gguf;
    std::string equals;
    std::string contains;
    int integer_equals = 0;
    bool has_integer_equals = false;
    bool case_insensitive = false;
};

struct Descriptor {
    std::string id;
    int specificity = 0;
    std::vector<std::vector<ProbeCondition>> probe_alternatives;
    std::unordered_map<std::string, Field> dimensions;
    std::unordered_map<std::string, Field> numbers;
    Field bos;
    Field eos;
    Field pad;
    std::string gguf_bos;
    std::string gguf_eos;
    std::string gguf_pad;
    std::string gguf_eot;
    std::string disable_rope_json;
    std::string disable_rope_gguf;
    std::string position_kind = "rope";
    std::optional<Field> position_kind_field;
    std::optional<Field> alibi_slopes;
    std::optional<Field> relative_bucket_count;
    std::optional<Field> relative_max_distance;
    bool relative_bidirectional = false;
    std::optional<Field> rotary_fraction;
    std::string rope_scaling_kind;
    std::optional<Field> rope_scaling_kind_field;
    std::optional<Field> rope_scaling_factor;
    std::optional<Field> rope_scaling_original_context;
    std::optional<Field> rope_scaling_attention_factor;
    std::optional<Field> rope_scaling_beta_fast;
    std::optional<Field> rope_scaling_beta_slow;
    std::optional<Field> rope_scaling_low_frequency_factor;
    std::optional<Field> rope_scaling_high_frequency_factor;
    std::optional<Field> rope_scaling_short_factors;
    std::optional<Field> rope_scaling_long_factors;
    std::optional<Field> mrope_sections;
    std::optional<Field> mrope_interleaved;
    bool repeated_layers = false;
    Field repeat_count;
    bool map_physical_layers = false;
    bool norm_after_physical_block = false;
    std::optional<AttentionVariant> full_attention_variant;
    std::optional<AttentionVariant> sliding_attention_variant;
    std::optional<Field> shared_kv_suffix_layers;
    std::optional<Field> mixer_schedule;
    std::optional<Field> kv_heads_schedule;
    std::string convolution_value = "conv";
    std::optional<Field> convolution_cache;
    std::optional<Field> convolution_channels;
    std::optional<Field> moe_dense_layers;
    std::optional<Field> moe_intermediate;
    std::optional<Field> moe_experts;
    std::optional<Field> moe_experts_per_token;
    std::optional<Field> moe_normalize_topk;
    std::optional<Field> moe_expert_bias;
    std::optional<Field> moe_routed_scaling;
    std::optional<Field> moe_shared_intermediate;
    std::optional<Field> recurrent_key_heads;
    std::optional<Field> recurrent_value_heads;
    std::optional<Field> recurrent_key_dim;
    std::optional<Field> recurrent_value_dim;
    std::optional<Field> recurrent_conv_kernel;
    std::optional<Field> mamba_intermediate;
    std::optional<Field> mamba_state_size;
    std::optional<Field> mamba_time_step_rank;
    std::optional<Field> mamba_heads;
    std::optional<Field> mamba_head_dim;
    std::optional<Field> mamba_groups;
    std::optional<Field> mamba_chunk_size;
    bool split_attention_norms = false;
    bool query_key_norm = false;
    bool query_gate = false;
    std::string attention_key_value_source = "current_sequence";
    std::optional<Field> attention_memory_slot;
    std::string attention_state_kind = "ordinary_kv";
    std::string state_key_storage = "bf16";
    std::string state_value_storage = "bf16";
    std::string state_latent_storage = "bf16";
    std::string state_rotary_storage = "bf16";
    std::string state_recurrent_storage = "fp32";
    std::string state_storage_granularity = "per_tensor";
    bool state_paged = true;
    std::optional<Field> latent_rank;
    std::optional<Field> latent_rope_head_dim;
    std::optional<Field> latent_nope_head_dim;
    bool latent_decoupled_rope = false;
    std::optional<Field> per_layer_input_size;
    bool double_wide_shared_suffix = false;
    ActivationKind feed_forward_activation = ActivationKind::SwiGLU;
    std::optional<Field> final_logit_softcap;
    std::optional<Field> attention_pattern;
    std::optional<Field> sliding_window;
    std::string sliding_pattern_value = "sliding_attention";
    std::vector<BindingPattern> bindings;
    bool tied_embeddings = false;
    std::optional<Field> tied_embeddings_field;
    std::string chat_profile;
};

std::unique_ptr<ITensorNamingPolicy> create_naming_policy(const Descriptor& descriptor);
void build_weight_plan(ResolvedModel& model, const Descriptor& descriptor,
                       const ITensorNamingPolicy& naming_policy);

} // namespace celeg::descriptor_detail
