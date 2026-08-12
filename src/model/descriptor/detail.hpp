#pragma once

#include "celeg/model/descriptor.hpp"
#include "celeg/checkpoint/formats/json.hpp"
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
    std::optional<Field> rope_theta_schedule;
    std::string position_kind = "rope";
    std::string rope_pairing = "split_half";
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
    std::optional<Field> moe_routing_group_count;
    std::optional<Field> moe_routing_experts_per_group;
    std::optional<Field> moe_routing_groups_per_token;
    std::optional<Field> moe_routing_group_score_top_k;
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
    NormWeightKind operator_norm_kind = NormWeightKind::Scale;
    NormWeightKind feed_forward_norm_kind = NormWeightKind::Scale;
    NormWeightKind final_norm_kind = NormWeightKind::Scale;
    NormWeightKind query_norm_kind = NormWeightKind::None;
    NormWeightKind key_norm_kind = NormWeightKind::None;
    bool query_norm_enabled = false;
    bool key_norm_enabled = false;
    std::optional<NormWeightKind> embedding_post_norm_kind;
    AttentionGateKind attention_gate_kind = AttentionGateKind::None;
    bool attention_gate_packed_with_query = false;
    std::optional<Field> orthogonalize_current_value;
    std::optional<Field> orthogonalize_current_value_minimum_norm_squared;
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
    std::string chat_template;
};

const Json& required(const Json& object, std::string_view key);
TensorRole parse_role(std::string_view name);
std::vector<BindingPattern> parse_bindings(const Json& object);
std::vector<ProbeCondition> parse_probe_conditions(const Json& value);
Field parse_field(const Json& value);
Descriptor parse_descriptor(const Json& value);
std::string selected_key(const CheckpointMetadata& metadata, const Field& field);
std::vector<std::string> selected_keys(const CheckpointMetadata& metadata, const Field& field);
int integer_value(const CheckpointMetadata& metadata, const Field& field,
                  int hidden = 0, int query_heads = 0);
double number_value(const CheckpointMetadata& metadata, const Field& field, int hidden = 0);
int scaling_integer_value(const CheckpointMetadata& metadata,
                          const std::optional<Field>& field, int fallback = 0);
double scaling_number_value(const CheckpointMetadata& metadata,
                            const std::optional<Field>& field, double fallback);
std::vector<float> scaling_factor_values(const CheckpointMetadata& metadata,
                                         const std::optional<Field>& field);
std::string position_kind_value(const CheckpointMetadata& metadata,
                                const Descriptor& descriptor);
RopeScalingKind parse_scaling_kind(std::string_view value);
StateScalarType parse_state_scalar(std::string_view value);
StateQuantizationGranularity parse_state_granularity(std::string_view value);
std::string scaling_kind_value(const CheckpointMetadata& metadata,
                               const Descriptor& descriptor);
std::vector<int> integer_values(const CheckpointMetadata& metadata,
                                const std::string& key);
std::vector<bool> attention_pattern_values(const CheckpointMetadata& metadata,
                                           const Descriptor& descriptor);
std::vector<int> field_integer_values(const CheckpointMetadata& metadata,
                                      const std::optional<Field>& field);
std::vector<bool> mixer_is_convolution(const CheckpointMetadata& metadata,
                                       const Descriptor& descriptor);
std::vector<std::string> mixer_schedule_values(const CheckpointMetadata& metadata,
                                               const Descriptor& descriptor);
bool boolean_value(const CheckpointMetadata& metadata, const std::optional<Field>& field,
                   bool fallback);
int token_value(const CheckpointMetadata& metadata, const Field& field,
                std::string_view gguf_override);
std::vector<int> eos_values(const CheckpointMetadata& metadata, const Descriptor& descriptor);
bool probe_condition(const CheckpointMetadata& metadata, const ProbeCondition& condition);
void build_descriptor_graph(ResolvedModel& model, const Descriptor& descriptor,
                            const RuntimeTopology& topology,
                            const CheckpointMetadata& metadata);
std::unique_ptr<IArchitecture> make_descriptor_architecture(Descriptor descriptor);
std::optional<Field> optional_field(const Json& object, std::string_view key);
std::string optional_string(const Json& object, std::string_view key,
                            std::string fallback = {});
bool optional_bool(const Json& object, std::string_view key, bool fallback = false);
AttentionVariant parse_attention_variant(const Json& value);
ActivationKind parse_activation_kind(std::string_view value);
NormWeightKind parse_norm_weight_kind(std::string_view value);
std::unique_ptr<ITensorNamingPolicy> create_naming_policy(const Descriptor& descriptor);

} // namespace celeg::descriptor_detail
