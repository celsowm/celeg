#include "celeg/model/descriptor.hpp"

#include "celeg/checkpoint/formats/json.hpp"
#include "celeg/model/graph_builder.hpp"
#include "celeg/model/weight_plan.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace celeg {
namespace {

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

const Json& required(const Json& object, std::string_view key) {
    if (!object.is_object() || !object.contains(key)) {
        throw std::invalid_argument("descriptor is missing field: " + std::string(key));
    }
    return object.at(key);
}

std::string optional_string(const Json& object, std::string_view key,
                           std::string fallback = {}) {
    return object.is_object() && object.contains(key) ? object.at(key).as_string()
                                                       : std::move(fallback);
}

bool optional_bool(const Json& object, std::string_view key, bool fallback = false) {
    return object.is_object() && object.contains(key) ? object.at(key).as_bool() : fallback;
}

Field parse_field(const Json& value) {
    Field result;
    result.json = optional_string(value, "json");
    if (value.is_object() && value.contains("json_alternatives")) {
        for (const Json& alternative : value.at("json_alternatives").as_array()) {
            result.json_alternatives.push_back(alternative.as_string());
        }
    }
    result.gguf = optional_string(value, "gguf");
    if (value.is_object() && value.contains("default")) {
        const Json& fallback = value.at("default");
        if (fallback.is_string()) result.fallback_expression = fallback.as_string();
        else result.fallback = fallback.as_number();
    }
    if (result.json.empty() && result.gguf.empty() && result.fallback_expression.empty() &&
        !(value.is_object() && value.contains("default"))) {
        throw std::invalid_argument("descriptor field has no metadata key or default");
    }
    return result;
}

std::optional<Field> optional_field(const Json& object, std::string_view key) {
    return object.is_object() && object.contains(key)
        ? std::optional<Field>(parse_field(object.at(key))) : std::nullopt;
}

AttentionVariant parse_attention_variant(const Json& value) {
    AttentionVariant result;
    result.query_heads = optional_field(value, "query_heads");
    result.key_value_heads = optional_field(value, "kv_heads");
    result.head_dim = optional_field(value, "head_dim");
    result.rope_theta = optional_field(value, "rope_theta");
    result.rotary_fraction = optional_field(value, "rotary_fraction");
    result.sliding_window = optional_field(value, "sliding_window");
    if (!result.query_heads.has_value() && !result.key_value_heads.has_value() &&
        !result.head_dim.has_value() && !result.rope_theta.has_value() &&
        !result.rotary_fraction.has_value() && !result.sliding_window.has_value()) {
        throw std::invalid_argument("descriptor attention variant is empty");
    }
    return result;
}

ActivationKind parse_activation(std::string_view value) {
    if (value == "swiglu") return ActivationKind::SwiGLU;
    if (value == "gelu_tanh") return ActivationKind::GeluTanh;
    if (value == "relu2") return ActivationKind::Relu2;
    throw std::invalid_argument("descriptor has unsupported feed-forward activation: " +
                                std::string(value));
}

TensorRole parse_role(std::string_view name) {
    static const std::unordered_map<std::string_view, TensorRole> roles = {
        {"TokenEmbedding", TensorRole::TokenEmbedding},
        {"LanguageModelHead", TensorRole::LanguageModelHead},
        {"FinalNorm", TensorRole::FinalNorm},
        {"AttentionInputNorm", TensorRole::AttentionInputNorm},
        {"AttentionQuery", TensorRole::AttentionQuery},
        {"AttentionQueryNorm", TensorRole::AttentionQueryNorm},
        {"AttentionKey", TensorRole::AttentionKey},
        {"AttentionKeyNorm", TensorRole::AttentionKeyNorm},
        {"AttentionValue", TensorRole::AttentionValue},
        {"AttentionRelativePositionBias", TensorRole::AttentionRelativePositionBias},
        {"AttentionLatentQuery", TensorRole::AttentionLatentQuery},
        {"AttentionLatentQueryRope", TensorRole::AttentionLatentQueryRope},
        {"AttentionLatentKey", TensorRole::AttentionLatentKey},
        {"AttentionLatentValue", TensorRole::AttentionLatentValue},
        {"AttentionLatentKeyRope", TensorRole::AttentionLatentKeyRope},
        {"AttentionLatentOutput", TensorRole::AttentionLatentOutput},
        {"AttentionOutput", TensorRole::AttentionOutput},
        {"AttentionPostNorm", TensorRole::AttentionPostNorm},
        {"FfnInputNorm", TensorRole::FfnInputNorm},
        {"FfnOutputNorm", TensorRole::FfnOutputNorm},
        {"FfnGate", TensorRole::FfnGate},
        {"FfnUp", TensorRole::FfnUp},
        {"FfnDown", TensorRole::FfnDown},
        {"PerLayerEmbedding", TensorRole::PerLayerEmbedding},
        {"PerLayerContextProjection", TensorRole::PerLayerContextProjection},
        {"PerLayerProjection", TensorRole::PerLayerProjection},
        {"PerLayerProjectionNorm", TensorRole::PerLayerProjectionNorm},
        {"PerLayerInputGate", TensorRole::PerLayerInputGate},
        {"PerLayerInputNorm", TensorRole::PerLayerInputNorm},
        {"LayerScalar", TensorRole::LayerScalar},
        {"ShortConvInput", TensorRole::ShortConvInput},
        {"ShortConvKernel", TensorRole::ShortConvKernel},
        {"ShortConvOutput", TensorRole::ShortConvOutput},
        {"GatedDeltaNetQkv", TensorRole::GatedDeltaNetQkv},
        {"GatedDeltaNetZ", TensorRole::GatedDeltaNetZ},
        {"GatedDeltaNetAlpha", TensorRole::GatedDeltaNetAlpha},
        {"GatedDeltaNetBeta", TensorRole::GatedDeltaNetBeta},
        {"GatedDeltaNetDtBias", TensorRole::GatedDeltaNetDtBias},
        {"GatedDeltaNetALog", TensorRole::GatedDeltaNetALog},
        {"GatedDeltaNetConv", TensorRole::GatedDeltaNetConv},
        {"GatedDeltaNetNorm", TensorRole::GatedDeltaNetNorm},
        {"GatedDeltaNetOutput", TensorRole::GatedDeltaNetOutput},
        {"Mamba2Input", TensorRole::Mamba2Input},
        {"Mamba2Conv", TensorRole::Mamba2Conv},
        {"Mamba2ConvBias", TensorRole::Mamba2ConvBias},
        {"Mamba2DtBias", TensorRole::Mamba2DtBias},
        {"Mamba2ALog", TensorRole::Mamba2ALog},
        {"Mamba2D", TensorRole::Mamba2D},
        {"Mamba2Norm", TensorRole::Mamba2Norm},
        {"Mamba2Output", TensorRole::Mamba2Output},
        {"MoeRouter", TensorRole::MoeRouter},
        {"MoeExpertGate", TensorRole::MoeExpertGate},
        {"MoeExpertUp", TensorRole::MoeExpertUp},
        {"MoeExpertDown", TensorRole::MoeExpertDown},
        {"MoePackedGateUp", TensorRole::MoePackedGateUp},
        {"MoePackedDown", TensorRole::MoePackedDown},
        {"MoeSharedGate", TensorRole::MoeSharedGate},
        {"MoeSharedUp", TensorRole::MoeSharedUp},
        {"MoeSharedDown", TensorRole::MoeSharedDown},
        {"MoeSharedGateWeight", TensorRole::MoeSharedGateWeight},
    };
    const auto it = roles.find(name);
    if (it == roles.end()) {
        throw std::invalid_argument("descriptor has unsupported tensor role: " +
                                    std::string(name));
    }
    return it->second;
}

std::vector<BindingPattern> parse_bindings(const Json& object) {
    std::vector<BindingPattern> result;
    for (const auto& [name, values] : required(object, "bindings").as_object()) {
        BindingPattern pattern{parse_role(name), {}};
        for (const Json& candidate : values.as_array()) pattern.candidates.push_back(candidate.as_string());
        if (pattern.candidates.empty()) throw std::invalid_argument("descriptor binding has no candidates");
        result.push_back(std::move(pattern));
    }
    return result;
}

std::vector<ProbeCondition> parse_conditions(const Json& value) {
    std::vector<ProbeCondition> result;
    for (const Json& item : value.as_array()) {
        ProbeCondition condition;
        condition.key = optional_string(item, "key");
        condition.json = optional_string(item, "json");
        condition.gguf = optional_string(item, "gguf");
        condition.equals = item.contains("equals") && item.at("equals").is_string()
            ? item.at("equals").as_string() : std::string{};
        condition.contains = optional_string(item, "contains");
        condition.case_insensitive = optional_bool(item, "case_insensitive");
        if (condition.key == "integer") {
            if (!item.contains("equals") || !item.at("equals").is_number()) {
                throw std::invalid_argument("integer descriptor probe has no numeric equals");
            }
            condition.has_integer_equals = true;
            condition.integer_equals = static_cast<int>(item.at("equals").as_i64());
        }
        if (condition.key.empty() && condition.json.empty() && condition.gguf.empty()) {
            throw std::invalid_argument("descriptor probe condition has no key");
        }
        result.push_back(std::move(condition));
    }
    return result;
}

Descriptor parse_descriptor(const Json& value) {
    Descriptor result;
    result.id = required(value, "id").as_string();
    result.specificity = static_cast<int>(required(value, "specificity").as_i64());
    for (const Json& alternative : required(value, "probe").as_array()) {
        result.probe_alternatives.push_back(parse_conditions(required(alternative, "all")));
    }
    for (const auto& [name, field] : required(value, "dimensions").as_object()) {
        result.dimensions.emplace(name, parse_field(field));
    }
    for (const auto& [name, field] : required(value, "numbers").as_object()) {
        result.numbers.emplace(name, parse_field(field));
    }
    const Json& tokens = required(value, "tokens");
    result.bos = parse_field(required(tokens, "bos"));
    result.eos = parse_field(required(tokens, "eos"));
    result.pad = parse_field(required(tokens, "pad"));
    result.gguf_bos = optional_string(tokens, "gguf_bos");
    result.gguf_eos = optional_string(tokens, "gguf_eos");
    result.gguf_pad = optional_string(tokens, "gguf_pad");
    result.gguf_eot = optional_string(tokens, "gguf_eot");
    if (value.contains("position")) {
        const Json& position = value.at("position");
        if (position.contains("kind")) {
            if (position.at("kind").is_string()) {
                result.position_kind = position.at("kind").as_string();
            } else {
                result.position_kind_field = parse_field(position.at("kind"));
            }
        }
        result.disable_rope_json = optional_string(position, "disable_rope_key");
        result.disable_rope_gguf = optional_string(position, "disable_rope_gguf_key");
        if (position.contains("alibi_slopes")) {
            result.alibi_slopes = parse_field(position.at("alibi_slopes"));
        }
        if (position.contains("relative_bias")) {
            const Json& bias = position.at("relative_bias");
            result.relative_bucket_count = bias.contains("bucket_count")
                ? std::optional<Field>(parse_field(bias.at("bucket_count"))) : std::nullopt;
            result.relative_max_distance = bias.contains("max_distance")
                ? std::optional<Field>(parse_field(bias.at("max_distance"))) : std::nullopt;
            result.relative_bidirectional = optional_bool(bias, "bidirectional");
        }
        if (position.contains("rotary_fraction")) {
            result.rotary_fraction = parse_field(position.at("rotary_fraction"));
        }
        if (position.contains("mrope_sections")) {
            result.mrope_sections = parse_field(position.at("mrope_sections"));
        }
        if (position.contains("mrope_interleaved")) {
            result.mrope_interleaved = parse_field(position.at("mrope_interleaved"));
        }
        if (position.contains("scaling")) {
            const Json& scaling = position.at("scaling");
            if (scaling.contains("kind")) {
                if (scaling.at("kind").is_string()) {
                    result.rope_scaling_kind = scaling.at("kind").as_string();
                } else {
                    result.rope_scaling_kind_field = parse_field(scaling.at("kind"));
                }
            }
            const auto parse_optional_field = [&scaling](std::string_view key)
                -> std::optional<Field> {
                return scaling.contains(key)
                    ? std::optional<Field>(parse_field(scaling.at(key))) : std::nullopt;
            };
            result.rope_scaling_factor = parse_optional_field("factor");
            result.rope_scaling_original_context = parse_optional_field("original_context");
            result.rope_scaling_attention_factor = parse_optional_field("attention_factor");
            result.rope_scaling_beta_fast = parse_optional_field("beta_fast");
            result.rope_scaling_beta_slow = parse_optional_field("beta_slow");
            result.rope_scaling_low_frequency_factor = parse_optional_field("low_frequency_factor");
            result.rope_scaling_high_frequency_factor = parse_optional_field("high_frequency_factor");
            result.rope_scaling_short_factors = parse_optional_field("short_factors");
            result.rope_scaling_long_factors = parse_optional_field("long_factors");
        }
    }
    if (value.contains("layer_schedule")) {
        const Json& schedule = value.at("layer_schedule");
        if (schedule.contains("repeat")) {
            result.repeated_layers = true;
            result.repeat_count = parse_field(schedule.at("repeat"));
        }
        result.map_physical_layers = optional_string(schedule, "physical_mapping") == "modulo";
        result.norm_after_physical_block = optional_string(schedule, "norm_after") ==
            "physical_block_end";
        result.shared_kv_suffix_layers = optional_field(schedule, "shared_kv_suffix_layers");
        if (schedule.contains("attention_variants")) {
            const Json& variants = schedule.at("attention_variants");
            if (variants.contains("full")) {
                result.full_attention_variant = parse_attention_variant(variants.at("full"));
            }
            if (variants.contains("sliding")) {
                result.sliding_attention_variant =
                    parse_attention_variant(variants.at("sliding"));
            }
        }
        if (schedule.contains("attention_pattern")) {
            result.attention_pattern = parse_field(schedule.at("attention_pattern"));
            if (!schedule.contains("sliding_window")) {
                throw std::invalid_argument(
                    "descriptor attention pattern has no sliding window field");
            }
            result.sliding_window = parse_field(schedule.at("sliding_window"));
            result.sliding_pattern_value = optional_string(
                schedule, "sliding_value", result.sliding_pattern_value);
        }
    }
    if (value.contains("graph")) {
        const Json& graph = value.at("graph");
        result.split_attention_norms = optional_bool(graph, "split_attention_norms");
        result.per_layer_input_size = optional_field(graph, "per_layer_input_size");
        result.double_wide_shared_suffix = optional_bool(graph, "double_wide_shared_suffix");
        if (graph.contains("feed_forward_activation")) {
            result.feed_forward_activation = parse_activation(
                graph.at("feed_forward_activation").as_string());
        }
        result.final_logit_softcap = optional_field(graph, "final_logit_softcap");
    }
    if (value.contains("attention")) {
        const Json& attention = value.at("attention");
        result.query_key_norm = optional_bool(attention, "query_key_norm");
        result.query_gate = optional_bool(attention, "query_gate");
        if (attention.contains("sources")) {
            const Json& sources = attention.at("sources");
            result.attention_key_value_source = optional_string(
                sources, "key_value", result.attention_key_value_source);
            result.attention_memory_slot = optional_field(sources, "memory_slot");
        }
        if (attention.contains("state")) {
            const Json& state = attention.at("state");
            result.attention_state_kind = optional_string(
                state, "kind", result.attention_state_kind);
            result.state_key_storage = optional_string(
                state, "key_storage", result.state_key_storage);
            result.state_value_storage = optional_string(
                state, "value_storage", result.state_value_storage);
            result.state_latent_storage = optional_string(
                state, "latent_storage", result.state_latent_storage);
            result.state_rotary_storage = optional_string(
                state, "rotary_storage", result.state_rotary_storage);
            result.state_recurrent_storage = optional_string(
                state, "recurrent_storage", result.state_recurrent_storage);
            result.state_storage_granularity = optional_string(
                state, "storage_granularity", result.state_storage_granularity);
            result.state_paged = optional_bool(state, "paged", result.state_paged);
            result.latent_rank = optional_field(state, "latent_rank");
            result.latent_rope_head_dim = optional_field(state, "rope_head_dim");
            result.latent_nope_head_dim = optional_field(state, "nope_head_dim");
            result.latent_decoupled_rope = optional_bool(state, "decoupled_rope");
        }
    }
    if (value.contains("hybrid")) {
        const Json& hybrid = value.at("hybrid");
        result.mixer_schedule = optional_field(hybrid, "mixer_schedule");
        result.kv_heads_schedule = optional_field(hybrid, "kv_heads_schedule");
        result.convolution_value = optional_string(hybrid, "convolution_value",
                                                   result.convolution_value);
        result.convolution_cache = optional_field(hybrid, "convolution_cache");
        result.convolution_channels = optional_field(hybrid, "convolution_channels");
        result.moe_dense_layers = optional_field(hybrid, "dense_layers");
        result.moe_intermediate = optional_field(hybrid, "expert_intermediate");
        result.moe_experts = optional_field(hybrid, "experts");
        result.moe_experts_per_token = optional_field(hybrid, "experts_per_token");
        result.moe_normalize_topk = optional_field(hybrid, "normalize_topk");
        result.moe_expert_bias = optional_field(hybrid, "expert_bias");
        result.moe_routed_scaling = optional_field(hybrid, "routed_scaling");
        result.moe_shared_intermediate = optional_field(hybrid, "shared_expert_intermediate");
        result.recurrent_key_heads = optional_field(hybrid, "linear_key_heads");
        result.recurrent_value_heads = optional_field(hybrid, "linear_value_heads");
        result.recurrent_key_dim = optional_field(hybrid, "linear_key_dim");
        result.recurrent_value_dim = optional_field(hybrid, "linear_value_dim");
        result.recurrent_conv_kernel = optional_field(hybrid, "linear_conv_kernel");
        result.mamba_intermediate = optional_field(hybrid, "mamba_intermediate");
        result.mamba_state_size = optional_field(hybrid, "mamba_state_size");
        result.mamba_time_step_rank = optional_field(hybrid, "mamba_time_step_rank");
        result.mamba_heads = optional_field(hybrid, "mamba_heads");
        result.mamba_head_dim = optional_field(hybrid, "mamba_head_dim");
        result.mamba_groups = optional_field(hybrid, "mamba_groups");
        result.mamba_chunk_size = optional_field(hybrid, "mamba_chunk_size");
    }
    result.bindings = parse_bindings(value);
    if (value.contains("tied_embeddings")) {
        result.tied_embeddings = value.at("tied_embeddings").is_bool()
            ? value.at("tied_embeddings").as_bool() : true;
        if (!value.at("tied_embeddings").is_bool()) {
            result.tied_embeddings_field = parse_field(value.at("tied_embeddings"));
        }
    } else {
        result.tied_embeddings = true;
    }
    result.chat_profile = required(value, "chat_profile").as_string();
    if (result.id.empty() || result.specificity <= 0 || result.probe_alternatives.empty()) {
        throw std::invalid_argument("descriptor has invalid identity or probe");
    }
    return result;
}

std::string selected_key(const CheckpointMetadata& metadata, const Field& field) {
    std::string key = metadata.is_gguf() ? field.gguf : field.json;
    const std::string needle = "{architecture}";
    size_t position = 0;
    while ((position = key.find(needle, position)) != std::string::npos) {
        const std::string architecture = metadata.architecture_type();
        key.replace(position, needle.size(), architecture);
        position += architecture.size();
    }
    return key;
}

std::vector<std::string> selected_keys(const CheckpointMetadata& metadata,
                                       const Field& field) {
    std::vector<std::string> keys;
    const std::string primary = selected_key(metadata, field);
    if (!primary.empty()) keys.push_back(primary);
    if (!metadata.is_gguf()) {
        for (const std::string& alternative : field.json_alternatives) {
            Field alias;
            alias.json = alternative;
            const std::string key = selected_key(metadata, alias);
            if (!key.empty()) keys.push_back(key);
        }
    }
    return keys;
}

int integer_value(const CheckpointMetadata& metadata, const Field& field,
                 int hidden = 0, int query_heads = 0) {
    const std::string key = selected_key(metadata, field);
    if (!key.empty() && metadata.contains(key)) {
        const MetadataValue& value = metadata.value(key);
        if (const auto* scalar = std::get_if<int64_t>(&value)) {
            return static_cast<int>(*scalar);
        }
        if (const auto* scalar = std::get_if<double>(&value)) {
            return static_cast<int>(*scalar);
        }
        if (std::holds_alternative<std::vector<int64_t>>(value)) {
            return field.fallback != 0.0 ? static_cast<int>(field.fallback) : 0;
        }
        throw std::runtime_error("descriptor dimension metadata is not an integer: " + key);
    }
    if (field.fallback_expression == "hidden_div_query_heads") {
        if (hidden <= 0 || query_heads <= 0) throw std::invalid_argument("invalid descriptor dimension expression");
        return hidden / query_heads;
    }
    if (field.fallback_expression.empty()) return static_cast<int>(field.fallback);
    throw std::invalid_argument("unsupported descriptor default expression: " + field.fallback_expression);
}

double number_value(const CheckpointMetadata& metadata, const Field& field, int hidden = 0) {
    const std::vector<std::string> keys = selected_keys(metadata, field);
    for (const std::string& key : keys) {
        if (metadata.contains(key)) return metadata.number(key);
    }
    const std::string key = keys.empty() ? selected_key(metadata, field) : keys.front();
    if (field.fallback_expression == "sqrt_hidden") {
        if (hidden <= 0) throw std::invalid_argument("invalid hidden size expression");
        return std::sqrt(static_cast<double>(hidden));
    }
    if (!field.fallback_expression.empty()) {
        throw std::invalid_argument("unsupported descriptor numeric expression: " +
                                    field.fallback_expression);
    }
    return field.fallback;
}

int scaling_integer_value(const CheckpointMetadata& metadata,
                          const std::optional<Field>& field, int fallback = 0) {
    return field.has_value() ? integer_value(metadata, *field) : fallback;
}

double scaling_number_value(const CheckpointMetadata& metadata,
                            const std::optional<Field>& field, double fallback) {
    return field.has_value() ? number_value(metadata, *field) : fallback;
}

std::vector<float> scaling_factor_values(const CheckpointMetadata& metadata,
                                         const std::optional<Field>& field) {
    if (!field.has_value()) return {};
    const std::string key = selected_key(metadata, *field);
    if (key.empty() || !metadata.contains(key)) return {};
    const MetadataValue& value = metadata.value(key);
    std::vector<float> result;
    if (const auto* list = std::get_if<std::vector<double>>(&value)) {
        result.reserve(list->size());
        for (double item : *list) result.push_back(static_cast<float>(item));
        return result;
    }
    if (const auto* list = std::get_if<std::vector<int64_t>>(&value)) {
        result.reserve(list->size());
        for (int64_t item : *list) result.push_back(static_cast<float>(item));
        return result;
    }
    if (const auto* item = std::get_if<double>(&value)) return {static_cast<float>(*item)};
    if (const auto* item = std::get_if<int64_t>(&value)) return {static_cast<float>(*item)};
    throw std::runtime_error("RoPE factor metadata must be numeric: " + key);
}

std::string position_kind_value(const CheckpointMetadata& metadata,
                                const Descriptor& descriptor) {
    if (!descriptor.position_kind_field.has_value()) return descriptor.position_kind;
    const Field& field = *descriptor.position_kind_field;
    const std::string key = selected_key(metadata, field);
    if (!key.empty() && metadata.contains(key)) return metadata.string(key);
    return field.fallback_expression.empty() ? "rope" : field.fallback_expression;
}

RopeScalingKind parse_scaling_kind(std::string_view value) {
    if (value.empty() || value == "none") return RopeScalingKind::None;
    if (value == "linear") return RopeScalingKind::Linear;
    if (value == "dynamic_ntk") return RopeScalingKind::DynamicNtk;
    if (value == "yarn") return RopeScalingKind::Yarn;
    if (value == "long" || value == "longrope") return RopeScalingKind::Long;
    if (value == "llama3_frequency") return RopeScalingKind::Llama3Frequency;
    throw std::invalid_argument("descriptor has unsupported RoPE scaling kind: " +
                                std::string(value));
}

StateScalarType parse_state_scalar(std::string_view value) {
    if (value == "fp32") return StateScalarType::FP32;
    if (value == "fp16") return StateScalarType::FP16;
    if (value == "bf16") return StateScalarType::BF16;
    if (value == "fp8") return StateScalarType::FP8;
    if (value == "int8") return StateScalarType::INT8;
    if (value == "int4") return StateScalarType::INT4;
    throw std::invalid_argument("descriptor has unsupported state storage: " +
                                std::string(value));
}

StateQuantizationGranularity parse_state_granularity(std::string_view value) {
    if (value == "per_tensor") return StateQuantizationGranularity::PerTensor;
    if (value == "per_head") return StateQuantizationGranularity::PerHead;
    if (value == "per_token") return StateQuantizationGranularity::PerToken;
    if (value == "per_block") return StateQuantizationGranularity::PerBlock;
    throw std::invalid_argument("descriptor has unsupported state quantization granularity: " +
                                std::string(value));
}

std::string scaling_kind_value(const CheckpointMetadata& metadata,
                               const Descriptor& descriptor) {
    if (!descriptor.rope_scaling_kind_field.has_value()) {
        return descriptor.rope_scaling_kind;
    }
    const Field& field = *descriptor.rope_scaling_kind_field;
    const std::string key = selected_key(metadata, field);
    if (!key.empty() && metadata.contains(key)) return metadata.string(key);
    return field.fallback_expression.empty() ? "none" : field.fallback_expression;
}

std::vector<int> integer_values(const CheckpointMetadata& metadata,
                               const std::string& key) {
    if (key.empty() || !metadata.contains(key)) return {};
    const MetadataValue& value = metadata.value(key);
    if (const auto* list = std::get_if<std::vector<int64_t>>(&value)) {
        std::vector<int> result;
        result.reserve(list->size());
        for (int64_t item : *list) result.push_back(static_cast<int>(item));
        return result;
    }
    return {};
}

std::vector<bool> attention_pattern_values(const CheckpointMetadata& metadata,
                                           const Descriptor& descriptor) {
    if (!descriptor.attention_pattern.has_value()) return {};
    const std::string key = selected_key(metadata, *descriptor.attention_pattern);
    if (key.empty() || !metadata.contains(key)) return {};
    const MetadataValue& value = metadata.value(key);
    const auto is_sliding = [&descriptor](std::string_view item) {
        return item == descriptor.sliding_pattern_value || item == "sliding";
    };
    if (const auto* list = std::get_if<std::vector<std::string>>(&value)) {
        std::vector<bool> result;
        result.reserve(list->size());
        for (const std::string& item : *list) result.push_back(is_sliding(item));
        return result;
    }
    if (const auto* list = std::get_if<std::vector<int64_t>>(&value)) {
        std::vector<bool> result;
        result.reserve(list->size());
        for (int64_t item : *list) result.push_back(item != 0);
        return result;
    }
    if (const auto* item = std::get_if<std::string>(&value)) return {is_sliding(*item)};
    if (const auto* item = std::get_if<int64_t>(&value)) return {*item != 0};
    throw std::runtime_error("attention pattern metadata must be strings or integers: " + key);
}

std::vector<int> field_integer_values(const CheckpointMetadata& metadata,
                                      const std::optional<Field>& field) {
    if (!field.has_value()) return {};
    const std::string key = selected_key(metadata, *field);
    if (key.empty() || !metadata.contains(key)) return {};
    const MetadataValue& value = metadata.value(key);
    if (const auto* list = std::get_if<std::vector<int64_t>>(&value)) {
        std::vector<int> result;
        result.reserve(list->size());
        for (int64_t item : *list) result.push_back(static_cast<int>(item));
        return result;
    }
    return {};
}

std::vector<bool> mixer_is_convolution(const CheckpointMetadata& metadata,
                                       const Descriptor& descriptor) {
    if (!descriptor.mixer_schedule.has_value()) return {};
    const std::string key = selected_key(metadata, *descriptor.mixer_schedule);
    if (key.empty() || !metadata.contains(key)) return {};
    const MetadataValue& value = metadata.value(key);
    const auto is_convolution = [&descriptor](std::string_view item) {
        return item == descriptor.convolution_value || item == "conv" ||
               item == "short_convolution";
    };
    if (const auto* list = std::get_if<std::vector<std::string>>(&value)) {
        std::vector<bool> result;
        result.reserve(list->size());
        for (const std::string& item : *list) result.push_back(is_convolution(item));
        return result;
    }
    if (const auto* list = std::get_if<std::vector<int64_t>>(&value)) {
        std::vector<bool> result;
        result.reserve(list->size());
        for (int64_t item : *list) result.push_back(item == 0);
        return result;
    }
    if (const auto* item = std::get_if<std::string>(&value)) return {is_convolution(*item)};
    if (const auto* item = std::get_if<int64_t>(&value)) return {*item == 0};
    throw std::runtime_error("mixer schedule metadata must be strings or integers: " + key);
}

std::vector<std::string> mixer_schedule_values(const CheckpointMetadata& metadata,
                                               const Descriptor& descriptor) {
    if (!descriptor.mixer_schedule.has_value()) return {};
    const std::string key = selected_key(metadata, *descriptor.mixer_schedule);
    if (key.empty() || !metadata.contains(key)) return {};
    const MetadataValue& value = metadata.value(key);
    if (const auto* list = std::get_if<std::vector<std::string>>(&value)) return *list;
    if (const auto* list = std::get_if<std::vector<int64_t>>(&value)) {
        std::vector<std::string> result;
        result.reserve(list->size());
        for (int64_t item : *list) result.push_back(std::to_string(item));
        return result;
    }
    if (const auto* item = std::get_if<std::string>(&value)) {
        if (item->size() > 1 && item->find_first_not_of("M-*_") == std::string::npos) {
            std::vector<std::string> result;
            result.reserve(item->size());
            for (char symbol : *item) result.emplace_back(1, symbol);
            return result;
        }
        return {*item};
    }
    if (const auto* item = std::get_if<int64_t>(&value)) return {std::to_string(*item)};
    throw std::runtime_error("mixer schedule metadata must be strings or integers: " + key);
}

bool boolean_value(const CheckpointMetadata& metadata, const std::optional<Field>& field,
                  bool fallback) {
    if (!field.has_value()) return fallback;
    const std::string key = selected_key(metadata, *field);
    if (key.empty() || !metadata.contains(key)) {
        if (field->fallback_expression == "true") return true;
        if (field->fallback_expression == "false" || !field->fallback_expression.empty()) {
            return false;
        }
        return field->fallback != 0.0 ? true : fallback;
    }
    return metadata.boolean(key);
}

int token_value(const CheckpointMetadata& metadata, const Field& field,
                std::string_view gguf_override) {
    if (metadata.is_gguf() && !gguf_override.empty() && metadata.contains(gguf_override)) {
        return static_cast<int>(metadata.integer(gguf_override));
    }
    return integer_value(metadata, field);
}

std::vector<int> eos_values(const CheckpointMetadata& metadata, const Descriptor& descriptor) {
    if (metadata.is_gguf()) {
        std::vector<int> result = integer_values(metadata, descriptor.gguf_eos);
        if (result.empty()) result.push_back(token_value(metadata, descriptor.eos, descriptor.gguf_eos));
        if (!descriptor.gguf_eot.empty()) {
            const std::vector<int> eot = integer_values(metadata, descriptor.gguf_eot);
            result.insert(result.end(), eot.begin(), eot.end());
            if (eot.empty() && metadata.contains(descriptor.gguf_eot)) {
                result.push_back(static_cast<int>(metadata.integer(descriptor.gguf_eot)));
            }
            std::sort(result.begin(), result.end());
            result.erase(std::unique(result.begin(), result.end()), result.end());
        }
        return result;
    }
    return integer_values(metadata, descriptor.eos.json).empty()
        ? std::vector<int>{token_value(metadata, descriptor.eos, {})}
        : integer_values(metadata, descriptor.eos.json);
}

bool equal_text(std::string_view left, std::string_view right, bool insensitive) {
    if (!insensitive) return left == right;
    if (left.size() != right.size()) return false;
    for (size_t i = 0; i < left.size(); ++i) {
        const auto lower = [](char c) { return static_cast<char>(c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c); };
        if (lower(left[i]) != lower(right[i])) return false;
    }
    return true;
}

bool probe_condition(const CheckpointMetadata& metadata, const ProbeCondition& condition) {
    if (condition.key == "architecture_type") {
        return condition.equals.empty() || equal_text(metadata.architecture_type(), condition.equals,
                                                      condition.case_insensitive);
    }
    if (condition.key == "repository_hint") {
        return condition.contains.empty() || metadata.repository_hint.find(condition.contains) != std::string::npos;
    }
    if (condition.key == "source_format") {
        const std::string format = metadata.is_gguf() ? "gguf" : "safetensors";
        return condition.equals.empty() || equal_text(format, condition.equals,
                                                       condition.case_insensitive);
    }
    const std::string key = metadata.is_gguf()
        ? (condition.gguf.empty() ? (condition.key == "integer" ? "" : condition.key)
                                  : condition.gguf)
        : (condition.json.empty() ? (condition.key == "integer" ? "" : condition.key)
                                  : condition.json);
    if (condition.has_integer_equals) {
        return metadata.contains(key) && metadata.integer(key) == condition.integer_equals;
    }
    if (!metadata.contains(key)) return false;
    if (!condition.equals.empty()) return equal_text(metadata.string(key), condition.equals,
                                                     condition.case_insensitive);
    if (!condition.contains.empty()) return metadata.string(key).find(condition.contains) != std::string::npos;
    return true;
}

class DescriptorNamingPolicy final : public ITensorNamingPolicy {
public:
    explicit DescriptorNamingPolicy(const Descriptor& descriptor) {
        for (const BindingPattern& pattern : descriptor.bindings) bindings_[pattern.role] = pattern.candidates;
    }

    std::vector<std::string> candidates(const TensorRequest& request) const override {
        const auto it = bindings_.find(request.role);
        if (it == bindings_.end()) return {};
        std::vector<std::string> result;
        result.reserve(it->second.size());
        for (const std::string& pattern : it->second) {
            std::string value = pattern;
            replace(value, "{layer}", request.layer >= 0 ? std::to_string(request.layer) : "");
            replace(value, "{physical_layer}", request.physical_layer >= 0
                ? std::to_string(request.physical_layer) : "");
            replace(value, "{expert}", request.expert >= 0 ? std::to_string(request.expert) : "");
            result.push_back(std::move(value));
        }
        return result;
    }

private:
    static void replace(std::string& value, std::string_view needle, std::string_view replacement) {
        size_t position = 0;
        while ((position = value.find(needle, position)) != std::string::npos) {
            value.replace(position, needle.size(), replacement);
            position += replacement.size();
        }
    }
    std::unordered_map<TensorRole, std::vector<std::string>> bindings_;
};

void build_descriptor_graph(ResolvedModel& model, const Descriptor& descriptor,
                            const CheckpointMetadata& metadata) {
    const RuntimeTopology& topology = model.topology;
    model.graph.embedding_multiplier = topology.numerical_policy.embedding_multiplier;
    model.graph.logits_divisor = topology.numerical_policy.logits_divisor;
    model.graph.final_norm.epsilon = topology.numerical_policy.norm_eps;
    model.graph.final_logit_softcap = descriptor.final_logit_softcap.has_value()
        ? static_cast<float>(number_value(metadata, *descriptor.final_logit_softcap)) : 0.0f;
    model.graph.norm_after_layers = topology.norm_after_layers;
    model.graph.layers.clear();
    model.graph.layers.reserve(static_cast<size_t>(topology.num_hidden_layers));
    for (int layer_index = 0; layer_index < topology.num_hidden_layers; ++layer_index) {
        LayerSpec layer;
        const float epsilon = topology.numerical_policy.norm_eps;
        layer.operator_norm.epsilon = epsilon;
        layer.feed_forward_norm.epsilon = epsilon;
        if (descriptor.split_attention_norms) {
            layer.post_attention_norm.epsilon = epsilon;
            layer.pre_feed_forward_norm.epsilon = epsilon;
            layer.post_feed_forward_norm.epsilon = epsilon;
        }
        if (topology.mixer_kinds[static_cast<size_t>(layer_index)] == MixerKind::Attention) {
            layer.mixer = topology.attention_layout(layer_index);
        } else if (topology.mixer_kinds[static_cast<size_t>(layer_index)] ==
                   MixerKind::ShortConvolution) {
            layer.mixer = ShortConvolutionSpec{topology.conv_cache, topology.conv_dim, false};
        } else if (topology.mixer_kinds[static_cast<size_t>(layer_index)] ==
                   MixerKind::GatedDeltaNet) {
            layer.mixer = topology.gated_delta_net_layouts.at(static_cast<size_t>(layer_index));
        } else if (topology.mixer_kinds[static_cast<size_t>(layer_index)] == MixerKind::Mamba2) {
            layer.mixer = topology.mamba2_layouts.at(static_cast<size_t>(layer_index));
        } else if (topology.mixer_kinds[static_cast<size_t>(layer_index)] == MixerKind::MlpOnly) {
            layer.mixer = topology.mlp_only_layouts.at(static_cast<size_t>(layer_index));
        } else {
            throw std::invalid_argument("descriptor has unsupported hybrid mixer kind");
        }
        const int intermediate = topology.feed_forward_intermediates.empty()
            ? topology.intermediate
            : topology.feed_forward_intermediates.at(static_cast<size_t>(layer_index));
        if (topology.feed_forward_kinds[static_cast<size_t>(layer_index)] ==
            FeedForwardKind::MixtureOfExperts) {
            layer.feed_forward = MixtureOfExpertsSpec{
                topology.moe_intermediate, topology.num_experts, topology.experts_per_token,
                topology.normalize_topk, topology.use_expert_bias, topology.routed_scaling_factor,
                0, 0, topology.shared_expert_intermediate > 0,
                topology.shared_expert_intermediate, false, topology.moe_router_softmax};
        } else {
            layer.feed_forward = DenseFeedForwardSpec{
                intermediate, topology.feed_forward_activations.empty()
                    ? descriptor.feed_forward_activation
                    : topology.feed_forward_activations.at(static_cast<size_t>(layer_index))};
        }
        layer.residual.multiplier = topology.numerical_policy.residual_multiplier;
        layer.execute_feed_forward = topology.mixer_kinds[static_cast<size_t>(layer_index)] !=
            MixerKind::MlpOnly;
        layer.per_layer_input = {topology.per_layer_input_size, ActivationKind::GeluTanh,
                                 topology.has_per_layer_input};
        model.graph.layers.push_back(std::move(layer));
    }
}

void add_descriptor_request(ResolvedModel& model, TensorRole role, int layer, int expert,
                            std::vector<int64_t> shape, int physical_layer = -1) {
    model.weight_plan.requests.push_back({role, layer, expert, std::move(shape),
                                          std::nullopt, physical_layer});
}

void build_descriptor_weight_plan(ResolvedModel& model,
                                  const ITensorNamingPolicy& naming_policy) {
    const RuntimeTopology& topology = model.topology;
    model.weight_plan.requests.clear();
    add_descriptor_request(model, TensorRole::TokenEmbedding, -1, -1,
                           {topology.vocab_size, topology.hidden});
    add_descriptor_request(model, TensorRole::LanguageModelHead, -1, -1,
                           {topology.vocab_size, topology.hidden});
    add_descriptor_request(model, TensorRole::FinalNorm, -1, -1, {topology.hidden});
    if (topology.has_per_layer_input) {
        const int width = topology.num_hidden_layers * topology.per_layer_input_size;
        add_descriptor_request(model, TensorRole::PerLayerEmbedding, -1, -1,
                               {topology.vocab_size, width});
        add_descriptor_request(model, TensorRole::PerLayerContextProjection, -1, -1,
                               {width, topology.hidden});
        add_descriptor_request(model, TensorRole::PerLayerProjectionNorm, -1, -1,
                               {topology.per_layer_input_size});
    }
    for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
        const int physical_layer = topology.checkpoint_layer_for_layer.empty()
            ? layer : topology.checkpoint_layer_for_layer.at(static_cast<size_t>(layer));
        const AttentionSpec& attention = topology.attention_layout(layer);
        const int query_width = attention.query_width();
        const int key_value_width = attention.key_value_width();
        const int intermediate = topology.feed_forward_intermediates.empty()
            ? topology.intermediate
            : topology.feed_forward_intermediates.at(static_cast<size_t>(layer));
        add_descriptor_request(model, TensorRole::AttentionInputNorm, layer, -1,
                               {topology.hidden}, physical_layer);
        if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::Attention) {
            if (attention.uses_external_memory()) {
                add_descriptor_request(model, TensorRole::AttentionQuery, layer, -1,
                                       {attention.query_projection_width(), topology.hidden},
                                       physical_layer);
                add_descriptor_request(model, TensorRole::AttentionOutput, layer, -1,
                                       {topology.hidden, attention.query_width()},
                                       physical_layer);
            } else if (attention.uses_latent_state()) {
                add_descriptor_request(model, TensorRole::AttentionLatentQuery, layer, -1,
                                       {attention.latent_query_content_width(), topology.hidden},
                                       physical_layer);
                if (attention.latent_query_rope_width() != 0) {
                    add_descriptor_request(model, TensorRole::AttentionLatentQueryRope,
                                           layer, -1,
                                           {attention.latent_query_rope_width(), topology.hidden},
                                           physical_layer);
                }
                const auto& latent = *attention.latent_state();
                add_descriptor_request(model, TensorRole::AttentionLatentKey, layer, -1,
                                       {latent.latent_rank, topology.hidden}, physical_layer);
                add_descriptor_request(model, TensorRole::AttentionLatentValue, layer, -1,
                                       {latent.latent_rank, topology.hidden}, physical_layer);
                if (latent.decoupled_rope && latent.rope_head_dim != 0) {
                    add_descriptor_request(model, TensorRole::AttentionLatentKeyRope, layer, -1,
                                           {latent.rope_head_dim, topology.hidden},
                                           physical_layer);
                }
                add_descriptor_request(model, TensorRole::AttentionLatentOutput, layer, -1,
                                       {topology.hidden, attention.latent_query_content_width()},
                                       physical_layer);
            } else {
            add_descriptor_request(model, TensorRole::AttentionQuery, layer, -1,
                                   {query_width, topology.hidden}, physical_layer);
            if (attention.query_key_norm) {
                add_descriptor_request(model, TensorRole::AttentionQueryNorm, layer, -1,
                                       {attention.head_dim}, physical_layer);
            }
            if (!attention.kv_sharing.shared() || attention.kv_sharing.publishes) {
                add_descriptor_request(model, TensorRole::AttentionKey, layer, -1,
                                       {key_value_width, topology.hidden}, physical_layer);
                add_descriptor_request(model, TensorRole::AttentionValue, layer, -1,
                                       {key_value_width, topology.hidden}, physical_layer);
                if (attention.query_key_norm) {
                    add_descriptor_request(model, TensorRole::AttentionKeyNorm, layer, -1,
                                           {attention.head_dim}, physical_layer);
                }
            }
            add_descriptor_request(model, TensorRole::AttentionOutput, layer, -1,
                                   {topology.hidden, query_width}, physical_layer);
            }
            if (const auto* relative =
                    std::get_if<RelativePositionBiasSpec>(&attention.bias)) {
                add_descriptor_request(
                    model, TensorRole::AttentionRelativePositionBias, layer, -1,
                    {attention.query_heads * relative->bucket_count}, physical_layer);
            }
            if (topology.has_split_attention_norms) {
                add_descriptor_request(model, TensorRole::AttentionPostNorm, layer, -1,
                                       {topology.hidden}, physical_layer);
            }
        } else if (topology.mixer_kinds[static_cast<size_t>(layer)] ==
                   MixerKind::ShortConvolution) {
            add_descriptor_request(model, TensorRole::ShortConvInput, layer, -1,
                                   {3 * topology.hidden, topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::ShortConvKernel, layer, -1,
                                   {topology.hidden, 1, topology.conv_cache}, physical_layer);
            add_descriptor_request(model, TensorRole::ShortConvOutput, layer, -1,
                                   {topology.hidden, topology.hidden}, physical_layer);
        } else if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::GatedDeltaNet) {
            const auto& recurrent = topology.gated_delta_net_layouts.at(static_cast<size_t>(layer));
            const int qkv = 2 * recurrent.key_heads * recurrent.key_head_dim +
                recurrent.value_heads * recurrent.value_head_dim;
            add_descriptor_request(model, TensorRole::GatedDeltaNetQkv, layer, -1,
                                   {qkv, topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::GatedDeltaNetZ, layer, -1,
                                   {recurrent.value_heads * recurrent.value_head_dim, topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::GatedDeltaNetAlpha, layer, -1,
                                   {recurrent.value_heads, topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::GatedDeltaNetBeta, layer, -1,
                                   {recurrent.value_heads, topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::GatedDeltaNetDtBias, layer, -1,
                                   {recurrent.value_heads}, physical_layer);
            add_descriptor_request(model, TensorRole::GatedDeltaNetALog, layer, -1,
                                   {recurrent.value_heads}, physical_layer);
            add_descriptor_request(model, TensorRole::GatedDeltaNetConv, layer, -1,
                                   {qkv, 1, recurrent.conv_kernel}, physical_layer);
            add_descriptor_request(model, TensorRole::GatedDeltaNetNorm, layer, -1,
                                   {recurrent.value_head_dim}, physical_layer);
            add_descriptor_request(model, TensorRole::GatedDeltaNetOutput, layer, -1,
                                   {topology.hidden, recurrent.value_heads * recurrent.value_head_dim}, physical_layer);
        } else if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::Mamba2) {
            const auto& recurrent = topology.mamba2_layouts.at(static_cast<size_t>(layer));
            const int conv_dim = recurrent.intermediate_size +
                2 * recurrent.group_count * recurrent.state_size;
            add_descriptor_request(model, TensorRole::Mamba2Input, layer, -1,
                                   {2 * recurrent.intermediate_size +
                                    2 * recurrent.group_count * recurrent.state_size +
                                    recurrent.num_heads, topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::Mamba2Conv, layer, -1,
                                   {conv_dim, 1, recurrent.conv_kernel}, physical_layer);
            add_descriptor_request(model, TensorRole::Mamba2ConvBias, layer, -1,
                                   {conv_dim}, physical_layer);
            add_descriptor_request(model, TensorRole::Mamba2DtBias, layer, -1,
                                   {recurrent.num_heads}, physical_layer);
            add_descriptor_request(model, TensorRole::Mamba2ALog, layer, -1,
                                   {recurrent.num_heads}, physical_layer);
            add_descriptor_request(model, TensorRole::Mamba2D, layer, -1,
                                   {recurrent.num_heads}, physical_layer);
            add_descriptor_request(model, TensorRole::Mamba2Norm, layer, -1,
                                   {recurrent.intermediate_size}, physical_layer);
            add_descriptor_request(model, TensorRole::Mamba2Output, layer, -1,
                                   {topology.hidden, recurrent.intermediate_size}, physical_layer);
        }
        const bool mlp_only = topology.mixer_kinds[static_cast<size_t>(layer)] ==
            MixerKind::MlpOnly;
        if (mlp_only) {
            const auto& mlp = topology.mlp_only_layouts.at(static_cast<size_t>(layer));
            add_descriptor_request(model, TensorRole::FfnUp, layer, -1,
                                   {mlp.intermediate_size, topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::FfnDown, layer, -1,
                                   {topology.hidden, mlp.intermediate_size}, physical_layer);
        } else {
            add_descriptor_request(model, TensorRole::FfnInputNorm, layer, -1,
                                   {topology.hidden}, physical_layer);
        }
        if (!mlp_only && topology.layer_uses_moe(layer)) {
            add_descriptor_request(model, TensorRole::MoeRouter, layer, -1,
                                   {topology.num_experts, topology.hidden}, physical_layer);
            if (topology.shared_expert_intermediate > 0) {
                add_descriptor_request(model, TensorRole::MoePackedGateUp, layer, -1,
                                       {topology.num_experts, 2 * topology.moe_intermediate,
                                        topology.hidden}, physical_layer);
                add_descriptor_request(model, TensorRole::MoePackedDown, layer, -1,
                                       {topology.num_experts, topology.hidden,
                                        topology.moe_intermediate}, physical_layer);
                add_descriptor_request(model, TensorRole::MoeSharedGate, layer, -1,
                                       {topology.shared_expert_intermediate, topology.hidden}, physical_layer);
                add_descriptor_request(model, TensorRole::MoeSharedUp, layer, -1,
                                       {topology.shared_expert_intermediate, topology.hidden}, physical_layer);
                add_descriptor_request(model, TensorRole::MoeSharedDown, layer, -1,
                                       {topology.hidden, topology.shared_expert_intermediate}, physical_layer);
                add_descriptor_request(model, TensorRole::MoeSharedGateWeight, layer, -1,
                                       {1, topology.hidden}, physical_layer);
            } else {
                for (int expert = 0; expert < topology.num_experts; ++expert) {
                    add_descriptor_request(model, TensorRole::MoeExpertGate, layer, expert,
                                           {topology.moe_intermediate, topology.hidden}, physical_layer);
                    add_descriptor_request(model, TensorRole::MoeExpertUp, layer, expert,
                                           {topology.moe_intermediate, topology.hidden}, physical_layer);
                    add_descriptor_request(model, TensorRole::MoeExpertDown, layer, expert,
                                           {topology.hidden, topology.moe_intermediate}, physical_layer);
                }
            }
        } else if (!mlp_only) {
            add_descriptor_request(model, TensorRole::FfnGate, layer, -1,
                                   {intermediate, topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::FfnUp, layer, -1,
                                   {intermediate, topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::FfnDown, layer, -1,
                                   {topology.hidden, intermediate}, physical_layer);
            if (topology.has_split_attention_norms) {
                add_descriptor_request(model, TensorRole::FfnOutputNorm, layer, -1,
                                       {topology.hidden}, physical_layer);
            }
        }
        if (topology.has_per_layer_input) {
            add_descriptor_request(model, TensorRole::PerLayerInputGate, layer, -1,
                                   {topology.per_layer_input_size, topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::PerLayerProjection, layer, -1,
                                   {topology.hidden, topology.per_layer_input_size}, physical_layer);
            add_descriptor_request(model, TensorRole::PerLayerInputNorm, layer, -1,
                                   {topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::LayerScalar, layer, -1, {1}, physical_layer);
        }
    }
    resolve_weight_plan(model, naming_policy);
}

class DescriptorArchitecture final : public IArchitecture {
public:
    explicit DescriptorArchitecture(Descriptor descriptor)
        : descriptor_(std::move(descriptor)) {}

    std::string_view id() const override { return descriptor_.id; }

    ProbeResult probe(const CheckpointMetadata& metadata) const override {
        for (const auto& alternative : descriptor_.probe_alternatives) {
            if (std::all_of(alternative.begin(), alternative.end(), [&](const ProbeCondition& condition) {
                    return probe_condition(metadata, condition);
                })) {
                return {true, descriptor_.specificity, "declarative model descriptor"};
            }
        }
        return {false, 0, "descriptor probe did not match"};
    }

    ResolvedModel resolve(const CheckpointView& checkpoint) const override {
        if (!probe(checkpoint.metadata).supported) {
            throw std::runtime_error("descriptor cannot resolve checkpoint: " + descriptor_.id);
        }
        const CheckpointMetadata& metadata = checkpoint.metadata;
        const int hidden = integer_value(metadata, descriptor_.dimensions.at("hidden"));
        const int query_heads = integer_value(metadata, descriptor_.dimensions.at("query_heads"));
        RuntimeTopology topology;
        topology.hidden = hidden;
        topology.intermediate = integer_value(metadata, descriptor_.dimensions.at("intermediate"));
        topology.dense_intermediate = topology.intermediate;
        topology.max_feed_forward_intermediate = topology.intermediate;
        const int physical_layer_count = integer_value(metadata, descriptor_.dimensions.at("layers"));
        const int repeat_count = descriptor_.repeated_layers
            ? integer_value(metadata, descriptor_.repeat_count) : 1;
        if (physical_layer_count <= 0 || repeat_count <= 0) {
            throw std::invalid_argument("descriptor has invalid layer schedule");
        }
        topology.num_hidden_layers = physical_layer_count * repeat_count;
        if (const auto mtp = descriptor_.dimensions.find("mtp_layers");
            mtp != descriptor_.dimensions.end()) {
            topology.mtp_num_hidden_layers = integer_value(metadata, mtp->second);
        }
        topology.vocab_size = integer_value(metadata, descriptor_.dimensions.at("vocab"));
        if (topology.vocab_size == 0 && metadata.is_gguf() && metadata.contains("tokenizer.ggml.tokens")) {
            topology.vocab_size = static_cast<int>(metadata.strings("tokenizer.ggml.tokens").size());
        }
        topology.max_position_embeddings = integer_value(metadata, descriptor_.dimensions.at("context"));
        topology.conv_cache = descriptor_.convolution_cache.has_value()
            ? integer_value(metadata, *descriptor_.convolution_cache) : 0;
        topology.conv_dim = descriptor_.convolution_channels.has_value()
            ? integer_value(metadata, *descriptor_.convolution_channels, hidden) : 0;
        const int kv_heads = integer_value(metadata, descriptor_.dimensions.at("kv_heads"));
        const int head_dim = integer_value(metadata, descriptor_.dimensions.at("head_dim"), hidden, query_heads);
        topology.token_policy.bos_token_id = token_value(metadata, descriptor_.bos, descriptor_.gguf_bos);
        topology.token_policy.eos_token_ids = eos_values(metadata, descriptor_);
        topology.token_policy.pad_token_id = token_value(metadata, descriptor_.pad, descriptor_.gguf_pad);
        const auto& numbers = descriptor_.numbers;
        topology.numerical_policy.norm_eps = static_cast<float>(number_value(metadata, numbers.at("norm_eps")));
        const double rope_theta = number_value(metadata, numbers.at("rope_theta"));
        RopeScalingSpec scaling;
        scaling.kind = parse_scaling_kind(scaling_kind_value(metadata, descriptor_));
        scaling.factor = scaling_number_value(metadata, descriptor_.rope_scaling_factor, 1.0);
        scaling.original_context = scaling_integer_value(
            metadata, descriptor_.rope_scaling_original_context);
        scaling.attention_factor = scaling_number_value(
            metadata, descriptor_.rope_scaling_attention_factor, 1.0);
        scaling.beta_fast = scaling_number_value(metadata, descriptor_.rope_scaling_beta_fast, 0.0);
        scaling.beta_slow = scaling_number_value(metadata, descriptor_.rope_scaling_beta_slow, 0.0);
        scaling.low_frequency_factor = scaling_number_value(
            metadata, descriptor_.rope_scaling_low_frequency_factor, 1.0);
        scaling.high_frequency_factor = scaling_number_value(
            metadata, descriptor_.rope_scaling_high_frequency_factor, 1.0);
        scaling.short_factors = scaling_factor_values(
            metadata, descriptor_.rope_scaling_short_factors);
        scaling.long_factors = scaling_factor_values(
            metadata, descriptor_.rope_scaling_long_factors);
        const double rotary_fraction = descriptor_.rotary_fraction.has_value()
            ? number_value(metadata, *descriptor_.rotary_fraction) : 1.0;
        topology.numerical_policy.embedding_multiplier = static_cast<float>(
            number_value(metadata, numbers.at("embedding_multiplier"), hidden));
        topology.numerical_policy.attention_multiplier = static_cast<float>(number_value(metadata, numbers.at("attention_multiplier")));
        topology.numerical_policy.residual_multiplier = static_cast<float>(number_value(metadata, numbers.at("residual_multiplier")));
        topology.numerical_policy.logits_divisor = static_cast<float>(number_value(metadata, numbers.at("logits_divisor")));
        const std::vector<std::string> scheduled_mixer = mixer_schedule_values(metadata, descriptor_);
        if (!scheduled_mixer.empty() && scheduled_mixer.size() !=
            static_cast<size_t>(topology.num_hidden_layers)) {
            throw std::invalid_argument("descriptor mixer schedule length does not match layer schedule");
        }
        const std::vector<int> scheduled_kv_heads = field_integer_values(
            metadata, descriptor_.kv_heads_schedule);
        if (!scheduled_kv_heads.empty() && scheduled_kv_heads.size() !=
            static_cast<size_t>(topology.num_hidden_layers)) {
            throw std::invalid_argument("descriptor KV-head schedule length does not match layer schedule");
        }
        topology.mixer_kinds.assign(static_cast<size_t>(topology.num_hidden_layers),
                                    MixerKind::Attention);
        for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
            if (!scheduled_mixer.empty()) {
                const std::string& kind = scheduled_mixer[static_cast<size_t>(layer)];
                if (kind == descriptor_.convolution_value || kind == "conv" ||
                    kind == "short_convolution") {
                    topology.mixer_kinds[static_cast<size_t>(layer)] = MixerKind::ShortConvolution;
                } else if (kind == "linear_attention" || kind == "gated_delta_net") {
                    topology.mixer_kinds[static_cast<size_t>(layer)] = MixerKind::GatedDeltaNet;
                } else if (kind == "mamba" || kind == "mamba2" || kind == "M") {
                    topology.mixer_kinds[static_cast<size_t>(layer)] = MixerKind::Mamba2;
                } else if (kind == "mlp_only" || kind == "MlpOnly" || kind == "-") {
                    topology.mixer_kinds[static_cast<size_t>(layer)] = MixerKind::MlpOnly;
                } else if (kind != "full_attention" && kind != "attention" && kind != "*") {
                    throw std::invalid_argument("descriptor has unsupported mixer kind: " + kind);
                }
            }
        }
        topology.feed_forward_kinds.assign(static_cast<size_t>(topology.num_hidden_layers), FeedForwardKind::Dense);
        topology.feed_forward_intermediates.assign(static_cast<size_t>(topology.num_hidden_layers), topology.intermediate);
        topology.feed_forward_activations.assign(static_cast<size_t>(topology.num_hidden_layers),
                                                 descriptor_.feed_forward_activation);
        const int dense_layers = descriptor_.moe_dense_layers.has_value()
            ? integer_value(metadata, *descriptor_.moe_dense_layers) : topology.num_hidden_layers;
        const int moe_experts = descriptor_.moe_experts.has_value()
            ? integer_value(metadata, *descriptor_.moe_experts) : 0;
        const bool has_moe = moe_experts > 0 && dense_layers < topology.num_hidden_layers;
        if (has_moe) {
            topology.num_dense_layers = dense_layers;
            topology.num_experts = moe_experts;
            topology.moe_intermediate = integer_value(metadata, *descriptor_.moe_intermediate);
            topology.experts_per_token = integer_value(metadata, *descriptor_.moe_experts_per_token);
            topology.shared_expert_intermediate = descriptor_.moe_shared_intermediate.has_value()
                ? integer_value(metadata, *descriptor_.moe_shared_intermediate) : 0;
            topology.normalize_topk = boolean_value(metadata, descriptor_.moe_normalize_topk, true);
            topology.moe_router_softmax = topology.normalize_topk;
            topology.use_expert_bias = boolean_value(metadata, descriptor_.moe_expert_bias, false);
            topology.routed_scaling_factor = static_cast<float>(
                descriptor_.moe_routed_scaling.has_value()
                    ? number_value(metadata, *descriptor_.moe_routed_scaling) : 1.0);
            for (int layer = dense_layers; layer < topology.num_hidden_layers; ++layer) {
                topology.feed_forward_kinds[static_cast<size_t>(layer)] =
                    FeedForwardKind::MixtureOfExperts;
            }
        }
        topology.gated_delta_net_layouts.resize(static_cast<size_t>(topology.num_hidden_layers));
        topology.mamba2_layouts.resize(static_cast<size_t>(topology.num_hidden_layers));
        topology.mlp_only_layouts.resize(static_cast<size_t>(topology.num_hidden_layers));
        const int gated_key_heads = descriptor_.recurrent_key_heads.has_value()
            ? integer_value(metadata, *descriptor_.recurrent_key_heads) : 0;
        const int gated_value_heads = descriptor_.recurrent_value_heads.has_value()
            ? integer_value(metadata, *descriptor_.recurrent_value_heads) : 0;
        const int gated_key_dim = descriptor_.recurrent_key_dim.has_value()
            ? integer_value(metadata, *descriptor_.recurrent_key_dim) : 0;
        const int gated_value_dim = descriptor_.recurrent_value_dim.has_value()
            ? integer_value(metadata, *descriptor_.recurrent_value_dim) : 0;
        const int gated_conv_kernel = descriptor_.recurrent_conv_kernel.has_value()
            ? integer_value(metadata, *descriptor_.recurrent_conv_kernel) : 0;
        int mamba_intermediate = descriptor_.mamba_intermediate.has_value()
            ? integer_value(metadata, *descriptor_.mamba_intermediate) : 0;
        const int mamba_state_size = descriptor_.mamba_state_size.has_value()
            ? integer_value(metadata, *descriptor_.mamba_state_size) : 0;
        const int mamba_time_step_rank = descriptor_.mamba_time_step_rank.has_value()
            ? integer_value(metadata, *descriptor_.mamba_time_step_rank) : 0;
        const int mamba_heads = descriptor_.mamba_heads.has_value()
            ? integer_value(metadata, *descriptor_.mamba_heads) : 0;
        const int mamba_head_dim = descriptor_.mamba_head_dim.has_value()
            ? integer_value(metadata, *descriptor_.mamba_head_dim) : 0;
        const int mamba_groups = descriptor_.mamba_groups.has_value()
            ? integer_value(metadata, *descriptor_.mamba_groups) : 0;
        const int mamba_chunk_size = descriptor_.mamba_chunk_size.has_value()
            ? integer_value(metadata, *descriptor_.mamba_chunk_size) : 0;
        if (mamba_intermediate == 0 && mamba_heads > 0 && mamba_head_dim > 0) {
            mamba_intermediate = mamba_heads * mamba_head_dim;
        }
        topology.mamba2_intermediate = mamba_intermediate;
        for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
            const MixerKind kind = topology.mixer_kinds[static_cast<size_t>(layer)];
            if (kind == MixerKind::GatedDeltaNet) {
                topology.gated_delta_net_layouts[static_cast<size_t>(layer)] =
                    GatedDeltaNetSpec{gated_conv_kernel, gated_key_dim, gated_value_dim,
                                      gated_key_heads, gated_value_heads};
            } else if (kind == MixerKind::Mamba2) {
                topology.mamba2_layouts[static_cast<size_t>(layer)] =
                    Mamba2Spec{gated_conv_kernel, mamba_intermediate, mamba_state_size,
                               mamba_time_step_rank, mamba_heads, mamba_head_dim,
                               mamba_groups, mamba_chunk_size, false, false};
            } else if (kind == MixerKind::MlpOnly) {
                topology.mlp_only_layouts[static_cast<size_t>(layer)] =
                    MlpBlockSpec{topology.intermediate, descriptor_.feed_forward_activation};
            }
        }
        topology.attention_layer_count = 0;
        topology.conv_layer_count = 0;
        topology.attention_slot_for_layer.assign(static_cast<size_t>(topology.num_hidden_layers), -1);
        topology.layer_for_attention_slot.clear();
        for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
            if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::Attention) {
                topology.attention_slot_for_layer[static_cast<size_t>(layer)] =
                    topology.attention_layer_count++;
                topology.layer_for_attention_slot.push_back(layer);
            } else if (topology.mixer_kinds[static_cast<size_t>(layer)] ==
                       MixerKind::ShortConvolution) {
                ++topology.conv_layer_count;
            } else if (topology.mixer_kinds[static_cast<size_t>(layer)] ==
                       MixerKind::GatedDeltaNet) {
                ++topology.gated_delta_net_layer_count;
            } else if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::Mamba2) {
                ++topology.mamba2_layer_count;
            } else if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::MlpOnly) {
                ++topology.mlp_only_layer_count;
            }
        }
        if (descriptor_.map_physical_layers) {
            topology.checkpoint_layer_for_layer.resize(static_cast<size_t>(topology.num_hidden_layers));
            for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
                topology.checkpoint_layer_for_layer[static_cast<size_t>(layer)] =
                    layer % physical_layer_count;
            }
        }
        if (descriptor_.norm_after_physical_block) {
            topology.norm_after_layers = {physical_layer_count - 1};
        }
        topology.attention_layouts.reserve(static_cast<size_t>(topology.num_hidden_layers));
        const std::string disable_key = metadata.is_gguf() ? descriptor_.disable_rope_gguf
                                                            : descriptor_.disable_rope_json;
        const std::vector<int> disabled = integer_values(metadata, disable_key);
        const std::string position_kind = position_kind_value(metadata, descriptor_);
        const std::vector<float> alibi_slopes = scaling_factor_values(
            metadata, descriptor_.alibi_slopes);
        const std::vector<int> mrope_sections = field_integer_values(
            metadata, descriptor_.mrope_sections);
        if (!mrope_sections.empty() && mrope_sections.size() != 3) {
            throw std::invalid_argument("descriptor M-RoPE requires three sections");
        }
        const bool mrope_interleaved = boolean_value(
            metadata, descriptor_.mrope_interleaved, false);
        const int relative_bucket_count = descriptor_.relative_bucket_count.has_value()
            ? integer_value(metadata, *descriptor_.relative_bucket_count) : 0;
        const int relative_max_distance = descriptor_.relative_max_distance.has_value()
            ? integer_value(metadata, *descriptor_.relative_max_distance) : 0;
        const std::vector<bool> scheduled_sliding = attention_pattern_values(metadata, descriptor_);
        if (descriptor_.attention_pattern.has_value()) {
            const std::string pattern_key = selected_key(metadata, *descriptor_.attention_pattern);
            if (!pattern_key.empty() && metadata.contains(pattern_key) &&
                scheduled_sliding.size() == 1 && topology.num_hidden_layers > 1) {
                const MetadataValue& pattern_value = metadata.value(pattern_key);
                if (std::holds_alternative<std::vector<std::string>>(pattern_value) ||
                    std::holds_alternative<std::vector<int64_t>>(pattern_value)) {
                    throw std::invalid_argument(
                        "descriptor attention pattern length does not match layer schedule");
                }
            }
        }
        if (!scheduled_sliding.empty() && scheduled_sliding.size() != 1 &&
            scheduled_sliding.size() != static_cast<size_t>(physical_layer_count) &&
            scheduled_sliding.size() != static_cast<size_t>(topology.num_hidden_layers)) {
            throw std::invalid_argument("descriptor attention pattern length does not match layer schedule");
        }
        const int sliding_window = descriptor_.sliding_window.has_value()
            ? integer_value(metadata, *descriptor_.sliding_window) : 0;
        const bool has_attention_variants = descriptor_.full_attention_variant.has_value() ||
            descriptor_.sliding_attention_variant.has_value();
        const int shared_layers = descriptor_.shared_kv_suffix_layers.has_value()
            ? integer_value(metadata, *descriptor_.shared_kv_suffix_layers) : 0;
        if (shared_layers < 0 || shared_layers > topology.num_hidden_layers) {
            throw std::invalid_argument("descriptor shared KV suffix is out of range");
        }
        const int shared_start = topology.num_hidden_layers - shared_layers;
        const auto scheduled_is_sliding = [&](int layer) {
            if (scheduled_sliding.empty()) return false;
            const size_t index = scheduled_sliding.size() == 1 ? 0 :
                scheduled_sliding.size() == static_cast<size_t>(physical_layer_count)
                    ? static_cast<size_t>(layer % physical_layer_count)
                    : static_cast<size_t>(layer);
            return scheduled_sliding[index];
        };
        for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
            const bool is_sliding = scheduled_is_sliding(layer);
            const AttentionVariant* variant = is_sliding
                ? (descriptor_.sliding_attention_variant.has_value()
                    ? &*descriptor_.sliding_attention_variant : nullptr)
                : (descriptor_.full_attention_variant.has_value()
                    ? &*descriptor_.full_attention_variant : nullptr);
            const int layer_query_heads = variant && variant->query_heads.has_value()
                ? integer_value(metadata, *variant->query_heads) : query_heads;
            const int layer_kv_heads = variant && variant->key_value_heads.has_value()
                ? integer_value(metadata, *variant->key_value_heads) : kv_heads;
            const int layer_head_dim = variant && variant->head_dim.has_value()
                ? integer_value(metadata, *variant->head_dim, hidden, layer_query_heads)
                : head_dim;
            const double layer_rope_theta = variant && variant->rope_theta.has_value()
                ? number_value(metadata, *variant->rope_theta) : rope_theta;
            const double layer_rotary_fraction = variant && variant->rotary_fraction.has_value()
                ? number_value(metadata, *variant->rotary_fraction) : rotary_fraction;
            const int layer_sliding_window = variant && variant->sliding_window.has_value()
                ? integer_value(metadata, *variant->sliding_window) : sliding_window;
            if (is_sliding && layer_sliding_window <= 0) {
                throw std::invalid_argument("descriptor sliding attention has invalid window");
            }
        const int scheduled_layer_kv_heads = scheduled_kv_heads.empty()
            ? layer_kv_heads : scheduled_kv_heads[static_cast<size_t>(layer)];
            AttentionSpec attention{layer_query_heads, scheduled_layer_kv_heads, layer_head_dim,
                descriptor_.query_key_norm,
                FullCausalPattern{}, {},
                topology.numerical_policy.attention_multiplier, descriptor_.query_gate};
            if (descriptor_.attention_state_kind == "latent") {
                if (!descriptor_.latent_rank.has_value()) {
                    throw std::invalid_argument("latent attention descriptor has no latent rank");
                }
                const int latent_rank = integer_value(metadata, *descriptor_.latent_rank);
                const int rope_head_dim = descriptor_.latent_rope_head_dim.has_value()
                    ? integer_value(metadata, *descriptor_.latent_rope_head_dim) : 0;
                const int nope_head_dim = descriptor_.latent_nope_head_dim.has_value()
                    ? integer_value(metadata, *descriptor_.latent_nope_head_dim) : latent_rank;
                attention.state = LatentAttentionStateSpec{
                    latent_rank, rope_head_dim, nope_head_dim,
                    descriptor_.latent_decoupled_rope};
            } else if (descriptor_.attention_state_kind != "ordinary_kv") {
                throw std::invalid_argument("descriptor has unsupported attention state kind: " +
                                            descriptor_.attention_state_kind);
            }
            attention.state_storage.key = parse_state_scalar(descriptor_.state_key_storage);
            attention.state_storage.value = parse_state_scalar(descriptor_.state_value_storage);
            attention.state_storage.latent = parse_state_scalar(descriptor_.state_latent_storage);
            attention.state_storage.rotary = parse_state_scalar(descriptor_.state_rotary_storage);
            attention.state_storage.recurrent = parse_state_scalar(
                descriptor_.state_recurrent_storage);
            attention.state_storage.granularity = parse_state_granularity(
                descriptor_.state_storage_granularity);
            attention.state_storage.paged = descriptor_.state_paged;
            if (descriptor_.attention_key_value_source == "external_memory") {
                if (!descriptor_.attention_memory_slot.has_value()) {
                    throw std::invalid_argument(
                        "external attention descriptor has no memory slot");
                }
                attention.sources.key_value = AttentionSourceKind::ExternalMemory;
                attention.sources.memory_slot = integer_value(
                    metadata, *descriptor_.attention_memory_slot);
                if (attention.sources.memory_slot < 0) {
                    throw std::invalid_argument(
                        "external attention descriptor has an invalid memory slot");
                }
            } else if (descriptor_.attention_key_value_source != "current_sequence") {
                throw std::invalid_argument(
                    "descriptor has unsupported attention key/value source: " +
                    descriptor_.attention_key_value_source);
            }
            if (is_sliding) {
                attention.pattern = SlidingWindowPattern{layer_sliding_window};
            }
            if (shared_layers > 0) {
                const int group = has_attention_variants ? (is_sliding ? 1 : 0) : 0;
                const bool is_shared = layer >= shared_start;
                bool publishes = false;
                if (!is_shared) {
                    publishes = true;
                    for (int later = layer + 1; later < shared_start; ++later) {
                        if ((has_attention_variants && scheduled_is_sliding(later)) == is_sliding) {
                            publishes = false;
                            break;
                        }
                    }
                }
                attention.kv_sharing = KvSharingSpec{group, publishes};
            }
            if (layer < static_cast<int>(disabled.size()) &&
                disabled[static_cast<size_t>(layer)] == 0) {
                attention.position = NoPositionEncodingSpec{};
            } else if (position_kind == "none") {
                attention.position = NoPositionEncodingSpec{};
            } else if (position_kind == "alibi") {
                attention.position = NoPositionEncodingSpec{};
                attention.bias = AlibiBiasSpec{alibi_slopes};
            } else if (position_kind == "relative_bias") {
                attention.position = NoPositionEncodingSpec{};
                attention.bias = RelativePositionBiasSpec{
                    relative_bucket_count, relative_max_distance, descriptor_.relative_bidirectional};
            } else if (position_kind == "rope") {
                RopePositionSpec rope{layer_rope_theta, layer_rotary_fraction, scaling};
                if (!mrope_sections.empty()) {
                    attention.position = MultiAxisRopeSpec{
                        rope, {mrope_sections[0], mrope_sections[1], mrope_sections[2]},
                        mrope_interleaved, 3};
                } else {
                    attention.position = rope;
                }
            } else {
                throw std::invalid_argument("descriptor has unsupported position kind: " + position_kind);
            }
            topology.attention_layouts.push_back(attention);
        }
        topology.shared_kv_group_count = shared_layers > 0
            ? (has_attention_variants ? 2 : 1) : 0;
        topology.has_split_attention_norms = descriptor_.split_attention_norms;
        topology.has_per_layer_input = descriptor_.per_layer_input_size.has_value();
        topology.per_layer_input_size = topology.has_per_layer_input
            ? integer_value(metadata, *descriptor_.per_layer_input_size) : 0;
        if (topology.has_per_layer_input && topology.per_layer_input_size <= 0) {
            throw std::invalid_argument("descriptor per-layer input size is invalid");
        }
        topology.feed_forward_activations.assign(
            static_cast<size_t>(topology.num_hidden_layers), descriptor_.feed_forward_activation);
        if (descriptor_.double_wide_shared_suffix && shared_layers > 0) {
            for (int layer = shared_start; layer < topology.num_hidden_layers; ++layer) {
                topology.feed_forward_intermediates[static_cast<size_t>(layer)] *= 2;
            }
            topology.max_feed_forward_intermediate = topology.intermediate * 2;
        }
        topology.validate();
        ArchitectureResolutionStages stages;
        stages.topology = [topology](const CheckpointView&) { return topology; };
        stages.graph = [this](ResolvedModel& model, const CheckpointView& view) {
            build_descriptor_graph(model, descriptor_, view.metadata);
        };
        stages.weights = [this](ResolvedModel& model, const CheckpointView&) {
            DescriptorNamingPolicy policy(descriptor_);
            build_descriptor_weight_plan(model, policy);
        };
        const bool tied_embeddings = descriptor_.tied_embeddings_field.has_value()
            ? boolean_value(metadata, descriptor_.tied_embeddings_field, descriptor_.tied_embeddings)
            : descriptor_.tied_embeddings;
        stages.capabilities = {true, true, false, tied_embeddings};
        stages.provenance.architecture_id = descriptor_.id;
        stages.provenance.source_format = metadata.is_gguf() ? "gguf" : "safetensors";
        stages.provenance.checkpoint_profile_id = descriptor_.id;
        stages.provenance.chat_profile_id = descriptor_.chat_profile;
        stages.provenance.profile = {descriptor_.id, "", {}, descriptor_.chat_profile};
        ResolvedModel result = resolve_architecture_stages(checkpoint, std::move(stages));
        result.provenance.identity = descriptor_.id + "-" + result.topology.fingerprint();
        return result;
    }

private:
    Descriptor descriptor_;
};

} // namespace

void register_descriptor_architectures(ArchitectureCatalog& catalog,
                                       const std::filesystem::path& directory) {
    const std::filesystem::path file = directory / "dense_transformers.json";
    if (!std::filesystem::exists(file)) {
        throw std::runtime_error("descriptor store is missing: " + file.string());
    }
    const Json root = Json::parse_file(file.string());
    for (const Json& descriptor : required(root, "architectures").as_array()) {
        const std::string descriptor_id = descriptor.is_object() && descriptor.contains("id")
            ? descriptor.at("id").as_string() : "<unknown>";
        try {
            catalog.add(std::make_unique<DescriptorArchitecture>(parse_descriptor(descriptor)));
        } catch (const std::exception& error) {
            throw std::invalid_argument("invalid model descriptor " + descriptor_id + ": " +
                                        std::string(error.what()));
        }
    }
}

} // namespace celeg
