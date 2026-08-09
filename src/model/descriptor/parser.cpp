#include "detail.hpp"

namespace celeg::descriptor_detail {

Descriptor parse_descriptor(const Json& value) {
    Descriptor result;
    result.id = required(value, "id").as_string();
    result.specificity = static_cast<int>(required(value, "specificity").as_i64());
    for (const Json& alternative : required(value, "probe").as_array()) {
        result.probe_alternatives.push_back(
            parse_probe_conditions(required(alternative, "all")));
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
        result.rope_pairing = optional_string(position, "pairing", result.rope_pairing);
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
            result.feed_forward_activation = parse_activation_kind(
                graph.at("feed_forward_activation").as_string());
        }
        result.final_logit_softcap = optional_field(graph, "final_logit_softcap");
    }
    if (value.contains("attention")) {
        const Json& attention = value.at("attention");
        result.query_key_norm = optional_bool(attention, "query_key_norm");
        result.query_gate = optional_bool(attention, "query_gate");
        result.orthogonalize_current_value = optional_field(
            attention, "orthogonalize_current_value");
        result.orthogonalize_current_value_minimum_norm_squared = optional_field(
            attention, "orthogonalize_current_value_minimum_norm_squared");
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


} // namespace celeg::descriptor_detail
