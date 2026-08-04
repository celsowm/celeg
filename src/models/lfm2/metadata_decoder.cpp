#include "celeg/detail/models/lfm2_metadata_decoder.hpp"

#include <string_view>

namespace celeg::detail {
namespace {

int integer(const CheckpointMetadata& metadata, bool gguf,
            std::string_view json_key, std::string_view gguf_key) {
    return static_cast<int>(metadata.integer(
        gguf ? std::string(gguf_key) : std::string(json_key)));
}

int integer_or(const CheckpointMetadata& metadata, bool gguf,
               std::string_view json_key, std::string_view gguf_key,
               int fallback) {
    const std::string key = gguf ? std::string(gguf_key) : std::string(json_key);
    return metadata.contains(key) ? static_cast<int>(metadata.integer(key)) : fallback;
}

float number_or(const CheckpointMetadata& metadata, bool gguf,
                std::string_view json_key, std::string_view gguf_key,
                float fallback) {
    const std::string key = gguf ? std::string(gguf_key) : std::string(json_key);
    return static_cast<float>(metadata.number_or(key, fallback));
}

} // namespace

Lfm2Metadata decode_lfm2_metadata(const CheckpointMetadata& metadata) {
    Lfm2Metadata result;
    result.gguf = metadata.is_gguf();
    const std::string type = metadata.architecture_type();
    result.moe = type == "lfm2moe" || type == "lfm2_moe";
    result.tensor_prefix = result.gguf ? (result.moe ? "lfm2moe" : "lfm2") : "";

    result.hidden = integer(metadata, result.gguf, "hidden_size", "embedding_length");
    result.intermediate = integer(metadata, result.gguf, "intermediate_size", "feed_forward_length");
    result.num_hidden_layers = integer(metadata, result.gguf, "num_hidden_layers", "block_count");
    result.num_attention_heads = integer(metadata, result.gguf, "num_attention_heads", "attention.head_count");
    result.num_key_value_heads = result.gguf
        ? 0 : integer(metadata, false, "num_key_value_heads", "attention.head_count_kv");
    result.head_dim = integer_or(metadata, result.gguf, "head_dim",
                                 "attention.key_length",
                                 result.hidden / result.num_attention_heads);
    result.vocab_size = integer_or(metadata, result.gguf, "vocab_size", "vocab_size",
                                   static_cast<int>(metadata.integer_or("llama.vocab_size", 0)));
    result.conv_cache = integer(metadata, result.gguf, "conv_L_cache", "shortconv.l_cache");
    result.conv_dim = integer_or(metadata, result.gguf, "conv_dim", "embedding_length", result.hidden);
    result.max_position_embeddings = integer(metadata, result.gguf, "max_position_embeddings", "context_length");
    result.norm_eps = number_or(metadata, result.gguf, "norm_eps", "attention.layer_norm_rms_epsilon", 1.0e-5f);
    result.rope_theta = number_or(metadata, result.gguf, "rope_theta", "rope.freq_base", 1.0e6f);
    // Transformers 5.x emits the RoPE setting under the nested
    // `rope_parameters` object. LFM2.5-2.6B uses this layout and a 10M base.
    if (!result.gguf && metadata.contains("rope_parameters.rope_theta")) {
        result.rope_theta = static_cast<float>(
            metadata.number("rope_parameters.rope_theta"));
    }
    result.bos_token_id = integer_or(metadata, result.gguf, "bos_token_id", "", 1);
    result.eos_token_id = integer_or(metadata, result.gguf, "eos_token_id", "", 7);
    result.pad_token_id = integer_or(metadata, result.gguf, "pad_token_id", "", 0);

    if (result.moe) {
        result.moe_intermediate = integer_or(metadata, result.gguf,
            "moe_intermediate_size", "expert_feed_forward_length", 0);
        result.num_dense_layers = integer_or(metadata, result.gguf,
            "num_dense_layers", "num_dense_layers", 0);
        result.num_experts = integer_or(metadata, result.gguf,
            "num_experts", "expert_count", 0);
        result.experts_per_token = integer_or(metadata, result.gguf,
            "num_experts_per_tok", "expert_used_count", 0);
        result.normalize_topk = result.gguf ? true : metadata.boolean_or("norm_topk_prob", true);
        result.use_expert_bias = result.gguf ? false : metadata.boolean_or("use_expert_bias", false);
        result.routed_scaling_factor = static_cast<float>(
            metadata.number_or("routed_scaling_factor", 1.0));
    }
    return result;
}

} // namespace celeg::detail
