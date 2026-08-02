#include "celeg/detail/models/granite_topology.hpp"

#include <cmath>

namespace celeg::detail {

RuntimeTopology resolve_granite_topology(const CheckpointMetadata& source) {
    const bool gguf = source.is_gguf();
    auto integer = [&](std::string_view json_key, std::string_view gguf_key) {
        return static_cast<int>(source.integer_for(json_key, "granite." + std::string(gguf_key)));
    };
    auto integer_or = [&](std::string_view json_key, std::string_view gguf_key, int fallback) {
        return static_cast<int>(source.integer_for_or(
            json_key, "granite." + std::string(gguf_key), fallback));
    };
    auto number_or = [&](std::string_view json_key, std::string_view gguf_key, double fallback) {
        return source.number_for_or(json_key, "granite." + std::string(gguf_key), fallback);
    };
    RuntimeTopology t;
    t.hidden = integer("hidden_size", "embedding_length");
    t.intermediate = integer("intermediate_size", "feed_forward_length");
    t.dense_intermediate = t.intermediate;
    t.num_hidden_layers = integer("num_hidden_layers", "block_count");
    const int query_heads = integer("num_attention_heads", "attention.head_count");
    const int key_value_heads = integer("num_key_value_heads", "attention.head_count_kv");
    const int head_dim = integer_or("head_dim", "attention.key_length",
                                    t.hidden / query_heads);
    t.vocab_size = integer_or("vocab_size", "vocab_size", 0);
    if (gguf && t.vocab_size == 0 && source.contains("tokenizer.ggml.tokens")) {
        t.vocab_size = static_cast<int>(source.strings("tokenizer.ggml.tokens").size());
    }
    t.conv_cache = 1;
    t.conv_dim = t.hidden;
    t.max_position_embeddings = integer("max_position_embeddings", "context_length");
    t.bos_token_id = integer_or("bos_token_id", "", 1);
    t.eos_token_id = integer_or("eos_token_id", "", 2);
    t.pad_token_id = integer_or("pad_token_id", "", 0);
    if (gguf) {
        t.bos_token_id = static_cast<int>(source.integer_or(
            "tokenizer.ggml.bos_token_id", t.bos_token_id));
        t.eos_token_id = static_cast<int>(source.integer_or(
            "tokenizer.ggml.eos_token_id", t.eos_token_id));
        t.pad_token_id = static_cast<int>(source.integer_or(
            "tokenizer.ggml.padding_token_id", t.pad_token_id));
    }
    t.norm_eps = static_cast<float>(number_or(
        "rms_norm_eps", "attention.layer_norm_rms_epsilon",
        source.number_or("norm_eps", 1.0e-5)));
    const float rope_theta = static_cast<float>(number_or("rope_theta", "rope.freq_base", 1.0e6));
    t.embedding_multiplier = static_cast<float>(number_or(
        "embedding_multiplier", "embedding_multiplier", 1.0));
    t.attention_multiplier = static_cast<float>(number_or(
        "attention_multiplier", "attention_multiplier", 1.0));
    t.residual_multiplier = static_cast<float>(number_or(
        "residual_multiplier", "residual_multiplier", 1.0));
    t.logits_divisor = static_cast<float>(number_or(
        "logits_scaling", "logits_scaling", 1.0));
    t.mixer_kinds.assign(static_cast<size_t>(t.num_hidden_layers), MixerKind::Attention);
    t.feed_forward_kinds.assign(static_cast<size_t>(t.num_hidden_layers),
                                 FeedForwardKind::Dense);
    t.attention_layer_count = t.num_hidden_layers;
    t.layer_for_attention_slot.resize(static_cast<size_t>(t.num_hidden_layers));
    t.attention_slot_for_layer.resize(static_cast<size_t>(t.num_hidden_layers));
    for (int i = 0; i < t.num_hidden_layers; ++i) {
        t.layer_for_attention_slot[static_cast<size_t>(i)] = i;
        t.attention_slot_for_layer[static_cast<size_t>(i)] = i;
    }
    t.max_feed_forward_intermediate = t.intermediate;
    t.feed_forward_intermediates.assign(static_cast<size_t>(t.num_hidden_layers), t.intermediate);
    t.feed_forward_activations.assign(static_cast<size_t>(t.num_hidden_layers),
                                       ActivationKind::SwiGLU);
    t.attention_layouts.assign(static_cast<size_t>(t.num_hidden_layers),
        AttentionSpec{query_heads, key_value_heads, head_dim,
                      false, AttentionMaskKind::Causal, 0, rope_theta, 1.0, {}});
    const float query_scale = t.attention_multiplier /
        (1.0f / std::sqrt(static_cast<float>(head_dim)));
    for (AttentionSpec& layout : t.attention_layouts) {
        layout.query_scale = query_scale;
    }
    t.validate();
    return t;
}

} // namespace celeg::detail
