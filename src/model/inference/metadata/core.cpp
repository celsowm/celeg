#include "aliases.hpp"
#include "detail.hpp"

#include "../support.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace celeg {

/// Core dimensions cluster of `normalize_model_metadata`.
void normalize_core_dims(const CheckpointMetadata& metadata, NormalizedModelMetadata& result) {
    result.core.hidden_size = aliases<int>(metadata, {"hidden_size", "n_embd", "d_model"},
                                           result.evidence, "hidden_size", "embedding_length");
    result.core.intermediate_size = scoped_aliases<int>(
        metadata, {"intermediate_size", "n_inner", "ffn_dim"}, result.evidence,
        "intermediate_size", "feed_forward_length");
    result.core.parallel_intermediate = aliases<int>(
        metadata, {"parallel_ffn_intermediate_size"}, result.evidence,
        "parallel_intermediate", "parallel_ffn_intermediate_size");
    result.core.layer_count = aliases<int>(
        metadata, {"num_hidden_layers", "n_layer", "num_layers"}, result.evidence,
        "layer_count", "block_count");
    validate_scoped_alias(result.core.intermediate_size, result.core.layer_count, "intermediate_size");
    result.core.layer_repeat_count = aliases<int>(
        metadata, {"num_loops"}, result.evidence, "layer_repeat_count", "num_loops");
    result.attention.query_heads = scoped_aliases<int>(
        metadata, {"num_attention_heads", "n_head"}, result.evidence, "query_heads",
        "attention.head_count");
    result.attention.key_value_heads = scoped_aliases<int>(
        metadata, {"num_key_value_heads", "n_kv_heads"}, result.evidence, "key_value_heads",
        "attention.head_count_kv");
    result.attention.head_dim = scoped_aliases<int>(metadata, {"head_dim"}, result.evidence, "head_dim",
                                                    "attention.key_length");
}

/// Mamba2 cluster of `normalize_model_metadata`.
void normalize_mamba2_facts(const CheckpointMetadata& metadata, NormalizedModelMetadata& result) {
    result.mamba2.intermediate = aliases<int>(
        metadata, {"mamba_intermediate", "ssm_inner_size"}, result.evidence,
        "mamba_intermediate", "ssm.inner_size");
    result.mamba2.state_size = aliases<int>(
        metadata, {"mamba_state_size", "ssm_state_size", "state_size"}, result.evidence,
        "mamba_state_size", "ssm.state_size");
    result.mamba2.time_step_rank = aliases<int>(
        metadata, {"mamba_time_step_rank", "ssm_time_step_rank", "time_step_rank"}, result.evidence,
        "mamba_time_step_rank", "ssm.time_step_rank");
    result.mamba2.num_heads = aliases<int>(
        metadata, {"mamba_num_heads", "mamba_heads", "num_heads"}, result.evidence,
        "mamba_num_heads", "ssm.time_step_rank");
    result.mamba2.head_dim = aliases<int>(
        metadata, {"mamba_head_dim"}, result.evidence, "mamba_head_dim");
    result.mamba2.group_count = aliases<int>(
        metadata, {"n_groups", "mamba_groups"}, result.evidence,
        "mamba_group_count", "ssm.group_count");
    result.mamba2.conv_kernel = aliases<int>(
        metadata, {"conv_kernel", "mamba_conv_kernel"}, result.evidence,
        "mamba_conv_kernel", "ssm.conv_kernel");
    result.mamba2.chunk_size = aliases<int>(
        metadata, {"chunk_size", "mamba_chunk_size"}, result.evidence,
        "mamba_chunk_size", "ssm.chunk_size");
    result.mamba2.decay_encoding = metadata.is_gguf()
        ? DecayParameterEncoding::Pretransformed
        : DecayParameterEncoding::LogA;
}

/// Vocabulary/tokenizer policy cluster of `normalize_model_metadata`; groups
/// the three token-policy blocks that were non-contiguous in the original
/// body, keeping their relative order.
void normalize_token_vocab_policy(const CheckpointMetadata& metadata, NormalizedModelMetadata& result) {
    result.core.vocab_size = aliases<int>(metadata, {"vocab_size", "n_vocab"}, result.evidence,
                                          "vocab_size", "vocab_size");
    if (!result.core.vocab_size.has_value()) {
        result.core.vocab_size = tokenizer_vocabulary_size(metadata, result.evidence);
    }
    result.core.context_length = aliases<int>(
        metadata, {"max_position_embeddings", "max_seq_len", "context_length"},
        result.evidence, "context_length", "context_length");
    result.core.norm_epsilon = aliases<float>(
        metadata, {"norm_eps", "rms_norm_eps", "rms_norm_epsilon", "layer_norm_epsilon"},
        result.evidence, "norm_epsilon", "attention.layer_norm_rms_epsilon");
    result.core.bos_token_id = aliases<int>(metadata,
                                            {"bos_token_id", "tokenizer.ggml.bos_token_id"},
                                            result.evidence, "bos_token_id");
    result.core.pad_token_id = aliases<int>(metadata,
                                            {"pad_token_id", "tokenizer.ggml.padding_token_id"},
                                            result.evidence, "pad_token_id");
    const std::vector<int> eos = token_list(metadata, "eos_token_id");
    result.core.eos_token_ids = eos.empty() ? token_list(metadata, "eos_token_ids") : eos;
}

/// `norm_type` gate of `normalize_model_metadata`.
void normalize_norm_type_gate(const CheckpointMetadata& metadata, NormalizedModelMetadata& result) {
    /// `norm_type` states the normalization kind explicitly: every norm celeg
    /// binds is RMS, so `rmsnorm` merely confirms it, while anything else
    /// fails loudly instead of resolving (e.g.) a LayerNorm as RMS.
    for (const std::string_view candidate : {
             std::string_view("norm_type"),
             std::string_view("text_config.norm_type")}) {
        if (!metadata.contains(candidate)) continue;
        inference_detail::record_metadata_consumption(candidate);
        const std::string norm = metadata.string(candidate);
        if (norm != "rmsnorm") {
            inference_detail::fail(ResolutionFailureKind::UnsupportedSemanticFeature,
                                   "unsupported normalization type: " + norm);
        }
        result.evidence.push_back({EvidenceKind::ExplicitMetadata, std::string(candidate),
                                   "norm_type = rmsnorm"});
    }
}

/// Logit multiplier cluster of `normalize_model_metadata`, plus the adjacent
/// KV-sharing, activation and short-conv statements that sat inside the same
/// contiguous span.
void normalize_logit_multipliers(const CheckpointMetadata& metadata, NormalizedModelMetadata& result) {
    result.core.embedding_multiplier = aliases<float>(
        metadata, {"embedding_multiplier"}, result.evidence,
        "embedding_multiplier");
    result.attention.attention_multiplier = aliases<float>(
        metadata, {"attention_multiplier"}, result.evidence,
        "attention_multiplier");
    /// Suffix KV sharing (`num_kv_shared_layers`): the last N layers consume KV
    /// from an earlier publisher of their own pattern type. Absent means private KV.
    result.attention.kv_shared_layers = aliases<int>(
        metadata, {"num_kv_shared_layers", "shared_kv_suffix_layers"}, result.evidence,
        "kv_shared_layers");
    result.core.residual_multiplier = aliases<float>(
        metadata, {"residual_multiplier"}, result.evidence,
        "residual_multiplier");
    result.core.logits_multiplier = aliases<float>(
        metadata, {"logits_multiplier"}, result.evidence,
        "logits_multiplier");
    result.core.logits_divisor = aliases<float>(
        metadata, {"logits_divisor", "logits_scaling"}, result.evidence,
        "logits_divisor");
    /// `ModelGraph::final_logit_softcap` is implemented on every backend but
    /// nothing in the automatic resolver ever assigned it. One key read here
    /// feeds both the numerical policy and the graph below.
    result.core.final_logit_softcap = aliases<float>(
        metadata, {"final_logit_softcapping", "final_logit_softcap", "logit_softcapping"},
        result.evidence, "final_logit_softcap");
    result.core.feed_forward_activation =
        feed_forward_activation(metadata, result.evidence);
    result.short_conv.cache_length = aliases<int>(metadata, {"conv_L_cache"}, result.evidence,
                                                  "shortconv_cache", "shortconv.l_cache");
}

}
