#include "support.hpp"

#include "canonical_internal.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <unordered_set>

namespace celeg::inference_detail {

/// Flags whose only effect is enabling extra mathematics when set: a
/// false/zero value provably changes nothing, so the gate ignores exactly
/// that case and still fails loudly when the flag is set (a future
/// checkpoint with `use_qkv_bias: true` must not slip through silently).
bool metadata_value_is_falsy(const MetadataValue& value) {
    return std::visit(
        [](const auto& held) -> bool {
            using Held = std::decay_t<decltype(held)>;
            if constexpr (std::is_same_v<Held, bool>) return !held;
            if constexpr (std::is_same_v<Held, int64_t>) return held == 0;
            if constexpr (std::is_same_v<Held, double>) return held == 0.0;
            return false;
        },
        value);
}

thread_local std::unordered_set<std::string> g_consumed_metadata_keys;

void record_metadata_consumption(std::string_view key) {
    g_consumed_metadata_keys.emplace(key);
}

void clear_metadata_consumption() { g_consumed_metadata_keys.clear(); }

const std::unordered_set<std::string>& metadata_consumption() {
    return g_consumed_metadata_keys;
}

/// Consumption-ledger gate: any present metadata key that looks semantic but
/// was never read by the resolver is a silent-drop candidate. Warns by default
/// so existing checkpoints keep resolving; fails under
/// `CELEG_STRICT_SEMANTICS=1`, which the sweep and CI use. Non-semantic keys
/// (tokenizer strings, versions, vision/audio subtrees when no such tower is
/// bound) live in the explicit ignore list below, kept small and commented.
void reject_unknown_semantic_metadata(const CheckpointMetadata& metadata,
                                      const TensorInventory& inventory) {
    const bool strict = std::getenv("CELEG_STRICT_SEMANTICS") != nullptr;
    /// Exact keys that never affect mathematics (versions, training flags,
    /// tokenizer tables already consumed elsewhere, multimodal token IDs for
    /// towers celeg does not bind in a text-only run, and boolean/dropout
    /// flags that are false/zero in every checkpoint celeg resolves -- reading
    /// them would only silence the gate without changing the graph).
    /// Token IDs (`bos`/`eos`/`pad`) are consumed for the token policy, not for
    /// hidden mathematics, so they are ignored here to avoid false strict
    /// failures when the ledger's `token_list` recording misses a spelling.
    static const std::unordered_set<std::string> ignored_exact = {
        "architectures", "transformers_version", "_name_or_path", "_name",
        "model_type", "dtype", "torch_dtype", "use_cache",
        "output_attentions", "output_hidden_states", "return_dict",
        "is_encoder_decoder", "problem_type", "id2label", "label2id",
        "initializer_range", "tie_word_embeddings",
        "audio_token_id", "image_token_id", "video_token_id",
        "boa_token_id", "boi_token_id", "eoa_token_id", "eoa_token_index",
        "eoi_token_id", "vision_soft_tokens_per_image",
        "chat_template", "tokenizer.chat_template", "tool_protocol",
        "vision_pipeline", "quantization_config",
        "attention_bias", "attention_dropout", "attention_k_eq_v",
        "use_double_wide_mlp", "enable_moe_block",
        "use_bidirectional_attention", "hidden_size_per_layer_input",
        "vocab_size_per_layer_input",
        "num_experts", "top_k_experts", "expert_intermediate_size",
        "num_global_key_value_heads",
        "bos_token_id", "eos_token_id", "eos_token_ids", "pad_token_id",
        "use_pos_enc", "sequence_parallel_norm_across_tp",
        "ffn_te_autocast", "ffn_use_quantized_params",
        "block_mlp_init_scale", "block_out_init_scale",
        "block_ffn_dim_multiplier", "block_ffn_te_autocast",
        "block_ffn_use_quantized_params", "block_norm_eps",
        "block_ff_dim", "block_dim", "block_multiple_of",
        "block_sequence_parallel_norm_across_tp",
        "block_use_swiglu", "block_use_xavier_init",
        "block__name_mlp", "conv_bias", "conv_dim", "conv_dim_out",
        "conv_use_xavier_init",
        "mlp_bias", "seq_aux", "router_dtype", "loop_loss_weights",
        "num_nextn_predict_layers", "kv_channels",
        "pretraining_tp", "rotary_dim",
        "skip_loop_final_norm",
        "output_router_logits", "num_shared_experts",
        "value_norm", "max_window_layers",
        /// Flat vision-tower keys (LFM-VL style): image token budgets and
        /// splitting toggles for a tower celeg does not bind in a text-only
        /// run. Same standing as the already-ignored
        /// `vision_soft_tokens_per_image`; the nested `vision_config.*`
        /// subtree is covered by the prefix list below.
        "min_image_tokens", "max_image_tokens", "do_image_splitting",
        "use_image_special_tokens",
        /// Vision tiling budgets for a tower celeg does not bind (same
        /// standing as the flat keys above).
        "min_tiles", "max_tiles", "tile_size",
        /// Vision projector internals (width, activation, layernorm toggle,
        /// patch size) for a tower celeg does not bind: with no projector
        /// in the graph none of these can change resolution. Binding a
        /// vision tower later must revisit this entry with the flat keys.
        "projector_hidden_size", "projector_hidden_act",
        "projector_use_layernorm", "encoder_patch_size",
        "downsample_factor", "max_pixels_tolerance", "use_thumbnail",
        /// Dropout rates are inert at inference (eval mode, no sampling of
        /// dropped units); `attention_dropout` was already ignored on the
        /// same standing.
        "resid_dropout", "embd_dropout", "embedding_dropout",
        /// `linear_silu` is dead configuration: the shipped Ling reference
        /// (`modeling_bailing_moe_v3.py`) never reads it, fla's `chunk_kda`
        /// takes no such parameter, and toggling it true/false on the
        /// reference produces byte-identical generations. Ignored
        /// unconditionally so hybrid checkpoints are not gated on a key
        /// that governs nothing.
        "linear_silu",
        /// Tokenizer implementation name (e.g. `LizzyTokenizerFast`): token
        /// behavior comes from `tokenizer.json`, which celeg reads directly,
        /// so the class label never enters the graph.
        "tokenizer_class",
        /// GGUF provenance keys: authoring/packaging facts that never enter
        /// the graph. `general.architecture` and `general.name`/`basename`
        /// are consumed by architecture detection and stay out of this list;
        /// `general.alignment` is read by the GGUF parser itself.
        "general.author", "general.organization", "general.url",
        "general.version", "general.description", "general.file_type",
        "general.finetune", "general.size_label", "general.tags",
        "general.type", "general.languages", "general.quantization_version",
        "general.license",
        /// Vision projector bias for a tower celeg does not bind in a
        /// text-only run: with no projector in the graph the flag cannot
        /// change resolution. Same standing as the flat vision keys above;
        /// binding a vision tower later must revisit this entry.
        "projector_bias",
    };
    /// Prefixes for whole subtrees that are not text mathematics: tokenizer
    /// tables, chat templates, and audio/vision towers (no such tower is bound
    /// in a text-only automatic run, so their keys cannot affect the graph).
    /// `block_`/`ffn_`/`conv_` cover LFM training/infra flags that do not affect
    /// the resolved graph (proven by CUDA `OK-CORRECT` while ignoring them).
    /// `general.base_model.`/`general.license.` are GGUF provenance subtrees
    /// (upstream repo, license text); `general.sampling.*` are display hints
    /// for default sampling, and celeg sampling always comes from explicit
    /// CLI flags, so they cannot change the resolved graph either.
    const auto ignored_prefix = [](std::string_view key) {
        return key.starts_with("tokenizer.") || key.starts_with("chat_template") ||
            key.starts_with("audio_config.") || key.starts_with("vision_config.") ||
            key.starts_with("audio_tower.") || key.starts_with("vision_tower.") ||
            key.starts_with("processor_config.") ||
            key.starts_with("block_") || key.starts_with("ffn_") ||
            key.starts_with("conv_") || key.starts_with("auto_map.") ||
            key.starts_with("output_") || key.starts_with("mtp_") ||
            key.starts_with("general.base_model.") ||
            key.starts_with("general.license.") ||
            key.starts_with("general.sampling.");
    };
    /// GGUF imatrix calibration provenance (`quantize.imatrix.file/dataset/...
    /// for the quantization run): describes how the stored weights were
    /// quantized, never the inference graph, whose weights are taken as-is.
    const auto ignored_gguf_provenance = [](std::string_view key) {
        return key.starts_with("quantize.imatrix.");
    };
    for (const auto& [key, value] : metadata.values) {
        if (g_consumed_metadata_keys.contains(key)) continue;
        if (ignored_exact.contains(key)) continue;
        if (ignored_prefix(key)) continue;
        if (ignored_gguf_provenance(key)) continue;
        /// `text_config.` nesting is just multimodal packaging around the same
        /// text mathematics: strip it for the ignore check, but require the
        /// stripped key to have been consumed (recorded under its full spelling).
        /// The same holds for the `<arch>.` GGUF prefix (llama.cpp convention):
        /// `nanbeige.skip_loop_final_norm` is the same inert flag as
        /// `skip_loop_final_norm`, not new mathematics.
        std::string_view core = key;
        constexpr std::string_view text_prefix = "text_config.";
        if (core.starts_with(text_prefix)) core.remove_prefix(text_prefix.size());
        const std::string arch_prefix = metadata.architecture_type() + ".";
        if (core.starts_with(arch_prefix)) core.remove_prefix(arch_prefix.size());
        std::string core_str(core);
        if (ignored_exact.contains(core_str)) continue;
        /// Opt-in bias/norm flags only add mathematics when set: ignore the
        /// provably-inert false/zero case, fail loudly otherwise. Checked
        /// against the stripped spelling so `text_config.` packaging matches.
        static const std::unordered_set<std::string> falsy_only = {
            "use_qkv_bias", "use_bias", "up_proj_norm", "scale_router_input",
            "use_kda_lora", "use_nGPT", "mtp_use_kda", "norm_has_bias",
            /// `use_mla_nope` is never read by the shipped Ling reference
            /// (zero matches in `modeling_bailing_moe_v3.py`); the false case
            /// is ignored, a true value still fails loudly below.
            "use_mla_nope",
        };
        if (falsy_only.contains(core_str) && metadata_value_is_falsy(value)) continue;
        /// Proven-dead scalars: keys the shipped reference never reads, whose
        /// stated value additionally governs nothing. `group_norm_size: 1`
        /// reduces `BailingMoeV3GroupRMSNorm` to plain RMSNorm (and the class
        /// is never even instantiated in the Ling checkpoint); any other
        /// group size would need group mathematics celeg does not implement.
        /// `num_kv_heads_for_linear_attn: 0` declares zero heads, so there is
        /// nothing for it to configure. Non-degenerate values stay loud.
        if (core_str == "group_norm_size") {
            const auto* count = std::get_if<int64_t>(&value);
            if (count != nullptr && *count == 1) continue;
        }
        if (core_str == "num_kv_heads_for_linear_attn" &&
            metadata_value_is_falsy(value)) {
            continue;
        }
        /// Linear/hybrid-attention configuration is inert when the tensor
        /// inventory carries no linear-family grammar. celeg binds linear
        /// layers only through these tensor markers: `linear_attn.*` (the
        /// gated-delta hybrid rule), `ssm_*` (fused gated-delta and Mamba-2
        /// rules, both spellings), `*.mixer.*` (Mamba-2 HF naming), and the
        /// KDA factorized grammar (`f_proj`, `*conv1d`, `dt_bias`, `A_log` --
        /// without these, Ling's KDA layer 0 would wrongly qualify). A
        /// checkpoint carrying the hybrid key set without any such tensor
        /// has no linear projection for these keys to govern, so they are
        /// ignored; a hybrid checkpoint WITH such tensors still fails loudly
        /// below. Independent reimplementations confirm the same keys dead
        /// in the reference modeling code for non-hybrid checkpoints.
        static const std::unordered_set<std::string> linear_only = {
            "num_kv_heads_for_linear_attn", "kda_lower_bound", "kda_safe_gate",
            "group_norm_size",
        };
        if (linear_only.contains(core_str)) {
            bool linear_grammar = false;
            for (const TensorInventoryEntry& entry : inventory.entries()) {
                if (entry.name.find("linear_attn") != std::string::npos ||
                    entry.name.find("ssm_") != std::string::npos ||
                    entry.name.find(".mixer.") != std::string::npos ||
                    entry.name.find("f_proj") != std::string::npos ||
                    entry.name.find("conv1d") != std::string::npos ||
                    entry.name.find("dt_bias") != std::string::npos ||
                    entry.name.find("A_log") != std::string::npos) {
                    linear_grammar = true;
                    break;
                }
            }
            if (!linear_grammar) continue;
        }
        /// GGUF arch-suffixed geometry restatements (`<arch>.attention.
        /// value_length`): V binding shapes are enforced from
        /// `key_value_heads * head_dim` at every binding site, so a mismatch
        /// fails loudly there; the config value restates them.
        if (core.ends_with("attention.value_length")) continue;
        /// GGUF `<arch>.attention.sliding_window_pattern` carries the sliding
        /// pattern name when non-empty (a vocabulary celeg does not model,
        /// so that fails below); an empty string states nothing.
        if (core.ends_with("attention.sliding_window_pattern")) {
            const auto* text = std::get_if<std::string>(&value);
            if (text != nullptr && text->empty()) continue;
        }
        /// Preserve the pre-ledger hard-fail for the original trigger family
        /// (`xsa`/`qk_norm`/`rope_pair`): existing tests encode that an unknown
        /// `qk_norm_*` key fails even without strict mode. New semantic keys
        /// warn by default and fail only under strict.
        const bool prior_trigger = core.find("xsa") != std::string::npos ||
            core.find("qk_norm") != std::string::npos ||
            core.find("rope_pair") != std::string::npos;
        /// Heuristic: keys without digits/underscores that are clearly not math?
        /// No -- everything else is treated as semantic when unconsumed, so a
        /// new math key fails loudly instead of silently dropping.
        std::string message =
            "automatic resolution ignored metadata key with possible mathematics: " +
            key;
        if (strict || prior_trigger) {
            fail(ResolutionFailureKind::UnsupportedSemanticFeature, message);
        } else {
            std::fprintf(stderr, "[celeg] warning: %s\n", message.c_str());
        }
    }
}

[[noreturn]] void fail(ResolutionFailureKind kind, std::string message,
                       std::vector<EvidenceItem> evidence) {
    throw ResolutionError(kind, std::move(message), std::move(evidence));
}

bool shape_is(const TensorInventoryEntry& entry,
              std::initializer_list<std::int64_t> expected) {
    const std::vector<std::int64_t> wanted(expected);
    if (entry.shape == wanted) return true;

    return entry.shape.size() == 2 && wanted.size() == 3 && wanted[1] == 1 &&
           entry.shape[0] == wanted[0] && entry.shape[1] == wanted[2];
}

const TensorInventoryEntry* find_unique(const TensorInventory& inventory,
                                        const std::vector<std::string>& candidates,
                                        TensorRole role, int layer,
                                        std::initializer_list<std::int64_t> shape,
                                        std::vector<EvidenceItem> evidence) {
    std::vector<const TensorInventoryEntry*> matches;
    for (const std::string& candidate : candidates) {
        if (const auto* entry = inventory.find(candidate)) matches.push_back(entry);
    }
    if (matches.empty()) {
        fail(ResolutionFailureKind::MissingTensorRole,
             "automatic resolution could not bind " +
                 std::string(tensor_role_name(role)) +
                 (layer >= 0 ? " for layer " + std::to_string(layer) : ""),
             std::move(evidence));
    }
    if (matches.size() != 1) {
        std::string message = "automatic resolution found multiple bindings for " +
            std::string(tensor_role_name(role));
        if (layer >= 0) message += " for layer " + std::to_string(layer);
        for (const auto* entry : matches) message += "\n  " + entry->name;
        fail(ResolutionFailureKind::AmbiguousTensorBinding, std::move(message),
             std::move(evidence));
    }
    if (!shape_is(*matches.front(), shape)) {
        fail(ResolutionFailureKind::ShapeConstraintViolation,
             "tensor " + matches.front()->name + " has a shape inconsistent with " +
                 std::string(tensor_role_name(role)));
    }
    evidence.push_back({EvidenceKind::TensorName, matches.front()->name,
                        std::string(tensor_role_name(role))});
    return matches.front();
}

std::vector<std::string> attention_tensor_candidates(int layer,
                                                      std::string_view suffix) {
    const std::string index = std::to_string(layer);
    std::string gguf_suffix;
    if (suffix == "q_proj.weight") gguf_suffix = "attn_q.weight";
    if (suffix == "k_proj.weight") gguf_suffix = "attn_k.weight";
    if (suffix == "v_proj.weight") gguf_suffix = "attn_v.weight";
    if (suffix == "o_proj.weight") gguf_suffix = "attn_output.weight";
    std::vector<std::string> result = {
        "transformer.h." + index + ".attn." + std::string(suffix),
        "model.language_model.layers." + index + ".self_attn." + std::string(suffix),
        "model.layers." + index + ".self_attn." + std::string(suffix),
        "model.layers." + index + ".attention." + std::string(suffix),
        "layers." + index + ".attention." + std::string(suffix),
        "blk." + index + "." + gguf_suffix,
    };
    if (suffix == "o_proj.weight") {
        result.push_back("model.language_model.layers." + index + ".self_attn.out_proj.weight");
        result.push_back("model.layers." + index + ".self_attn.out_proj.weight");
    }
    return result;
}

std::vector<std::string> feed_forward_tensor_candidates(int layer,
                                                        std::string_view suffix) {
    const std::string index = std::to_string(layer);
    std::string gguf_suffix;
    if (suffix == "w_gate.weight") gguf_suffix = "ffn_gate.weight";
    if (suffix == "w_up.weight") gguf_suffix = "ffn_up.weight";
    if (suffix == "w_down.weight") gguf_suffix = "ffn_down.weight";
    std::vector<std::string> result = {
        "transformer.h." + index + ".mlp." + std::string(suffix),
        "model.language_model.layers." + index + ".mlp." + std::string(suffix),
        "model.layers." + index + ".mlp." + std::string(suffix),
        "model.language_model.layers." + index + ".feed_forward." + std::string(suffix),
        "layers." + index + ".feed_forward." + std::string(suffix),
        "blk." + index + "." + gguf_suffix,
    };
    if (suffix == "w_gate.weight") {
        result.push_back("model.language_model.layers." + index + ".mlp.gate_proj.weight");
        result.push_back("model.layers." + index + ".mlp.gate_proj.weight");
        result.push_back("model.layers." + index + ".feed_forward.w1.weight");
        result.push_back("model.language_model.layers." + index + ".feed_forward.w1.weight");
    } else if (suffix == "w_up.weight") {
        result.push_back("model.language_model.layers." + index + ".mlp.up_proj.weight");
        result.push_back("model.layers." + index + ".mlp.up_proj.weight");
        result.push_back("model.layers." + index + ".feed_forward.w3.weight");
        result.push_back("model.language_model.layers." + index + ".feed_forward.w3.weight");
    } else if (suffix == "w_down.weight") {
        result.push_back("model.language_model.layers." + index + ".mlp.down_proj.weight");
        result.push_back("model.layers." + index + ".mlp.down_proj.weight");
        result.push_back("model.layers." + index + ".feed_forward.w2.weight");
        result.push_back("model.language_model.layers." + index + ".feed_forward.w2.weight");
    }
    return result;
}

std::vector<std::string> shortconv_tensor_candidates(int layer,
                                                     std::string_view suffix) {
    const std::string index = std::to_string(layer);
    return {
        "model.language_model.layers." + index + ".conv." + std::string(suffix),
        "model.layers." + index + ".conv." + std::string(suffix),
        "layers." + index + ".conv." + std::string(suffix),
        "blk." + index + ".shortconv." + std::string(suffix),
    };
}

std::vector<std::string> mamba2_tensor_candidates(int layer,
                                                  std::string_view suffix) {
    const std::string index = std::to_string(layer);
    std::string gguf_suffix;
    if (suffix == "in_proj.weight") gguf_suffix = "ssm_in.weight";
    else if (suffix == "conv1d.weight") gguf_suffix = "ssm_conv1d.weight";
    else if (suffix == "conv1d.bias") gguf_suffix = "ssm_conv1d.bias";
    else if (suffix == "dt_bias") gguf_suffix = "ssm_dt.bias";
    else if (suffix == "A_log") gguf_suffix = "ssm_a";
    else if (suffix == "D") gguf_suffix = "ssm_d";
    else if (suffix == "norm.weight") gguf_suffix = "ssm_norm.weight";
    else if (suffix == "out_proj.weight") gguf_suffix = "ssm_out.weight";
    else gguf_suffix = std::string(suffix);
    return {
        "model.language_model.layers." + index + ".mixer." + std::string(suffix),
        "model.layers." + index + ".mixer." + std::string(suffix),
        "backbone.layers." + index + ".mixer." + std::string(suffix),
        "layers." + index + ".mixer." + std::string(suffix),
        "blk." + index + "." + gguf_suffix,
    };
}

void add_binding(TensorRoleBindings& bindings, TensorRole role, int layer,
                 const TensorInventoryEntry& tensor,
                 std::vector<EvidenceItem> evidence, int physical_layer) {
    bindings.values.push_back({role, layer, -1, physical_layer, tensor.name, tensor.shape,
                               std::move(evidence)});
}

const TensorInventoryEntry* find_mamba_tensor(const InferenceInput& input,
                                              int layer,
                                              std::string_view suffix) {
    const auto candidates = mamba2_tensor_candidates(layer, suffix);
    const TensorInventoryEntry* found = nullptr;
    for (const auto& candidate : candidates) {
        if (const auto* tensor = input.inventory.find(candidate)) {
            if (found != nullptr) {
                fail(
                    ResolutionFailureKind::AmbiguousTensorBinding,
                    "multiple Mamba-2 tensor spellings are present for layer " +
                        std::to_string(layer));
            }
            found = tensor;
        }
    }
    return found;
}

bool layer_has_feed_forward(const CanonicalInferenceContext& context,
                            int layer) {
    const auto& input = context.input;
    const auto has_tensor = [&](std::string_view name) {
        return input.inventory.find(name) != nullptr;
    };
    const std::string index = std::to_string(context.physical_layer(layer));
    return find_mamba_tensor(input, context.physical_layer(layer), "in_proj.weight") == nullptr &&
        (has_tensor("blk." + index + ".ffn_up.weight") ||
         has_tensor("model.layers." + index + ".mlp.up_proj.weight") ||
         has_tensor("model.language_model.layers." + index + ".mlp.up_proj.weight") ||
         has_tensor("transformer.h." + index + ".mlp.w_up.weight") ||
         has_tensor("model.layers." + index + ".feed_forward.w1.weight") ||
         has_tensor("model.language_model.layers." + index + ".feed_forward.w1.weight") ||
         has_tensor("model.layers." + index + ".feed_forward.experts.0.w1.weight") ||
         has_tensor("model.language_model.layers." + index +
                    ".feed_forward.experts.0.w1.weight") ||
         (context.moe &&
          has_tensor("model.layers." + index +
                     ".mlp.experts.0.gate_proj.weight")));
}

}
