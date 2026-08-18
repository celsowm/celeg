#include "canonical_internal.hpp"
#include "rules.hpp"
#include "support.hpp"

#include <cmath>
#include <utility>

namespace celeg::inference_detail {
namespace {

void add_global_binding(CanonicalInferenceContext& context,
                        TensorRole role,
                        const TensorInventoryEntry& tensor) {
    add_binding(
        context.facts.bindings,
        role,
        -1,
        tensor,
        {{EvidenceKind::TensorName,
          tensor.name,
          std::string(tensor_role_name(role))}});
}

void bind_global_tensors(CanonicalInferenceContext& context) {
    const auto& input = context.input;
    const auto& m = input.metadata;
    auto& facts = context.facts;

    add_global_binding(context, TensorRole::TokenEmbedding, *context.embedding);

    const std::vector<std::string> head_names = {
        "lm_head.weight",
        "transformer.lm_head.weight",
        "output.weight",
    };
    const TensorInventoryEntry* head = nullptr;
    for (const std::string& name : head_names) {
        if (const auto* candidate = input.inventory.find(name)) {
            if (head != nullptr) {
                fail(
                    ResolutionFailureKind::AmbiguousTensorBinding,
                    "multiple language-model heads are present");
            }
            head = candidate;
        }
    }
    if (head == nullptr) {
        const std::vector<std::string> head_suffixes = {
            ".lm_head.weight",
            ".output.weight",
        };
        for (const TensorInventoryEntry& entry : input.inventory.entries()) {
            const std::string& name = entry.name;
            for (const std::string& suffix : head_suffixes) {
                if (name.size() >= suffix.size() &&
                    name.compare(name.size() - suffix.size(),
                                 suffix.size(), suffix) == 0) {
                    if (head != nullptr) {
                        fail(
                            ResolutionFailureKind::AmbiguousTensorBinding,
                            "multiple language-model heads are present");
                    }
                    head = &entry;
                    break;
                }
            }
        }
    }
    if (head == nullptr) {
        if (const bool already_determined =
                m.core.tied_embeddings.has_value();
            !already_determined && context.embedding != nullptr) {
            facts.tied_embeddings = true;
            facts.evidence.push_back({
                EvidenceKind::Derived,
                context.embedding->name,
                "safetensors checkpoint has no independent language-model head; "
                "tied to token embedding"});
        }
        if (!facts.tied_embeddings) {
            fail(
                ResolutionFailureKind::MissingTensorRole,
                "untied checkpoint has no language-model head");
        }
        head = context.embedding;
        facts.evidence.push_back({
            EvidenceKind::Derived,
            context.embedding->name,
            "language-model head is tied to token embedding"});
    } else if (!shape_is(*head, {*m.core.vocab_size, *m.core.hidden_size})) {
        fail(
            ResolutionFailureKind::ShapeConstraintViolation,
            "language-model head shape does not agree with normalized metadata");
    }
    add_global_binding(context, TensorRole::LanguageModelHead, *head);

    const std::vector<std::string> final_norm_names = {
        "transformer.ln_f.weight",
        "model.norm.weight",
        "norm.weight",
        "output_norm.weight",
        "model.language_model.norm.weight",
        "backbone.norm_f.weight",
        "model.embedding_norm.weight",
        "token_embd_norm.weight",
    };
    const TensorInventoryEntry* final_norm = nullptr;
    for (const std::string& name : final_norm_names) {
        if (const auto* candidate = input.inventory.find(name)) {
            if (final_norm != nullptr) {
                fail(
                    ResolutionFailureKind::AmbiguousTensorBinding,
                    "multiple final norms are present");
            }
            final_norm = candidate;
        }
    }
    if (final_norm == nullptr) {
        const std::vector<std::string> final_norm_suffixes = {
            ".embedding_norm.weight",
            ".norm.weight",
            ".ln_f.weight",
            ".output_norm.weight",
            ".norm_f.weight",
            ".token_embd_norm.weight",
        };
        for (const TensorInventoryEntry& entry : input.inventory.entries()) {
            const std::string& name = entry.name;
            for (const std::string& suffix : final_norm_suffixes) {
                if (name.size() >= suffix.size() &&
                    name.compare(name.size() - suffix.size(),
                                 suffix.size(), suffix) == 0) {
                    if (final_norm != nullptr) {
                        fail(
                            ResolutionFailureKind::AmbiguousTensorBinding,
                            "multiple final norms are present");
                    }
                    final_norm = &entry;
                    break;
                }
            }
        }
    }
    if (final_norm == nullptr) {
        fail(
            ResolutionFailureKind::MissingTensorRole,
            "automatic resolution could not find final norm");
    }
    if (!shape_is(*final_norm, {*m.core.hidden_size})) {
        fail(
            ResolutionFailureKind::ShapeConstraintViolation,
            "final norm shape mismatch");
    }
    add_global_binding(context, TensorRole::FinalNorm, *final_norm);
}

const TensorInventoryEntry* find_optional_norm(
    const TensorInventory& inventory,
    const std::vector<std::string>& candidates,
    TensorRole role,
    int layer,
    int hidden) {
    const TensorInventoryEntry* match = nullptr;
    for (const std::string& candidate : candidates) {
        if (const auto* tensor = inventory.find(candidate)) {
            if (match != nullptr) {
                fail(
                    ResolutionFailureKind::AmbiguousTensorBinding,
                    "automatic resolution found multiple bindings for " +
                        std::string(tensor_role_name(role)) + " for layer " +
                        std::to_string(layer));
            }
            match = tensor;
        }
    }
    if (match != nullptr && !shape_is(*match, {hidden})) {
        fail(
            ResolutionFailureKind::ShapeConstraintViolation,
            "tensor " + match->name + " has a shape inconsistent with " +
                std::string(tensor_role_name(role)));
    }
    return match;
}

void require_explicit_norm_consistency(std::optional<bool> explicit_value,
                                       const TensorInventoryEntry* tensor,
                                       TensorRole role,
                                       int layer) {
    if (!explicit_value.has_value()) return;
    if (*explicit_value && tensor == nullptr) {
        fail(
            ResolutionFailureKind::MissingTensorRole,
            "metadata requires " + std::string(tensor_role_name(role)) +
                " but checkpoint has no matching tensor for layer " +
                std::to_string(layer));
    }
    if (!*explicit_value && tensor != nullptr) {
        fail(
            ResolutionFailureKind::ConflictingInferenceFacts,
            "metadata disables " + std::string(tensor_role_name(role)) +
                " but checkpoint tensor proves it for layer " +
                std::to_string(layer));
    }
}

void bind_structural_norm(CanonicalInferenceContext& context,
                          int layer,
                          TensorRole role,
                          const TensorInventoryEntry* tensor,
                          std::optional<NormSpec>& slot) {
    if (tensor == nullptr) {
        slot.reset();
        return;
    }
    slot = NormSpec{
        context.facts.numerical_policy.norm_eps,
        NormWeightKind::Scale};
    add_binding(
        context.facts.bindings,
        role,
        layer,
        *tensor,
        {{EvidenceKind::TensorName,
          tensor->name,
          std::string(tensor_role_name(role))}});
}

void infer_and_bind_layer_norms(CanonicalInferenceContext& context,
                                int layer) {
    const auto& input = context.input;
    const auto& norm_facts = input.metadata.norms;
    const int hidden = *input.metadata.core.hidden_size;
    const std::string index = std::to_string(layer);
    LayerSpec& semantic_layer =
        context.facts.graph.layers[static_cast<size_t>(layer)];

    const std::vector<std::string> mixer_before_candidates = {
        "transformer.h." + index + ".ln_1.weight",
        "model.layers." + index + ".input_layernorm.weight",
        "model.layers." + index + ".self_attn_layer_norm.weight",
        "model.language_model.layers." + index + ".input_layernorm.weight",
        "model.language_model.layers." + index + ".operator_norm.weight",
        "model.layers." + index + ".operator_norm.weight",
        "backbone.layers." + index + ".norm.weight",
        "blk." + index + ".attn_norm.weight",
    };
    const std::vector<std::string> explicit_mixer_after_candidates = {
        "model.layers." + index + ".post_attention_norm.weight",
        "model.language_model.layers." + index + ".post_attention_norm.weight",
        "blk." + index + ".post_attention_norm.weight",
    };
    const std::vector<std::string> post_attention_layernorm_candidates = {
        "model.layers." + index + ".post_attention_layernorm.weight",
        "model.language_model.layers." + index + ".post_attention_layernorm.weight",
    };
    const std::vector<std::string> explicit_ffn_before_candidates = {
        "transformer.h." + index + ".ln_2.weight",
        "model.language_model.layers." + index + ".ffn_norm.weight",
        "model.layers." + index + ".ffn_norm.weight",
        "blk." + index + ".ffn_norm.weight",
    };
    const std::vector<std::string> pre_feedforward_candidates = {
        "model.layers." + index + ".pre_feedforward_layernorm.weight",
        "model.language_model.layers." + index + ".pre_feedforward_layernorm.weight",
        "model.layers." + index + ".pre_feed_forward_layernorm.weight",
        "model.language_model.layers." + index + ".pre_feed_forward_layernorm.weight",
    };
    const std::vector<std::string> ffn_after_candidates = {
        "model.layers." + index + ".post_feedforward_layernorm.weight",
        "model.language_model.layers." + index + ".post_feedforward_layernorm.weight",
        "model.layers." + index + ".post_feed_forward_layernorm.weight",
        "model.language_model.layers." + index + ".post_feed_forward_layernorm.weight",
        "model.layers." + index + ".post_ffn_norm.weight",
        "model.language_model.layers." + index + ".post_ffn_norm.weight",
        "blk." + index + ".post_ffn_norm.weight",
        "blk." + index + ".ffn_post_norm.weight",
    };

    const TensorInventoryEntry* mixer_before = find_optional_norm(
        input.inventory, mixer_before_candidates,
        TensorRole::AttentionInputNorm, layer, hidden);
    const TensorInventoryEntry* explicit_mixer_after = find_optional_norm(
        input.inventory, explicit_mixer_after_candidates,
        TensorRole::AttentionPostNorm, layer, hidden);
    const TensorInventoryEntry* post_attention_layernorm = find_optional_norm(
        input.inventory, post_attention_layernorm_candidates,
        TensorRole::AttentionPostNorm, layer, hidden);
    const TensorInventoryEntry* explicit_ffn_before = find_optional_norm(
        input.inventory, explicit_ffn_before_candidates,
        TensorRole::FfnInputNorm, layer, hidden);
    const TensorInventoryEntry* pre_feedforward = find_optional_norm(
        input.inventory, pre_feedforward_candidates,
        TensorRole::FfnInputNorm, layer, hidden);
    const TensorInventoryEntry* ffn_after = find_optional_norm(
        input.inventory, ffn_after_candidates,
        TensorRole::FfnOutputNorm, layer, hidden);

    const std::optional<bool> metadata_mixer_after =
        norm_facts.mixer_after.value_for(layer);
    const std::optional<bool> metadata_ffn_before =
        norm_facts.feed_forward_before.value_for(layer);

    const TensorInventoryEntry* mixer_after = explicit_mixer_after;
    const TensorInventoryEntry* ffn_before = explicit_ffn_before;
    if (pre_feedforward != nullptr) {
        if (explicit_ffn_before != nullptr) {
            fail(
                ResolutionFailureKind::AmbiguousTensorBinding,
                "automatic resolution found multiple pre-feed-forward norms for layer " +
                    std::to_string(layer));
        }
        ffn_before = pre_feedforward;
    }

    if (post_attention_layernorm != nullptr) {
        if (metadata_mixer_after == true && metadata_ffn_before == true) {
            fail(
                ResolutionFailureKind::ConflictingMetadata,
                "one post_attention_layernorm tensor cannot satisfy both mixer-after and "
                "feed-forward-before metadata for layer " + std::to_string(layer));
        }
        const bool explicitly_mixer_after = metadata_mixer_after == true;
        const bool explicitly_ffn_before = metadata_ffn_before == true;
        const bool post_norm_topology =
            explicitly_mixer_after ||
            (!explicitly_ffn_before &&
             (pre_feedforward != nullptr || ffn_after != nullptr));
        if (post_norm_topology) {
            if (explicit_mixer_after != nullptr) {
                fail(
                    ResolutionFailureKind::AmbiguousTensorBinding,
                    "automatic resolution found multiple post-attention norms for layer " +
                        std::to_string(layer));
            }
            mixer_after = post_attention_layernorm;
        } else {
            if (explicit_ffn_before != nullptr) {
                fail(
                    ResolutionFailureKind::AmbiguousTensorBinding,
                    "post_attention_layernorm conflicts with another feed-forward input norm "
                    "for layer " + std::to_string(layer));
            }
            ffn_before = post_attention_layernorm;
        }
    }

    require_explicit_norm_consistency(
        norm_facts.mixer_before.value_for(layer), mixer_before,
        TensorRole::AttentionInputNorm, layer);
    require_explicit_norm_consistency(
        metadata_mixer_after, mixer_after,
        TensorRole::AttentionPostNorm, layer);
    require_explicit_norm_consistency(
        metadata_ffn_before, ffn_before,
        TensorRole::FfnInputNorm, layer);
    require_explicit_norm_consistency(
        norm_facts.feed_forward_after.value_for(layer), ffn_after,
        TensorRole::FfnOutputNorm, layer);

    bind_structural_norm(
        context, layer, TensorRole::AttentionInputNorm,
        mixer_before, semantic_layer.mixer_norm.before);
    bind_structural_norm(
        context, layer, TensorRole::AttentionPostNorm,
        mixer_after, semantic_layer.mixer_norm.after);

    const bool has_feed_forward =
        !std::holds_alternative<std::monostate>(semantic_layer.feed_forward);
    if (!has_feed_forward && (ffn_before != nullptr || ffn_after != nullptr)) {
        fail(
            ResolutionFailureKind::ConflictingInferenceFacts,
            "checkpoint exposes feed-forward normalization for a layer with no "
            "feed-forward semantics: " + std::to_string(layer));
    }
    if (has_feed_forward) {
        bind_structural_norm(
            context, layer, TensorRole::FfnInputNorm,
            ffn_before, semantic_layer.feed_forward_norm.before);
        bind_structural_norm(
            context, layer, TensorRole::FfnOutputNorm,
            ffn_after, semantic_layer.feed_forward_norm.after);
    } else {
        semantic_layer.feed_forward_norm = {};
    }
}

void bind_dense_ffn(CanonicalInferenceContext& context,
                    int layer,
                    int layer_intermediate) {
    const auto& input = context.input;
    const auto& m = input.metadata;
    auto& bindings = context.facts.bindings;

    const auto* gate = find_unique(
        input.inventory,
        feed_forward_tensor_candidates(layer, "w_gate.weight"),
        TensorRole::FfnGate,
        layer,
        {layer_intermediate, *m.core.hidden_size},
        {});
    add_binding(bindings, TensorRole::FfnGate, layer, *gate, {});

    const auto* up = find_unique(
        input.inventory,
        feed_forward_tensor_candidates(layer, "w_up.weight"),
        TensorRole::FfnUp,
        layer,
        {layer_intermediate, *m.core.hidden_size},
        {});
    add_binding(bindings, TensorRole::FfnUp, layer, *up, {});

    const auto* down = find_unique(
        input.inventory,
        feed_forward_tensor_candidates(layer, "w_down.weight"),
        TensorRole::FfnDown,
        layer,
        {*m.core.hidden_size, layer_intermediate},
        {});
    add_binding(bindings, TensorRole::FfnDown, layer, *down, {});
}

void bind_moe(CanonicalInferenceContext& context,
              int layer,
              std::string_view index) {
    const auto& input = context.input;
    const auto& m = input.metadata;
    auto& facts = context.facts;
    const std::string prefix =
        "model.layers." + std::string(index) + ".mlp.";
    const std::string feed_forward_prefix =
        "model.layers." + std::string(index) + ".feed_forward.";
    const auto has_tensor = [&](std::string_view name) {
        return input.inventory.find(name) != nullptr;
    };

    const auto bind = [&](TensorRole role,
                          std::string name,
                          std::initializer_list<std::int64_t> shape,
                          int expert = -1) {
        const auto* tensor = find_unique(
            input.inventory,
            {std::move(name)},
            role,
            layer,
            shape,
            {});
        facts.bindings.values.push_back({
            role,
            layer,
            expert,
            -1,
            tensor->name,
            tensor->shape,
            {},
        });
    };

    const auto bind_packed = [&](TensorRole role, const std::string& name) {
        const auto* tensor = input.inventory.find(name);
        if (tensor == nullptr) {
            fail(ResolutionFailureKind::MissingTensorRole,
                 "automatic resolution could not bind " +
                     std::string(tensor_role_name(role)) + " for layer " +
                     std::to_string(layer));
        }
        facts.bindings.values.push_back({
            role, layer, -1, -1, tensor->name, tensor->shape, {},
        });
    };

    const int num_experts = context.moe->num_experts;
    bind(
        TensorRole::MoeRouter,
        prefix + "gate.weight",
        {num_experts, *m.core.hidden_size});
    std::optional<std::string> bias_name;
    for (const std::string& candidate : {
             prefix + "gate.expert_bias",
             feed_forward_prefix + "expert_bias.weight",
             feed_forward_prefix + "expert_bias",
         }) {
        if (has_tensor(candidate)) {
            bias_name = candidate;
            break;
        }
    }
    if (bias_name) {
        bind(
            TensorRole::MoeRouterBias,
            *bias_name,
            {num_experts});
    }

    const auto* moe = std::get_if<MixtureOfExpertsSpec>(
        &facts.graph.layers[static_cast<size_t>(layer)].feed_forward);
    if (moe == nullptr || moe->intermediate_size <= 0) {
        fail(
            ResolutionFailureKind::UnsupportedGraphPrimitive,
            "MoE binding resolution has no per-layer routed width: " +
                std::to_string(layer));
    }

    const int expert_intermediate = moe->intermediate_size;
    const bool named_individual = has_tensor(prefix + "experts.0.gate_proj.weight");
    const bool raw_individual = !named_individual &&
        has_tensor(feed_forward_prefix + "experts.0.w1.weight");
    std::optional<std::string> packed_prefix;
    if (!named_individual && !raw_individual) {
        const std::string alternate_prefix =
            "model.language_model.layers." + std::string(index) + ".mlp.";
        for (const std::string& candidate_prefix : {prefix, alternate_prefix}) {
            if (has_tensor(candidate_prefix + "experts.gate_up_proj") &&
                has_tensor(candidate_prefix + "experts.down_proj")) {
                packed_prefix = candidate_prefix;
                break;
            }
        }
    }

    if (named_individual) {
        for (int expert = 0; expert < num_experts; ++expert) {
            const std::string expert_prefix =
                prefix + "experts." + std::to_string(expert) + ".";
            bind(
                TensorRole::MoeExpertGate,
                expert_prefix + "gate_proj.weight",
                {expert_intermediate, *m.core.hidden_size},
                expert);
            bind(
                TensorRole::MoeExpertUp,
                expert_prefix + "up_proj.weight",
                {expert_intermediate, *m.core.hidden_size},
                expert);
            bind(
                TensorRole::MoeExpertDown,
                expert_prefix + "down_proj.weight",
                {*m.core.hidden_size, expert_intermediate},
                expert);
        }
    } else if (raw_individual) {
        for (int expert = 0; expert < num_experts; ++expert) {
            const std::string expert_prefix =
                feed_forward_prefix + "experts." + std::to_string(expert) + ".";
            bind(
                TensorRole::MoeExpertGate,
                expert_prefix + "w1.weight",
                {expert_intermediate, *m.core.hidden_size},
                expert);
            bind(
                TensorRole::MoeExpertUp,
                expert_prefix + "w3.weight",
                {expert_intermediate, *m.core.hidden_size},
                expert);
            bind(
                TensorRole::MoeExpertDown,
                expert_prefix + "w2.weight",
                {*m.core.hidden_size, expert_intermediate},
                expert);
        }
    } else if (packed_prefix) {
        bind_packed(TensorRole::MoePackedGateUp, *packed_prefix + "experts.gate_up_proj");
        bind_packed(TensorRole::MoePackedDown, *packed_prefix + "experts.down_proj");
    } else {
        fail(
            ResolutionFailureKind::MissingTensorRole,
            "automatic resolution could not determine the MoE routed-expert "
            "checkpoint layout for layer " + std::to_string(layer));
    }

    if (moe->shared) {
        const int shared_intermediate = moe->shared->intermediate_size;
        const std::string shared = prefix + "shared_experts.";
        bind(
            TensorRole::MoeSharedGate,
            shared + "gate_proj.weight",
            {shared_intermediate, *m.core.hidden_size});
        bind(
            TensorRole::MoeSharedUp,
            shared + "up_proj.weight",
            {shared_intermediate, *m.core.hidden_size});
        bind(
            TensorRole::MoeSharedDown,
            shared + "down_proj.weight",
            {*m.core.hidden_size, shared_intermediate});
    }
}

void resolve_layer_feed_forward(CanonicalInferenceContext& context,
                                int layer) {
    auto& facts = context.facts;
    const LayerSpec& semantic_layer =
        facts.graph.layers[static_cast<size_t>(layer)];

    if (std::holds_alternative<MlpBlockSpec>(semantic_layer.mixer)) return;
    if (std::holds_alternative<std::monostate>(semantic_layer.feed_forward)) return;

    if (std::holds_alternative<MixtureOfExpertsSpec>(semantic_layer.feed_forward)) {
        bind_moe(context, layer, std::to_string(layer));
    } else {
        bind_dense_ffn(context, layer,
                       context.intermediate_sizes.at(static_cast<size_t>(layer)));
    }
}

void apply_attention_output_scale(CanonicalInferenceContext& context) {
    const auto& m = context.input.metadata;
    auto& graph = context.facts.graph;
    auto& numerical_policy = context.facts.numerical_policy;

    if (m.attention.attention_multiplier.has_value()) {
        numerical_policy.attention_multiplier =
            *m.attention.attention_multiplier;
    } else {
        int attention_head_dim = 0;
        for (int layer = 0; layer < context.layer_count; ++layer) {
            if (std::holds_alternative<AttentionSpec>(
                    graph.layers[static_cast<size_t>(layer)].mixer)) {
                attention_head_dim = std::get<AttentionSpec>(
                    graph.layers[static_cast<size_t>(layer)].mixer)
                                         .head_dim;
                break;
            }
        }
        numerical_policy.attention_multiplier =
            attention_head_dim > 0
                ? 1.0f /
                      std::sqrt(
                          static_cast<float>(attention_head_dim))
                : 1.0f;
    }

    for (LayerSpec& semantic_layer : graph.layers) {
        if (auto* attention =
                std::get_if<AttentionSpec>(&semantic_layer.mixer);
            attention != nullptr && attention->query_heads > 0) {
            attention->query_scale *=
                numerical_policy.attention_multiplier;
        }
    }
}

}

void resolve_canonical_layers(CanonicalInferenceContext& context) {
    auto& facts = context.facts;
    const auto rules = make_builtin_layer_inference_rules();

    bind_global_tensors(context);
    for (int layer = 0; layer < context.layer_count; ++layer) {
        const ILayerInferenceRule& rule =
            select_layer_inference_rule(rules, context, layer);
        rule.resolve(context, layer);
        infer_and_bind_layer_norms(context, layer);
        resolve_layer_feed_forward(context, layer);
    }

    apply_attention_output_scale(context);
    facts.graph.validate();
    facts.bindings = BindingSolver{}.solve(facts.bindings.values);
}

}
