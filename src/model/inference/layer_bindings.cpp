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
        if (input.is_gguf() && !m.core.tied_embeddings.has_value()) {
            facts.tied_embeddings = true;
            facts.evidence.push_back({
                EvidenceKind::FormatGuarantee,
                "output.weight",
                "GGUF omits an independent language-model head"});
        } else if (!facts.tied_embeddings) {
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

    // Binds a tensor whose on-disk shape may legitimately vary between a
    // packed [num_experts, rows, cols] layout and a flattened
    // [num_experts * rows, cols] layout. The physical family (packed vs
    // individual) is decided here, once, from real checkpoint evidence, so
    // that backends never need to re-derive it from tensor-name spellings.
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
    /// The router-bias tensor spelling is a checkpoint-family fact: resolve
    /// it here, once, so backends read the bound name from the plan instead
    /// of each probing their own literal fallback chain.
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

    // Decide the routed-expert checkpoint's physical layout (family) once,
    // here, from real checkpoint evidence: individually-named
    // "gate_proj/up_proj/down_proj" tensors per expert, individually-named
    // "w1/w2/w3" tensors per expert (the convention GGUF native storage
    // answers to), or a single tensor packing every expert's weights
    // together. Backends must consume the resulting role set rather than
    // re-probing the repository for these spellings themselves.
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

/// Binds the orthogonal feed-forward axis for one layer whose mixer rule
/// already ran: the input norm plus either the routed-expert MoE grammar or
/// the dense projections. Layers whose feed-forward is monostate (no FFN
/// grammar, Mamba-2, or MLP-only mixers) bind nothing here.
void resolve_layer_feed_forward(CanonicalInferenceContext& context,
                                int layer,
                                const std::vector<std::string>& ffn_norm_candidates) {
    const auto& input = context.input;
    const auto& m = input.metadata;
    auto& facts = context.facts;
    const LayerSpec& semantic_layer =
        facts.graph.layers[static_cast<size_t>(layer)];

    if (std::holds_alternative<MlpBlockSpec>(semantic_layer.mixer)) return;
    if (std::holds_alternative<std::monostate>(semantic_layer.feed_forward)) return;

    const auto* ffn_norm = find_unique(
        input.inventory,
        ffn_norm_candidates,
        TensorRole::FfnInputNorm,
        layer,
        {*m.core.hidden_size},
        {});
    add_binding(
        facts.bindings,
        TensorRole::FfnInputNorm,
        layer,
        *ffn_norm,
        {});

    if (std::holds_alternative<MixtureOfExpertsSpec>(semantic_layer.feed_forward)) {
        bind_moe(context, layer, std::to_string(layer));
    } else {
        bind_dense_ffn(context, layer,
                       context.intermediate_sizes.at(static_cast<size_t>(layer)));
    }
}

/// Derives the default attention output scale from the first attention
/// layer when the checkpoint metadata does not state one, and applies it to
/// every attention layer's query scale.
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
    const auto& input = context.input;
    const auto& m = input.metadata;
    auto& facts = context.facts;
    const auto rules = make_builtin_layer_inference_rules();

    bind_global_tensors(context);
    for (int layer = 0; layer < context.layer_count; ++layer) {
        const std::string index = std::to_string(layer);

        const std::vector<std::string> norm_candidates = {
            "transformer.h." + index + ".ln_1.weight",
            "model.layers." + index + ".input_layernorm.weight",
            "model.layers." + index + ".self_attn_layer_norm.weight",
            "model.language_model.layers." + index + ".input_layernorm.weight",
            "model.layers." + index + ".operator_norm.weight",
            "backbone.layers." + index + ".norm.weight",
            "blk." + index + ".attn_norm.weight",
        };
        const std::vector<std::string> ffn_norm_candidates = {
            "transformer.h." + index + ".ln_2.weight",
            "model.layers." + index + ".post_attention_layernorm.weight",
            "model.language_model.layers." + index + ".post_attention_layernorm.weight",
            "model.layers." + index + ".ffn_norm.weight",
            "blk." + index + ".ffn_norm.weight",
            "blk." + index + ".post_attention_norm.weight",
        };

        const auto* attention_norm = find_unique(
            input.inventory,
            norm_candidates,
            TensorRole::AttentionInputNorm,
            layer,
            {*m.core.hidden_size},
            {});
        add_binding(
            facts.bindings,
            TensorRole::AttentionInputNorm,
            layer,
            *attention_norm,
            {});

        const ILayerInferenceRule& rule =
            select_layer_inference_rule(rules, context, layer);
        rule.resolve(context, layer);
        resolve_layer_feed_forward(context, layer, ffn_norm_candidates);
    }

    apply_attention_output_scale(context);
    facts.graph.validate();
    facts.bindings = BindingSolver{}.solve(facts.bindings.values);
}

}
