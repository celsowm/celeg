#include "canonical_internal.hpp"
#include "support.hpp"

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
        if (input.source_format == CheckpointSourceFormat::Gguf &&
            !m.tied_embeddings.has_value()) {
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
    } else if (!shape_is(*head, {*m.vocab_size, *m.hidden_size})) {
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
    if (!shape_is(*final_norm, {*m.hidden_size})) {
        fail(
            ResolutionFailureKind::ShapeConstraintViolation,
            "final norm shape mismatch");
    }
    add_global_binding(context, TensorRole::FinalNorm, *final_norm);
}

void bind_mamba2(CanonicalInferenceContext& context, int layer) {
    const auto& input = context.input;
    const auto& m = input.metadata;
    auto& bindings = context.facts.bindings;
    const auto& spec = std::get<Mamba2Spec>(
        context.facts.graph.layers[static_cast<size_t>(layer)].mixer);

    const auto bind = [&](TensorRole role,
                          std::string_view suffix,
                          std::initializer_list<std::int64_t> shape) {
        const auto* tensor = find_unique(
            input.inventory,
            mamba2_tensor_candidates(layer, suffix),
            role,
            layer,
            shape,
            {});
        add_binding(bindings, role, layer, *tensor, {});
    };

    const int conv_dim = spec.intermediate_size +
        2 * spec.group_count * spec.state_size;
    bind(
        TensorRole::Mamba2Input,
        "in_proj.weight",
        {2 * spec.intermediate_size +
             2 * spec.group_count * spec.state_size + spec.num_heads,
         *m.hidden_size});
    bind(
        TensorRole::Mamba2Conv,
        "conv1d.weight",
        {conv_dim, 1, spec.conv_kernel});
    bind(TensorRole::Mamba2ConvBias, "conv1d.bias", {conv_dim});
    bind(TensorRole::Mamba2DtBias, "dt_bias", {spec.num_heads});
    bind(TensorRole::Mamba2ALog, "A_log", {spec.num_heads});
    bind(TensorRole::Mamba2D, "D", {spec.num_heads});
    bind(
        TensorRole::Mamba2Norm,
        "norm.weight",
        {spec.intermediate_size});
    bind(
        TensorRole::Mamba2Output,
        "out_proj.weight",
        {*m.hidden_size, spec.intermediate_size});
}

void bind_factorized_gated_delta(CanonicalInferenceContext& context,
                                 int layer,
                                 std::string_view index) {
    const auto& input = context.input;
    const auto& m = input.metadata;
    auto& bindings = context.facts.bindings;
    const auto& spec = std::get<GatedDeltaNetSpec>(
        context.facts.graph.layers[static_cast<size_t>(layer)].mixer);
    const std::string prefix =
        "model.layers." + std::string(index) + ".attention.";

    const auto bind = [&](TensorRole role,
                          std::string_view suffix,
                          std::initializer_list<std::int64_t> shape) {
        const auto* tensor = find_unique(
            input.inventory,
            {prefix + std::string(suffix)},
            role,
            layer,
            shape,
            {});
        add_binding(bindings, role, layer, *tensor, {});
    };

    bind(
        TensorRole::GatedDeltaNetQuery,
        "q_proj.weight",
        {spec.key_heads * spec.key_head_dim, *m.hidden_size});
    bind(
        TensorRole::GatedDeltaNetKey,
        "k_proj.weight",
        {spec.key_heads * spec.key_head_dim, *m.hidden_size});
    bind(
        TensorRole::GatedDeltaNetValue,
        "v_proj.weight",
        {spec.value_width(), *m.hidden_size});
    bind(
        TensorRole::GatedDeltaNetDecay,
        "f_proj.weight",
        {spec.decay_width(), *m.hidden_size});
    bind(
        TensorRole::GatedDeltaNetOutputGate,
        "g_proj.weight",
        {spec.value_width(), *m.hidden_size});
    bind(
        TensorRole::GatedDeltaNetQueryConv,
        "q_conv1d.weight",
        {spec.key_heads * spec.key_head_dim, 1, spec.conv_kernel});
    bind(
        TensorRole::GatedDeltaNetKeyConv,
        "k_conv1d.weight",
        {spec.key_heads * spec.key_head_dim, 1, spec.conv_kernel});
    bind(
        TensorRole::GatedDeltaNetValueConv,
        "v_conv1d.weight",
        {spec.value_width(), 1, spec.conv_kernel});
    bind(
        TensorRole::GatedDeltaNetBeta,
        "b_proj.weight",
        {spec.value_heads, *m.hidden_size});
    bind(
        TensorRole::GatedDeltaNetDtBias,
        "dt_bias",
        {spec.decay_width()});
    bind(TensorRole::GatedDeltaNetALog, "A_log", {spec.value_heads});
    bind(
        TensorRole::GatedDeltaNetNorm,
        "o_norm.weight",
        {spec.value_head_dim});
    bind(
        TensorRole::GatedDeltaNetOutput,
        "o_proj.weight",
        {*m.hidden_size, spec.value_width()});
}

void bind_fused_gated_delta(CanonicalInferenceContext& context,
                            int layer,
                            std::string_view layer_prefix) {
    const auto& input = context.input;
    const auto& m = input.metadata;
    auto& bindings = context.facts.bindings;
    const auto& spec = std::get<GatedDeltaNetSpec>(
        context.facts.graph.layers[static_cast<size_t>(layer)].mixer);
    const int qkv_width = spec.qkv_width();

    const auto bind = [&](TensorRole role,
                          std::string_view suffix,
                          std::initializer_list<std::int64_t> shape) {
        const auto* tensor = find_unique(
            input.inventory,
            {std::string(layer_prefix) + std::string(suffix)},
            role,
            layer,
            shape,
            {});
        add_binding(bindings, role, layer, *tensor, {});
    };

    bind(
        TensorRole::GatedDeltaNetQkv,
        "attn_qkv.weight",
        {qkv_width, *m.hidden_size});
    bind(
        TensorRole::GatedDeltaNetZ,
        "attn_gate.weight",
        {spec.value_width(), *m.hidden_size});
    bind(
        TensorRole::GatedDeltaNetAlpha,
        "ssm_alpha.weight",
        {spec.value_heads, *m.hidden_size});
    bind(
        TensorRole::GatedDeltaNetBeta,
        "ssm_beta.weight",
        {spec.value_heads, *m.hidden_size});
    bind(
        TensorRole::GatedDeltaNetDtBias,
        "ssm_dt.bias",
        {spec.value_heads});
    bind(TensorRole::GatedDeltaNetALog, "ssm_a", {spec.value_heads});
    bind(
        TensorRole::GatedDeltaNetConv,
        "ssm_conv1d.weight",
        {qkv_width, 1, spec.conv_kernel});
    bind(
        TensorRole::GatedDeltaNetNorm,
        "ssm_norm.weight",
        {spec.value_head_dim});
    bind(
        TensorRole::GatedDeltaNetOutput,
        "ssm_out.weight",
        {*m.hidden_size, spec.value_width()});
}

void bind_short_convolution(CanonicalInferenceContext& context, int layer) {
    const auto& input = context.input;
    const auto& m = input.metadata;
    auto& bindings = context.facts.bindings;
    const auto& spec = std::get<ShortConvolutionSpec>(
        context.facts.graph.layers[static_cast<size_t>(layer)].mixer);

    const auto* input_projection = find_unique(
        input.inventory,
        shortconv_tensor_candidates(layer, "in_proj.weight"),
        TensorRole::ShortConvInput,
        layer,
        {3 * *m.hidden_size, *m.hidden_size},
        {});
    add_binding(
        bindings,
        TensorRole::ShortConvInput,
        layer,
        *input_projection,
        {});

    const auto* kernel = find_unique(
        input.inventory,
        shortconv_tensor_candidates(layer, "conv.weight"),
        TensorRole::ShortConvKernel,
        layer,
        {*m.hidden_size, 1, spec.cache_length},
        {});
    add_binding(
        bindings,
        TensorRole::ShortConvKernel,
        layer,
        *kernel,
        {});

    const auto* output_projection = find_unique(
        input.inventory,
        shortconv_tensor_candidates(layer, "out_proj.weight"),
        TensorRole::ShortConvOutput,
        layer,
        {*m.hidden_size, *m.hidden_size},
        {});
    add_binding(
        bindings,
        TensorRole::ShortConvOutput,
        layer,
        *output_projection,
        {});
}

void bind_latent_attention(CanonicalInferenceContext& context,
                           int layer,
                           std::string_view index,
                           const AttentionSpec& attention) {
    const auto& input = context.input;
    const auto& m = input.metadata;
    auto& bindings = context.facts.bindings;
    const auto& latent = *attention.latent_state();
    const int query_head_count = *m.query_heads.value_for(layer);
    const std::string prefix =
        "model.layers." + std::string(index) + ".attention.";

    const auto bind = [&](TensorRole role,
                          std::string_view suffix,
                          std::initializer_list<std::int64_t> shape) {
        const auto* tensor = find_unique(
            input.inventory,
            {prefix + std::string(suffix)},
            role,
            layer,
            shape,
            {});
        add_binding(bindings, role, layer, *tensor, {});
    };

    bind(
        TensorRole::AttentionLatentQueryProjection,
        "q_a_proj.weight",
        {latent.query_rank, *m.hidden_size});
    bind(
        TensorRole::AttentionLatentQueryNorm,
        "q_a_layernorm.weight",
        {latent.query_rank});
    bind(
        TensorRole::AttentionLatentQueryExpansion,
        "q_b_proj.weight",
        {query_head_count * (latent.nope_head_dim + latent.rope_head_dim),
         latent.query_rank});
    bind(
        TensorRole::AttentionLatentKeyProjection,
        "kv_a_proj_with_mqa.weight",
        {latent.latent_rank + latent.rope_head_dim, *m.hidden_size});
    bind(
        TensorRole::AttentionLatentKeyNorm,
        "kv_a_layernorm.weight",
        {latent.latent_rank});
    bind(
        TensorRole::AttentionLatentExpansion,
        "kv_b_proj.weight",
        {query_head_count * (latent.nope_head_dim + latent.value_head_dim),
         latent.latent_rank});

    const auto* output = find_unique(
        input.inventory,
        {prefix + "dense.weight", prefix + "o_proj.weight"},
        TensorRole::AttentionLatentOutput,
        layer,
        {*m.hidden_size, attention.latent_output_width()},
        {});
    add_binding(
        bindings,
        TensorRole::AttentionLatentOutput,
        layer,
        *output,
        {});
    bind(
        TensorRole::AttentionGate,
        "g_proj.weight",
        {attention.output_gate_width(), *m.hidden_size});
}

void bind_standard_attention(CanonicalInferenceContext& context,
                             int layer,
                             std::string_view index,
                             std::string_view layer_prefix,
                             const AttentionSpec& attention) {
    const auto& input = context.input;
    const auto& m = input.metadata;
    auto& bindings = context.facts.bindings;
    const int query_head_count = *m.query_heads.value_for(layer);
    const int key_value_head_count = *m.key_value_heads.value_for(layer);
    const int layer_head_dim = m.head_dim.value_for(layer).value_or(
        *m.hidden_size / query_head_count);

    const auto* q = find_unique(
        input.inventory,
        attention_tensor_candidates(layer, "q_proj.weight"),
        TensorRole::AttentionQuery,
        layer,
        {attention.query_projection_width(), *m.hidden_size},
        {});
    add_binding(bindings, TensorRole::AttentionQuery, layer, *q, {});

    const auto* k = find_unique(
        input.inventory,
        attention_tensor_candidates(layer, "k_proj.weight"),
        TensorRole::AttentionKey,
        layer,
        {key_value_head_count * layer_head_dim, *m.hidden_size},
        {});
    add_binding(bindings, TensorRole::AttentionKey, layer, *k, {});

    const auto* v = find_unique(
        input.inventory,
        attention_tensor_candidates(layer, "v_proj.weight"),
        TensorRole::AttentionValue,
        layer,
        {key_value_head_count * layer_head_dim, *m.hidden_size},
        {});
    add_binding(bindings, TensorRole::AttentionValue, layer, *v, {});

    const auto* o = find_unique(
        input.inventory,
        attention_tensor_candidates(layer, "o_proj.weight"),
        TensorRole::AttentionOutput,
        layer,
        {*m.hidden_size, query_head_count * layer_head_dim},
        {});
    add_binding(bindings, TensorRole::AttentionOutput, layer, *o, {});

    const auto has_tensor = [&](std::string_view name) {
        return input.inventory.find(name) != nullptr;
    };
    const std::string q_norm_alias =
        "model.layers." + std::string(index) + ".self_attn.q_layernorm.weight";
    const std::string k_norm_alias =
        "model.layers." + std::string(index) + ".self_attn.k_layernorm.weight";

    if (has_tensor(std::string(layer_prefix) + "attn_q_norm.weight")) {
        const auto* q_norm = find_unique(
            input.inventory,
            {std::string(layer_prefix) + "attn_q_norm.weight"},
            TensorRole::AttentionQueryNorm,
            layer,
            {layer_head_dim},
            {});
        add_binding(
            bindings,
            TensorRole::AttentionQueryNorm,
            layer,
            *q_norm,
            {});
    } else if (has_tensor(q_norm_alias)) {
        const auto* q_norm = find_unique(
            input.inventory,
            {q_norm_alias},
            TensorRole::AttentionQueryNorm,
            layer,
            {layer_head_dim},
            {});
        add_binding(
            bindings,
            TensorRole::AttentionQueryNorm,
            layer,
            *q_norm,
            {});
    }

    if (has_tensor(std::string(layer_prefix) + "attn_k_norm.weight")) {
        const auto* k_norm = find_unique(
            input.inventory,
            {std::string(layer_prefix) + "attn_k_norm.weight"},
            TensorRole::AttentionKeyNorm,
            layer,
            {layer_head_dim},
            {});
        add_binding(
            bindings,
            TensorRole::AttentionKeyNorm,
            layer,
            *k_norm,
            {});
    } else if (has_tensor(k_norm_alias)) {
        const auto* k_norm = find_unique(
            input.inventory,
            {k_norm_alias},
            TensorRole::AttentionKeyNorm,
            layer,
            {layer_head_dim},
            {});
        add_binding(
            bindings,
            TensorRole::AttentionKeyNorm,
            layer,
            *k_norm,
            {});
    }
}

void bind_mixer(CanonicalInferenceContext& context,
                int layer,
                std::string_view index,
                std::string_view layer_prefix) {
    const LayerSpec& semantic_layer =
        context.facts.graph.layers[static_cast<size_t>(layer)];

    if (std::holds_alternative<Mamba2Spec>(semantic_layer.mixer)) {
        bind_mamba2(context, layer);
        return;
    }
    if (const auto* spec =
            std::get_if<GatedDeltaNetSpec>(&semantic_layer.mixer)) {
        if (spec->factorized_projections) {
            bind_factorized_gated_delta(context, layer, index);
        } else {
            bind_fused_gated_delta(context, layer, layer_prefix);
        }
        return;
    }
    if (std::holds_alternative<ShortConvolutionSpec>(semantic_layer.mixer)) {
        bind_short_convolution(context, layer);
        return;
    }
    if (const auto* attention = std::get_if<AttentionSpec>(&semantic_layer.mixer)) {
        if (attention->uses_latent_state() &&
            attention->latent_state()->factorized) {
            bind_latent_attention(context, layer, index, *attention);
        } else {
            bind_standard_attention(
                context, layer, index, layer_prefix, *attention);
        }
        return;
    }
    if (std::holds_alternative<MlpBlockSpec>(semantic_layer.mixer)) {
        return;
    }
    static_assert(std::variant_size_v<MixerSpec> == 5);
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
        {layer_intermediate, *m.hidden_size},
        {});
    add_binding(bindings, TensorRole::FfnGate, layer, *gate, {});

    const auto* up = find_unique(
        input.inventory,
        feed_forward_tensor_candidates(layer, "w_up.weight"),
        TensorRole::FfnUp,
        layer,
        {layer_intermediate, *m.hidden_size},
        {});
    add_binding(bindings, TensorRole::FfnUp, layer, *up, {});

    const auto* down = find_unique(
        input.inventory,
        feed_forward_tensor_candidates(layer, "w_down.weight"),
        TensorRole::FfnDown,
        layer,
        {*m.hidden_size, layer_intermediate},
        {});
    add_binding(bindings, TensorRole::FfnDown, layer, *down, {});
}

void bind_mlp_only(CanonicalInferenceContext& context,
                   int layer,
                   int layer_intermediate) {
    const auto& input = context.input;
    const auto& m = input.metadata;
    auto& bindings = context.facts.bindings;

    const auto* up = find_unique(
        input.inventory,
        feed_forward_tensor_candidates(layer, "w_up.weight"),
        TensorRole::FfnUp,
        layer,
        {layer_intermediate, *m.hidden_size},
        {});
    add_binding(bindings, TensorRole::FfnUp, layer, *up, {});

    const auto* down = find_unique(
        input.inventory,
        feed_forward_tensor_candidates(layer, "w_down.weight"),
        TensorRole::FfnDown,
        layer,
        {*m.hidden_size, layer_intermediate},
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

    bind(
        TensorRole::MoeRouter,
        prefix + "gate.weight",
        {context.num_experts, *m.hidden_size});
    const std::string bias_name = prefix + "gate.expert_bias";
    if (has_tensor(bias_name)) {
        bind(
            TensorRole::MoeRouterBias,
            bias_name,
            {context.num_experts});
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
    for (int expert = 0; expert < context.num_experts; ++expert) {
        const std::string expert_prefix =
            prefix + "experts." + std::to_string(expert) + ".";
        bind(
            TensorRole::MoeExpertGate,
            expert_prefix + "gate_proj.weight",
            {expert_intermediate, *m.hidden_size},
            expert);
        bind(
            TensorRole::MoeExpertUp,
            expert_prefix + "up_proj.weight",
            {expert_intermediate, *m.hidden_size},
            expert);
        bind(
            TensorRole::MoeExpertDown,
            expert_prefix + "down_proj.weight",
            {*m.hidden_size, expert_intermediate},
            expert);
    }

    if (moe->shared) {
        const int shared_intermediate = moe->shared->intermediate_size;
        const std::string shared = prefix + "shared_experts.";
        bind(
            TensorRole::MoeSharedGate,
            shared + "gate_proj.weight",
            {shared_intermediate, *m.hidden_size});
        bind(
            TensorRole::MoeSharedUp,
            shared + "up_proj.weight",
            {shared_intermediate, *m.hidden_size});
        bind(
            TensorRole::MoeSharedDown,
            shared + "down_proj.weight",
            {*m.hidden_size, shared_intermediate});
    }
}

void bind_feed_forward(CanonicalInferenceContext& context,
                       int layer,
                       std::string_view index,
                       const std::vector<std::string>& ffn_norm_candidates,
                       int layer_intermediate) {
    const auto& input = context.input;
    const auto& m = input.metadata;
    auto& facts = context.facts;
    const LayerSpec& semantic_layer =
        facts.graph.layers[static_cast<size_t>(layer)];

    if (std::holds_alternative<MlpBlockSpec>(semantic_layer.mixer)) {
        bind_mlp_only(context, layer, layer_intermediate);
        return;
    }
    if (std::holds_alternative<std::monostate>(semantic_layer.feed_forward)) return;

    const auto* ffn_norm = find_unique(
        input.inventory,
        ffn_norm_candidates,
        TensorRole::FfnInputNorm,
        layer,
        {*m.hidden_size},
        {});
    add_binding(
        facts.bindings,
        TensorRole::FfnInputNorm,
        layer,
        *ffn_norm,
        {});

    if (std::holds_alternative<MixtureOfExpertsSpec>(semantic_layer.feed_forward)) {
        bind_moe(context, layer, index);
    } else {
        bind_dense_ffn(context, layer, layer_intermediate);
    }
}

void bind_layer(CanonicalInferenceContext& context, int layer) {
    const auto& input = context.input;
    const auto& m = input.metadata;
    auto& facts = context.facts;
    const std::string index = std::to_string(layer);
    const std::string layer_prefix = "blk." + index + ".";
    const int layer_intermediate =
        context.intermediate_sizes.at(static_cast<size_t>(layer));

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
        {*m.hidden_size},
        {});
    add_binding(
        facts.bindings,
        TensorRole::AttentionInputNorm,
        layer,
        *attention_norm,
        {});

    bind_mixer(context, layer, index, layer_prefix);
    bind_feed_forward(
        context,
        layer,
        index,
        ffn_norm_candidates,
        layer_intermediate);
}

} // namespace

void bind_canonical_tensors(CanonicalInferenceContext& context) {
    bind_global_tensors(context);
    for (int layer = 0; layer < context.layer_count; ++layer) {
        bind_layer(context, layer);
    }
    context.facts.bindings =
        BindingSolver{}.solve(context.facts.bindings.values);
}

} // namespace celeg::inference_detail
