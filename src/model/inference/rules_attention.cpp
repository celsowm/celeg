#include "rules.hpp"

#include "support.hpp"

#include "celeg/model/definition.hpp"

#include <cmath>
#include <type_traits>
#include <utility>

namespace celeg::inference_detail {
namespace {

AttentionSpec make_attention(
    const NormalizedModelMetadata& metadata,
    int query_heads,
    int key_value_heads,
    int head_dim,
    bool query_key_norm) {
    AttentionSpec attention;
    attention.query_heads = query_heads;
    attention.key_value_heads = key_value_heads;
    attention.head_dim = head_dim;
    const auto optional_qk_norm = [&]() -> std::optional<NormSpec> {
        if (!query_key_norm || !std::isfinite(*metadata.core.norm_epsilon) ||
            *metadata.core.norm_epsilon <= 0.0f) {
            return std::nullopt;
        }
        return NormSpec{*metadata.core.norm_epsilon, NormWeightKind::Scale};
    };
    attention.query_norm = optional_qk_norm();
    attention.key_norm = attention.query_norm;
    attention.pattern = FullCausalPattern{};
    attention.query_scale = 1.0f;
    std::visit([&](const auto& position) {
        using T = std::decay_t<decltype(position)>;
        if constexpr (std::is_same_v<T, NoPositionEncodingSpec>) {
            attention.position = NoPositionEncodingSpec{};
        } else if constexpr (std::is_same_v<T, InferredRopePosition>) {
            RopePositionSpec rope{position.theta, position.rotary_fraction, RopeScalingSpec{}};
            rope.pairing = position.pairing;
            attention.position = rope;
        } else if constexpr (std::is_same_v<T, UnresolvedPositionEncoding>) {
            fail(ResolutionFailureKind::MissingRequiredMetadata,
                "positional encoding could not be resolved from checkpoint metadata");
        } else {
            static_assert(always_false_v<T>, "unhandled inferred position encoding alternative");
        }
    }, metadata.attention.position_encoding);
    if (*metadata.attention.xsa_projection) {
        attention.output_transform = OrthogonalizeCurrentValueSpec{
            *metadata.attention.xsa_minimum_norm_squared};
    }
    return attention;
}

std::vector<std::string> query_norm_candidates(int layer) {
    const std::string index = std::to_string(layer);
    return {
        "blk." + index + ".attn_q_norm.weight",
        "model.layers." + index + ".self_attn.q_layernorm.weight",
        "model.layers." + index + ".self_attn.q_norm.weight",
        "model.language_model.layers." + index + ".self_attn.q_norm.weight",
        "layers." + index + ".self_attn.q_norm.weight",
    };
}

std::vector<std::string> key_norm_candidates(int layer) {
    const std::string index = std::to_string(layer);
    return {
        "blk." + index + ".attn_k_norm.weight",
        "model.layers." + index + ".self_attn.k_layernorm.weight",
        "model.layers." + index + ".self_attn.k_norm.weight",
        "model.language_model.layers." + index + ".self_attn.k_norm.weight",
        "layers." + index + ".self_attn.k_norm.weight",
    };
}

const TensorInventoryEntry* find_optional_unique(
    const TensorInventory& inventory,
    const std::vector<std::string>& candidates,
    TensorRole role,
    int layer,
    int head_dim) {
    const TensorInventoryEntry* found = nullptr;
    for (const std::string& candidate : candidates) {
        if (const auto* tensor = inventory.find(candidate)) {
            if (found != nullptr) {
                fail(
                    ResolutionFailureKind::AmbiguousTensorBinding,
                    "multiple query/key norm spellings are present for " +
                        std::string(tensor_role_name(role)) + " for layer " +
                        std::to_string(layer));
            }
            found = tensor;
        }
    }
    if (found != nullptr && !shape_is(*found, {head_dim})) {
        fail(
            ResolutionFailureKind::ShapeConstraintViolation,
            "tensor " + found->name + " has a shape inconsistent with " +
                std::string(tensor_role_name(role)));
    }
    return found;
}

/// Recognizes the ordinary GQA attention grammar (q/k/v/o projections and
/// optional q/k norms) in any known spelling convention.
class StandardAttentionRule final : public ILayerInferenceRule {
public:
    std::string_view id() const override { return "standard_attention"; }
    int specificity() const override { return 2; }
    MixerFamily family() const override { return MixerFamily::Attention; }

    bool probe(const CanonicalInferenceContext& context, int layer)
        const override {
        const auto& inventory = context.input.inventory;
        for (const std::string& candidate :
             attention_tensor_candidates(layer, "q_proj.weight")) {
            if (inventory.find(candidate) != nullptr) return true;
        }
        return false;
    }

    void resolve(CanonicalInferenceContext& context, int layer)
        const override {
        const auto& input = context.input;
        const auto& m = input.metadata;
        auto& bindings = context.facts.bindings;
        LayerSpec& semantic_layer =
            context.facts.graph.layers[static_cast<size_t>(layer)];

        const auto query_heads = m.attention.query_heads.value_for(layer);
        const auto key_value_heads = m.attention.key_value_heads.value_for(layer);
        const auto explicit_head_dim = m.attention.head_dim.value_for(layer);
        if (!query_heads.has_value() || !key_value_heads.has_value()) {
            fail(
                ResolutionFailureKind::MissingRequiredMetadata,
                "automatic resolution requires attention geometry for layer " +
                    std::to_string(layer));
        }
        if (*query_heads <= 0) {
            fail(
                ResolutionFailureKind::ConflictingInferenceFacts,
                "query_heads is non-positive for layer " +
                    std::to_string(layer));
        }

        const int head_dim =
            explicit_head_dim.value_or(*m.core.hidden_size / *query_heads);
        if (*key_value_heads <= 0 ||
            *query_heads % *key_value_heads != 0 ||
            head_dim <= 0 || head_dim % 2 != 0 ||
            *key_value_heads * head_dim > *m.core.hidden_size) {
            fail(
                ResolutionFailureKind::ConflictingInferenceFacts,
                "attention head geometry is not a valid GQA layout for layer " +
                    std::to_string(layer));
        }

        const TensorInventoryEntry* query_norm = find_optional_unique(
            input.inventory, query_norm_candidates(layer),
            TensorRole::AttentionQueryNorm, layer, head_dim);
        const TensorInventoryEntry* key_norm = find_optional_unique(
            input.inventory, key_norm_candidates(layer),
            TensorRole::AttentionKeyNorm, layer, head_dim);
        const bool metadata_qk_norm = *m.attention.query_key_norm;
        const bool has_query_norm = metadata_qk_norm || query_norm != nullptr;
        const bool has_key_norm = metadata_qk_norm || key_norm != nullptr;
        if (has_query_norm != has_key_norm) {
            fail(
                ResolutionFailureKind::ConflictingInferenceFacts,
                "query/key normalization evidence is incomplete for layer " +
                    std::to_string(layer));
        }

        AttentionSpec attention = make_attention(
            m,
            *query_heads,
            *key_value_heads,
            head_dim,
            has_query_norm);

        const auto q_candidates =
            attention_tensor_candidates(layer, "q_proj.weight");
        const TensorInventoryEntry* query = nullptr;
        for (const std::string& name : q_candidates) {
            if (const auto* candidate = input.inventory.find(name)) {
                if (query != nullptr) {
                    fail(
                        ResolutionFailureKind::AmbiguousTensorBinding,
                        "multiple query projections are present for layer " +
                            std::to_string(layer));
                }
                query = candidate;
            }
        }
        if (query &&
            shape_is(
                *query,
                {2 * attention.query_width(), *m.core.hidden_size})) {
            attention.output_gate = SigmoidAttentionGateSpec{
                true,
                AttentionGateGranularity::ElementWise};
        }

        semantic_layer.mixer = std::move(attention);
        if (!layer_has_feed_forward(context, layer)) {
            semantic_layer.feed_forward = std::monostate{};
        }

        const AttentionSpec& resolved =
            std::get<AttentionSpec>(semantic_layer.mixer);
        const int query_head_count = *m.attention.query_heads.value_for(layer);
        const int key_value_head_count = *m.attention.key_value_heads.value_for(layer);
        const int layer_head_dim = m.attention.head_dim.value_for(layer).value_or(
            *m.core.hidden_size / query_head_count);

        const auto* q = find_unique(
            input.inventory,
            attention_tensor_candidates(layer, "q_proj.weight"),
            TensorRole::AttentionQuery,
            layer,
            {resolved.query_projection_width(), *m.core.hidden_size},
            {});
        add_binding(bindings, TensorRole::AttentionQuery, layer, *q, {});

        const auto* k = find_unique(
            input.inventory,
            attention_tensor_candidates(layer, "k_proj.weight"),
            TensorRole::AttentionKey,
            layer,
            {key_value_head_count * layer_head_dim, *m.core.hidden_size},
            {});
        add_binding(bindings, TensorRole::AttentionKey, layer, *k, {});

        const auto* v = find_unique(
            input.inventory,
            attention_tensor_candidates(layer, "v_proj.weight"),
            TensorRole::AttentionValue,
            layer,
            {key_value_head_count * layer_head_dim, *m.core.hidden_size},
            {});
        add_binding(bindings, TensorRole::AttentionValue, layer, *v, {});

        const auto* o = find_unique(
            input.inventory,
            attention_tensor_candidates(layer, "o_proj.weight"),
            TensorRole::AttentionOutput,
            layer,
            {*m.core.hidden_size, query_head_count * layer_head_dim},
            {});
        add_binding(bindings, TensorRole::AttentionOutput, layer, *o, {});

        if (query_norm != nullptr) {
            add_binding(
                bindings,
                TensorRole::AttentionQueryNorm,
                layer,
                *query_norm,
                {});
        }
        if (key_norm != nullptr) {
            add_binding(
                bindings,
                TensorRole::AttentionKeyNorm,
                layer,
                *key_norm,
                {});
        }
    }
};

/// Recognizes the factorized latent-attention grammar (q_a/q_b/kv_a/kv_b
/// projections with latent norms and a sigmoid output gate).
class LatentAttentionRule final : public ILayerInferenceRule {
public:
    std::string_view id() const override { return "latent_attention"; }
    int specificity() const override { return 4; }
    MixerFamily family() const override { return MixerFamily::Attention; }

    bool probe(const CanonicalInferenceContext& context, int layer)
        const override {
        return context.input.inventory.find(
                   "model.layers." + std::to_string(layer) +
                   ".attention.q_a_proj.weight") != nullptr;
    }

    void resolve(CanonicalInferenceContext& context, int layer)
        const override {
        const auto& input = context.input;
        const auto& m = input.metadata;
        LayerSpec& semantic_layer =
            context.facts.graph.layers[static_cast<size_t>(layer)];
        const std::string prefix =
            "model.layers." + std::to_string(layer) + ".attention.";

        const auto* q_a = input.inventory.find(prefix + "q_a_proj.weight");
        const auto* q_a_norm =
            input.inventory.find(prefix + "q_a_layernorm.weight");
        const auto* q_b = input.inventory.find(prefix + "q_b_proj.weight");
        const auto* kv_a =
            input.inventory.find(prefix + "kv_a_proj_with_mqa.weight");
        const auto* kv_a_norm =
            input.inventory.find(prefix + "kv_a_layernorm.weight");
        const auto* kv_b = input.inventory.find(prefix + "kv_b_proj.weight");
        const auto* out = input.inventory.find(prefix + "dense.weight");
        if (!out) {
            out = input.inventory.find(prefix + "o_proj.weight");
        }
        const auto* gate = input.inventory.find(prefix + "g_proj.weight");

        const int q_rank = m.latent_attention.query_rank.value_or(0);
        const int kv_rank = m.latent_attention.kv_rank.value_or(0);
        const int nope = m.latent_attention.query_nope_dim.value_or(0);
        const int rope = m.latent_attention.query_rope_dim.value_or(0);
        const int value_dim = m.latent_attention.value_head_dim.value_or(0);
        const int heads = *m.attention.query_heads.value_for(layer);

        const std::vector<std::int64_t> head_gate_shape = {
            heads,
            *m.core.hidden_size};
        const std::vector<std::int64_t> element_gate_shape = {
            heads * value_dim,
            *m.core.hidden_size};
        const bool head_wise_gate =
            gate && gate->shape == head_gate_shape;
        const bool element_wise_gate =
            gate && gate->shape == element_gate_shape;

        if (!q_a || !q_a_norm || !q_b || !kv_a || !kv_a_norm || !kv_b ||
            !out || !gate || q_rank <= 0 || kv_rank <= 0 ||
            nope <= 0 || rope <= 0 || value_dim <= 0 ||
            q_a->shape !=
                std::vector<std::int64_t>{q_rank, *m.core.hidden_size} ||
            q_a_norm->shape != std::vector<std::int64_t>{q_rank} ||
            q_b->shape !=
                std::vector<std::int64_t>{heads * (nope + rope), q_rank} ||
            kv_a->shape !=
                std::vector<std::int64_t>{kv_rank + rope, *m.core.hidden_size} ||
            kv_a_norm->shape != std::vector<std::int64_t>{kv_rank} ||
            kv_b->shape !=
                std::vector<std::int64_t>{
                    heads * (nope + value_dim),
                    kv_rank} ||
            out->shape !=
                std::vector<std::int64_t>{
                    *m.core.hidden_size,
                    heads * value_dim} ||
            (!head_wise_gate && !element_wise_gate)) {
            fail(
                ResolutionFailureKind::ShapeConstraintViolation,
                "factorized latent-attention tensor shapes do not agree with "
                "geometry for layer " +
                    std::to_string(layer));
        }

        AttentionSpec attention =
            make_attention(m, heads, 1, value_dim, false);
        attention.query_heads = heads;
        attention.key_value_heads = 1;
        attention.head_dim = value_dim;
        attention.query_norm = std::nullopt;
        attention.key_norm = std::nullopt;
        attention.output_gate = SigmoidAttentionGateSpec{
            false,
            head_wise_gate
                ? AttentionGateGranularity::HeadWise
                : AttentionGateGranularity::ElementWise};
        attention.state = LatentAttentionStateSpec{
            kv_rank,
            rope,
            nope,
            true,
            FactorizedLatentProjection{
                q_rank,
                value_dim,
                NormSpec{*m.core.norm_epsilon, NormWeightKind::Scale},
                NormSpec{*m.core.norm_epsilon, NormWeightKind::Scale}}};
        attention.query_scale =
            std::sqrt(static_cast<float>(value_dim) /
                      static_cast<float>(nope + rope));

        semantic_layer.mixer = std::move(attention);
        if (!layer_has_feed_forward(context, layer)) {
            semantic_layer.feed_forward = std::monostate{};
        }

        const AttentionSpec& resolved =
            std::get<AttentionSpec>(semantic_layer.mixer);
        const auto& latent = *resolved.latent_state();
        const auto& factorized = *latent.factorized_projection();
        const int query_head_count = *m.attention.query_heads.value_for(layer);
        auto& bindings = context.facts.bindings;

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
            {factorized.query_rank, *m.core.hidden_size});
        bind(
            TensorRole::AttentionLatentQueryNorm,
            "q_a_layernorm.weight",
            {factorized.query_rank});
        bind(
            TensorRole::AttentionLatentQueryExpansion,
            "q_b_proj.weight",
            {query_head_count * (latent.nope_head_dim + latent.rope_head_dim),
             factorized.query_rank});
        bind(
            TensorRole::AttentionLatentKeyProjection,
            "kv_a_proj_with_mqa.weight",
            {latent.latent_rank + latent.rope_head_dim, *m.core.hidden_size});
        bind(
            TensorRole::AttentionLatentKeyNorm,
            "kv_a_layernorm.weight",
            {latent.latent_rank});
        bind(
            TensorRole::AttentionLatentExpansion,
            "kv_b_proj.weight",
            {query_head_count * (latent.nope_head_dim + factorized.value_head_dim),
             latent.latent_rank});

        const auto* output = find_unique(
            input.inventory,
            {prefix + "dense.weight", prefix + "o_proj.weight"},
            TensorRole::AttentionLatentOutput,
            layer,
            {*m.core.hidden_size, resolved.latent_output_width()},
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
            {resolved.output_gate_width(), *m.core.hidden_size});
    }
};

}

std::unique_ptr<ILayerInferenceRule> make_standard_attention_rule() {
    return std::make_unique<StandardAttentionRule>();
}

std::unique_ptr<ILayerInferenceRule> make_latent_attention_rule() {
    return std::make_unique<LatentAttentionRule>();
}

}
