#include "attention_detail.hpp"

#include "../rules.hpp"
#include "../support.hpp"

#include "celeg/model/definition.hpp"

#include <cmath>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace celeg::inference_detail {
namespace {

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

        /// A stated `gated_attention_proj_granularity_type` must agree with
        /// the shape-inferred granularity: a checkpoint claiming `head_wise`
        /// while carrying an element-wise `g_proj` (or vice versa) is a
        /// configuration/weight mismatch, not a resolvable model.
        if (m.latent_attention.output_gate_granularity.has_value()) {
            const std::string& stated = *m.latent_attention.output_gate_granularity;
            const bool agrees = (stated == "head_wise" && head_wise_gate) ||
                (stated == "element_wise" && element_wise_gate);
            if (!agrees) {
                fail(
                    ResolutionFailureKind::ConflictingMetadata,
                    "stated attention gate granularity disagrees with g_proj "
                    "shape for layer " +
                        std::to_string(layer));
            }
        }

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
            make_attention(m, layer, heads, 1, value_dim, false);
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
                NormSpec{*m.core.norm_epsilon, NormWeightKind::Scale}}, {}};
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

std::unique_ptr<ILayerInferenceRule> make_latent_attention_rule() {
    return std::make_unique<LatentAttentionRule>();
}

}
