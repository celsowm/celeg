#include "rules.hpp"

#include "support.hpp"

#include <string>

namespace celeg::inference_detail {
namespace {

/// Recognizes the short-convolution grammar: in/conv/out projections under
/// a conv prefix in any known spelling convention.
class ShortConvolutionRule final : public ILayerInferenceRule {
public:
    std::string_view id() const override { return "short_convolution"; }
    int specificity() const override { return 1; }
    MixerFamily family() const override { return MixerFamily::Convolutional; }

    bool probe(const CanonicalInferenceContext& context, int layer)
        const override {
        const auto& inventory = context.input.inventory;
        for (const std::string& candidate :
             shortconv_tensor_candidates(layer, "in_proj.weight")) {
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

        if (!m.short_conv.cache_length.has_value() || *m.short_conv.cache_length <= 0) {
            fail(
                ResolutionFailureKind::MissingRequiredMetadata,
                "short-convolution layer has no positive cache length");
        }
        const ShortConvolutionSpec spec{*m.short_conv.cache_length, *m.core.hidden_size,
                                        false};
        semantic_layer.mixer = spec;
        if (!layer_has_feed_forward(context, layer)) {
            semantic_layer.feed_forward = std::monostate{};
        }

        const auto* input_projection = find_unique(
            input.inventory,
            shortconv_tensor_candidates(layer, "in_proj.weight"),
            TensorRole::ShortConvInput,
            layer,
            {3 * *m.core.hidden_size, *m.core.hidden_size},
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
            {*m.core.hidden_size, 1, spec.cache_length},
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
            {*m.core.hidden_size, *m.core.hidden_size},
            {});
        add_binding(
            bindings,
            TensorRole::ShortConvOutput,
            layer,
            *output_projection,
            {});
    }
};

/// Recognizes the MLP-only fallback grammar: no mixer tensors at all, but
/// the layer carries feed-forward projections that become the mixer's
/// up/down blocks.
class MlpOnlyRule final : public ILayerInferenceRule {
public:
    std::string_view id() const override { return "mlp_only"; }
    int specificity() const override { return 0; }
    MixerFamily family() const override { return MixerFamily::PureMlp; }

    bool probe(const CanonicalInferenceContext& context, int layer)
        const override {
        return layer_has_feed_forward(context, layer);
    }

    void resolve(CanonicalInferenceContext& context, int layer)
        const override {
        const auto& input = context.input;
        const auto& m = input.metadata;
        auto& bindings = context.facts.bindings;
        LayerSpec& semantic_layer =
            context.facts.graph.layers[static_cast<size_t>(layer)];

        const int layer_intermediate =
            context.intermediate_sizes.at(static_cast<size_t>(layer));
        semantic_layer.mixer =
            MlpBlockSpec{layer_intermediate, ActivationKind::Relu2};
        semantic_layer.feed_forward = std::monostate{};

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
};

}

std::unique_ptr<ILayerInferenceRule> make_short_convolution_rule() {
    return std::make_unique<ShortConvolutionRule>();
}

std::unique_ptr<ILayerInferenceRule> make_mlp_only_rule() {
    return std::make_unique<MlpOnlyRule>();
}

std::vector<std::unique_ptr<ILayerInferenceRule>>
make_builtin_layer_inference_rules() {
    std::vector<std::unique_ptr<ILayerInferenceRule>> rules;
    rules.push_back(make_fused_gated_delta_rule());
    rules.push_back(make_linear_attn_gated_delta_rule());
    rules.push_back(make_factorized_gated_delta_rule());
    rules.push_back(make_latent_attention_rule());
    rules.push_back(make_mamba2_rule());
    rules.push_back(make_standard_attention_rule());
    rules.push_back(make_short_convolution_rule());
    rules.push_back(make_mlp_only_rule());
    return rules;
}

const ILayerInferenceRule& select_layer_inference_rule(
    std::span<const std::unique_ptr<ILayerInferenceRule>> rules,
    const CanonicalInferenceContext& context,
    int layer) {
    const ILayerInferenceRule* best = nullptr;
    bool attention_matched = false;
    bool recurrent_matched = false;
    for (const auto& rule : rules) {
        if (!rule->probe(context, layer)) continue;
        switch (rule->family()) {
            case MixerFamily::Attention:
                attention_matched = true;
                break;
            case MixerFamily::Recurrent:
                recurrent_matched = true;
                break;
            default:
                break;
        }
        if (best == nullptr) {
            best = rule.get();
            continue;
        }
        if (rule->specificity() == best->specificity()) {
            fail(
                ResolutionFailureKind::ConflictingInferenceFacts,
                "layer " + std::to_string(layer) +
                    " matched two inference rules at the same specificity: " +
                    std::string(best->id()) + " and " + std::string(rule->id()));
        }
        if (rule->specificity() > best->specificity()) {
            best = rule.get();
        }
    }
    if (attention_matched && recurrent_matched) {
        fail(
            ResolutionFailureKind::ConflictingInferenceFacts,
            "layer has mutually exclusive attention and recurrent tensor "
            "grammars: " +
                std::to_string(layer));
    }
    if (best == nullptr) {
        fail(
            ResolutionFailureKind::UnsupportedGraphPrimitive,
            "layer has no recognized mixer or FFN tensor grammar: " +
                std::to_string(layer));
    }
    return *best;
}

}
