#include "celeg/model/inference.hpp"

#include "inference/canonical_internal.hpp"
#include "inference/support.hpp"

#include <sstream>
#include <utility>

namespace celeg {

ResolutionError::ResolutionError(
    ResolutionFailureKind kind,
    std::string message,
    std::vector<EvidenceItem> evidence)
    : std::runtime_error(std::move(message)),
      kind_(kind),
      evidence_(std::move(evidence)) {}

CanonicalModelFacts infer_canonical_model_facts(const InferenceInput& input) {
    auto context = inference_detail::initialize_canonical_facts(input);
    inference_detail::resolve_canonical_layers(context);
    context.facts.validate();
    return std::move(context.facts);
}

std::string CanonicalModelFacts::fingerprint() const {
    std::ostringstream out;
    out << resolution_mode << ':' << source_format << ":graph=" << graph.fingerprint()
        << ":layers=" << graph.layers.size()
        << ":vocab=" << checkpoint.vocab_size
        << ":num=" << numerical_policy.norm_eps << ':'
        << numerical_policy.post_norm_eps << ':'
        << numerical_policy.embedding_multiplier << ':'
        << numerical_policy.attention_multiplier << ':'
        << numerical_policy.residual_multiplier << ':'
        << numerical_policy.logits_multiplier << ':'
        << numerical_policy.logits_divisor << ':'
        << numerical_policy.final_logit_softcap
        << ":tied=" << tied_embeddings;
    for (const auto& binding : bindings.values) {
        out << ':' << static_cast<int>(binding.role) << ':' << binding.layer << ':'
            << binding.expert;
        for (const auto dimension : binding.shape) {
            out << ':' << dimension;
        }
    }
    return out.str();
}

void CanonicalModelFacts::validate() const {
    checkpoint.validate();
    numerical_policy.validate();
    graph.validate();
    bindings.validate();

    const auto require = [&](TensorRole role, int layer) {
        if (bindings.find(role, layer) == nullptr) {
            inference_detail::fail(
                ResolutionFailureKind::MissingTensorRole,
                "canonical facts omit required tensor role " +
                    std::string(tensor_role_name(role)));
        }
    };

    require(TensorRole::TokenEmbedding, -1);
    require(TensorRole::LanguageModelHead, -1);
    require(TensorRole::FinalNorm, -1);

    for (int layer = 0;
         layer < static_cast<int>(graph.layers.size());
         ++layer) {
        const LayerSpec& semantic_layer =
            graph.layers[static_cast<size_t>(layer)];
        const auto require_norm = [&](const std::optional<NormSpec>& norm,
                                      TensorRole role) {
            if (norm.has_value() && !norm->weightless()) require(role, layer);
        };
        require_norm(semantic_layer.mixer_norm.before, TensorRole::AttentionInputNorm);
        require_norm(semantic_layer.mixer_norm.after, TensorRole::AttentionPostNorm);

        if (const auto* attention_ptr =
                std::get_if<AttentionSpec>(&semantic_layer.mixer)) {
            const AttentionSpec& attention = *attention_ptr;
            require_norm(attention.query_norm, TensorRole::AttentionQueryNorm);
            require_norm(attention.key_norm, TensorRole::AttentionKeyNorm);
            if (attention.uses_latent_state()) {
                if (attention.latent_state()->factorized()) {
                    require(TensorRole::AttentionLatentQueryProjection, layer);
                    require(TensorRole::AttentionLatentQueryExpansion, layer);
                    require(TensorRole::AttentionLatentQueryNorm, layer);
                    require(TensorRole::AttentionLatentKeyProjection, layer);
                    require(TensorRole::AttentionLatentKeyNorm, layer);
                    require(TensorRole::AttentionLatentExpansion, layer);
                } else {
                    require(TensorRole::AttentionLatentQuery, layer);
                    require(TensorRole::AttentionLatentKey, layer);
                    require(TensorRole::AttentionLatentValue, layer);
                }
            } else {
                require(TensorRole::AttentionQuery, layer);
                require(TensorRole::AttentionKey, layer);
                require(TensorRole::AttentionValue, layer);
            }
            require(
                attention.uses_latent_state()
                    ? TensorRole::AttentionLatentOutput
                    : TensorRole::AttentionOutput,
                layer);
        } else if (std::holds_alternative<ShortConvolutionSpec>(semantic_layer.mixer)) {
            require(TensorRole::ShortConvInput, layer);
            require(TensorRole::ShortConvKernel, layer);
            require(TensorRole::ShortConvOutput, layer);
        } else if (std::holds_alternative<Mamba2Spec>(semantic_layer.mixer)) {
            require(TensorRole::Mamba2Input, layer);
            require(TensorRole::Mamba2Conv, layer);
            require(TensorRole::Mamba2ConvBias, layer);
            require(TensorRole::Mamba2DtBias, layer);
            require(TensorRole::Mamba2ALog, layer);
            require(TensorRole::Mamba2D, layer);
            require(TensorRole::Mamba2Norm, layer);
            require(TensorRole::Mamba2Output, layer);
        } else if (std::holds_alternative<GatedDeltaNetSpec>(semantic_layer.mixer)) {
            const GatedDeltaNetSpec& spec =
                std::get<GatedDeltaNetSpec>(semantic_layer.mixer);
            if (spec.factorized_projections) {
                require(TensorRole::GatedDeltaNetQuery, layer);
                require(TensorRole::GatedDeltaNetKey, layer);
                require(TensorRole::GatedDeltaNetValue, layer);
                require(TensorRole::GatedDeltaNetDecay, layer);
                require(TensorRole::GatedDeltaNetOutputGate, layer);
                require(TensorRole::GatedDeltaNetQueryConv, layer);
                require(TensorRole::GatedDeltaNetKeyConv, layer);
                require(TensorRole::GatedDeltaNetValueConv, layer);
            } else {
                require(TensorRole::GatedDeltaNetQkv, layer);
                require(TensorRole::GatedDeltaNetZ, layer);
                require(TensorRole::GatedDeltaNetAlpha, layer);
            }
            require(TensorRole::GatedDeltaNetBeta, layer);
            require(TensorRole::GatedDeltaNetDtBias, layer);
            require(TensorRole::GatedDeltaNetALog, layer);
            require(TensorRole::GatedDeltaNetNorm, layer);
            require(TensorRole::GatedDeltaNetOutput, layer);
        } else if (std::holds_alternative<MlpBlockSpec>(semantic_layer.mixer)) {
            require(TensorRole::FfnUp, layer);
            require(TensorRole::FfnDown, layer);
        } else {
            inference_detail::fail(
                ResolutionFailureKind::UnsupportedGraphPrimitive,
                "automatic resolution has no binding contract for mixer");
        }

        if (!std::holds_alternative<MlpBlockSpec>(semantic_layer.mixer) &&
            !std::holds_alternative<std::monostate>(semantic_layer.feed_forward)) {
            require_norm(semantic_layer.feed_forward_norm.before, TensorRole::FfnInputNorm);
            require_norm(semantic_layer.feed_forward_norm.after, TensorRole::FfnOutputNorm);
            if (const auto* moe_ptr =
                    std::get_if<MixtureOfExpertsSpec>(&semantic_layer.feed_forward)) {
                require(TensorRole::MoeRouter, layer);
                const auto& moe = *moe_ptr;
                for (int expert = 0; expert < moe.num_experts; ++expert) {
                    if (bindings.find(
                            TensorRole::MoeExpertGate,
                            layer,
                            expert) == nullptr ||
                        bindings.find(
                            TensorRole::MoeExpertUp,
                            layer,
                            expert) == nullptr ||
                        bindings.find(
                            TensorRole::MoeExpertDown,
                            layer,
                            expert) == nullptr) {
                        inference_detail::fail(
                            ResolutionFailureKind::MissingTensorRole,
                            "canonical facts omit an MoE expert tensor role");
                    }
                }
            } else {
                require(TensorRole::FfnGate, layer);
                require(TensorRole::FfnUp, layer);
                require(TensorRole::FfnDown, layer);
            }
        }
    }
}

}
