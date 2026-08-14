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
    inference_detail::infer_layer_semantics(context);
    inference_detail::bind_canonical_tensors(context);
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
            << binding.expert << ':' << binding.source_name;
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
        const MixerKind mixer = semantic_layer.mixer_kind();
        require(TensorRole::AttentionInputNorm, layer);

        if (mixer == MixerKind::Attention) {
            const AttentionSpec& attention =
                std::get<AttentionSpec>(semantic_layer.mixer);
            if (attention.uses_latent_state()) {
                if (attention.latent_state()->factorized) {
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
        } else if (mixer == MixerKind::ShortConvolution) {
            require(TensorRole::ShortConvInput, layer);
            require(TensorRole::ShortConvKernel, layer);
            require(TensorRole::ShortConvOutput, layer);
        } else if (mixer == MixerKind::Mamba2) {
            require(TensorRole::Mamba2Input, layer);
            require(TensorRole::Mamba2Conv, layer);
            require(TensorRole::Mamba2ConvBias, layer);
            require(TensorRole::Mamba2DtBias, layer);
            require(TensorRole::Mamba2ALog, layer);
            require(TensorRole::Mamba2D, layer);
            require(TensorRole::Mamba2Norm, layer);
            require(TensorRole::Mamba2Output, layer);
        } else if (mixer == MixerKind::GatedDeltaNet) {
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
        } else if (mixer == MixerKind::MlpOnly) {
            require(TensorRole::FfnUp, layer);
            require(TensorRole::FfnDown, layer);
        } else {
            inference_detail::fail(
                ResolutionFailureKind::UnsupportedGraphPrimitive,
                "automatic resolution has no binding contract for mixer");
        }

        if (mixer != MixerKind::MlpOnly &&
            semantic_layer.execute_feed_forward) {
            require(TensorRole::FfnInputNorm, layer);
            if (semantic_layer.feed_forward_kind() ==
                FeedForwardKind::MixtureOfExperts) {
                require(TensorRole::MoeRouter, layer);
                const auto& moe = std::get<MixtureOfExpertsSpec>(
                    semantic_layer.feed_forward);
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

} // namespace celeg
