#include "mixer_registry.hpp"

#include "detail.hpp"

#include <stdexcept>

namespace celeg {

void mixer_encode_token(
    MetalLayer& layer,
    const CompiledLayerProgram& program_layer,
    id<MTLComputeCommandEncoder> encoder,
    MetalModel::Impl& impl,
    const std::array<int32_t, 3>* rope_position) {
    switch (layer.mixer_kind) {
        case MetalLayer::MixerKind::ShortConvolution:
            impl.encode_short_convolution(encoder, layer);
            break;
        case MetalLayer::MixerKind::GatedDelta:
            impl.encode_gated_delta_layer(encoder, layer);
            break;
        case MetalLayer::MixerKind::Mamba2:
            impl.encode_mamba2_layer(encoder, layer);
            break;
        case MetalLayer::MixerKind::Attention:
            impl.encode_attention(
                encoder, layer, std::get<CompiledAttentionProgram>(program_layer.mixer),
                rope_position);
            break;
    }
}

void mixer_encode_batch(
    MetalLayer& layer,
    const CompiledLayerProgram& program_layer,
    id<MTLComputeCommandEncoder> encoder,
    uint32_t rows,
    uint32_t base_position,
    MetalModel::Impl& impl) {
    switch (layer.mixer_kind) {
        case MetalLayer::MixerKind::ShortConvolution:
            impl.encode_short_convolution_batch(encoder, layer, rows, base_position);
            break;
        case MetalLayer::MixerKind::Attention:
            impl.encode_attention_batch(
                encoder, layer, std::get<CompiledAttentionProgram>(program_layer.mixer), rows,
                base_position);
            break;
        case MetalLayer::MixerKind::GatedDelta:
        case MetalLayer::MixerKind::Mamba2:
            throw std::logic_error("recurrent Metal mixer reached batched prefill");
    }
}

bool mixer_supports_batch(const CompiledLayerProgram& layer) noexcept {
    if (std::holds_alternative<GatedDeltaNetSpec>(layer.mixer) ||
        std::holds_alternative<Mamba2Spec>(layer.mixer) ||
        std::holds_alternative<MlpBlockSpec>(layer.mixer)) {
        return false;
    }
    if (const auto* attention = std::get_if<CompiledAttentionProgram>(&layer.mixer)) {
        const bool supported_pattern =
            std::holds_alternative<FullCausalPattern>(attention->semantics.pattern) ||
            std::holds_alternative<SlidingWindowPattern>(attention->semantics.pattern);
        const bool supported_bias =
            std::holds_alternative<NoAttentionBiasSpec>(attention->semantics.bias) ||
            std::holds_alternative<AlibiBiasSpec>(attention->semantics.bias) ||
            std::holds_alternative<RelativePositionBiasSpec>(attention->semantics.bias);
        const bool supported_output_transform =
            std::holds_alternative<NoAttentionOutputTransformSpec>(attention->semantics.output_transform) ||
            std::holds_alternative<OrthogonalizeCurrentValueSpec>(attention->semantics.output_transform);
        if (!std::holds_alternative<OrdinaryKvStateSpec>(attention->semantics.state) ||
            !supported_pattern || !supported_bias || !supported_output_transform) {
            return false;
        }
    }
    if (std::holds_alternative<MoeLayerProgram>(layer.feed_forward)) return false;
    return true;
}

}
