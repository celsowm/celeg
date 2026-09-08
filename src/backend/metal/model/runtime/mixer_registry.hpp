#pragma once

#include "celeg/backend/metal/model.hpp"
#include "celeg/model/program.hpp"
#include "layer.hpp"

#import <Metal/Metal.h>

#include <array>

namespace celeg {

struct MetalModelImplFriend;

/**
 * @brief Registry for mixer-specific encode.
 *
 * Centralizes the `MixerKind` switches that previously duplicated
 * across `execution.mm`. A new mixer touches exactly one translation
 * unit.
 */
void mixer_encode_token(
    MetalLayer& layer,
    const CompiledLayerProgram& program_layer,
    id<MTLComputeCommandEncoder> encoder,
    class MetalModel::Impl& impl,
    const std::array<int32_t, 3>* rope_position = nullptr);

void mixer_encode_batch(
    MetalLayer& layer,
    const CompiledLayerProgram& program_layer,
    id<MTLComputeCommandEncoder> encoder,
    uint32_t rows,
    uint32_t base_position,
    class MetalModel::Impl& impl);

bool mixer_supports_batch(const CompiledLayerProgram& layer) noexcept;

}
