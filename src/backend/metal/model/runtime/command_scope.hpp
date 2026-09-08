#pragma once

#include "detail.hpp"

#import <Metal/Metal.h>

#include <string_view>

namespace celeg {

/**
 * @brief Command-scope helpers for Metal encode.
 *
 * Extracted from `pipelines.mm` to give `Impl` a single
 * responsibility (SRP) and to make command lifecycle testable
 * without pulling the full matvec/matmul encode.
 */
void command_begin(
    MetalModel::Impl& impl,
    id<MTLCommandBuffer>& command_buffer,
    id<MTLComputeCommandEncoder>& encoder);

void command_finish(
    MetalModel::Impl& impl,
    id<MTLCommandBuffer>& command_buffer,
    id<MTLComputeCommandEncoder>& encoder);

void command_dispatch(
    MetalModel::Impl& impl,
    id<MTLComputeCommandEncoder> encoder,
    std::string_view name,
    NSUInteger count);

void command_dispatch_cooperative(
    MetalModel::Impl& impl,
    id<MTLComputeCommandEncoder> encoder,
    std::string_view name,
    NSUInteger groups);

id<MTLComputeCommandEncoder> command_encoder(
    MetalModel::Impl& impl,
    id<MTLComputeCommandEncoder> fallback);

void command_order_before_dispatch(
    MetalModel::Impl& impl,
    id<MTLComputeCommandEncoder> encoder);

void command_begin_parallel_group(
    MetalModel::Impl& impl,
    id<MTLComputeCommandEncoder> encoder);

void command_end_parallel_group(MetalModel::Impl& impl);

void command_record_dispatch(
    MetalModel::Impl& impl,
    std::string_view name);

void command_set_buffer(
    MetalModel::Impl& impl,
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> value,
    NSUInteger index,
    NSUInteger offset = 0);

void command_set_bytes(
    MetalModel::Impl& impl,
    id<MTLComputeCommandEncoder> encoder,
    const void* value,
    NSUInteger length,
    NSUInteger index);

}
