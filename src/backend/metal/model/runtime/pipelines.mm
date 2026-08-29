#include "detail.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace celeg {

using metal_model_detail::ns_string;

void MetalModel::Impl::begin_commands(
    id<MTLCommandBuffer>& command_buffer,
    id<MTLComputeCommandEncoder>& encoder) {
    command_buffer = [queue commandBuffer];
    if (!command_buffer) {
        throw std::runtime_error("Metal command buffer creation failed");
    }
    encoder = [command_buffer computeCommandEncoder];
    if (!encoder) {
        throw std::runtime_error("Metal compute encoder creation failed");
    }
}

void MetalModel::Impl::finish_commands(
    id<MTLCommandBuffer>& command_buffer,
    id<MTLComputeCommandEncoder>& encoder) {
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        const std::string message = command_buffer.error
            ? metal_model_detail::ns_string(command_buffer.error.localizedDescription)
            : "unknown Metal command-buffer error";
        throw std::runtime_error("Metal inference dispatch failed: " + message);
    }
    encoder = nil;
    command_buffer = nil;
}

id<MTLComputePipelineState> MetalModel::Impl::pipeline(std::string_view name) {
        const auto found = pipelines.find(std::string(name));
        if (found != pipelines.end()) return found->second;
        NSString* function_name = [NSString stringWithUTF8String:std::string(name).c_str()];
        id<MTLFunction> function = [library newFunctionWithName:function_name];
        if (!function) throw std::runtime_error("Metal inference function is missing: " + std::string(name));
        NSError* error = nil;
        id<MTLComputePipelineState> result =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!result) {
            const std::string message = error ? ns_string(error.localizedDescription)
                                              : "unknown Metal pipeline error";
            throw std::runtime_error("Metal inference pipeline failed: " + message);
        }
        pipelines.emplace(std::string(name), result);
        return result;
    }


void MetalModel::Impl::dispatch(id<MTLComputeCommandEncoder> encoder, std::string_view name,
                  NSUInteger count) {
        if (count == 0) return;
        id<MTLComputePipelineState> state = pipeline(name);
        [encoder setComputePipelineState:state];
        const NSUInteger threads = std::min(count, state.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    }


void MetalModel::Impl::set_buffer(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> value,
                    NSUInteger index, NSUInteger offset) {
        [encoder setBuffer:value offset:offset atIndex:index];
    }


void MetalModel::Impl::set_bytes(id<MTLComputeCommandEncoder> encoder, const void* value,
                   NSUInteger length, NSUInteger index) {
        [encoder setBytes:value length:length atIndex:index];
    }


void MetalModel::Impl::encode_matvec(id<MTLComputeCommandEncoder> encoder,
                       const MetalModel::Impl::Linear& weight,
                       id<MTLBuffer> input, id<MTLBuffer> output,
                       NSUInteger output_offset) {
        set_buffer(encoder, weight.buffer, 0);
        set_buffer(encoder, input, 1);
        set_buffer(encoder, output, 2, output_offset);
        set_bytes(encoder, &weight.rows, sizeof(weight.rows), 3);
        set_bytes(encoder, &weight.cols, sizeof(weight.cols), 4);
        if (weight.storage == MetalModel::Impl::LinearStorage::Float32) {
            dispatch(encoder, "celeg_matvec", weight.rows);
            return;
        }
        set_bytes(encoder, &weight.row_bytes, sizeof(weight.row_bytes), 5);
        dispatch(encoder, weight.storage == MetalModel::Impl::LinearStorage::Q4K
                           ? "celeg_matvec_q4k" : "celeg_matvec_q6k",
                 weight.rows);
    }


void MetalModel::Impl::encode_embedding(id<MTLComputeCommandEncoder> encoder, uint32_t width,
                          uint32_t token) {
        if (embedding.storage == MetalModel::Impl::LinearStorage::Float32) {
            set_buffer(encoder, embedding.buffer, 0);
            set_buffer(encoder, hidden, 1);
            set_bytes(encoder, &width, sizeof(width), 2);
            set_bytes(encoder, &token, sizeof(token), 3);
            dispatch(encoder, "celeg_embedding", width);
            return;
        }
        set_buffer(encoder, embedding.buffer, 0);
        set_buffer(encoder, hidden, 1);
        set_bytes(encoder, &width, sizeof(width), 2);
        set_bytes(encoder, &token, sizeof(token), 3);
        dispatch(encoder, embedding.storage == MetalModel::Impl::LinearStorage::Q4K
                           ? "celeg_embedding_q4k" : "celeg_embedding_q6k",
                 width);
    }


void MetalModel::Impl::encode_rmsnorm(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> input,
                        id<MTLBuffer> weight, id<MTLBuffer> output, uint32_t width,
                        float epsilon) {
        set_buffer(encoder, input, 0);
        set_buffer(encoder, weight, 1);
        set_buffer(encoder, output, 2);
        set_bytes(encoder, &width, sizeof(width), 3);
        set_bytes(encoder, &epsilon, sizeof(epsilon), 4);
        dispatch(encoder, "celeg_rmsnorm", 1);
    }


void MetalModel::Impl::encode_weighted_add(id<MTLComputeCommandEncoder> encoder,
                             id<MTLBuffer> input, id<MTLBuffer> output,
                             uint32_t count, float weight) {
        set_buffer(encoder, input, 0);
        set_buffer(encoder, output, 1);
        set_bytes(encoder, &count, sizeof(count), 2);
        set_bytes(encoder, &weight, sizeof(weight), 3);
        dispatch(encoder, "celeg_weighted_add", count);
    }


}
