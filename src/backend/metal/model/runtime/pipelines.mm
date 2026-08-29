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
    command_started = std::chrono::steady_clock::now();
    command_dispatches = 0;
}

void MetalModel::Impl::finish_commands(
    id<MTLCommandBuffer>& command_buffer,
    id<MTLComputeCommandEncoder>& encoder) {
    [encoder endEncoding];
    const auto encoding_finished = std::chrono::steady_clock::now();
    [command_buffer commit];
    const auto waiting_started = std::chrono::steady_clock::now();
    [command_buffer waitUntilCompleted];
    const auto completed = std::chrono::steady_clock::now();
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        const std::string message = command_buffer.error
            ? metal_model_detail::ns_string(command_buffer.error.localizedDescription)
            : "unknown Metal command-buffer error";
        throw std::runtime_error("Metal inference dispatch failed: " + message);
    }
    execution_metrics.command_encoding_ms += std::chrono::duration<double, std::milli>(
        encoding_finished - command_started).count();
    execution_metrics.command_wait_ms += std::chrono::duration<double, std::milli>(
        completed - waiting_started).count();
    const double gpu_started = command_buffer.GPUStartTime;
    const double gpu_completed = command_buffer.GPUEndTime;
    if (gpu_completed > gpu_started) {
        execution_metrics.gpu_execution_ms += (gpu_completed - gpu_started) * 1000.0;
    }
    ++execution_metrics.command_buffers;
    execution_metrics.dispatches += command_dispatches;
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
    ++command_dispatches;
}

void MetalModel::Impl::dispatch_cooperative(id<MTLComputeCommandEncoder> encoder,
                                            std::string_view name, NSUInteger groups) {
        if (groups == 0) return;
        id<MTLComputePipelineState> state = pipeline(name);
        constexpr NSUInteger threads = 256;
        if (state.maxTotalThreadsPerThreadgroup < threads) {
            throw std::runtime_error("Metal pipeline cannot run cooperative kernel");
        }
        [encoder setComputePipelineState:state];
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        ++command_dispatches;
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
        const NSUInteger groups = (weight.rows + 7u) / 8u;
        if (weight.storage == MetalModel::Impl::LinearStorage::Float32) {
            dispatch_cooperative(encoder, "celeg_matvec", groups);
            return;
        }
        if (weight.storage == MetalModel::Impl::LinearStorage::Float16) {
            dispatch_cooperative(encoder, "celeg_matvec_f16", groups);
            return;
        }
        if (weight.storage == MetalModel::Impl::LinearStorage::BFloat16) {
            dispatch_cooperative(encoder, "celeg_matvec_bf16", groups);
            return;
        }
        set_bytes(encoder, &weight.row_bytes, sizeof(weight.row_bytes), 5);
        std::string_view kernel;
        switch (weight.storage) {
            case MetalModel::Impl::LinearStorage::Q4_0: kernel = "celeg_matvec_q4_0"; break;
            case MetalModel::Impl::LinearStorage::Q4K: kernel = "celeg_matvec_q4k"; break;
            case MetalModel::Impl::LinearStorage::Q5K: kernel = "celeg_matvec_q5k"; break;
            case MetalModel::Impl::LinearStorage::Q6K: kernel = "celeg_matvec_q6k"; break;
            case MetalModel::Impl::LinearStorage::Q8_0: kernel = "celeg_matvec_q8_0"; break;
            default: throw std::runtime_error("invalid Metal linear storage");
        }
        dispatch_cooperative(encoder, kernel, groups);
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
        if (embedding.storage == MetalModel::Impl::LinearStorage::Float16) {
            set_buffer(encoder, embedding.buffer, 0);
            set_buffer(encoder, hidden, 1);
            set_bytes(encoder, &width, sizeof(width), 2);
            set_bytes(encoder, &token, sizeof(token), 3);
            dispatch(encoder, "celeg_embedding_f16", width);
            return;
        }
        if (embedding.storage == MetalModel::Impl::LinearStorage::BFloat16) {
            set_buffer(encoder, embedding.buffer, 0);
            set_buffer(encoder, hidden, 1);
            set_bytes(encoder, &width, sizeof(width), 2);
            set_bytes(encoder, &token, sizeof(token), 3);
            dispatch(encoder, "celeg_embedding_bf16", width);
            return;
        }
        set_buffer(encoder, embedding.buffer, 0);
        set_buffer(encoder, hidden, 1);
        set_bytes(encoder, &width, sizeof(width), 2);
        set_bytes(encoder, &token, sizeof(token), 3);
        std::string_view kernel;
        switch (embedding.storage) {
            case MetalModel::Impl::LinearStorage::Q4_0: kernel = "celeg_embedding_q4_0"; break;
            case MetalModel::Impl::LinearStorage::Q4K: kernel = "celeg_embedding_q4k"; break;
            case MetalModel::Impl::LinearStorage::Q5K: kernel = "celeg_embedding_q5k"; break;
            case MetalModel::Impl::LinearStorage::Q6K: kernel = "celeg_embedding_q6k"; break;
            case MetalModel::Impl::LinearStorage::Q8_0: kernel = "celeg_embedding_q8_0"; break;
            default: throw std::runtime_error("invalid Metal embedding storage");
        }
        dispatch(encoder, kernel, width);
    }


void MetalModel::Impl::encode_rmsnorm(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> input,
                        id<MTLBuffer> weight, id<MTLBuffer> output, uint32_t width,
                        float epsilon) {
        set_buffer(encoder, input, 0);
        set_buffer(encoder, weight, 1);
        set_buffer(encoder, output, 2);
        set_bytes(encoder, &width, sizeof(width), 3);
        set_bytes(encoder, &epsilon, sizeof(epsilon), 4);
        dispatch_cooperative(encoder, "celeg_rmsnorm", 1);
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
