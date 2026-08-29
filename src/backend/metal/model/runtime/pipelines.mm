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

id<MTLComputePipelineState> MetalModel::Impl::tensor_pipeline(std::string_view name) {
        const auto found = pipelines.find(std::string("tensor:") + std::string(name));
        if (found != pipelines.end()) return found->second;
        if (!tensor_library) throw std::runtime_error("Metal tensor library is unavailable");
        NSString* function_name = [NSString stringWithUTF8String:std::string(name).c_str()];
        id<MTLFunction> function = [tensor_library newFunctionWithName:function_name];
        if (!function) throw std::runtime_error("Metal tensor function is missing: " + std::string(name));
        NSError* error = nil;
        id<MTLComputePipelineState> result =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!result) {
            const std::string message = error ? ns_string(error.localizedDescription)
                                              : "unknown Metal tensor pipeline error";
            throw std::runtime_error("Metal tensor pipeline failed: " + message);
        }
        pipelines.emplace(std::string("tensor:") + std::string(name), result);
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

bool MetalModel::Impl::encode_matvec_pair(
    id<MTLComputeCommandEncoder> encoder, const Linear& first,
    const Linear& second, id<MTLBuffer> input, id<MTLBuffer> output,
    uint32_t width) {
    if (first.storage != second.storage || first.rows != second.rows ||
        first.cols != second.cols || first.rows != width) {
        return false;
    }
    set_buffer(encoder, first.buffer, 0);
    set_buffer(encoder, second.buffer, 1);
    set_buffer(encoder, input, 2);
    set_buffer(encoder, output, 3);
    set_bytes(encoder, &first.rows, sizeof(first.rows), 4);
    set_bytes(encoder, &first.cols, sizeof(first.cols), 5);
    if (first.storage == LinearStorage::Float32) {
        dispatch_cooperative(encoder, "celeg_matvec_pair", (width + 3u) / 4u);
        return true;
    }
    if (first.storage == LinearStorage::Float16) {
        dispatch_cooperative(encoder, "celeg_matvec_pair_f16", (width + 3u) / 4u);
        return true;
    }
    if (first.storage == LinearStorage::BFloat16) {
        dispatch_cooperative(encoder, "celeg_matvec_pair_bf16", (width + 3u) / 4u);
        return true;
    }
    if (first.row_bytes != second.row_bytes) return false;
    set_bytes(encoder, &first.row_bytes, sizeof(first.row_bytes), 6);
    std::string_view kernel;
    switch (first.storage) {
        case LinearStorage::Q4_0: kernel = "celeg_matvec_pair_q4_0"; break;
        case LinearStorage::Q4K: kernel = "celeg_matvec_pair_q4k"; break;
        case LinearStorage::Q5K: kernel = "celeg_matvec_pair_q5k"; break;
        case LinearStorage::Q6K: kernel = "celeg_matvec_pair_q6k"; break;
        case LinearStorage::Q8_0: kernel = "celeg_matvec_pair_q8_0"; break;
        default: return false;
    }
    dispatch_cooperative(encoder, kernel, (width + 3u) / 4u);
    return true;
}

void MetalModel::Impl::encode_matmul(id<MTLComputeCommandEncoder> encoder,
                                     const Linear& weight, id<MTLBuffer> input,
                                     id<MTLBuffer> output, uint32_t rows,
                                     NSUInteger input_offset, NSUInteger output_offset,
                                     uint32_t output_stride) {
    set_buffer(encoder, weight.buffer, 0);
    set_buffer(encoder, input, 1, input_offset);
    set_buffer(encoder, output, 2, output_offset);
    set_bytes(encoder, &rows, sizeof(rows), 3);
    set_bytes(encoder, &weight.cols, sizeof(weight.cols), 4);
    set_bytes(encoder, &weight.rows, sizeof(weight.rows), 5);
    const uint32_t stride = output_stride == 0 ? weight.rows : output_stride;
    set_bytes(encoder, &stride, sizeof(stride), 6);
    if (weight.storage == LinearStorage::Float32) {
        const NSUInteger groups = (weight.rows + 7u) / 8u;
        id<MTLComputePipelineState> state = pipeline("celeg_matmul");
        [encoder setComputePipelineState:state];
        [encoder dispatchThreadgroups:MTLSizeMake(groups, rows, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ++command_dispatches;
        return;
    }
    if (weight.storage == LinearStorage::Float16 || weight.storage == LinearStorage::BFloat16) {
        const bool use_tensor = weight.storage == LinearStorage::Float16
            ? tensor_matmul_f16 : tensor_matmul_bf16;
        if (use_tensor) {
            set_buffer(encoder, weight.buffer, 0);
            set_buffer(encoder, input, 1, input_offset);
            set_buffer(encoder, output, 2, output_offset);
            set_bytes(encoder, &rows, sizeof(rows), 3);
            set_bytes(encoder, &weight.cols, sizeof(weight.cols), 4);
            set_bytes(encoder, &weight.rows, sizeof(weight.rows), 5);
            const uint32_t stride = output_stride == 0 ? weight.rows : output_stride;
            set_bytes(encoder, &stride, sizeof(stride), 6);
            const std::string_view kernel = weight.storage == LinearStorage::Float16
                ? "celeg_matmul_tensor_f16" : "celeg_matmul_tensor_bf16";
            id<MTLComputePipelineState> state = tensor_pipeline(kernel);
            [encoder setComputePipelineState:state];
            [encoder dispatchThreadgroups:MTLSizeMake(
                (weight.rows + 63u) / 64u, (rows + 127u) / 128u, 1)
               threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            ++command_dispatches;
            return;
        }
        const NSUInteger groups = (weight.rows + 7u) / 8u;
        const std::string_view kernel = weight.storage == LinearStorage::Float16
            ? "celeg_matmul_f16" : "celeg_matmul_bf16";
        id<MTLComputePipelineState> state = pipeline(kernel);
        [encoder setComputePipelineState:state];
        [encoder dispatchThreadgroups:MTLSizeMake(groups, rows, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ++command_dispatches;
        return;
    }
    set_bytes(encoder, &weight.row_bytes, sizeof(weight.row_bytes), 7);
    std::string_view kernel;
    switch (weight.storage) {
        case LinearStorage::Q4_0: kernel = "celeg_matmul_q4_0"; break;
        case LinearStorage::Q4K: kernel = "celeg_matmul_q4k"; break;
        case LinearStorage::Q5K: kernel = "celeg_matmul_q5k"; break;
        case LinearStorage::Q6K: kernel = "celeg_matmul_q6k"; break;
        case LinearStorage::Q8_0: kernel = "celeg_matmul_q8_0"; break;
        default: throw std::runtime_error("invalid Metal linear storage");
    }
    const NSUInteger groups = (weight.rows + 7u) / 8u;
    id<MTLComputePipelineState> state = pipeline(kernel);
    [encoder setComputePipelineState:state];
    [encoder dispatchThreadgroups:MTLSizeMake(groups, rows, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ++command_dispatches;
}

bool MetalModel::Impl::encode_matmul_pair(
    id<MTLComputeCommandEncoder> encoder, const Linear& first,
    const Linear& second, id<MTLBuffer> input, id<MTLBuffer> output,
    uint32_t rows, uint32_t width) {
    if (first.storage != second.storage || first.rows != second.rows ||
        first.cols != second.cols || first.rows != width) {
        return false;
    }
    set_buffer(encoder, first.buffer, 0);
    set_buffer(encoder, second.buffer, 1);
    set_buffer(encoder, input, 2);
    set_buffer(encoder, output, 3);
    set_bytes(encoder, &first.rows, sizeof(first.rows), 4);
    set_bytes(encoder, &first.cols, sizeof(first.cols), 5);
    const uint32_t stride = width * 2;
    set_bytes(encoder, &stride, sizeof(stride), 6);
    auto dispatch_pair = [&](std::string_view name) {
        id<MTLComputePipelineState> state = pipeline(name);
        [encoder setComputePipelineState:state];
        [encoder dispatchThreadgroups:MTLSizeMake((width + 3u) / 4u, rows, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ++command_dispatches;
    };
    if (first.storage == LinearStorage::Float32) {
        dispatch_pair("celeg_matmul_pair");
        return true;
    }
    if (first.storage == LinearStorage::Float16) {
        dispatch_pair("celeg_matmul_pair_f16");
        return true;
    }
    if (first.storage == LinearStorage::BFloat16) {
        dispatch_pair("celeg_matmul_pair_bf16");
        return true;
    }
    if (first.row_bytes != second.row_bytes) return false;
    set_bytes(encoder, &first.row_bytes, sizeof(first.row_bytes), 7);
    std::string_view kernel;
    switch (first.storage) {
        case LinearStorage::Q4_0: kernel = "celeg_matmul_pair_q4_0"; break;
        case LinearStorage::Q4K: kernel = "celeg_matmul_pair_q4k"; break;
        case LinearStorage::Q5K: kernel = "celeg_matmul_pair_q5k"; break;
        case LinearStorage::Q6K: kernel = "celeg_matmul_pair_q6k"; break;
        case LinearStorage::Q8_0: kernel = "celeg_matmul_pair_q8_0"; break;
        default: return false;
    }
    dispatch_pair(kernel);
    return true;
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

void MetalModel::Impl::encode_embedding_batch(
    id<MTLComputeCommandEncoder> encoder, uint32_t rows,
    const std::vector<int32_t>& tokens) {
    const uint32_t width = static_cast<uint32_t>(model.graph.hidden);
    auto* values = static_cast<uint32_t*>(batch_tokens.contents);
    for (uint32_t index = 0; index < rows; ++index) {
        values[index] = static_cast<uint32_t>(tokens[index]);
    }
    set_buffer(encoder, embedding.buffer, 0);
    set_buffer(encoder, batch_hidden, 1);
    set_bytes(encoder, &width, sizeof(width), 2);
    set_buffer(encoder, batch_tokens, 3);
    std::string_view kernel;
    switch (embedding.storage) {
        case LinearStorage::Float32: kernel = "celeg_embedding_batch"; break;
        case LinearStorage::Float16: kernel = "celeg_embedding_f16_batch"; break;
        case LinearStorage::BFloat16: kernel = "celeg_embedding_bf16_batch"; break;
        case LinearStorage::Q4_0: kernel = "celeg_embedding_q4_0_batch"; break;
        case LinearStorage::Q4K: kernel = "celeg_embedding_q4k_batch"; break;
        case LinearStorage::Q5K: kernel = "celeg_embedding_q5k_batch"; break;
        case LinearStorage::Q6K: kernel = "celeg_embedding_q6k_batch"; break;
        case LinearStorage::Q8_0: kernel = "celeg_embedding_q8_0_batch"; break;
        default: throw std::runtime_error("invalid Metal embedding storage");
    }
    dispatch(encoder, kernel, static_cast<NSUInteger>(rows) * width);
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

void MetalModel::Impl::encode_rmsnorm_save(id<MTLComputeCommandEncoder> encoder,
                                           id<MTLBuffer> input, id<MTLBuffer> residual,
                                           id<MTLBuffer> weight, id<MTLBuffer> output,
                                           uint32_t width, float epsilon) {
    set_buffer(encoder, input, 0);
    set_buffer(encoder, residual, 1);
    set_buffer(encoder, weight, 2);
    set_buffer(encoder, output, 3);
    set_bytes(encoder, &width, sizeof(width), 4);
    set_bytes(encoder, &epsilon, sizeof(epsilon), 5);
    dispatch_cooperative(encoder, "celeg_rmsnorm_save", 1);
}

void MetalModel::Impl::encode_rmsnorm_batch(id<MTLComputeCommandEncoder> encoder,
                                            id<MTLBuffer> input, id<MTLBuffer> weight,
                                            id<MTLBuffer> output, uint32_t rows,
                                            uint32_t width, float epsilon) {
    set_buffer(encoder, input, 0);
    set_buffer(encoder, weight, 1);
    set_buffer(encoder, output, 2);
    set_bytes(encoder, &rows, sizeof(rows), 3);
    set_bytes(encoder, &width, sizeof(width), 4);
    set_bytes(encoder, &epsilon, sizeof(epsilon), 5);
    id<MTLComputePipelineState> state = pipeline("celeg_rmsnorm_batch");
    [encoder setComputePipelineState:state];
    [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ++command_dispatches;
}

void MetalModel::Impl::encode_residual_batch(id<MTLComputeCommandEncoder> encoder,
                                             id<MTLBuffer> input, id<MTLBuffer> residual,
                                             id<MTLBuffer> output, uint32_t count,
                                             float multiplier) {
    set_buffer(encoder, input, 0);
    set_buffer(encoder, residual, 1);
    set_buffer(encoder, output, 2);
    set_bytes(encoder, &count, sizeof(count), 3);
    set_bytes(encoder, &multiplier, sizeof(multiplier), 4);
    dispatch(encoder, "celeg_residual_batch", count);
}

void MetalModel::Impl::encode_swiglu_batch(id<MTLComputeCommandEncoder> encoder,
                                           id<MTLBuffer> input, id<MTLBuffer> output,
                                           uint32_t rows, uint32_t width) {
    set_buffer(encoder, input, 0);
    set_buffer(encoder, output, 1);
    set_bytes(encoder, &rows, sizeof(rows), 2);
    set_bytes(encoder, &width, sizeof(width), 3);
    dispatch(encoder, "celeg_swiglu_batch", static_cast<NSUInteger>(rows) * width);
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
