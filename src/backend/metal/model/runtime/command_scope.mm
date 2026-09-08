#include "detail.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace celeg {

namespace {

/// @brief Concurrent decode encoding only when explicitly requested.
bool concurrent_decode_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("CELEG_METAL_CONCURRENT_DECODE");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

constexpr NSUInteger kGpuCounterSampleCapacity = 4096;

}

void MetalModel::Impl::begin_commands(
    id<MTLCommandBuffer>& command_buffer,
    id<MTLComputeCommandEncoder>& encoder) {
    command_buffer = [queue commandBuffer];
    if (!command_buffer) {
        throw std::runtime_error("Metal command buffer creation failed");
    }
    command_started = std::chrono::steady_clock::now();
    command_dispatches = 0;
    gpu_counter_samples = nil;
    gpu_counter_dispatches.clear();
    gpu_counter_next_sample = 0;
    gpu_profile_command_buffer = nil;
    gpu_profile_encoder = nil;

    if (metal_model_detail::dispatch_profile_mode() ==
        metal_model_detail::DispatchProfileMode::GpuStage) {
        if (@available(macOS 11.0, *)) {
            if ([device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
                id<MTLCounterSet> timestamp_set = nil;
                for (id<MTLCounterSet> counter_set in device.counterSets) {
                    if ([counter_set.name isEqualToString:MTLCommonCounterSetTimestamp]) {
                        timestamp_set = counter_set;
                        break;
                    }
                }
                if (timestamp_set) {
                    MTLCounterSampleBufferDescriptor* descriptor =
                        [[MTLCounterSampleBufferDescriptor alloc] init];
                    descriptor.counterSet = timestamp_set;
                    descriptor.storageMode = MTLStorageModeShared;
                    descriptor.sampleCount = kGpuCounterSampleCapacity;
                    NSError* counter_error = nil;
                    gpu_counter_samples = [device
                        newCounterSampleBufferWithDescriptor:descriptor
                                                     error:&counter_error];
                    if (!gpu_counter_samples) {
                        const std::string detail = counter_error
                            ? metal_model_detail::ns_string(counter_error.localizedDescription)
                            : "timestamp counter buffer creation failed";
                        throw std::runtime_error("Metal gpu-stage profile unavailable: " + detail);
                    }
                }
            }
        }
        if (!gpu_counter_samples) {
            throw std::runtime_error(
                "Metal gpu-stage profile requires timestamp sampling at stage boundaries");
        }
    }

    concurrent_decode = false;
    parallel_group = false;
    if (gpu_counter_samples) {
        gpu_profile_command_buffer = command_buffer;
        encoder = compute_encoder(nil);
    } else if (concurrent_decode_enabled()) {
        encoder = [command_buffer
            computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        concurrent_decode = encoder != nil;
        if (!concurrent_decode) {
            encoder = [command_buffer computeCommandEncoder];
        }
    } else {
        encoder = [command_buffer computeCommandEncoder];
    }
    if (!encoder) {
        throw std::runtime_error("Metal compute encoder creation failed");
    }

}

void MetalModel::Impl::finish_commands(
    id<MTLCommandBuffer>& command_buffer,
    id<MTLComputeCommandEncoder>& encoder) {
    if (gpu_counter_samples) {
        if (gpu_profile_encoder) [gpu_profile_encoder endEncoding];
    } else {
        [encoder endEncoding];
    }
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

    if (gpu_counter_samples && !gpu_counter_dispatches.empty()) {
        const NSUInteger resolved_sample_count = gpu_counter_dispatches.size() * 2;
        NSData* resolved = [gpu_counter_samples resolveCounterRange:
            NSMakeRange(0, resolved_sample_count)];
        const NSUInteger expected =
            resolved_sample_count * sizeof(MTLCounterResultTimestamp);
        if (resolved.length >= expected) {
            const auto* samples = static_cast<const MTLCounterResultTimestamp*>(resolved.bytes);
            for (size_t index = 0; index < gpu_counter_dispatches.size(); ++index) {
                const uint64_t start = samples[index * 2].timestamp;
                const uint64_t end = samples[index * 2 + 1].timestamp;
                if (start == MTLCounterErrorValue || end == MTLCounterErrorValue ||
                    end < start) {
                    continue;
                }
                metal_model_detail::record_dispatch_gpu_time(
                    gpu_counter_dispatches[index],
                    static_cast<double>(end - start) / 1.0e6);
            }
        }
    }
    gpu_counter_samples = nil;
    gpu_counter_dispatches.clear();
    gpu_counter_next_sample = 0;
    gpu_profile_command_buffer = nil;
    gpu_profile_encoder = nil;
    concurrent_decode = false;
    parallel_group = false;
    encoder = nil;
    command_buffer = nil;
}

void MetalModel::Impl::order_before_dispatch(
    id<MTLComputeCommandEncoder> encoder) {
    if (!concurrent_decode || parallel_group) return;
    [compute_encoder(encoder) memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void MetalModel::Impl::begin_parallel_group(
    id<MTLComputeCommandEncoder> encoder) {
    if (!concurrent_decode) return;
    [compute_encoder(encoder) memoryBarrierWithScope:MTLBarrierScopeBuffers];
    parallel_group = true;
}

void MetalModel::Impl::end_parallel_group() {
    parallel_group = false;
}

id<MTLComputeCommandEncoder> MetalModel::Impl::compute_encoder(
    id<MTLComputeCommandEncoder> fallback) {
    if (!gpu_counter_samples) return fallback;
    if (gpu_profile_encoder) return gpu_profile_encoder;
    if (gpu_counter_next_sample + 2 > gpu_counter_samples.sampleCount) {
        throw std::runtime_error("Metal gpu-stage profile exceeded its sample capacity");
    }
    MTLComputePassDescriptor* pass_descriptor =
        [MTLComputePassDescriptor computePassDescriptor];
    MTLComputePassSampleBufferAttachmentDescriptor* attachment =
        pass_descriptor.sampleBufferAttachments[0];
    attachment.sampleBuffer = gpu_counter_samples;
    attachment.startOfEncoderSampleIndex = gpu_counter_next_sample;
    attachment.endOfEncoderSampleIndex = gpu_counter_next_sample + 1;
    gpu_counter_next_sample += 2;
    gpu_profile_encoder =
        [gpu_profile_command_buffer computeCommandEncoderWithDescriptor:pass_descriptor];
    if (!gpu_profile_encoder) {
        throw std::runtime_error("Metal gpu-stage compute encoder creation failed");
    }
    return gpu_profile_encoder;
}

void MetalModel::Impl::record_dispatch(std::string_view name) {
    ++command_dispatches;
    metal_model_detail::record_dispatch_count(name);
    if (gpu_counter_samples && gpu_profile_encoder) {
        gpu_counter_dispatches.emplace_back(name);
        [gpu_profile_encoder endEncoding];
        gpu_profile_encoder = nil;
    }
}

void MetalModel::Impl::dispatch(id<MTLComputeCommandEncoder> encoder, std::string_view name,
                                 NSUInteger count) {
    if (count == 0) return;
    id<MTLComputePipelineState> state = pipeline(name);
    encoder = compute_encoder(encoder);
    order_before_dispatch(encoder);
    [encoder setComputePipelineState:state];
    const NSUInteger threads = std::min(count, state.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    record_dispatch(name);
}

void MetalModel::Impl::dispatch_cooperative(id<MTLComputeCommandEncoder> encoder, std::string_view name,
                                             NSUInteger groups) {
    if (groups == 0) return;
    id<MTLComputePipelineState> state = pipeline(name);
    constexpr NSUInteger threads = 256;
    if (state.maxTotalThreadsPerThreadgroup < threads) {
        throw std::runtime_error("Metal pipeline cannot run cooperative kernel");
    }
    encoder = compute_encoder(encoder);
    order_before_dispatch(encoder);
    [encoder setComputePipelineState:state];
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    record_dispatch(name);
}

void MetalModel::Impl::set_buffer(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> value,
                                   NSUInteger index, NSUInteger offset) {
    [compute_encoder(encoder) setBuffer:value offset:offset atIndex:index];
}

void MetalModel::Impl::set_bytes(id<MTLComputeCommandEncoder> encoder, const void* value,
                                  NSUInteger length, NSUInteger index) {
    [compute_encoder(encoder) setBytes:value length:length atIndex:index];
}

}
