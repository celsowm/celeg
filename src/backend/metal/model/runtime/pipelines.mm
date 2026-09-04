#include "detail.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace celeg {

using metal_model_detail::ns_string;

namespace {

/// @brief Weight-tile geometry; must match the constants in `tensor.metal`.
constexpr NSUInteger kTensorTileRows = 64;
constexpr NSUInteger kTensorTileTokens = 128;
constexpr NSUInteger kTensorTileK = 64;
constexpr NSUInteger kTensorTileThreads = 128;
constexpr NSUInteger kQ4KStrictStageK = 128;
constexpr NSUInteger kTensorTileBytes =
    kTensorTileRows * kTensorTileK * sizeof(uint16_t);
constexpr NSUInteger kQ4KStrictStageBytes =
    kTensorTileRows * kQ4KStrictStageK * sizeof(uint16_t);
constexpr NSUInteger kGpuCounterSampleCapacity = 4096;
constexpr std::string_view kQ4KStrictKernel =
    "celeg_matmul_tensor_q4k_static_stage128";
constexpr std::string_view kF16RelaxedKernel =
    "celeg_matmul_tensor_f16_fast";
constexpr std::string_view kBF16RelaxedKernel =
    "celeg_matmul_tensor_bf16_fast";
constexpr std::string_view kQ40RelaxedKernel =
    "celeg_matmul_tensor_q4_0_relaxed";
constexpr std::string_view kQ4KRelaxedKernel =
    "celeg_matmul_tensor_q4k_relaxed";
constexpr std::string_view kQ5KRelaxedKernel =
    "celeg_matmul_tensor_q5k_relaxed";
constexpr std::string_view kQ6KRelaxedKernel =
    "celeg_matmul_tensor_q6k_fast";
constexpr std::string_view kQ80RelaxedKernel =
    "celeg_matmul_tensor_q8_0_relaxed";
constexpr std::string_view kF16RelaxedN32Kernel =
    "celeg_matmul_tensor_f16_fast_n32";
constexpr std::string_view kBF16RelaxedN32Kernel =
    "celeg_matmul_tensor_bf16_fast_n32";
constexpr std::string_view kQ40RelaxedN32Kernel =
    "celeg_matmul_tensor_q4_0_relaxed_n32";
constexpr std::string_view kQ4KRelaxedN32Kernel =
    "celeg_matmul_tensor_q4k_relaxed_n32";
constexpr std::string_view kQ5KRelaxedN32Kernel =
    "celeg_matmul_tensor_q5k_relaxed_n32";
constexpr std::string_view kQ6KRelaxedN32Kernel =
    "celeg_matmul_tensor_q6k_fast_n32";
constexpr std::string_view kQ6KStrictFastKernel =
    "celeg_matmul_tensor_q6k_fast_strict";
constexpr std::string_view kQ6KStrictFastN32Kernel =
    "celeg_matmul_tensor_q6k_fast_strict_n32";
constexpr std::string_view kQ80RelaxedN32Kernel =
    "celeg_matmul_tensor_q8_0_relaxed_n32";
}

std::optional<std::string_view> MetalModel::Impl::linear_kernel(
    LinearStorage storage, LinearOperationKind operation) const {
    static constexpr const char* kGeneric[8][8] = {
        {nullptr, nullptr, "celeg_matmul", nullptr, nullptr,
         "celeg_embedding", "celeg_embedding_batch", nullptr},
        {nullptr, nullptr, "celeg_matmul_f16", nullptr,
         "celeg_matmul_tensor_f16", "celeg_embedding_f16", "celeg_embedding_f16_batch", nullptr},
        {nullptr, nullptr, "celeg_matmul_bf16", nullptr,
         "celeg_matmul_tensor_bf16", "celeg_embedding_bf16", "celeg_embedding_bf16_batch", nullptr},
        {nullptr, nullptr, "celeg_matmul_q4_0", nullptr, "celeg_matmul_tensor_q4_0",
         "celeg_embedding_q4_0", "celeg_embedding_q4_0_batch", nullptr},
        {nullptr, nullptr, "celeg_matmul_q4k", nullptr, "celeg_matmul_tensor_q4k",
         "celeg_embedding_q4k", "celeg_embedding_q4k_batch", nullptr},
        {nullptr, nullptr, "celeg_matmul_q5k", nullptr, "celeg_matmul_tensor_q5k",
         "celeg_embedding_q5k", "celeg_embedding_q5k_batch", nullptr},
        {nullptr, nullptr, "celeg_matmul_q6k", nullptr, "celeg_matmul_tensor_q6k",
         "celeg_embedding_q6k", "celeg_embedding_q6k_batch", nullptr},
        {nullptr, nullptr, "celeg_matmul_q8_0", nullptr, "celeg_matmul_tensor_q8_0",
         "celeg_embedding_q8_0", "celeg_embedding_q8_0_batch", nullptr},
    };
    const auto storage_index = static_cast<std::size_t>(storage);
    const auto operation_index = static_cast<std::size_t>(operation);
    if (storage_index >= 8 || operation_index >= 8) return std::nullopt;
    const char* name = kGeneric[storage_index][operation_index];
    if (name == nullptr) return std::nullopt;
    return std::string_view{name};
}

MetalMatvecKernel MetalModel::Impl::matvec_kernel(LinearStorage storage,
                                                  uint32_t rows,
                                                  uint32_t cols) const {
    const bool ffn_expansion = rows >= 4096 && rows < 32768 && cols <= 2048;
    const bool ffn_contraction = rows <= 2048 && cols >= 4096 && cols < 32768;
    const bool m5_fast = options.numerical_policy == MetalNumericalPolicy::Fast &&
        ns_string(device.name).find("Apple M5") != std::string::npos;
    switch (storage) {
        case LinearStorage::Float32: return {"celeg_matvec", 8, 256, 0};
        case LinearStorage::Float16: return {"celeg_matvec_f16", 2, 128, 8};
        case LinearStorage::BFloat16: return {"celeg_matvec_bf16", 2, 128, 8};
        case LinearStorage::Q4_0: return {"celeg_matvec_q4_0", 16, 128, 0};
        case LinearStorage::Q4K: return ffn_expansion
            ? MetalMatvecKernel{"celeg_matvec_q4k_rows8", 32, 128, 0}
            : MetalMatvecKernel{"celeg_matvec_q4k", 16, 128, 0};
        case LinearStorage::Q5K: return ffn_expansion
            ? MetalMatvecKernel{"celeg_matvec_q5k_rows8", 32, 128, 0}
            : MetalMatvecKernel{"celeg_matvec_q5k", 16, 128, 0};
        case LinearStorage::Q6K: return {"celeg_matvec_q6k", 16, 128, 0};
        case LinearStorage::Q8_0: return m5_fast
            ? MetalMatvecKernel{"celeg_matvec_q8_0_m5", 2, 128, 8}
            : ffn_expansion || ffn_contraction
                ? MetalMatvecKernel{"celeg_matvec_q8_0_rows8", 32, 128, 0}
            : MetalMatvecKernel{"celeg_matvec_q8_0", 16, 128, 0};
    }
    return {};
}

MetalMatvecKernel MetalModel::Impl::swiglu_matvec_kernel(LinearStorage storage,
                                                         uint32_t rows,
                                                         uint32_t cols) const {
    const bool m5_fast = options.numerical_policy == MetalNumericalPolicy::Fast &&
        ns_string(device.name).find("Apple M5") != std::string::npos;
    switch (storage) {
        case LinearStorage::Q4_0: return {"celeg_swiglu_matvec_q4_0", 16, 128, 0};
        case LinearStorage::Q4K: return {"celeg_swiglu_matvec_q4k", 16, 128, 0};
        case LinearStorage::Q5K: return {"celeg_swiglu_matvec_q5k", 16, 128, 0};
        case LinearStorage::Q6K: return {"celeg_swiglu_matvec_q6k", 16, 128, 0};
        case LinearStorage::Q8_0: return rows <= 2048 && cols >= 4096 && cols < 32768
            ? m5_fast
                ? MetalMatvecKernel{"celeg_swiglu_matvec_q8_0_m5", 2, 128, 8}
                : MetalMatvecKernel{"celeg_swiglu_matvec_q8_0_rows8", 32, 128, 0}
            : MetalMatvecKernel{};
        default: return {};
    }
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
                            ? ns_string(counter_error.localizedDescription)
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

    if (gpu_counter_samples) {
        gpu_profile_command_buffer = command_buffer;
        encoder = compute_encoder(nil);
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
    encoder = nil;
    command_buffer = nil;
}

id<MTLComputePipelineState> MetalPipelineCache::pipeline(std::string_view name) {
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

id<MTLComputePipelineState> MetalPipelineCache::tensor_pipeline(std::string_view name) {
    const std::string key = "tensor:" + std::string(name);
    const auto found = pipelines.find(key);
    if (found != pipelines.end()) return found->second;
    id<MTLLibrary> selected_library = tensor_library;
    if (name.find("_relaxed") != std::string_view::npos ||
        name.find("_fast") != std::string_view::npos) {
        if (name.find("_f16_") != std::string_view::npos ||
            name.find("_bf16_") != std::string_view::npos) {
            selected_library = tensor_fast_dense_library;
        } else if (name.find("_q4_0_") != std::string_view::npos) {
            selected_library = tensor_fast_q4_0_library;
        } else if (name.find("_q4k_") != std::string_view::npos) {
            selected_library = tensor_fast_q4k_library;
        } else if (name.find("_q5k_") != std::string_view::npos) {
            selected_library = tensor_fast_q5k_library;
        } else if (name.find("_q6k_") != std::string_view::npos) {
            selected_library = tensor_fast_q6k_library;
        } else if (name.find("_q8_0_") != std::string_view::npos) {
            selected_library = tensor_fast_q8_0_library;
        } else {
            selected_library = nil;
        }
    }
    if (!selected_library) throw std::runtime_error("Metal tensor library is unavailable");
    NSString* function_name = [NSString stringWithUTF8String:std::string(name).c_str()];
    id<MTLFunction> function = [selected_library newFunctionWithName:function_name];
    if (!function) throw std::runtime_error("Metal tensor function is missing: " + std::string(name));
    NSError* error = nil;
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!result) {
        const std::string message = error ? ns_string(error.localizedDescription)
                                          : "unknown Metal tensor pipeline error";
        throw std::runtime_error("Metal tensor pipeline failed: " + message);
    }
    pipelines.emplace(key, result);
    return result;
}

id<MTLComputePipelineState> MetalModel::Impl::pipeline(std::string_view name) {
    return pipeline_cache.pipeline(name);
}

id<MTLComputePipelineState> MetalModel::Impl::tensor_pipeline(std::string_view name) {
    return pipeline_cache.tensor_pipeline(name);
}

void MetalModel::Impl::dispatch(id<MTLComputeCommandEncoder> encoder, std::string_view name,
                                NSUInteger count) {
    if (count == 0) return;
    id<MTLComputePipelineState> state = pipeline(name);
    encoder = compute_encoder(encoder);
    [encoder setComputePipelineState:state];
    const NSUInteger threads = std::min(count, state.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    record_dispatch(name);
}

void MetalModel::Impl::dispatch_cooperative(id<MTLComputeCommandEncoder> encoder,
                                            std::string_view name, NSUInteger groups) {
    if (groups == 0) return;
    id<MTLComputePipelineState> state = pipeline(name);
    constexpr NSUInteger threads = 256;
    if (state.maxTotalThreadsPerThreadgroup < threads) {
        throw std::runtime_error("Metal pipeline cannot run cooperative kernel");
    }
    encoder = compute_encoder(encoder);
    [encoder setComputePipelineState:state];
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    record_dispatch(name);
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

void MetalModel::Impl::set_buffer(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> value,
                                  NSUInteger index, NSUInteger offset) {
    [compute_encoder(encoder) setBuffer:value offset:offset atIndex:index];
}

void MetalModel::Impl::set_bytes(id<MTLComputeCommandEncoder> encoder, const void* value,
                                 NSUInteger length, NSUInteger index) {
    [compute_encoder(encoder) setBytes:value length:length atIndex:index];
}

void MetalModel::Impl::encode_matvec(id<MTLComputeCommandEncoder> encoder,
                                     const Linear& weight,
                                     id<MTLBuffer> input,
                                     id<MTLBuffer> output,
                                     NSUInteger output_offset) {
    set_buffer(encoder, weight.buffer, 0);
    set_buffer(encoder, input, 1);
    set_buffer(encoder, output, 2, output_offset);
    set_bytes(encoder, &weight.rows, sizeof(weight.rows), 3);
    set_bytes(encoder, &weight.cols, sizeof(weight.cols), 4);
    if (weight.row_bytes != 0) {
        set_bytes(encoder, &weight.row_bytes, sizeof(weight.row_bytes), 5);
    }
    const MetalMatvecKernel selected = matvec_kernel(
        weight.storage, weight.rows, weight.cols);
    if (!selected.name) throw std::runtime_error("unsupported Metal matvec binding");
    id<MTLComputePipelineState> state = pipeline(selected.name);
    encoder = compute_encoder(encoder);
    [encoder setComputePipelineState:state];
    if (selected.threadgroup_floats != 0) {
        [encoder setThreadgroupMemoryLength:selected.threadgroup_floats * sizeof(float)
                                    atIndex:0];
    }
    const NSUInteger groups =
        (weight.rows + selected.rows_per_threadgroup - 1u) / selected.rows_per_threadgroup;
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(selected.threads, 1, 1)];
    record_dispatch(selected.name);
}

bool MetalModel::Impl::encode_swiglu_matvec(
    id<MTLComputeCommandEncoder> encoder, const Linear& weight,
    id<MTLBuffer> gate_up, id<MTLBuffer> output) {
    const MetalMatvecKernel selected = swiglu_matvec_kernel(
        weight.storage, weight.rows, weight.cols);
    if (!selected.name) return false;
    set_buffer(encoder, weight.buffer, 0);
    set_buffer(encoder, gate_up, 1);
    set_buffer(encoder, output, 2);
    set_bytes(encoder, &weight.rows, sizeof(weight.rows), 3);
    set_bytes(encoder, &weight.cols, sizeof(weight.cols), 4);
    set_bytes(encoder, &weight.row_bytes, sizeof(weight.row_bytes), 5);
    id<MTLComputePipelineState> state = pipeline(selected.name);
    encoder = compute_encoder(encoder);
    [encoder setComputePipelineState:state];
    if (selected.threadgroup_floats != 0) {
        [encoder setThreadgroupMemoryLength:selected.threadgroup_floats * sizeof(float)
                                    atIndex:0];
    }
    const NSUInteger groups =
        (weight.rows + selected.rows_per_threadgroup - 1u) / selected.rows_per_threadgroup;
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(selected.threads, 1, 1)];
    record_dispatch(selected.name);
    return true;
}

bool MetalModel::Impl::tensor_matmul_available(LinearStorage storage,
                                               uint32_t rows) const {
    if (rows < 16) return false;
    switch (storage) {
        case LinearStorage::Float16: return pipeline_cache.tensor_matmul_f16;
        case LinearStorage::BFloat16: return pipeline_cache.tensor_matmul_bf16;
        case LinearStorage::Q4_0: return pipeline_cache.tensor_matmul_q4_0;
        case LinearStorage::Q4K: return pipeline_cache.tensor_matmul_q4k;
        case LinearStorage::Q5K: return pipeline_cache.tensor_matmul_q5k;
        case LinearStorage::Q6K: return pipeline_cache.tensor_matmul_q6k;
        case LinearStorage::Q8_0: return pipeline_cache.tensor_matmul_q8_0;
        case LinearStorage::Float32: return false;
    }
    return false;
}

bool MetalModel::Impl::fast_tensor_matmul_available(LinearStorage storage) const {
    switch (storage) {
        case LinearStorage::Float16: return pipeline_cache.tensor_fast_f16;
        case LinearStorage::BFloat16: return pipeline_cache.tensor_fast_bf16;
        case LinearStorage::Q4_0: return pipeline_cache.tensor_fast_q4_0;
        case LinearStorage::Q4K: return pipeline_cache.tensor_fast_q4k;
        case LinearStorage::Q5K: return pipeline_cache.tensor_fast_q5k;
        case LinearStorage::Q6K: return pipeline_cache.tensor_fast_q6k;
        case LinearStorage::Q8_0: return pipeline_cache.tensor_fast_q8_0;
        case LinearStorage::Float32: return false;
    }
    return false;
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
    const bool dense = weight.row_bytes == 0;
    if (!dense) set_bytes(encoder, &weight.row_bytes, sizeof(weight.row_bytes), 7);
    const bool tensor = tensor_matmul_available(weight.storage, rows);
    if (tensor) {
        const auto generic_kernel = linear_kernel(
            weight.storage, LinearOperationKind::MatMulTensor);
        if (!generic_kernel) {
            throw std::runtime_error("unsupported Metal tensor matmul binding");
        }

        std::string_view selected_kernel = *generic_kernel;
        NSUInteger shared_bytes = kTensorTileBytes;
        NSUInteger tile_tokens = kTensorTileTokens;
        bool exact_groups = false;
        bool custom_tensor = false;
        const bool relaxed = options.numerical_policy == MetalNumericalPolicy::Fast &&
            fast_tensor_matmul_available(weight.storage);
        const bool m5_q6_gate = rows >= 128 &&
            ns_string(device.name).find("Apple M5") != std::string::npos &&
            weight.role == TensorRole::FfnGate && weight.layer >= 0 &&
            weight.layer < 8;

        if (relaxed && weight.storage == LinearStorage::Float16) {
            selected_kernel = rows <= 32 ? kF16RelaxedN32Kernel : kF16RelaxedKernel;
            custom_tensor = true;
        } else if (relaxed && weight.storage == LinearStorage::BFloat16) {
            selected_kernel = rows <= 32 ? kBF16RelaxedN32Kernel : kBF16RelaxedKernel;
            custom_tensor = true;
        } else if (relaxed && weight.storage == LinearStorage::Q4_0) {
            selected_kernel = rows <= 32 ? kQ40RelaxedN32Kernel : kQ40RelaxedKernel;
            custom_tensor = true;
        } else if (relaxed && weight.storage == LinearStorage::Q4K) {
            selected_kernel = rows <= 32 ? kQ4KRelaxedN32Kernel : kQ4KRelaxedKernel;
            custom_tensor = true;
        } else if (relaxed && weight.storage == LinearStorage::Q5K) {
            selected_kernel = rows <= 32 ? kQ5KRelaxedN32Kernel : kQ5KRelaxedKernel;
            custom_tensor = true;
        } else if (relaxed && weight.storage == LinearStorage::Q6K) {
            if (m5_q6_gate) {
                selected_kernel = rows <= 32
                    ? kQ6KRelaxedN32Kernel : kQ6KRelaxedKernel;
            } else {
                selected_kernel = rows <= 32
                    ? kQ6KStrictFastN32Kernel : kQ6KStrictFastKernel;
            }
            custom_tensor = true;
        } else if (relaxed && weight.storage == LinearStorage::Q8_0) {
            selected_kernel = rows <= 32 ? kQ80RelaxedN32Kernel : kQ80RelaxedKernel;
            custom_tensor = true;
        } else if (weight.storage == LinearStorage::Q4K &&
                   (rows % kTensorTileTokens) == 0u &&
                   (weight.cols % kQ4KStrictStageK) == 0u &&
                   (weight.rows % kTensorTileRows) == 0u &&
                   device.maxThreadgroupMemoryLength >= kQ4KStrictStageBytes) {
            selected_kernel = kQ4KStrictKernel;
            shared_bytes = kQ4KStrictStageBytes;
            exact_groups = true;
            custom_tensor = true;
        }
        if (relaxed && rows <= 32) tile_tokens = 32;

        id<MTLComputePipelineState> state = tensor_pipeline(selected_kernel);
        if (custom_tensor &&
            (state.maxTotalThreadsPerThreadgroup < kTensorTileThreads ||
             state.staticThreadgroupMemoryLength + shared_bytes >
                 device.maxThreadgroupMemoryLength)) {
            selected_kernel = *generic_kernel;
            shared_bytes = kTensorTileBytes;
            exact_groups = false;
            state = tensor_pipeline(selected_kernel);
        }

        encoder = compute_encoder(encoder);
        [encoder setComputePipelineState:state];
        [encoder setThreadgroupMemoryLength:shared_bytes atIndex:0];
        const NSUInteger row_groups = exact_groups
            ? weight.rows / kTensorTileRows
            : (weight.rows + kTensorTileRows - 1u) / kTensorTileRows;
        const NSUInteger token_groups = exact_groups
            ? rows / kTensorTileTokens
            : (rows + tile_tokens - 1u) / tile_tokens;
        [encoder dispatchThreadgroups:MTLSizeMake(row_groups, token_groups, 1)
             threadsPerThreadgroup:MTLSizeMake(kTensorTileThreads, 1, 1)];
        record_dispatch(selected_kernel);
        return;
    }
    const auto kernel = linear_kernel(weight.storage, LinearOperationKind::MatMul);
    if (!kernel) throw std::runtime_error("unsupported Metal matmul binding");
    const NSUInteger groups = (weight.rows + 7u) / 8u;
    id<MTLComputePipelineState> state = pipeline(*kernel);
    encoder = compute_encoder(encoder);
    [encoder setComputePipelineState:state];
    [encoder dispatchThreadgroups:MTLSizeMake(groups, rows, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    record_dispatch(*kernel);
}

void MetalModel::Impl::encode_embedding(id<MTLComputeCommandEncoder> encoder,
                                        uint32_t width, uint32_t token) {
    set_buffer(encoder, embedding.buffer, 0);
    set_buffer(encoder, hidden, 1);
    set_bytes(encoder, &width, sizeof(width), 2);
    set_bytes(encoder, &token, sizeof(token), 3);
    const auto kernel = linear_kernel(embedding.storage,
                                      LinearOperationKind::Embedding);
    if (!kernel) throw std::runtime_error("unsupported Metal embedding binding");
    dispatch(encoder, *kernel, width);
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
    const auto kernel = linear_kernel(embedding.storage,
                                      LinearOperationKind::EmbeddingBatch);
    if (!kernel) throw std::runtime_error("unsupported Metal embedding batch binding");
    dispatch(encoder, *kernel, static_cast<NSUInteger>(rows) * width);
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

void MetalModel::Impl::encode_residual_rmsnorm(
    id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> input,
    id<MTLBuffer> residual, id<MTLBuffer> weight, id<MTLBuffer> output,
    id<MTLBuffer> normed, uint32_t width, float multiplier, float epsilon) {
    set_buffer(encoder, input, 0);
    set_buffer(encoder, residual, 1);
    set_buffer(encoder, weight, 2);
    set_buffer(encoder, output, 3);
    set_buffer(encoder, normed, 4);
    set_bytes(encoder, &width, sizeof(width), 5);
    set_bytes(encoder, &multiplier, sizeof(multiplier), 6);
    set_bytes(encoder, &epsilon, sizeof(epsilon), 7);
    dispatch_cooperative(encoder, "celeg_residual_rmsnorm", 1);
}

void MetalModel::Impl::encode_residual_rmsnorm_save(
    id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> input,
    id<MTLBuffer> residual, id<MTLBuffer> weight, id<MTLBuffer> output,
    id<MTLBuffer> next_residual, id<MTLBuffer> normed, uint32_t width,
    float multiplier, float epsilon) {
    set_buffer(encoder, input, 0);
    set_buffer(encoder, residual, 1);
    set_buffer(encoder, weight, 2);
    set_buffer(encoder, output, 3);
    set_buffer(encoder, next_residual, 4);
    set_buffer(encoder, normed, 5);
    set_bytes(encoder, &width, sizeof(width), 6);
    set_bytes(encoder, &multiplier, sizeof(multiplier), 7);
    set_bytes(encoder, &epsilon, sizeof(epsilon), 8);
    dispatch_cooperative(encoder, "celeg_residual_rmsnorm_save", 1);
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
    encoder = compute_encoder(encoder);
    [encoder setComputePipelineState:state];
    [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    record_dispatch("celeg_rmsnorm_batch");
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
