#include "celeg/backend/cpu/rope.hpp"
#include "metal_inference_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kHeads = 2;
constexpr uint32_t kHeadDim = 4;
constexpr uint32_t kWidth = kHeads * kHeadDim;
constexpr uint32_t kRows = 3;
constexpr float kTolerance = 2.0e-5f;

std::string ns_string(NSString* value) {
    return value ? std::string(value.UTF8String) : std::string{};
}

id<MTLBuffer> float_buffer(id<MTLDevice> device, const std::vector<float>& values) {
    id<MTLBuffer> result = [device newBufferWithBytes:values.data()
                                                length:values.size() * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (!result) throw std::runtime_error("Metal value norm buffer allocation failed");
    return result;
}

id<MTLBuffer> zero_buffer(id<MTLDevice> device, size_t elements) {
    return float_buffer(device, std::vector<float>(elements, 0.0f));
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                     const char* name) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:
        [NSString stringWithUTF8String:name]];
    if (!function) {
        throw std::runtime_error(std::string("missing Metal value norm kernel: ") + name);
    }
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!result) {
        throw std::runtime_error("Metal value norm pipeline failed: " +
            (error ? ns_string(error.localizedDescription) : "unknown error"));
    }
    return result;
}

void wait(id<MTLCommandBuffer> command, const char* label) {
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        const std::string detail = command.error
            ? ": " + ns_string(command.error.localizedDescription)
            : std::string{};
        throw std::runtime_error(std::string(label) + " dispatch failed" + detail);
    }
}

void check_close(const float* actual, const std::vector<float>& expected,
                 const char* label, float tolerance = kTolerance) {
    for (size_t index = 0; index < expected.size(); ++index) {
        const float delta = std::abs(actual[index] - expected[index]);
        if (delta > tolerance) {
            throw std::runtime_error(std::string(label) + " differs at " +
                std::to_string(index) + ": actual=" + std::to_string(actual[index]) +
                " expected=" + std::to_string(expected[index]) +
                " delta=" + std::to_string(delta));
        }
    }
}

void check_single_normalization_is_observable(const std::vector<float>& single,
                                               const std::vector<float>& doubled,
                                               const char* label) {
    float maximum_delta = 0.0f;
    for (size_t index = 0; index < single.size(); ++index) {
        maximum_delta = std::max(maximum_delta,
                                 std::abs(single[index] - doubled[index]));
    }
    if (maximum_delta <= 1.0e-3f) {
        throw std::runtime_error(std::string(label) +
                                 " does not distinguish single from double normalization");
    }
}

std::vector<float> values(size_t count, float phase) {
    std::vector<float> result(count);
    for (size_t index = 0; index < count; ++index) {
        result[index] = 0.15f +
            0.9f * std::sin(phase + static_cast<float>(index) * 0.43f) +
            0.35f * std::cos(phase * 0.7f + static_cast<float>(index) * 0.17f);
    }
    return result;
}

std::vector<float> per_head_weight() {
    return {0.70f, 0.95f, 1.20f, 1.45f};
}

std::vector<float> whole_weight() {
    return {0.63f, 0.77f, 0.91f, 1.05f, 1.19f, 1.33f, 1.47f, 1.61f};
}

void cpu_normalize_row(float* row, const std::vector<float>& weight,
                       bool per_head, float epsilon) {
    if (per_head) {
        celeg::cpu_qk_norm_only(row, weight.data(), static_cast<int>(kHeads),
                                static_cast<int>(kHeadDim), epsilon);
    } else {
        celeg::cpu_qk_norm_only(row, weight.data(), 1,
                                static_cast<int>(kWidth), epsilon);
    }
}

std::vector<float> cpu_normalized(std::vector<float> input,
                                  const std::vector<float>& weight,
                                  bool per_head, float epsilon,
                                  uint32_t rows) {
    for (uint32_t row = 0; row < rows; ++row) {
        cpu_normalize_row(input.data() + static_cast<size_t>(row) * kWidth,
                          weight, per_head, epsilon);
    }
    return input;
}

void encode_token_norm(id<MTLDevice> device, id<MTLLibrary> library,
                       id<MTLComputeCommandEncoder> encoder,
                       id<MTLBuffer> data, id<MTLBuffer> weight,
                       bool per_head, float epsilon) {
    if (per_head) {
        id<MTLComputePipelineState> state =
            pipeline(device, library, "celeg_head_rmsnorm_inplace");
        [encoder setComputePipelineState:state];
        [encoder setBuffer:data offset:0 atIndex:0];
        [encoder setBuffer:weight offset:0 atIndex:1];
        [encoder setBytes:&kHeads length:sizeof(kHeads) atIndex:2];
        [encoder setBytes:&kHeadDim length:sizeof(kHeadDim) atIndex:3];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(kHeads, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(kHeads, 1, 1)];
        return;
    }

    id<MTLComputePipelineState> state = pipeline(device, library, "celeg_rmsnorm");
    if (state.maxTotalThreadsPerThreadgroup < 256) {
        throw std::runtime_error("Metal value norm RMSNorm pipeline cannot run 256 threads");
    }
    [encoder setComputePipelineState:state];
    [encoder setBuffer:data offset:0 atIndex:0];
    [encoder setBuffer:weight offset:0 atIndex:1];
    [encoder setBuffer:data offset:0 atIndex:2];
    [encoder setBytes:&kWidth length:sizeof(kWidth) atIndex:3];
    [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void encode_batch_norm(id<MTLDevice> device, id<MTLLibrary> library,
                       id<MTLComputeCommandEncoder> encoder,
                       id<MTLBuffer> data, id<MTLBuffer> weight,
                       bool per_head, float epsilon) {
    if (per_head) {
        id<MTLComputePipelineState> state =
            pipeline(device, library, "celeg_head_rmsnorm_batch_inplace");
        [encoder setComputePipelineState:state];
        [encoder setBuffer:data offset:0 atIndex:0];
        [encoder setBuffer:weight offset:0 atIndex:1];
        [encoder setBytes:&kRows length:sizeof(kRows) atIndex:2];
        [encoder setBytes:&kHeads length:sizeof(kHeads) atIndex:3];
        [encoder setBytes:&kHeadDim length:sizeof(kHeadDim) atIndex:4];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:5];
        const uint32_t count = kRows * kHeads;
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(count, 1, 1)];
        return;
    }

    id<MTLComputePipelineState> state = pipeline(device, library, "celeg_rmsnorm_batch");
    if (state.maxTotalThreadsPerThreadgroup < 256) {
        throw std::runtime_error("Metal batch value norm pipeline cannot run 256 threads");
    }
    [encoder setComputePipelineState:state];
    [encoder setBuffer:data offset:0 atIndex:0];
    [encoder setBuffer:weight offset:0 atIndex:1];
    [encoder setBuffer:data offset:0 atIndex:2];
    [encoder setBytes:&kRows length:sizeof(kRows) atIndex:3];
    [encoder setBytes:&kWidth length:sizeof(kWidth) atIndex:4];
    [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(kRows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void run_token_case(id<MTLDevice> device, id<MTLLibrary> library,
                    id<MTLCommandQueue> queue, bool per_head) {
    constexpr uint32_t cache_position = 2;
    constexpr uint32_t page_tokens = 16;
    constexpr uint32_t position_mode = 0;
    constexpr float theta = 10000.0f;
    constexpr float query_scale = 1.0f;
    const float epsilon = per_head ? 1.0e-5f : 2.0e-5f;
    const std::vector<float> weight = per_head ? per_head_weight() : whole_weight();
    const std::vector<float> original = values(kWidth, per_head ? 0.25f : 0.65f);
    const std::vector<float> expected =
        cpu_normalized(original, weight, per_head, epsilon, 1);
    const std::vector<float> doubled =
        cpu_normalized(expected, weight, per_head, epsilon, 1);
    check_single_normalization_is_observable(expected, doubled,
                                             per_head ? "token per-head" : "token whole");

    id<MTLBuffer> value = float_buffer(device, original);
    id<MTLBuffer> weight_buffer = float_buffer(device, weight);
    id<MTLBuffer> query = float_buffer(device, values(kWidth, 1.1f));
    id<MTLBuffer> key = float_buffer(device, values(kWidth, 1.7f));
    const size_t cache_elements = static_cast<size_t>(cache_position + 1) * kWidth;
    id<MTLBuffer> key_cache = zero_buffer(device, cache_elements);
    id<MTLBuffer> value_cache = zero_buffer(device, cache_elements);

    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    encode_token_norm(device, library, encoder, value, weight_buffer,
                      per_head, epsilon);

    id<MTLComputePipelineState> store =
        pipeline(device, library, "celeg_qk_position_store_kv");
    [encoder setComputePipelineState:store];
    [encoder setBuffer:query offset:0 atIndex:0];
    [encoder setBuffer:key offset:0 atIndex:1];
    [encoder setBuffer:value offset:0 atIndex:2];
    [encoder setBuffer:key_cache offset:0 atIndex:3];
    [encoder setBuffer:value_cache offset:0 atIndex:4];
    [encoder setBytes:&kHeads length:sizeof(kHeads) atIndex:5];
    [encoder setBytes:&kHeads length:sizeof(kHeads) atIndex:6];
    [encoder setBytes:&kHeadDim length:sizeof(kHeadDim) atIndex:7];
    [encoder setBytes:&cache_position length:sizeof(cache_position) atIndex:8];
    [encoder setBytes:&position_mode length:sizeof(position_mode) atIndex:9];
    [encoder setBytes:&theta length:sizeof(theta) atIndex:10];
    [encoder setBytes:&query_scale length:sizeof(query_scale) atIndex:11];
    [encoder setBytes:&page_tokens length:sizeof(page_tokens) atIndex:12];
    [encoder dispatchThreads:MTLSizeMake(kHeads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(kHeads, 1, 1)];
    [encoder endEncoding];
    wait(command, per_head ? "token per-head value norm" : "token whole value norm");

    check_close(static_cast<const float*>(value.contents), expected,
                per_head ? "token per-head value" : "token whole value");
    const float* cached = static_cast<const float*>(value_cache.contents) +
        static_cast<size_t>(cache_position) * kWidth;
    check_close(cached, expected,
                per_head ? "token per-head cache" : "token whole cache");
}

void run_batch_case(id<MTLDevice> device, id<MTLLibrary> library,
                    id<MTLCommandQueue> queue, bool per_head) {
    constexpr uint32_t base_position = 3;
    const float epsilon = per_head ? 3.0e-5f : 4.0e-5f;
    const std::vector<float> weight = per_head ? per_head_weight() : whole_weight();
    const std::vector<float> original =
        values(static_cast<size_t>(kRows) * kWidth, per_head ? 0.9f : 1.35f);
    const std::vector<float> expected =
        cpu_normalized(original, weight, per_head, epsilon, kRows);
    const std::vector<float> doubled =
        cpu_normalized(expected, weight, per_head, epsilon, kRows);
    check_single_normalization_is_observable(expected, doubled,
                                             per_head ? "batch per-head" : "batch whole");

    id<MTLBuffer> value = float_buffer(device, original);
    id<MTLBuffer> weight_buffer = float_buffer(device, weight);
    id<MTLBuffer> key = float_buffer(device, values(original.size(), 1.8f));
    const size_t cache_rows = static_cast<size_t>(base_position + kRows + 1);
    id<MTLBuffer> key_cache = zero_buffer(device, cache_rows * kWidth);
    id<MTLBuffer> value_cache = zero_buffer(device, cache_rows * kWidth);

    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    encode_batch_norm(device, library, encoder, value, weight_buffer,
                      per_head, epsilon);

    id<MTLComputePipelineState> store =
        pipeline(device, library, "celeg_store_kv_batch_2d");
    if (store.maxTotalThreadsPerThreadgroup < 256) {
        throw std::runtime_error("Metal value norm KV store cannot run 256 threads");
    }
    [encoder setComputePipelineState:store];
    [encoder setBuffer:key offset:0 atIndex:0];
    [encoder setBuffer:value offset:0 atIndex:1];
    [encoder setBuffer:key_cache offset:0 atIndex:2];
    [encoder setBuffer:value_cache offset:0 atIndex:3];
    [encoder setBytes:&kRows length:sizeof(kRows) atIndex:4];
    [encoder setBytes:&base_position length:sizeof(base_position) atIndex:5];
    [encoder setBytes:&kWidth length:sizeof(kWidth) atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake(1, kRows, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    wait(command, per_head ? "batch per-head value norm" : "batch whole value norm");

    check_close(static_cast<const float*>(value.contents), expected,
                per_head ? "batch per-head value" : "batch whole value");
    const float* cached = static_cast<const float*>(value_cache.contents) +
        static_cast<size_t>(base_position) * kWidth;
    check_close(cached, expected,
                per_head ? "batch per-head cache" : "batch whole cache");
}

}

int main() {
    try {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("no default Metal device is available");
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:
            [NSString stringWithUTF8String:celeg::metal_detail::kInferenceShader]
                                                          options:nil error:&error];
        if (!library) {
            throw std::runtime_error("Metal value norm shader compilation failed: " +
                (error ? ns_string(error.localizedDescription) : "unknown error"));
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) throw std::runtime_error("Metal value norm command queue failed");

        run_token_case(device, library, queue, true);
        run_token_case(device, library, queue, false);
        run_batch_case(device, library, queue, true);
        run_batch_case(device, library, queue, false);

        std::cout << "metal value normalization matches CPU before KV publication\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
