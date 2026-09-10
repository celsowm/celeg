#include "metal_inference_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

constexpr uint32_t kRows = 4;
constexpr uint32_t kHeads = 1;
constexpr uint32_t kHeadDim = 32;
constexpr uint32_t kPageTokens = 16;
constexpr float kScale = 1.0f;
constexpr std::array<float, kRows> kValues{1.0f, 3.0f, 5.0f, 9.0f};

id<MTLBuffer> float_buffer(id<MTLDevice> device, const std::vector<float>& values) {
    return [device newBufferWithBytes:values.data()
                               length:values.size() * sizeof(float)
                              options:MTLResourceStorageModeShared];
}

std::vector<float> run_attention(id<MTLDevice> device,
                                 id<MTLCommandQueue> queue,
                                 id<MTLLibrary> library,
                                 std::string_view kernel,
                                 uint32_t pattern_mode,
                                 uint32_t prefix_length) {
    const size_t elements = static_cast<size_t>(kRows) * kHeadDim;
    std::vector<float> zeros(elements, 0.0f);
    std::vector<float> values(elements);
    for (uint32_t row = 0; row < kRows; ++row) {
        for (uint32_t d = 0; d < kHeadDim; ++d) {
            values[static_cast<size_t>(row) * kHeadDim + d] = kValues[row];
        }
    }

    id<MTLBuffer> query = float_buffer(device, zeros);
    id<MTLBuffer> key = float_buffer(device, zeros);
    id<MTLBuffer> value = float_buffer(device, values);
    id<MTLBuffer> output = [device newBufferWithLength:elements * sizeof(float)
                                              options:MTLResourceStorageModeShared];

    NSString* name = [NSString stringWithUTF8String:std::string(kernel).c_str()];
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) throw std::runtime_error("missing Metal attention kernel");
    NSError* error = nil;
    id<MTLComputePipelineState> state =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!state) throw std::runtime_error("failed to create Metal attention pipeline");

    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setBuffer:query offset:0 atIndex:0];
    [encoder setBuffer:key offset:0 atIndex:1];
    [encoder setBuffer:value offset:0 atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    const uint32_t base_position = 0;
    [encoder setBytes:&kRows length:sizeof(kRows) atIndex:4];
    [encoder setBytes:&base_position length:sizeof(base_position) atIndex:5];
    [encoder setBytes:&kHeads length:sizeof(kHeads) atIndex:6];
    [encoder setBytes:&kHeads length:sizeof(kHeads) atIndex:7];
    [encoder setBytes:&kHeadDim length:sizeof(kHeadDim) atIndex:8];
    [encoder setBytes:&kScale length:sizeof(kScale) atIndex:9];
    [encoder setBytes:&kPageTokens length:sizeof(kPageTokens) atIndex:10];
    if (kernel == "celeg_attention_batch_dense_pattern") {
        [encoder setBytes:&pattern_mode length:sizeof(pattern_mode) atIndex:11];
        [encoder setBytes:&prefix_length length:sizeof(prefix_length) atIndex:12];
    }

    constexpr NSUInteger simdgroups = 8;
    constexpr NSUInteger threads = 32 * simdgroups;
    const NSUInteger shared_floats = 2 * simdgroups + simdgroups * kHeadDim;
    [encoder setThreadgroupMemoryLength:shared_floats * sizeof(float) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(kHeads, kRows, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal dense noncausal attention dispatch failed");
    }

    const auto* actual = static_cast<const float*>(output.contents);
    return std::vector<float>(actual, actual + elements);
}

void expect_rows(const std::vector<float>& actual,
                 const std::array<float, kRows>& expected,
                 const char* label) {
    for (uint32_t row = 0; row < kRows; ++row) {
        for (uint32_t d = 0; d < kHeadDim; ++d) {
            const float value = actual[static_cast<size_t>(row) * kHeadDim + d];
            if (std::abs(value - expected[row]) > 1.0e-5f) {
                throw std::runtime_error(std::string(label) + " output mismatch");
            }
        }
    }
}

}

int main() {
    try {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("no default Metal device is available");
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:
            celeg::metal_detail::kInferenceShader];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) throw std::runtime_error("Metal dense noncausal shader compilation failed");
        id<MTLCommandQueue> queue = [device newCommandQueue];

        const std::vector<float> causal = run_attention(
            device, queue, library, "celeg_attention_batch", 0u, 0u);
        const std::vector<float> bidirectional = run_attention(
            device, queue, library, "celeg_attention_batch_dense_pattern", 1u, 0u);
        const std::vector<float> prefix = run_attention(
            device, queue, library, "celeg_attention_batch_dense_pattern", 2u, 3u);

        expect_rows(causal, {1.0f, 2.0f, 3.0f, 4.5f}, "causal baseline");
        expect_rows(bidirectional, {4.5f, 4.5f, 4.5f, 4.5f}, "bidirectional");
        expect_rows(prefix, {3.0f, 3.0f, 3.0f, 4.5f}, "Prefix-LM");

        if (std::abs(causal[0] - bidirectional[0]) < 1.0f ||
            std::abs(causal[0] - prefix[0]) < 1.0f) {
            throw std::runtime_error("future-read fixture did not distinguish causal output");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
