#include "celeg/attention/bias_semantics.hpp"
#include "metal_inference_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct BiasCase {
    int query_position;
    int key_position;
    uint32_t bucket_count;
    uint32_t max_distance;
    uint32_t bidirectional;
    float slope;
};

id<MTLComputePipelineState> pipeline(id<MTLDevice> device,
                                     id<MTLLibrary> library) {
    id<MTLFunction> function = [library newFunctionWithName:
        @"celeg_attention_bias_semantics_probe"];
    if (!function) throw std::runtime_error("missing Metal bias semantics probe");
    NSError* error = nil;
    id<MTLComputePipelineState> state =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!state) throw std::runtime_error("failed Metal bias semantics pipeline");
    return state;
}

void run_case(id<MTLDevice> device, id<MTLCommandQueue> queue,
              id<MTLComputePipelineState> state, const BiasCase& c) {
    id<MTLBuffer> bucket = [device newBufferWithLength:sizeof(uint32_t)
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> alibi = [device newBufferWithLength:sizeof(float)
                                             options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setBuffer:bucket offset:0 atIndex:0];
    [encoder setBuffer:alibi offset:0 atIndex:1];
    [encoder setBytes:&c.query_position length:sizeof(c.query_position) atIndex:2];
    [encoder setBytes:&c.key_position length:sizeof(c.key_position) atIndex:3];
    [encoder setBytes:&c.bucket_count length:sizeof(c.bucket_count) atIndex:4];
    [encoder setBytes:&c.max_distance length:sizeof(c.max_distance) atIndex:5];
    [encoder setBytes:&c.bidirectional length:sizeof(c.bidirectional) atIndex:6];
    [encoder setBytes:&c.slope length:sizeof(c.slope) atIndex:7];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal bias semantics dispatch failed");
    }

    const auto actual_bucket = *static_cast<const uint32_t*>(bucket.contents);
    const auto actual_alibi = *static_cast<const float*>(alibi.contents);
    const int expected_bucket = celeg::attention_semantics::relative_position_bucket(
        c.query_position, c.key_position, static_cast<int>(c.bucket_count),
        static_cast<int>(c.max_distance), c.bidirectional != 0);
    const float expected_alibi = celeg::attention_semantics::alibi_bias(
        c.slope, c.query_position, c.key_position);
    if (actual_bucket != static_cast<uint32_t>(expected_bucket)) {
        throw std::runtime_error("Metal relative-position bucket drifted from canonical semantics");
    }
    if (std::abs(actual_alibi - expected_alibi) > 1.0e-6f) {
        throw std::runtime_error("Metal ALiBi drifted from canonical semantics");
    }
}

}

int main() {
    try {
        constexpr std::array<BiasCase, 8> cases{{
            {0, 0, 32, 128, 0, 0.5f},
            {7, 0, 32, 128, 0, 0.5f},
            {31, 15, 32, 128, 0, 0.25f},
            {127, 0, 32, 128, 0, 1.0f},
            {8, 9, 32, 128, 1, 0.5f},
            {8, 7, 32, 128, 1, 0.5f},
            {64, 127, 32, 128, 1, 0.125f},
            {127, 64, 32, 128, 1, 0.125f},
        }};
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("no default Metal device is available");
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:
            celeg::metal_detail::kInferenceShader];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) throw std::runtime_error("Metal bias semantics shader compilation failed");
        id<MTLComputePipelineState> state = pipeline(device, library);
        id<MTLCommandQueue> queue = [device newCommandQueue];
        for (const BiasCase& c : cases) run_case(device, queue, state, c);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
