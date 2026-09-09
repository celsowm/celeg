#include "celeg/attention/pattern_semantics.hpp"
#include "metal_inference_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

struct PatternCase {
    int query_position;
    int key_position;
    uint32_t window_size;
};

id<MTLComputePipelineState> pipeline(id<MTLDevice> device,
                                     id<MTLLibrary> library) {
    id<MTLFunction> function = [library newFunctionWithName:
        @"celeg_attention_pattern_semantics_probe"];
    if (!function) throw std::runtime_error("missing Metal pattern semantics probe");
    NSError* error = nil;
    id<MTLComputePipelineState> state =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!state) throw std::runtime_error("failed Metal pattern semantics pipeline");
    return state;
}

void run_case(id<MTLDevice> device, id<MTLCommandQueue> queue,
              id<MTLComputePipelineState> state, const PatternCase& c) {
    id<MTLBuffer> output = [device newBufferWithLength:3 * sizeof(uint32_t)
                                              options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setBuffer:output offset:0 atIndex:0];
    [encoder setBytes:&c.query_position length:sizeof(c.query_position) atIndex:1];
    [encoder setBytes:&c.key_position length:sizeof(c.key_position) atIndex:2];
    [encoder setBytes:&c.window_size length:sizeof(c.window_size) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal pattern semantics dispatch failed");
    }

    const auto* actual = static_cast<const uint32_t*>(output.contents);
    const bool expected_causal = celeg::attention_semantics::causal_visible(
        c.query_position, c.key_position);
    const int expected_first = celeg::attention_semantics::sliding_window_first_candidate(
        c.query_position, static_cast<int>(c.window_size));
    const bool expected_sliding = celeg::attention_semantics::sliding_window_visible(
        c.query_position, c.key_position, static_cast<int>(c.window_size));

    if (actual[0] != static_cast<uint32_t>(expected_causal)) {
        throw std::runtime_error("Metal causal visibility drifted from canonical semantics");
    }
    if (actual[1] != static_cast<uint32_t>(expected_first)) {
        throw std::runtime_error("Metal sliding first-candidate drifted from canonical semantics");
    }
    if (actual[2] != static_cast<uint32_t>(expected_sliding)) {
        throw std::runtime_error("Metal sliding visibility drifted from canonical semantics");
    }
}

}

int main() {
    try {
        constexpr std::array<PatternCase, 10> cases{{
            {0, 0, 0},
            {7, 7, 0},
            {7, 8, 0},
            {7, 0, 4},
            {7, 4, 4},
            {7, 7, 4},
            {7, 8, 4},
            {31, 0, 16},
            {31, 16, 16},
            {31, 31, 16},
        }};
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("no default Metal device is available");
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:
            celeg::metal_detail::kInferenceShader];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) throw std::runtime_error("Metal pattern semantics shader compilation failed");
        id<MTLComputePipelineState> state = pipeline(device, library);
        id<MTLCommandQueue> queue = [device newCommandQueue];
        for (const PatternCase& c : cases) run_case(device, queue, state, c);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
