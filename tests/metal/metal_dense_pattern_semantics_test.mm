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
    uint32_t available_sequence_length;
    uint32_t prefix_length;
};

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
    [encoder setBytes:&c.available_sequence_length
                length:sizeof(c.available_sequence_length) atIndex:3];
    [encoder setBytes:&c.prefix_length length:sizeof(c.prefix_length) atIndex:4];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal dense-pattern semantics dispatch failed");
    }

    const auto* actual = static_cast<const uint32_t*>(output.contents);
    const bool expected_bidirectional =
        celeg::attention_semantics::bidirectional_visible(
            c.query_position, c.key_position);
    const bool expected_prefix = celeg::attention_semantics::prefix_lm_visible(
        c.query_position, c.key_position, static_cast<int>(c.prefix_length));
    const int expected_length =
        celeg::attention_semantics::prefix_lm_visible_sequence_length(
            c.query_position, static_cast<int>(c.available_sequence_length),
            static_cast<int>(c.prefix_length));

    if (actual[0] != static_cast<uint32_t>(expected_bidirectional)) {
        throw std::runtime_error("Metal bidirectional visibility drifted from host semantics");
    }
    if (actual[1] != static_cast<uint32_t>(expected_prefix)) {
        throw std::runtime_error("Metal Prefix-LM visibility drifted from host semantics");
    }
    if (actual[2] != static_cast<uint32_t>(expected_length)) {
        throw std::runtime_error("Metal Prefix-LM visible extent drifted from host semantics");
    }
}

}

int main() {
    try {
        constexpr std::array<PatternCase, 10> cases{{
            {0, 0, 8, 4},
            {0, 3, 8, 4},
            {0, 4, 8, 4},
            {2, 3, 8, 4},
            {3, 0, 8, 4},
            {4, 3, 8, 4},
            {4, 5, 8, 4},
            {7, 7, 8, 4},
            {2, 3, 3, 4},
            {-1, 0, 8, 4},
        }};

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("no default Metal device is available");
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:
            celeg::metal_detail::kInferenceShader];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            throw std::runtime_error("Metal dense-pattern semantics shader compilation failed");
        }
        id<MTLFunction> function = [library newFunctionWithName:
            @"celeg_attention_dense_pattern_semantics_probe"];
        if (!function) throw std::runtime_error("missing Metal dense-pattern semantics probe");
        id<MTLComputePipelineState> state =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!state) throw std::runtime_error("failed Metal dense-pattern semantics pipeline");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        for (const PatternCase& c : cases) run_case(device, queue, state, c);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
