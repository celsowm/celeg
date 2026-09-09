#include "celeg/attention/online_semantics.hpp"
#include "metal_inference_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

struct OnlineCase {
    float previous_maximum;
    float previous_denominator;
    float score;
    float accumulator;
    float value;
};

void run_case(id<MTLDevice> device, id<MTLCommandQueue> queue,
              id<MTLComputePipelineState> state, const OnlineCase& c) {
    id<MTLBuffer> output = [device newBufferWithLength:5 * sizeof(float)
                                              options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setBuffer:output offset:0 atIndex:0];
    [encoder setBytes:&c.previous_maximum length:sizeof(float) atIndex:1];
    [encoder setBytes:&c.previous_denominator length:sizeof(float) atIndex:2];
    [encoder setBytes:&c.score length:sizeof(float) atIndex:3];
    [encoder setBytes:&c.accumulator length:sizeof(float) atIndex:4];
    [encoder setBytes:&c.value length:sizeof(float) atIndex:5];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal online semantics dispatch failed");
    }

    const auto expected = celeg::attention_semantics::online_transition(
        c.previous_maximum, c.previous_denominator, c.score);
    const float expected_accumulator = celeg::attention_semantics::online_accumulate(
        c.accumulator, c.value, expected);
    const float* actual = static_cast<const float*>(output.contents);
    const std::array<float, 5> reference{
        expected.maximum, expected.denominator, expected.previous_scale,
        expected.current_scale, expected_accumulator};
    for (size_t i = 0; i < reference.size(); ++i) {
        if (std::abs(actual[i] - reference[i]) > 1.0e-6f) {
            throw std::runtime_error("Metal online attention semantics drifted from canonical semantics");
        }
    }
}

}

int main() {
    try {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("no default Metal device is available");
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:celeg::metal_detail::kInferenceShader];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) throw std::runtime_error("Metal online semantics shader compilation failed");
        id<MTLFunction> function = [library newFunctionWithName:@"celeg_attention_online_semantics_probe"];
        if (!function) throw std::runtime_error("missing Metal online semantics probe");
        id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function error:&error];
        if (!state) throw std::runtime_error("failed Metal online semantics pipeline");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        const float negative_infinity = -std::numeric_limits<float>::infinity();
        const std::array<OnlineCase, 4> cases{{
            {negative_infinity, 0.0f, 2.0f, 0.0f, 5.0f},
            {1.0f, 2.0f, 3.0f, 4.0f, 6.0f},
            {4.0f, 1.5f, 2.0f, -2.0f, 3.0f},
            {-10.0f, 7.0f, -10.0f, 1.0f, -4.0f},
        }};
        for (const OnlineCase& c : cases) run_case(device, queue, state, c);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
