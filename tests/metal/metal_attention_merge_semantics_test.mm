#include "celeg/attention/merge_semantics.hpp"
#include "metal_inference_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

struct MergeCase {
    float destination_maximum;
    float destination_denominator;
    float source_maximum;
    float source_denominator;
    float destination_accumulator;
    float source_accumulator;
    float global_maximum;
};

void run_case(id<MTLDevice> device, id<MTLCommandQueue> queue,
              id<MTLComputePipelineState> state, const MergeCase& c) {
    id<MTLBuffer> output = [device newBufferWithLength:6 * sizeof(float)
                                              options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setBuffer:output offset:0 atIndex:0];
    [encoder setBytes:&c.destination_maximum length:sizeof(float) atIndex:1];
    [encoder setBytes:&c.destination_denominator length:sizeof(float) atIndex:2];
    [encoder setBytes:&c.source_maximum length:sizeof(float) atIndex:3];
    [encoder setBytes:&c.source_denominator length:sizeof(float) atIndex:4];
    [encoder setBytes:&c.destination_accumulator length:sizeof(float) atIndex:5];
    [encoder setBytes:&c.source_accumulator length:sizeof(float) atIndex:6];
    [encoder setBytes:&c.global_maximum length:sizeof(float) atIndex:7];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal merge semantics dispatch failed");
    }

    const auto transition = celeg::attention_semantics::merge_pair(
        c.destination_maximum, c.destination_denominator,
        c.source_maximum, c.source_denominator);
    const std::array<float, 6> reference{
        transition.maximum,
        transition.denominator,
        transition.destination_scale,
        transition.source_scale,
        celeg::attention_semantics::merge_accumulate(
            c.destination_accumulator, c.source_accumulator, transition),
        celeg::attention_semantics::partial_rescale(
            c.source_maximum, c.global_maximum)};
    const float* actual = static_cast<const float*>(output.contents);
    for (size_t i = 0; i < reference.size(); ++i) {
        if (std::abs(actual[i] - reference[i]) > 1.0e-6f) {
            throw std::runtime_error("Metal partial attention merge semantics drifted from canonical semantics");
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
        if (!library) throw std::runtime_error("Metal merge semantics shader compilation failed");
        id<MTLFunction> function = [library newFunctionWithName:@"celeg_attention_merge_semantics_probe"];
        if (!function) throw std::runtime_error("missing Metal merge semantics probe");
        id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function error:&error];
        if (!state) throw std::runtime_error("failed Metal merge semantics pipeline");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        const float negative_infinity = -std::numeric_limits<float>::infinity();
        const std::array<MergeCase, 4> cases{{
            {negative_infinity, 0.0f, 2.0f, 3.0f, 0.0f, 5.0f, 2.0f},
            {2.0f, 3.0f, 4.0f, 2.0f, 5.0f, 7.0f, 4.0f},
            {4.0f, 1.5f, 2.0f, 5.0f, -2.0f, 3.0f, 4.0f},
            {-3.0f, 2.0f, -3.0f, 6.0f, 1.0f, -4.0f, -1.0f},
        }};
        for (const MergeCase& c : cases) run_case(device, queue, state, c);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
