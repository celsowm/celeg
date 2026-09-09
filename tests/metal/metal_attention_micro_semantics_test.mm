#include "celeg/attention/micro_semantics.hpp"
#include "metal_inference_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
    try {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("no default Metal device is available");
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:celeg::metal_detail::kInferenceShader];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) throw std::runtime_error("Metal micro semantics shader compilation failed");
        id<MTLFunction> function = [library newFunctionWithName:@"celeg_attention_micro_semantics_probe"];
        if (!function) throw std::runtime_error("missing Metal micro semantics probe");
        id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function error:&error];
        if (!state) throw std::runtime_error("failed Metal micro semantics pipeline");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> integers = [device newBufferWithLength:3 * sizeof(uint32_t)
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> scale = [device newBufferWithLength:sizeof(float)
                                              options:MTLResourceStorageModeShared];
        uint32_t query_head = 7;
        uint32_t query_heads = 8;
        uint32_t kv_heads = 2;
        uint32_t query_position = 31;
        uint32_t head_dim = 64;
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:state];
        [encoder setBuffer:integers offset:0 atIndex:0];
        [encoder setBuffer:scale offset:0 atIndex:1];
        [encoder setBytes:&query_head length:sizeof(query_head) atIndex:2];
        [encoder setBytes:&query_heads length:sizeof(query_heads) atIndex:3];
        [encoder setBytes:&kv_heads length:sizeof(kv_heads) atIndex:4];
        [encoder setBytes:&query_position length:sizeof(query_position) atIndex:5];
        [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:6];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("Metal micro semantics dispatch failed");
        }
        const uint32_t* actual = static_cast<const uint32_t*>(integers.contents);
        const int expected_kv = celeg::attention_semantics::gqa_kv_head(7, 8, 2);
        const int expected_length = celeg::attention_semantics::sequence_length_from_query_position(31);
        const int expected_position = celeg::attention_semantics::query_position_from_sequence_length(32);
        const float expected_scale = celeg::attention_semantics::attention_scale(64);
        if (actual[0] != static_cast<uint32_t>(expected_kv) ||
            actual[1] != static_cast<uint32_t>(expected_length) ||
            actual[2] != static_cast<uint32_t>(expected_position)) {
            throw std::runtime_error("Metal attention micro semantics drifted");
        }
        if (std::abs(*static_cast<const float*>(scale.contents) - expected_scale) > 1.0e-6f) {
            throw std::runtime_error("Metal attention scale semantics drifted");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
