#include "metal_inference_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                     const char* name) {
    id<MTLFunction> function = [library newFunctionWithName:
        [NSString stringWithUTF8String:name]];
    if (!function) throw std::runtime_error(std::string("missing Metal function: ") + name);
    NSError* error = nil;
    id<MTLComputePipelineState> state =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!state) throw std::runtime_error(std::string("failed Metal pipeline: ") + name);
    return state;
}

void finish(id<MTLCommandBuffer> command_buffer) {
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal attention gate dispatch failed");
    }
}

void check_close(float actual, float expected) {
    if (std::abs(actual - expected) > 1.0e-5f) {
        throw std::runtime_error("Metal attention gate result mismatch");
    }
}

void test_packed_query_extract(id<MTLDevice> device, id<MTLLibrary> library) {
    const std::vector<float> packed{1.0f, 2.0f, 10.0f, 20.0f,
                                    3.0f, 4.0f, 30.0f, 40.0f};
    id<MTLBuffer> source = [device newBufferWithBytes:packed.data()
                                                length:packed.size() * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> query = [device newBufferWithLength:4 * sizeof(float)
                                             options:MTLResourceStorageModeShared];
    const uint32_t rows = 1;
    const uint32_t query_width = 4;
    const uint32_t head_dim = 2;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline(device, library,
        "celeg_extract_attention_query_batch")];
    [encoder setBuffer:source offset:0 atIndex:0];
    [encoder setBuffer:query offset:0 atIndex:1];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
    [encoder setBytes:&query_width length:sizeof(query_width) atIndex:3];
    [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:4];
    [encoder dispatchThreads:MTLSizeMake(query_width, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(query_width, 1, 1)];
    [encoder endEncoding];
    finish(command_buffer);
    const float* values = static_cast<const float*>(query.contents);
    check_close(values[0], 1.0f);
    check_close(values[1], 2.0f);
    check_close(values[2], 3.0f);
    check_close(values[3], 4.0f);
}

void test_headwise_gate(id<MTLDevice> device, id<MTLLibrary> library) {
    const std::vector<float> initial{2.0f, 4.0f, 6.0f, 8.0f};
    const std::vector<float> gates{0.0f, std::log(3.0f)};
    id<MTLBuffer> output = [device newBufferWithBytes:initial.data()
                                                length:initial.size() * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> gate = [device newBufferWithBytes:gates.data()
                                              length:gates.size() * sizeof(float)
                                             options:MTLResourceStorageModeShared];
    const uint32_t width = 4;
    const uint32_t head_dim = 2;
    const uint32_t head_wise = 1;
    const uint32_t packed = 0;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline(device, library,
        "celeg_attention_output_gate")];
    [encoder setBuffer:output offset:0 atIndex:0];
    [encoder setBuffer:gate offset:0 atIndex:1];
    [encoder setBytes:&width length:sizeof(width) atIndex:2];
    [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:3];
    [encoder setBytes:&head_wise length:sizeof(head_wise) atIndex:4];
    [encoder setBytes:&packed length:sizeof(packed) atIndex:5];
    [encoder dispatchThreads:MTLSizeMake(width, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    finish(command_buffer);
    const float* values = static_cast<const float*>(output.contents);
    check_close(values[0], 1.0f);
    check_close(values[1], 2.0f);
    check_close(values[2], 4.5f);
    check_close(values[3], 6.0f);
}

}

int main() {
    try {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("no default Metal device is available");
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:celeg::metal_detail::kInferenceShader];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            throw std::runtime_error("Metal attention gate shader compilation failed");
        }
        pipeline(device, library, "celeg_attention_output_gate_batch");
        test_packed_query_extract(device, library);
        test_headwise_gate(device, library);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
