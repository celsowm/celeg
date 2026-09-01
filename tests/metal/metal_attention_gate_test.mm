#include "metal_inference_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
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

void test_mrope_batch(id<MTLDevice> device, id<MTLLibrary> library) {
    const uint32_t rows = 2;
    const uint32_t query_heads = 1;
    const uint32_t key_heads = 1;
    const uint32_t head_dim = 4;
    const std::array<uint32_t, 3> sections{1, 1, 0};
    const float theta = 100.0f;
    const float query_scale = 0.5f;
    const std::array<int32_t, 6> positions{1, 2, 3, 4, 5, 6};
    const std::vector<float> initial_query{1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
    const std::vector<float> initial_key{2.0f, 1.0f, 4.0f, 3.0f,
                                         6.0f, 5.0f, 8.0f, 7.0f};
    id<MTLBuffer> query = [device newBufferWithBytes:initial_query.data()
                                               length:initial_query.size() * sizeof(float)
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> key = [device newBufferWithBytes:initial_key.data()
                                             length:initial_key.size() * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> rope = [device newBufferWithBytes:positions.data()
                                              length:positions.size() * sizeof(int32_t)
                                             options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline(device, library,
        "celeg_qk_mrope_position_batch")];
    [encoder setBuffer:query offset:0 atIndex:0];
    [encoder setBuffer:key offset:0 atIndex:1];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
    [encoder setBytes:&query_heads length:sizeof(query_heads) atIndex:3];
    [encoder setBytes:&key_heads length:sizeof(key_heads) atIndex:4];
    [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:5];
    [encoder setBuffer:rope offset:0 atIndex:6];
    [encoder setBytes:sections.data() length:sizeof(sections) atIndex:7];
    [encoder setBytes:&theta length:sizeof(theta) atIndex:8];
    [encoder setBytes:&query_scale length:sizeof(query_scale) atIndex:9];
    [encoder dispatchThreads:MTLSizeMake(rows, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(rows, 1, 1)];
    [encoder endEncoding];
    finish(command_buffer);

    const float* q = static_cast<const float*>(query.contents);
    const float* k = static_cast<const float*>(key.contents);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t pair = 0; pair < 2; ++pair) {
            const uint32_t axis = pair % 3;
            const float frequency = std::pow(theta, -2.0f * static_cast<float>(pair) /
                                                       static_cast<float>(head_dim));
            const float angle = static_cast<float>(positions[row * 3 + axis]) * frequency;
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            const size_t first = static_cast<size_t>(row) * head_dim + pair;
            const size_t second = static_cast<size_t>(row) * head_dim + 2 + pair;
            const float qx = initial_query[first];
            const float qy = initial_query[second];
            const float kx = initial_key[first];
            const float ky = initial_key[second];
            check_close(q[first], (qx * c - qy * s) * query_scale);
            check_close(q[second], (qy * c + qx * s) * query_scale);
            check_close(k[first], kx * c - ky * s);
            check_close(k[second], ky * c + kx * s);
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
        if (!library) {
            throw std::runtime_error("Metal attention gate shader compilation failed");
        }
        pipeline(device, library, "celeg_attention_output_gate_batch");
        test_packed_query_extract(device, library);
        test_headwise_gate(device, library);
        test_mrope_batch(device, library);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}