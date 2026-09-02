#include "metal_inference_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string ns_string(NSString* value) {
    return value ? std::string(value.UTF8String) : std::string{};
}

id<MTLBuffer> buffer(id<MTLDevice> device, const std::vector<float>& values) {
    id<MTLBuffer> result = [device newBufferWithBytes:values.data()
                                                length:values.size() * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (!result) throw std::runtime_error("Metal KV test buffer allocation failed");
    return result;
}

id<MTLBuffer> zero_buffer(id<MTLDevice> device, size_t elements) {
    id<MTLBuffer> result = [device newBufferWithLength:elements * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (!result) throw std::runtime_error("Metal KV test zero buffer allocation failed");
    std::fill_n(static_cast<float*>(result.contents), elements, 0.0f);
    return result;
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                     const char* name) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:
        [NSString stringWithUTF8String:name]];
    if (!function) throw std::runtime_error(std::string("Metal KV function missing: ") + name);
    id<MTLComputePipelineState> result = [device newComputePipelineStateWithFunction:function
                                                                                  error:&error];
    if (!result) {
        throw std::runtime_error("Metal KV pipeline failed: " +
            (error ? ns_string(error.localizedDescription) : "unknown error"));
    }
    return result;
}

void check_equal(const float* actual, const std::vector<float>& expected,
                 const char* label) {
    for (size_t index = 0; index < expected.size(); ++index) {
        if (actual[index] != expected[index]) {
            throw std::runtime_error(std::string(label) + " differs at index " +
                                     std::to_string(index));
        }
    }
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
            throw std::runtime_error("Metal KV shader compilation failed: " +
                (error ? ns_string(error.localizedDescription) : "unknown error"));
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) throw std::runtime_error("Metal KV command queue failed");

        constexpr uint32_t rows = 5;
        constexpr uint32_t width = 513;
        constexpr uint32_t base_position = 3;
        constexpr uint32_t page_tokens = 16;
        constexpr uint32_t cache_rows = base_position + rows + 2;
        const size_t source_elements = static_cast<size_t>(rows) * width;
        const size_t cache_elements = static_cast<size_t>(cache_rows) * width;

        std::vector<float> key_values(source_elements);
        std::vector<float> value_values(source_elements);
        for (size_t index = 0; index < source_elements; ++index) {
            key_values[index] = static_cast<float>(index + 1);
            value_values[index] = -static_cast<float>(index + 1);
        }
        std::vector<float> expected_key(cache_elements, 0.0f);
        std::vector<float> expected_value(cache_elements, 0.0f);
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t column = 0; column < width; ++column) {
                const size_t source = static_cast<size_t>(row) * width + column;
                const size_t destination =
                    static_cast<size_t>(base_position + row) * width + column;
                expected_key[destination] = key_values[source];
                expected_value[destination] = value_values[source];
            }
        }

        id<MTLBuffer> key = buffer(device, key_values);
        id<MTLBuffer> value = buffer(device, value_values);
        id<MTLBuffer> old_key_cache = zero_buffer(device, cache_elements);
        id<MTLBuffer> old_value_cache = zero_buffer(device, cache_elements);
        id<MTLBuffer> new_key_cache = zero_buffer(device, cache_elements);
        id<MTLBuffer> new_value_cache = zero_buffer(device, cache_elements);
        id<MTLComputePipelineState> old_state = pipeline(device, library, "celeg_store_kv_batch");
        id<MTLComputePipelineState> new_state = pipeline(device, library, "celeg_store_kv_batch_2d");

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:old_state];
        [encoder setBuffer:key offset:0 atIndex:0];
        [encoder setBuffer:value offset:0 atIndex:1];
        [encoder setBuffer:old_key_cache offset:0 atIndex:2];
        [encoder setBuffer:old_value_cache offset:0 atIndex:3];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
        [encoder setBytes:&base_position length:sizeof(base_position) atIndex:5];
        [encoder setBytes:&width length:sizeof(width) atIndex:6];
        [encoder setBytes:&page_tokens length:sizeof(page_tokens) atIndex:7];
        const NSUInteger old_count = static_cast<NSUInteger>(rows) * width;
        [encoder dispatchThreads:MTLSizeMake(old_count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(
               std::min<NSUInteger>(old_count, old_state.maxTotalThreadsPerThreadgroup), 1, 1)];

        constexpr NSUInteger threads = 256;
        if (new_state.maxTotalThreadsPerThreadgroup < threads) {
            throw std::runtime_error("Metal KV 2D pipeline cannot run 256 threads");
        }
        [encoder setComputePipelineState:new_state];
        [encoder setBuffer:key offset:0 atIndex:0];
        [encoder setBuffer:value offset:0 atIndex:1];
        [encoder setBuffer:new_key_cache offset:0 atIndex:2];
        [encoder setBuffer:new_value_cache offset:0 atIndex:3];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
        [encoder setBytes:&base_position length:sizeof(base_position) atIndex:5];
        [encoder setBytes:&width length:sizeof(width) atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake((width + threads - 1u) / threads, rows, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("Metal KV dispatch failed");
        }

        check_equal(static_cast<const float*>(old_key_cache.contents), expected_key,
                    "legacy key cache");
        check_equal(static_cast<const float*>(old_value_cache.contents), expected_value,
                    "legacy value cache");
        check_equal(static_cast<const float*>(new_key_cache.contents), expected_key,
                    "2D key cache");
        check_equal(static_cast<const float*>(new_value_cache.contents), expected_value,
                    "2D value cache");

        std::cout << "metal KV store kernels passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
