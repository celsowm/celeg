#include "metal_inference_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
    if (!result) throw std::runtime_error("Metal QK store buffer allocation failed");
    return result;
}

id<MTLBuffer> zero_buffer(id<MTLDevice> device, size_t elements) {
    id<MTLBuffer> result = [device newBufferWithLength:elements * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (!result) throw std::runtime_error("Metal QK store zero buffer allocation failed");
    std::fill_n(static_cast<float*>(result.contents), elements, 0.0f);
    return result;
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                     const char* name) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:
        [NSString stringWithUTF8String:name]];
    if (!function) {
        throw std::runtime_error(std::string("Metal QK store function missing: ") + name);
    }
    id<MTLComputePipelineState> result = [device newComputePipelineStateWithFunction:function
                                                                                  error:&error];
    if (!result) {
        throw std::runtime_error("Metal QK store pipeline failed: " +
            (error ? ns_string(error.localizedDescription) : "unknown error"));
    }
    return result;
}

void check_identical(id<MTLBuffer> left, id<MTLBuffer> right, size_t elements,
                     const char* label) {
    const auto* a = static_cast<const float*>(left.contents);
    const auto* b = static_cast<const float*>(right.contents);
    for (size_t index = 0; index < elements; ++index) {
        if (std::memcmp(a + index, b + index, sizeof(float)) != 0) {
            throw std::runtime_error(std::string(label) + " differs at index " +
                                     std::to_string(index) + ": " +
                                     std::to_string(a[index]) + " vs " +
                                     std::to_string(b[index]));
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
            throw std::runtime_error("Metal QK store shader compilation failed: " +
                (error ? ns_string(error.localizedDescription) : "unknown error"));
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) throw std::runtime_error("Metal QK store command queue failed");

        constexpr uint32_t rows = 3;
        constexpr uint32_t query_heads = 4;
        constexpr uint32_t key_heads = 2;
        constexpr uint32_t head_dim = 8;
        constexpr uint32_t base_position = 2;
        constexpr float theta = 10000.0f;
        constexpr float query_scale = 0.75f;
        constexpr float query_epsilon = 1.0e-5f;
        constexpr float key_epsilon = 2.0e-5f;
        constexpr NSUInteger store_threads = 256;
        constexpr NSUInteger cooperative_threads = 32;

        const size_t query_elements = static_cast<size_t>(rows) * query_heads * head_dim;
        const size_t key_elements = static_cast<size_t>(rows) * key_heads * head_dim;
        const size_t cache_rows = base_position + rows + 1;
        const size_t cache_elements = cache_rows * key_heads * head_dim;

        std::vector<float> query_values(query_elements);
        std::vector<float> key_values(key_elements);
        std::vector<float> value_values(key_elements);
        std::vector<float> query_weight(head_dim);
        std::vector<float> key_weight(head_dim);
        for (size_t i = 0; i < query_elements; ++i) {
            query_values[i] = static_cast<float>((static_cast<int>(i % 17) - 8) * 0.125);
        }
        for (size_t i = 0; i < key_elements; ++i) {
            key_values[i] = static_cast<float>((static_cast<int>(i % 13) - 6) * 0.1875);
            value_values[i] = static_cast<float>((static_cast<int>(i % 19) - 9) * 0.0625);
        }
        for (uint32_t i = 0; i < head_dim; ++i) {
            query_weight[i] = 0.75f + static_cast<float>(i) * 0.03125f;
            key_weight[i] = 0.875f - static_cast<float>(i) * 0.015625f;
        }

        id<MTLBuffer> baseline_query = buffer(device, query_values);
        id<MTLBuffer> baseline_key = buffer(device, key_values);
        id<MTLBuffer> fused_query = buffer(device, query_values);
        id<MTLBuffer> fused_key = buffer(device, key_values);
        id<MTLBuffer> cooperative_query = buffer(device, query_values);
        id<MTLBuffer> cooperative_key = buffer(device, key_values);
        id<MTLBuffer> value = buffer(device, value_values);
        id<MTLBuffer> query_norm = buffer(device, query_weight);
        id<MTLBuffer> key_norm = buffer(device, key_weight);
        id<MTLBuffer> baseline_key_cache = zero_buffer(device, cache_elements);
        id<MTLBuffer> baseline_value_cache = zero_buffer(device, cache_elements);
        id<MTLBuffer> fused_key_cache = zero_buffer(device, cache_elements);
        id<MTLBuffer> fused_value_cache = zero_buffer(device, cache_elements);
        id<MTLBuffer> cooperative_key_cache = zero_buffer(device, cache_elements);
        id<MTLBuffer> cooperative_value_cache = zero_buffer(device, cache_elements);

        id<MTLComputePipelineState> qk =
            pipeline(device, library, "celeg_qk_norm_rope_batch_split");
        id<MTLComputePipelineState> store =
            pipeline(device, library, "celeg_store_kv_batch_2d");
        id<MTLComputePipelineState> fused =
            pipeline(device, library, "celeg_qk_norm_rope_batch_split_store_kv");
        id<MTLComputePipelineState> cooperative = pipeline(
            device, library, "celeg_qk_norm_rope_batch_split_cooperative_store_kv");
        if (store.maxTotalThreadsPerThreadgroup < store_threads) {
            throw std::runtime_error("Metal QK store test needs 256 KV-store threads");
        }
        if (cooperative.maxTotalThreadsPerThreadgroup < cooperative_threads) {
            throw std::runtime_error("Metal QK store test needs 32 cooperative threads");
        }

        const NSUInteger head_dispatch = static_cast<NSUInteger>(rows) * query_heads;
        const NSUInteger qk_threads =
            std::min<NSUInteger>(head_dispatch, qk.maxTotalThreadsPerThreadgroup);
        const NSUInteger fused_threads =
            std::min<NSUInteger>(head_dispatch, fused.maxTotalThreadsPerThreadgroup);
        const uint32_t kv_width = key_heads * head_dim;

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        [encoder setComputePipelineState:qk];
        [encoder setBuffer:baseline_query offset:0 atIndex:0];
        [encoder setBuffer:query_norm offset:0 atIndex:1];
        [encoder setBuffer:baseline_key offset:0 atIndex:2];
        [encoder setBuffer:key_norm offset:0 atIndex:3];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
        [encoder setBytes:&query_heads length:sizeof(query_heads) atIndex:5];
        [encoder setBytes:&key_heads length:sizeof(key_heads) atIndex:6];
        [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:7];
        [encoder setBytes:&base_position length:sizeof(base_position) atIndex:8];
        [encoder setBytes:&theta length:sizeof(theta) atIndex:9];
        [encoder setBytes:&query_scale length:sizeof(query_scale) atIndex:10];
        [encoder setBytes:&query_epsilon length:sizeof(query_epsilon) atIndex:11];
        [encoder setBytes:&key_epsilon length:sizeof(key_epsilon) atIndex:12];
        [encoder dispatchThreads:MTLSizeMake(head_dispatch, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(qk_threads, 1, 1)];

        [encoder setComputePipelineState:store];
        [encoder setBuffer:baseline_key offset:0 atIndex:0];
        [encoder setBuffer:value offset:0 atIndex:1];
        [encoder setBuffer:baseline_key_cache offset:0 atIndex:2];
        [encoder setBuffer:baseline_value_cache offset:0 atIndex:3];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
        [encoder setBytes:&base_position length:sizeof(base_position) atIndex:5];
        [encoder setBytes:&kv_width length:sizeof(kv_width) atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake(
            (kv_width + store_threads - 1u) / store_threads, rows, 1)
               threadsPerThreadgroup:MTLSizeMake(store_threads, 1, 1)];

        [encoder setComputePipelineState:fused];
        [encoder setBuffer:fused_query offset:0 atIndex:0];
        [encoder setBuffer:query_norm offset:0 atIndex:1];
        [encoder setBuffer:fused_key offset:0 atIndex:2];
        [encoder setBuffer:key_norm offset:0 atIndex:3];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
        [encoder setBytes:&query_heads length:sizeof(query_heads) atIndex:5];
        [encoder setBytes:&key_heads length:sizeof(key_heads) atIndex:6];
        [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:7];
        [encoder setBytes:&base_position length:sizeof(base_position) atIndex:8];
        [encoder setBytes:&theta length:sizeof(theta) atIndex:9];
        [encoder setBytes:&query_scale length:sizeof(query_scale) atIndex:10];
        [encoder setBytes:&query_epsilon length:sizeof(query_epsilon) atIndex:11];
        [encoder setBytes:&key_epsilon length:sizeof(key_epsilon) atIndex:12];
        [encoder setBuffer:value offset:0 atIndex:13];
        [encoder setBuffer:fused_key_cache offset:0 atIndex:14];
        [encoder setBuffer:fused_value_cache offset:0 atIndex:15];
        [encoder dispatchThreads:MTLSizeMake(head_dispatch, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(fused_threads, 1, 1)];

        [encoder setComputePipelineState:cooperative];
        [encoder setBuffer:cooperative_query offset:0 atIndex:0];
        [encoder setBuffer:query_norm offset:0 atIndex:1];
        [encoder setBuffer:cooperative_key offset:0 atIndex:2];
        [encoder setBuffer:key_norm offset:0 atIndex:3];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
        [encoder setBytes:&query_heads length:sizeof(query_heads) atIndex:5];
        [encoder setBytes:&key_heads length:sizeof(key_heads) atIndex:6];
        [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:7];
        [encoder setBytes:&base_position length:sizeof(base_position) atIndex:8];
        [encoder setBytes:&theta length:sizeof(theta) atIndex:9];
        [encoder setBytes:&query_scale length:sizeof(query_scale) atIndex:10];
        [encoder setBytes:&query_epsilon length:sizeof(query_epsilon) atIndex:11];
        [encoder setBytes:&key_epsilon length:sizeof(key_epsilon) atIndex:12];
        [encoder setBuffer:value offset:0 atIndex:13];
        [encoder setBuffer:cooperative_key_cache offset:0 atIndex:14];
        [encoder setBuffer:cooperative_value_cache offset:0 atIndex:15];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(cooperative_threads, 1, 1)];

        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("Metal QK store dispatch failed");
        }

        check_identical(baseline_query, fused_query, query_elements, "fused query");
        check_identical(baseline_key, fused_key, key_elements, "fused key");
        check_identical(baseline_key_cache, fused_key_cache, cache_elements, "fused key cache");
        check_identical(baseline_value_cache, fused_value_cache, cache_elements,
                        "fused value cache");
        check_identical(baseline_query, cooperative_query, query_elements,
                        "cooperative query");
        check_identical(baseline_key_cache, cooperative_key_cache, cache_elements,
                        "cooperative key cache");
        check_identical(baseline_value_cache, cooperative_value_cache, cache_elements,
                        "cooperative value cache");

        std::cout << "metal SplitHalf KV publication kernels passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
