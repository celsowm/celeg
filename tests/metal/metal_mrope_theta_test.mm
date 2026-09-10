#include "celeg/backend/cpu/rope.hpp"
#include "metal_inference_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kHeadDim = 12;
constexpr uint32_t kHeads = 1;
constexpr uint32_t kCachePosition = 3;
constexpr uint32_t kPageTokens = 16;
constexpr float kQueryScale = 0.75f;
constexpr float kTolerance = 6.0e-5f;
constexpr std::array<uint32_t, 3> kMetalSections{2, 3, 1};
constexpr std::array<int, 3> kCpuSections{2, 3, 1};
constexpr std::array<int32_t, 3> kPosition{2, 7, 13};

std::string ns_string(NSString* value) {
    return value ? std::string(value.UTF8String) : std::string{};
}

id<MTLBuffer> float_buffer(id<MTLDevice> device, const std::vector<float>& values) {
    id<MTLBuffer> result = [device newBufferWithBytes:values.data()
                                                length:values.size() * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (!result) throw std::runtime_error("Metal MRoPE theta buffer allocation failed");
    return result;
}

std::vector<float> values(float phase) {
    std::vector<float> result(kHeadDim);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = std::sin(phase + static_cast<float>(i) * 0.31f) * 0.8f +
                    std::cos(phase * 0.6f + static_cast<float>(i) * 0.17f) * 0.2f;
    }
    return result;
}

void check_close(const float* actual, const std::vector<float>& expected,
                 const std::string& label) {
    for (size_t i = 0; i < expected.size(); ++i) {
        const float delta = std::abs(actual[i] - expected[i]);
        if (delta > kTolerance) {
            throw std::runtime_error(label + " differs at " + std::to_string(i) +
                                     ": actual=" + std::to_string(actual[i]) +
                                     " expected=" + std::to_string(expected[i]) +
                                     " delta=" + std::to_string(delta));
        }
    }
}

void run_case(id<MTLDevice> device, id<MTLLibrary> library,
              id<MTLCommandQueue> queue, float theta, bool interleaved) {
    celeg::RopePositionSpec rope;
    rope.theta = theta;
    rope.rotary_fraction = 1.0;
    rope.pairing = celeg::RopePairingKind::SplitHalf;
    rope.scaling = celeg::NoRopeScaling{};

    std::vector<float> query_values = values(0.25f);
    std::vector<float> key_values = values(0.9f);
    const std::vector<float> value_values = values(1.4f);
    std::vector<float> expected_query = query_values;
    std::vector<float> expected_key = key_values;
    celeg::cpu_rope_mrope(expected_query.data(), 1, static_cast<int>(kHeadDim),
                          kPosition, kCpuSections, interleaved, rope);
    celeg::cpu_rope_mrope(expected_key.data(), 1, static_cast<int>(kHeadDim),
                          kPosition, kCpuSections, interleaved, rope);
    for (float& value : expected_query) value *= kQueryScale;

    id<MTLBuffer> query = float_buffer(device, query_values);
    id<MTLBuffer> key = float_buffer(device, key_values);
    id<MTLBuffer> value = float_buffer(device, value_values);
    const size_t cache_count = static_cast<size_t>(kCachePosition + 1u) * kHeadDim;
    id<MTLBuffer> key_cache = float_buffer(device, std::vector<float>(cache_count, 0.0f));
    id<MTLBuffer> value_cache = float_buffer(device, std::vector<float>(cache_count, 0.0f));

    NSError* error = nil;
    id<MTLFunction> function =
        [library newFunctionWithName:@"celeg_qk_mrope_position_store_kv"];
    if (!function) throw std::runtime_error("missing production MRoPE kernel");
    id<MTLComputePipelineState> state =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!state) {
        throw std::runtime_error("MRoPE theta pipeline failed: " +
            (error ? ns_string(error.localizedDescription) : "unknown error"));
    }

    const uint32_t interleaved_value = interleaved ? 1u : 0u;
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setBuffer:query offset:0 atIndex:0];
    [encoder setBuffer:key offset:0 atIndex:1];
    [encoder setBuffer:value offset:0 atIndex:2];
    [encoder setBuffer:key_cache offset:0 atIndex:3];
    [encoder setBuffer:value_cache offset:0 atIndex:4];
    [encoder setBytes:&kHeads length:sizeof(kHeads) atIndex:5];
    [encoder setBytes:&kHeads length:sizeof(kHeads) atIndex:6];
    [encoder setBytes:&kHeadDim length:sizeof(kHeadDim) atIndex:7];
    [encoder setBytes:&kCachePosition length:sizeof(kCachePosition) atIndex:8];
    [encoder setBytes:kPosition.data() length:sizeof(kPosition) atIndex:9];
    [encoder setBytes:kMetalSections.data() length:sizeof(kMetalSections) atIndex:10];
    [encoder setBytes:&theta length:sizeof(theta) atIndex:11];
    [encoder setBytes:&kQueryScale length:sizeof(kQueryScale) atIndex:12];
    [encoder setBytes:&kPageTokens length:sizeof(kPageTokens) atIndex:13];
    [encoder setBytes:&interleaved_value length:sizeof(interleaved_value) atIndex:14];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        const std::string detail = command.error
            ? ": " + ns_string(command.error.localizedDescription)
            : std::string{};
        throw std::runtime_error("MRoPE theta dispatch failed" + detail);
    }

    check_close(static_cast<const float*>(query.contents), expected_query, "query");
    check_close(static_cast<const float*>(key.contents), expected_key, "key");
    const float* cached_key = static_cast<const float*>(key_cache.contents) +
        static_cast<size_t>(kCachePosition) * kHeadDim;
    const float* cached_value = static_cast<const float*>(value_cache.contents) +
        static_cast<size_t>(kCachePosition) * kHeadDim;
    check_close(cached_key, expected_key, "key cache");
    check_close(cached_value, value_values, "value cache");
}

}

int main() {
    try {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("no default Metal device is available");
        NSError* error = nil;
        NSString* source =
            [NSString stringWithUTF8String:celeg::metal_detail::kInferenceShader];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            throw std::runtime_error("Metal MRoPE theta shader compilation failed: " +
                (error ? ns_string(error.localizedDescription) : "unknown error"));
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        run_case(device, library, queue, 10000.0f, false);
        run_case(device, library, queue, 10000.0f, true);
        run_case(device, library, queue, 1000000.0f, false);
        run_case(device, library, queue, 1000000.0f, true);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
