#include "backend/metal/model/runtime/rope_scaling.hpp"
#include "celeg/backend/cpu/rope.hpp"
#include "metal_inference_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kHeads = 2;
constexpr uint32_t kHeadDim = 8;
constexpr uint32_t kWidth = kHeads * kHeadDim;
constexpr uint32_t kPosition = 7;
constexpr uint32_t kPageTokens = 16;
constexpr float kTheta = 10000.0f;
constexpr float kQueryScale = 0.73f;
constexpr float kTolerance = 6.0e-5f;

std::string ns_string(NSString* value) {
    return value ? std::string(value.UTF8String) : std::string{};
}

id<MTLBuffer> float_buffer(id<MTLDevice> device, const std::vector<float>& values) {
    id<MTLBuffer> result = [device newBufferWithBytes:values.data()
                                                length:values.size() * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (!result) throw std::runtime_error("Metal scaled RoPE buffer allocation failed");
    return result;
}

id<MTLBuffer> zero_buffer(id<MTLDevice> device, size_t count) {
    return float_buffer(device, std::vector<float>(count, 0.0f));
}

std::vector<float> values(float phase) {
    std::vector<float> result(kWidth);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = 0.35f + 0.7f * std::sin(phase + static_cast<float>(i) * 0.27f) +
                    0.21f * std::cos(phase * 0.4f + static_cast<float>(i) * 0.13f);
    }
    return result;
}

uint32_t position_mode(celeg::RopePairingKind pairing, uint32_t rotary_dim) {
    const uint32_t pairing_mode =
        pairing == celeg::RopePairingKind::SplitHalf ? 1u : 2u;
    if (rotary_dim == kHeadDim) return pairing_mode;
    return pairing_mode | ((rotary_dim + 1u) << 2u);
}

celeg::RopePositionSpec rope_spec(celeg::RopePairingKind pairing,
                                  uint32_t rotary_dim) {
    celeg::RopePositionSpec rope;
    rope.theta = kTheta;
    rope.rotary_fraction = static_cast<double>(rotary_dim) /
                           static_cast<double>(kHeadDim);
    rope.scaling = celeg::LinearRopeScaling{2.5};
    rope.pairing = pairing;
    return rope;
}

void check_close(const float* actual, const std::vector<float>& expected,
                 const char* label) {
    for (size_t i = 0; i < expected.size(); ++i) {
        const float delta = std::abs(actual[i] - expected[i]);
        if (delta > kTolerance) {
            throw std::runtime_error(std::string(label) + " differs at " +
                                     std::to_string(i) + ": actual=" +
                                     std::to_string(actual[i]) + " expected=" +
                                     std::to_string(expected[i]) + " delta=" +
                                     std::to_string(delta));
        }
    }
}

void run_case(id<MTLDevice> device, id<MTLLibrary> library,
              id<MTLCommandQueue> queue, celeg::RopePairingKind pairing,
              uint32_t rotary_dim) {
    const celeg::RopePositionSpec rope = rope_spec(pairing, rotary_dim);
    const auto scaling =
        celeg::metal_model_detail::make_metal_rope_scaling_binding(rope);
    if (!scaling.scaled()) throw std::logic_error("scaled RoPE fixture is unscaled");

    std::vector<float> query_values = values(0.4f);
    std::vector<float> key_values = values(1.2f);
    const std::vector<float> value_values = values(2.0f);
    std::vector<float> expected_query = query_values;
    std::vector<float> expected_key = key_values;

    celeg::cpu_rope(expected_query.data(), static_cast<int>(kHeads),
                    static_cast<int>(kHeadDim), static_cast<int>(kPosition), rope);
    celeg::cpu_rope(expected_key.data(), static_cast<int>(kHeads),
                    static_cast<int>(kHeadDim), static_cast<int>(kPosition), rope);
    for (float& value : expected_query) value *= kQueryScale;

    id<MTLBuffer> query = float_buffer(device, query_values);
    id<MTLBuffer> key = float_buffer(device, key_values);
    id<MTLBuffer> value = float_buffer(device, value_values);
    const size_t cache_elements = static_cast<size_t>(kPosition + 1u) * kWidth;
    id<MTLBuffer> key_cache = zero_buffer(device, cache_elements);
    id<MTLBuffer> value_cache = zero_buffer(device, cache_elements);
    id<MTLBuffer> dummy_factors = float_buffer(device, {1.0f});

    NSError* error = nil;
    id<MTLFunction> function =
        [library newFunctionWithName:@"celeg_qk_position_store_kv_scaled"];
    if (!function) throw std::runtime_error("missing scaled QK RoPE kernel");
    id<MTLComputePipelineState> state =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!state) {
        throw std::runtime_error("scaled QK RoPE pipeline failed: " +
            (error ? ns_string(error.localizedDescription) : "unknown error"));
    }

    const uint32_t mode = position_mode(pairing, rotary_dim);
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
    [encoder setBytes:&kPosition length:sizeof(kPosition) atIndex:8];
    [encoder setBytes:&mode length:sizeof(mode) atIndex:9];
    [encoder setBytes:&kTheta length:sizeof(kTheta) atIndex:10];
    [encoder setBytes:&kQueryScale length:sizeof(kQueryScale) atIndex:11];
    [encoder setBytes:&kPageTokens length:sizeof(kPageTokens) atIndex:12];
    [encoder setBytes:&scaling.spec length:sizeof(scaling.spec) atIndex:13];
    [encoder setBuffer:dummy_factors offset:0 atIndex:14];
    [encoder setBuffer:dummy_factors offset:0 atIndex:15];
    [encoder dispatchThreads:MTLSizeMake(kHeads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(kHeads, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        const std::string detail = command.error
            ? ": " + ns_string(command.error.localizedDescription)
            : std::string{};
        throw std::runtime_error("scaled QK RoPE dispatch failed" + detail);
    }

    check_close(static_cast<const float*>(query.contents), expected_query, "query");
    check_close(static_cast<const float*>(key.contents), expected_key, "key");
    const float* cached_key = static_cast<const float*>(key_cache.contents) +
        static_cast<size_t>(kPosition) * kWidth;
    const float* cached_value = static_cast<const float*>(value_cache.contents) +
        static_cast<size_t>(kPosition) * kWidth;
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
            throw std::runtime_error("Metal scaled RoPE shader compilation failed: " +
                (error ? ns_string(error.localizedDescription) : "unknown error"));
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        run_case(device, library, queue, celeg::RopePairingKind::SplitHalf, kHeadDim);
        run_case(device, library, queue, celeg::RopePairingKind::AdjacentPairs, kHeadDim);
        run_case(device, library, queue, celeg::RopePairingKind::SplitHalf, 4u);
        run_case(device, library, queue, celeg::RopePairingKind::AdjacentPairs, 4u);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
