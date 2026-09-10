#include "celeg/backend/cpu/rope.hpp"
#include "metal_inference_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kHeadDim = 12;
constexpr float kTheta = 10000.0f;
constexpr float kTolerance = 5.0e-5f;
constexpr std::array<uint32_t, 3> kMetalSections{2, 3, 1};
constexpr std::array<int, 3> kCpuSections{2, 3, 1};

celeg::RopePositionSpec rope_spec() {
    celeg::RopePositionSpec rope;
    rope.theta = kTheta;
    rope.rotary_fraction = 1.0;
    rope.pairing = celeg::RopePairingKind::SplitHalf;
    rope.scaling = celeg::NoRopeScaling{};
    return rope;
}

std::string ns_string(NSString* value) {
    return value ? std::string(value.UTF8String) : std::string{};
}

id<MTLBuffer> float_buffer(id<MTLDevice> device, const std::vector<float>& values) {
    id<MTLBuffer> result = [device newBufferWithBytes:values.data()
                                                length:values.size() * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (!result) throw std::runtime_error("Metal MRoPE float buffer allocation failed");
    return result;
}

id<MTLBuffer> int_buffer(id<MTLDevice> device, const std::vector<int32_t>& values) {
    id<MTLBuffer> result = [device newBufferWithBytes:values.data()
                                                length:values.size() * sizeof(int32_t)
                                               options:MTLResourceStorageModeShared];
    if (!result) throw std::runtime_error("Metal MRoPE position buffer allocation failed");
    return result;
}

id<MTLBuffer> zero_float_buffer(id<MTLDevice> device, size_t count) {
    return float_buffer(device, std::vector<float>(count, 0.0f));
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                     const char* name) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:
        [NSString stringWithUTF8String:name]];
    if (!function) throw std::runtime_error(std::string("missing Metal MRoPE kernel: ") + name);
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!result) {
        throw std::runtime_error("Metal MRoPE pipeline failed: " +
            (error ? ns_string(error.localizedDescription) : "unknown error"));
    }
    return result;
}

void wait(id<MTLCommandBuffer> command, const char* label) {
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        const std::string detail = command.error
            ? ": " + ns_string(command.error.localizedDescription)
            : std::string{};
        throw std::runtime_error(std::string(label) + " dispatch failed" + detail);
    }
}

void check_close(const float* actual, const std::vector<float>& expected,
                 const char* label, float tolerance = kTolerance) {
    for (size_t index = 0; index < expected.size(); ++index) {
        const float delta = std::abs(actual[index] - expected[index]);
        if (delta > tolerance) {
            throw std::runtime_error(std::string(label) + " differs at " +
                std::to_string(index) + ": actual=" + std::to_string(actual[index]) +
                " expected=" + std::to_string(expected[index]) +
                " delta=" + std::to_string(delta));
        }
    }
}

void scale_values(float* values, float scale) {
    for (uint32_t d = 0; d < kHeadDim; ++d) values[d] *= scale;
}

void cpu_mrope(float* values,
               const std::array<int32_t, 3>& position,
               bool interleaved,
               float scale) {
    const celeg::RopePositionSpec rope = rope_spec();
    celeg::cpu_rope_mrope(values, 1, static_cast<int>(kHeadDim), position,
                          kCpuSections, interleaved, rope);
    if (scale != 1.0f) scale_values(values, scale);
}

void cpu_norm_mrope(float* values,
                    const std::vector<float>& weight,
                    float epsilon,
                    const std::array<int32_t, 3>& position,
                    bool interleaved,
                    float scale) {
    const celeg::RopePositionSpec rope = rope_spec();
    celeg::cpu_qk_norm_rope_mrope(
        values, weight.data(), 1, static_cast<int>(kHeadDim), position,
        kCpuSections, interleaved, rope, epsilon);
    if (scale != 1.0f) scale_values(values, scale);
}

std::vector<float> initial_values(float phase, size_t count) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = std::sin(phase + static_cast<float>(i) * 0.37f) * 0.8f +
                    std::cos(phase * 0.5f + static_cast<float>(i) * 0.11f) * 0.2f;
    }
    return values;
}

std::vector<float> weights(float base, float step) {
    std::vector<float> result(kHeadDim);
    for (uint32_t d = 0; d < kHeadDim; ++d) {
        result[d] = base + step * static_cast<float>(d);
    }
    return result;
}

void run_position_store_case(id<MTLDevice> device, id<MTLLibrary> library,
                             id<MTLCommandQueue> queue, bool is_interleaved) {
    constexpr uint32_t query_heads = 1;
    constexpr uint32_t key_heads = 1;
    constexpr uint32_t cache_position = 3;
    constexpr uint32_t page_tokens = 16;
    constexpr float query_scale = 0.75f;
    const std::array<int32_t, 3> position{2, 7, 13};
    const uint32_t interleaved = is_interleaved ? 1u : 0u;

    std::vector<float> query_values = initial_values(0.2f, kHeadDim);
    std::vector<float> key_values = initial_values(0.7f, kHeadDim);
    const std::vector<float> value_values = initial_values(1.3f, kHeadDim);
    std::vector<float> expected_query = query_values;
    std::vector<float> expected_key = key_values;
    cpu_mrope(expected_query.data(), position, is_interleaved, query_scale);
    cpu_mrope(expected_key.data(), position, is_interleaved, 1.0f);

    const size_t cache_count = static_cast<size_t>(cache_position + 1) * kHeadDim;
    id<MTLBuffer> query = float_buffer(device, query_values);
    id<MTLBuffer> key = float_buffer(device, key_values);
    id<MTLBuffer> value = float_buffer(device, value_values);
    id<MTLBuffer> key_cache = zero_float_buffer(device, cache_count);
    id<MTLBuffer> value_cache = zero_float_buffer(device, cache_count);
    id<MTLComputePipelineState> state =
        pipeline(device, library, "celeg_qk_mrope_position_store_kv");

    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setBuffer:query offset:0 atIndex:0];
    [encoder setBuffer:key offset:0 atIndex:1];
    [encoder setBuffer:value offset:0 atIndex:2];
    [encoder setBuffer:key_cache offset:0 atIndex:3];
    [encoder setBuffer:value_cache offset:0 atIndex:4];
    [encoder setBytes:&query_heads length:sizeof(query_heads) atIndex:5];
    [encoder setBytes:&key_heads length:sizeof(key_heads) atIndex:6];
    [encoder setBytes:&kHeadDim length:sizeof(kHeadDim) atIndex:7];
    [encoder setBytes:&cache_position length:sizeof(cache_position) atIndex:8];
    [encoder setBytes:position.data() length:sizeof(position) atIndex:9];
    [encoder setBytes:kMetalSections.data() length:sizeof(kMetalSections) atIndex:10];
    [encoder setBytes:&kTheta length:sizeof(kTheta) atIndex:11];
    [encoder setBytes:&query_scale length:sizeof(query_scale) atIndex:12];
    [encoder setBytes:&page_tokens length:sizeof(page_tokens) atIndex:13];
    [encoder setBytes:&interleaved length:sizeof(interleaved) atIndex:14];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    wait(command, "Metal position-store MRoPE");

    check_close(static_cast<const float*>(query.contents), expected_query, "position query");
    check_close(static_cast<const float*>(key.contents), expected_key, "position key");
    const auto* key_cache_values = static_cast<const float*>(key_cache.contents) +
        static_cast<size_t>(cache_position) * kHeadDim;
    const auto* value_cache_values = static_cast<const float*>(value_cache.contents) +
        static_cast<size_t>(cache_position) * kHeadDim;
    check_close(key_cache_values, expected_key, "position key cache");
    check_close(value_cache_values, value_values, "position value cache", 0.0f);
}

void run_fused_norm_case(id<MTLDevice> device, id<MTLLibrary> library,
                         id<MTLCommandQueue> queue, bool is_interleaved) {
    constexpr uint32_t query_heads = 1;
    constexpr uint32_t key_heads = 1;
    constexpr uint32_t cache_position = 2;
    constexpr uint32_t page_tokens = 16;
    constexpr float query_scale = 0.625f;
    constexpr float query_epsilon = 1.0e-5f;
    constexpr float key_epsilon = 2.0e-5f;
    const std::array<int32_t, 3> position{3, 9, 17};
    const uint32_t interleaved = is_interleaved ? 1u : 0u;

    std::vector<float> query_values = initial_values(0.4f, kHeadDim);
    std::vector<float> key_values = initial_values(1.1f, kHeadDim);
    const std::vector<float> value_values = initial_values(1.9f, kHeadDim);
    const std::vector<float> query_weight = weights(0.8f, 0.025f);
    const std::vector<float> key_weight = weights(0.95f, -0.018f);
    std::vector<float> expected_query = query_values;
    std::vector<float> expected_key = key_values;
    cpu_norm_mrope(expected_query.data(), query_weight, query_epsilon,
                   position, is_interleaved, query_scale);
    cpu_norm_mrope(expected_key.data(), key_weight, key_epsilon,
                   position, is_interleaved, 1.0f);

    const size_t cache_count = static_cast<size_t>(cache_position + 1) * kHeadDim;
    id<MTLBuffer> query = float_buffer(device, query_values);
    id<MTLBuffer> key = float_buffer(device, key_values);
    id<MTLBuffer> value = float_buffer(device, value_values);
    id<MTLBuffer> query_norm = float_buffer(device, query_weight);
    id<MTLBuffer> key_norm = float_buffer(device, key_weight);
    id<MTLBuffer> key_cache = zero_float_buffer(device, cache_count);
    id<MTLBuffer> value_cache = zero_float_buffer(device, cache_count);
    id<MTLComputePipelineState> state =
        pipeline(device, library, "celeg_qk_norm_mrope_store_kv");

    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setBuffer:query offset:0 atIndex:0];
    [encoder setBuffer:query_norm offset:0 atIndex:1];
    [encoder setBuffer:key offset:0 atIndex:2];
    [encoder setBuffer:key_norm offset:0 atIndex:3];
    [encoder setBuffer:value offset:0 atIndex:4];
    [encoder setBuffer:key_cache offset:0 atIndex:5];
    [encoder setBuffer:value_cache offset:0 atIndex:6];
    [encoder setBytes:&query_heads length:sizeof(query_heads) atIndex:7];
    [encoder setBytes:&key_heads length:sizeof(key_heads) atIndex:8];
    [encoder setBytes:&kHeadDim length:sizeof(kHeadDim) atIndex:9];
    [encoder setBytes:&cache_position length:sizeof(cache_position) atIndex:10];
    [encoder setBytes:position.data() length:sizeof(position) atIndex:11];
    [encoder setBytes:kMetalSections.data() length:sizeof(kMetalSections) atIndex:12];
    [encoder setBytes:&kTheta length:sizeof(kTheta) atIndex:13];
    [encoder setBytes:&query_scale length:sizeof(query_scale) atIndex:14];
    [encoder setBytes:&query_epsilon length:sizeof(query_epsilon) atIndex:15];
    [encoder setBytes:&key_epsilon length:sizeof(key_epsilon) atIndex:16];
    [encoder setBytes:&page_tokens length:sizeof(page_tokens) atIndex:17];
    [encoder setBytes:&interleaved length:sizeof(interleaved) atIndex:18];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    wait(command, "Metal fused norm MRoPE");

    check_close(static_cast<const float*>(query.contents), expected_query, "fused query", 2.0e-4f);
    check_close(static_cast<const float*>(key.contents), expected_key, "fused key", 2.0e-4f);
    const auto* key_cache_values = static_cast<const float*>(key_cache.contents) +
        static_cast<size_t>(cache_position) * kHeadDim;
    const auto* value_cache_values = static_cast<const float*>(value_cache.contents) +
        static_cast<size_t>(cache_position) * kHeadDim;
    check_close(key_cache_values, expected_key, "fused key cache", 2.0e-4f);
    check_close(value_cache_values, value_values, "fused value cache", 0.0f);
}

void run_batch_case(id<MTLDevice> device, id<MTLLibrary> library,
                    id<MTLCommandQueue> queue, bool is_interleaved) {
    constexpr uint32_t rows = 2;
    constexpr uint32_t query_heads = 1;
    constexpr uint32_t key_heads = 1;
    constexpr float query_scale = 0.875f;
    const std::array<std::array<int32_t, 3>, rows> positions{{
        {2, 5, 11},
        {4, 13, 19},
    }};
    const uint32_t interleaved = is_interleaved ? 1u : 0u;

    std::vector<int32_t> flattened_positions;
    for (const auto& position : positions) {
        flattened_positions.insert(flattened_positions.end(), position.begin(), position.end());
    }
    std::vector<float> query_values = initial_values(0.15f, rows * kHeadDim);
    std::vector<float> key_values = initial_values(0.95f, rows * kHeadDim);
    std::vector<float> expected_query = query_values;
    std::vector<float> expected_key = key_values;
    for (uint32_t row = 0; row < rows; ++row) {
        float* expected_query_row = expected_query.data() + static_cast<size_t>(row) * kHeadDim;
        float* expected_key_row = expected_key.data() + static_cast<size_t>(row) * kHeadDim;
        cpu_mrope(expected_query_row, positions[row], is_interleaved, query_scale);
        cpu_mrope(expected_key_row, positions[row], is_interleaved, 1.0f);
    }

    id<MTLBuffer> query = float_buffer(device, query_values);
    id<MTLBuffer> key = float_buffer(device, key_values);
    id<MTLBuffer> rope_positions = int_buffer(device, flattened_positions);
    id<MTLComputePipelineState> state =
        pipeline(device, library, "celeg_qk_mrope_position_batch");

    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setBuffer:query offset:0 atIndex:0];
    [encoder setBuffer:key offset:0 atIndex:1];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
    [encoder setBytes:&query_heads length:sizeof(query_heads) atIndex:3];
    [encoder setBytes:&key_heads length:sizeof(key_heads) atIndex:4];
    [encoder setBytes:&kHeadDim length:sizeof(kHeadDim) atIndex:5];
    [encoder setBuffer:rope_positions offset:0 atIndex:6];
    [encoder setBytes:kMetalSections.data() length:sizeof(kMetalSections) atIndex:7];
    [encoder setBytes:&kTheta length:sizeof(kTheta) atIndex:8];
    [encoder setBytes:&query_scale length:sizeof(query_scale) atIndex:9];
    [encoder setBytes:&interleaved length:sizeof(interleaved) atIndex:10];
    [encoder dispatchThreads:MTLSizeMake(rows, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(rows, 1, 1)];
    [encoder endEncoding];
    wait(command, "Metal batched MRoPE");

    check_close(static_cast<const float*>(query.contents), expected_query, "batch query");
    check_close(static_cast<const float*>(key.contents), expected_key, "batch key");
}

void prove_layouts_are_distinct() {
    const std::array<int32_t, 3> position{2, 7, 13};
    std::vector<float> interleaved = initial_values(0.31f, kHeadDim);
    std::vector<float> sectioned = interleaved;
    cpu_mrope(interleaved.data(), position, true, 1.0f);
    cpu_mrope(sectioned.data(), position, false, 1.0f);
    bool differs = false;
    for (size_t i = 0; i < interleaved.size(); ++i) {
        differs = differs || std::abs(interleaved[i] - sectioned[i]) > 1.0e-4f;
    }
    if (!differs) throw std::runtime_error("MRoPE test vectors do not distinguish layouts");
}

}

int main() {
    try {
        prove_layouts_are_distinct();
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("no default Metal device is available");
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:
            [NSString stringWithUTF8String:celeg::metal_detail::kInferenceShader]
                                                          options:nil error:&error];
        if (!library) {
            throw std::runtime_error("Metal MRoPE shader compilation failed: " +
                (error ? ns_string(error.localizedDescription) : "unknown error"));
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) throw std::runtime_error("Metal MRoPE command queue failed");

        for (bool interleaved : std::array<bool, 2>{true, false}) {
            run_position_store_case(device, library, queue, interleaved);
            run_fused_norm_case(device, library, queue, interleaved);
            run_batch_case(device, library, queue, interleaved);
        }

        std::cout << "metal interleaved and sectioned MRoPE CPU parity passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
