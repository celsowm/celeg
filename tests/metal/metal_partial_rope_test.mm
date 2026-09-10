#include "celeg/backend/cpu/rope.hpp"
#include "metal_inference_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kHeads = 2;
constexpr uint32_t kHeadDim = 8;
constexpr uint32_t kRows = 3;
constexpr uint32_t kWidth = kHeads * kHeadDim;
constexpr float kTheta = 10000.0f;
constexpr float kEpsilon = 1.0e-5f;
constexpr float kQueryScale = 0.73f;
constexpr float kTolerance = 5.0e-5f;

std::string ns_string(NSString* value) {
    return value ? std::string(value.UTF8String) : std::string{};
}

id<MTLBuffer> float_buffer(id<MTLDevice> device, const std::vector<float>& values) {
    id<MTLBuffer> result = [device newBufferWithBytes:values.data()
                                                length:values.size() * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (!result) throw std::runtime_error("Metal partial RoPE buffer allocation failed");
    return result;
}

id<MTLBuffer> zero_buffer(id<MTLDevice> device, size_t count) {
    return float_buffer(device, std::vector<float>(count, 0.0f));
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                     const char* name) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:
        [NSString stringWithUTF8String:name]];
    if (!function) {
        throw std::runtime_error(std::string("missing Metal partial RoPE kernel: ") + name);
    }
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!result) {
        throw std::runtime_error("Metal partial RoPE pipeline failed: " +
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
                 const char* label) {
    for (size_t index = 0; index < expected.size(); ++index) {
        const float delta = std::abs(actual[index] - expected[index]);
        if (delta > kTolerance) {
            throw std::runtime_error(std::string(label) + " differs at " +
                std::to_string(index) + ": actual=" + std::to_string(actual[index]) +
                " expected=" + std::to_string(expected[index]) +
                " delta=" + std::to_string(delta));
        }
    }
}

std::vector<float> values(size_t count, float phase) {
    std::vector<float> result(count);
    for (size_t index = 0; index < count; ++index) {
        result[index] = 0.2f +
            0.8f * std::sin(phase + static_cast<float>(index) * 0.31f) +
            0.27f * std::cos(phase * 0.6f + static_cast<float>(index) * 0.19f);
    }
    return result;
}

std::vector<float> norm_weight() {
    return {0.71f, 0.83f, 0.95f, 1.07f, 1.19f, 1.31f, 1.43f, 1.55f};
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
    rope.scaling = celeg::NoRopeScaling{};
    rope.pairing = pairing;
    if (rope.resolved_rotary_dimension(static_cast<int>(kHeadDim)) !=
        static_cast<int>(rotary_dim)) {
        throw std::logic_error("partial RoPE fixture resolved unexpected rotary dimension");
    }
    return rope;
}

void cpu_prepare(std::vector<float>& data, uint32_t rows,
                 uint32_t base_position, celeg::RopePairingKind pairing,
                 uint32_t rotary_dim, bool query) {
    const std::vector<float> weight = norm_weight();
    const celeg::RopePositionSpec rope = rope_spec(pairing, rotary_dim);
    for (uint32_t row = 0; row < rows; ++row) {
        float* row_data = data.data() + static_cast<size_t>(row) * kWidth;
        celeg::cpu_qk_norm_only(row_data, weight.data(), static_cast<int>(kHeads),
                                static_cast<int>(kHeadDim), kEpsilon);
        celeg::cpu_rope(row_data, static_cast<int>(kHeads), static_cast<int>(kHeadDim),
                        static_cast<int>(base_position + row), rope);
        if (query) {
            for (uint32_t d = 0; d < kWidth; ++d) row_data[d] *= kQueryScale;
        }
    }
}

void encode_norm(id<MTLDevice> device, id<MTLLibrary> library,
                 id<MTLComputeCommandEncoder> encoder,
                 id<MTLBuffer> data, id<MTLBuffer> weight,
                 uint32_t rows) {
    id<MTLComputePipelineState> state = pipeline(
        device, library, rows == 1 ? "celeg_head_rmsnorm_inplace"
                                   : "celeg_head_rmsnorm_batch_inplace");
    [encoder setComputePipelineState:state];
    [encoder setBuffer:data offset:0 atIndex:0];
    [encoder setBuffer:weight offset:0 atIndex:1];
    if (rows == 1) {
        [encoder setBytes:&kHeads length:sizeof(kHeads) atIndex:2];
        [encoder setBytes:&kHeadDim length:sizeof(kHeadDim) atIndex:3];
        [encoder setBytes:&kEpsilon length:sizeof(kEpsilon) atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(kHeads, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(kHeads, 1, 1)];
    } else {
        [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
        [encoder setBytes:&kHeads length:sizeof(kHeads) atIndex:3];
        [encoder setBytes:&kHeadDim length:sizeof(kHeadDim) atIndex:4];
        [encoder setBytes:&kEpsilon length:sizeof(kEpsilon) atIndex:5];
        const uint32_t count = rows * kHeads;
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(count, 1, 1)];
    }
}

void run_token_case(id<MTLDevice> device, id<MTLLibrary> library,
                    id<MTLCommandQueue> queue,
                    celeg::RopePairingKind pairing, uint32_t rotary_dim) {
    constexpr uint32_t cache_position = 3;
    constexpr uint32_t page_tokens = 16;
    std::vector<float> query_values = values(kWidth, 0.35f);
    std::vector<float> key_values = values(kWidth, 1.05f);
    const std::vector<float> value_values = values(kWidth, 1.75f);
    std::vector<float> expected_query = query_values;
    std::vector<float> expected_key = key_values;
    cpu_prepare(expected_query, 1, cache_position, pairing, rotary_dim, true);
    cpu_prepare(expected_key, 1, cache_position, pairing, rotary_dim, false);

    id<MTLBuffer> query = float_buffer(device, query_values);
    id<MTLBuffer> key = float_buffer(device, key_values);
    id<MTLBuffer> value = float_buffer(device, value_values);
    id<MTLBuffer> weight = float_buffer(device, norm_weight());
    const size_t cache_elements = static_cast<size_t>(cache_position + 1) * kWidth;
    id<MTLBuffer> key_cache = zero_buffer(device, cache_elements);
    id<MTLBuffer> value_cache = zero_buffer(device, cache_elements);
    const uint32_t mode = position_mode(pairing, rotary_dim);

    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    encode_norm(device, library, encoder, query, weight, 1);
    encode_norm(device, library, encoder, key, weight, 1);

    id<MTLComputePipelineState> state =
        pipeline(device, library, "celeg_qk_position_store_kv");
    [encoder setComputePipelineState:state];
    [encoder setBuffer:query offset:0 atIndex:0];
    [encoder setBuffer:key offset:0 atIndex:1];
    [encoder setBuffer:value offset:0 atIndex:2];
    [encoder setBuffer:key_cache offset:0 atIndex:3];
    [encoder setBuffer:value_cache offset:0 atIndex:4];
    [encoder setBytes:&kHeads length:sizeof(kHeads) atIndex:5];
    [encoder setBytes:&kHeads length:sizeof(kHeads) atIndex:6];
    [encoder setBytes:&kHeadDim length:sizeof(kHeadDim) atIndex:7];
    [encoder setBytes:&cache_position length:sizeof(cache_position) atIndex:8];
    [encoder setBytes:&mode length:sizeof(mode) atIndex:9];
    [encoder setBytes:&kTheta length:sizeof(kTheta) atIndex:10];
    [encoder setBytes:&kQueryScale length:sizeof(kQueryScale) atIndex:11];
    [encoder setBytes:&page_tokens length:sizeof(page_tokens) atIndex:12];
    [encoder dispatchThreads:MTLSizeMake(kHeads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(kHeads, 1, 1)];
    [encoder endEncoding];
    wait(command, "token partial RoPE");

    check_close(static_cast<const float*>(query.contents), expected_query,
                "token query");
    check_close(static_cast<const float*>(key.contents), expected_key,
                "token key");
    const float* cached_key = static_cast<const float*>(key_cache.contents) +
        static_cast<size_t>(cache_position) * kWidth;
    const float* cached_value = static_cast<const float*>(value_cache.contents) +
        static_cast<size_t>(cache_position) * kWidth;
    check_close(cached_key, expected_key, "token key cache");
    check_close(cached_value, value_values, "token value cache");
}

void run_batch_case(id<MTLDevice> device, id<MTLLibrary> library,
                    id<MTLCommandQueue> queue,
                    celeg::RopePairingKind pairing, uint32_t rotary_dim) {
    constexpr uint32_t base_position = 2;
    std::vector<float> query_values = values(static_cast<size_t>(kRows) * kWidth, 0.6f);
    std::vector<float> key_values = values(static_cast<size_t>(kRows) * kWidth, 1.4f);
    const std::vector<float> value_values = values(static_cast<size_t>(kRows) * kWidth, 2.2f);
    std::vector<float> expected_query = query_values;
    std::vector<float> expected_key = key_values;
    cpu_prepare(expected_query, kRows, base_position, pairing, rotary_dim, true);
    cpu_prepare(expected_key, kRows, base_position, pairing, rotary_dim, false);

    id<MTLBuffer> query = float_buffer(device, query_values);
    id<MTLBuffer> key = float_buffer(device, key_values);
    id<MTLBuffer> value = float_buffer(device, value_values);
    id<MTLBuffer> weight = float_buffer(device, norm_weight());
    const size_t cache_rows = static_cast<size_t>(base_position + kRows + 1);
    id<MTLBuffer> key_cache = zero_buffer(device, cache_rows * kWidth);
    id<MTLBuffer> value_cache = zero_buffer(device, cache_rows * kWidth);
    const uint32_t mode = position_mode(pairing, rotary_dim);

    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    encode_norm(device, library, encoder, query, weight, kRows);
    encode_norm(device, library, encoder, key, weight, kRows);

    id<MTLComputePipelineState> rope =
        pipeline(device, library, "celeg_qk_position_batch");
    [encoder setComputePipelineState:rope];
    [encoder setBuffer:query offset:0 atIndex:0];
    [encoder setBuffer:key offset:0 atIndex:1];
    [encoder setBytes:&kRows length:sizeof(kRows) atIndex:2];
    [encoder setBytes:&kHeads length:sizeof(kHeads) atIndex:3];
    [encoder setBytes:&kHeads length:sizeof(kHeads) atIndex:4];
    [encoder setBytes:&kHeadDim length:sizeof(kHeadDim) atIndex:5];
    [encoder setBytes:&base_position length:sizeof(base_position) atIndex:6];
    [encoder setBytes:&mode length:sizeof(mode) atIndex:7];
    [encoder setBytes:&kTheta length:sizeof(kTheta) atIndex:8];
    [encoder setBytes:&kQueryScale length:sizeof(kQueryScale) atIndex:9];
    const uint32_t head_count = kRows * kHeads;
    [encoder dispatchThreads:MTLSizeMake(head_count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(head_count, 1, 1)];

    id<MTLComputePipelineState> store =
        pipeline(device, library, "celeg_store_kv_batch_2d");
    [encoder setComputePipelineState:store];
    [encoder setBuffer:key offset:0 atIndex:0];
    [encoder setBuffer:value offset:0 atIndex:1];
    [encoder setBuffer:key_cache offset:0 atIndex:2];
    [encoder setBuffer:value_cache offset:0 atIndex:3];
    [encoder setBytes:&kRows length:sizeof(kRows) atIndex:4];
    [encoder setBytes:&base_position length:sizeof(base_position) atIndex:5];
    [encoder setBytes:&kWidth length:sizeof(kWidth) atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake(1, kRows, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    wait(command, "batch partial RoPE");

    check_close(static_cast<const float*>(query.contents), expected_query,
                "batch query");
    check_close(static_cast<const float*>(key.contents), expected_key,
                "batch key");
    const float* cached_key = static_cast<const float*>(key_cache.contents) +
        static_cast<size_t>(base_position) * kWidth;
    const float* cached_value = static_cast<const float*>(value_cache.contents) +
        static_cast<size_t>(base_position) * kWidth;
    check_close(cached_key, expected_key, "batch key cache");
    check_close(cached_value, value_values, "batch value cache");
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
            throw std::runtime_error("Metal partial RoPE shader compilation failed: " +
                (error ? ns_string(error.localizedDescription) : "unknown error"));
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) throw std::runtime_error("failed to create Metal command queue");

        for (celeg::RopePairingKind pairing : {
                 celeg::RopePairingKind::SplitHalf,
                 celeg::RopePairingKind::AdjacentPairs}) {
            run_token_case(device, library, queue, pairing, 4);
            run_batch_case(device, library, queue, pairing, 4);
            run_token_case(device, library, queue, pairing, kHeadDim);
            run_batch_case(device, library, queue, pairing, kHeadDim);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
