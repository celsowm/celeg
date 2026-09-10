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
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kHeads = 2;
constexpr uint32_t kHeadDim = 8;
constexpr uint32_t kWidth = kHeads * kHeadDim;
constexpr uint32_t kPageTokens = 16;
constexpr float kTheta = 10000.0f;
constexpr float kQueryScale = 0.73f;
constexpr float kTolerance = 1.5e-4f;

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
                                  uint32_t rotary_dim,
                                  celeg::RopeScalingSpec scaling) {
    celeg::RopePositionSpec rope;
    rope.theta = kTheta;
    rope.rotary_fraction = static_cast<double>(rotary_dim) /
                           static_cast<double>(kHeadDim);
    rope.scaling = std::move(scaling);
    rope.pairing = pairing;
    return rope;
}

void check_close(const float* actual, const std::vector<float>& expected,
                 const std::string& label) {
    for (size_t i = 0; i < expected.size(); ++i) {
        const float delta = std::abs(actual[i] - expected[i]);
        if (delta > kTolerance) {
            throw std::runtime_error(label + " differs at " +
                                     std::to_string(i) + ": actual=" +
                                     std::to_string(actual[i]) + " expected=" +
                                     std::to_string(expected[i]) + " delta=" +
                                     std::to_string(delta));
        }
    }
}

void run_case(id<MTLDevice> device, id<MTLLibrary> library,
              id<MTLCommandQueue> queue, const std::string& label,
              celeg::RopePairingKind pairing, uint32_t rotary_dim,
              uint32_t position, celeg::RopeScalingSpec scaling_spec) {
    const celeg::RopePositionSpec rope =
        rope_spec(pairing, rotary_dim, std::move(scaling_spec));
    const auto scaling =
        celeg::metal_model_detail::make_metal_rope_scaling_binding(rope);
    if (!scaling.scaled()) throw std::logic_error(label + " fixture is unscaled");

    std::vector<float> query_values = values(0.4f);
    std::vector<float> key_values = values(1.2f);
    const std::vector<float> value_values = values(2.0f);
    std::vector<float> expected_query = query_values;
    std::vector<float> expected_key = key_values;

    celeg::cpu_rope(expected_query.data(), static_cast<int>(kHeads),
                    static_cast<int>(kHeadDim), static_cast<int>(position), rope);
    celeg::cpu_rope(expected_key.data(), static_cast<int>(kHeads),
                    static_cast<int>(kHeadDim), static_cast<int>(position), rope);
    for (float& value : expected_query) value *= kQueryScale;

    id<MTLBuffer> query = float_buffer(device, query_values);
    id<MTLBuffer> key = float_buffer(device, key_values);
    id<MTLBuffer> value = float_buffer(device, value_values);
    const size_t cache_elements = static_cast<size_t>(position + 1u) * kWidth;
    id<MTLBuffer> key_cache = zero_buffer(device, cache_elements);
    id<MTLBuffer> value_cache = zero_buffer(device, cache_elements);
    const std::vector<float> dummy_factors{1.0f};
    const std::vector<float>& short_values =
        scaling.short_factors && !scaling.short_factors->empty()
            ? *scaling.short_factors
            : dummy_factors;
    const std::vector<float>& long_values =
        scaling.long_factors && !scaling.long_factors->empty()
            ? *scaling.long_factors
            : dummy_factors;
    id<MTLBuffer> short_factors = float_buffer(device, short_values);
    id<MTLBuffer> long_factors = float_buffer(device, long_values);

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
    [encoder setBytes:&position length:sizeof(position) atIndex:8];
    [encoder setBytes:&mode length:sizeof(mode) atIndex:9];
    [encoder setBytes:&kTheta length:sizeof(kTheta) atIndex:10];
    [encoder setBytes:&kQueryScale length:sizeof(kQueryScale) atIndex:11];
    [encoder setBytes:&kPageTokens length:sizeof(kPageTokens) atIndex:12];
    [encoder setBytes:&scaling.spec length:sizeof(scaling.spec) atIndex:13];
    [encoder setBuffer:short_factors offset:0 atIndex:14];
    [encoder setBuffer:long_factors offset:0 atIndex:15];
    [encoder dispatchThreads:MTLSizeMake(kHeads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(kHeads, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        const std::string detail = command.error
            ? ": " + ns_string(command.error.localizedDescription)
            : std::string{};
        throw std::runtime_error(label + " dispatch failed" + detail);
    }

    check_close(static_cast<const float*>(query.contents), expected_query,
                label + " query");
    check_close(static_cast<const float*>(key.contents), expected_key,
                label + " key");
    const float* cached_key = static_cast<const float*>(key_cache.contents) +
        static_cast<size_t>(position) * kWidth;
    const float* cached_value = static_cast<const float*>(value_cache.contents) +
        static_cast<size_t>(position) * kWidth;
    check_close(cached_key, expected_key, label + " key cache");
    check_close(cached_value, value_values, label + " value cache");
}

celeg::YarnRopeScaling yarn() {
    celeg::YarnRopeScaling value;
    value.factor = 4.0;
    value.attention_factor = 1.25;
    value.beta_fast = 16.0;
    value.beta_slow = 2.0;
    value.original_context = 4;
    return value;
}

celeg::LongRopeScaling long_rope() {
    celeg::LongRopeScaling value;
    value.original_context = 4;
    value.short_factors = {1.0f, 1.15f, 1.3f, 1.45f};
    value.long_factors = {2.0f, 2.2f, 2.4f, 2.6f};
    return value;
}

celeg::Llama3FrequencyScaling llama3() {
    celeg::Llama3FrequencyScaling value;
    value.factor = 8.0;
    value.original_context = 8;
    value.low_frequency_factor = 1.0;
    value.high_frequency_factor = 4.0;
    return value;
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

        run_case(device, library, queue, "linear split full",
                 celeg::RopePairingKind::SplitHalf, kHeadDim, 7,
                 celeg::LinearRopeScaling{2.5});
        run_case(device, library, queue, "linear adjacent full",
                 celeg::RopePairingKind::AdjacentPairs, kHeadDim, 7,
                 celeg::LinearRopeScaling{2.5});
        run_case(device, library, queue, "linear split partial",
                 celeg::RopePairingKind::SplitHalf, 4u, 7,
                 celeg::LinearRopeScaling{2.5});
        run_case(device, library, queue, "linear adjacent partial",
                 celeg::RopePairingKind::AdjacentPairs, 4u, 7,
                 celeg::LinearRopeScaling{2.5});

        run_case(device, library, queue, "dynamic ntk boundary",
                 celeg::RopePairingKind::SplitHalf, kHeadDim, 4,
                 celeg::DynamicNtkRopeScaling{2.0, 4});
        run_case(device, library, queue, "dynamic ntk extended",
                 celeg::RopePairingKind::SplitHalf, kHeadDim, 7,
                 celeg::DynamicNtkRopeScaling{2.0, 4});

        run_case(device, library, queue, "yarn split full",
                 celeg::RopePairingKind::SplitHalf, kHeadDim, 7, yarn());
        run_case(device, library, queue, "yarn adjacent partial",
                 celeg::RopePairingKind::AdjacentPairs, 4u, 7, yarn());

        run_case(device, library, queue, "longrope boundary",
                 celeg::RopePairingKind::SplitHalf, kHeadDim, 4, long_rope());
        run_case(device, library, queue, "longrope extended",
                 celeg::RopePairingKind::SplitHalf, kHeadDim, 7, long_rope());

        run_case(device, library, queue, "llama3 split full",
                 celeg::RopePairingKind::SplitHalf, kHeadDim, 7, llama3());
        run_case(device, library, queue, "llama3 adjacent partial",
                 celeg::RopePairingKind::AdjacentPairs, 4u, 7, llama3());

        run_case(device, library, queue, "proportional partial",
                 celeg::RopePairingKind::SplitHalf, 4u, 7,
                 celeg::ProportionalRopeScaling{1.75});
        run_case(device, library, queue, "proportional adjacent full",
                 celeg::RopePairingKind::AdjacentPairs, kHeadDim, 7,
                 celeg::ProportionalRopeScaling{1.75});
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
