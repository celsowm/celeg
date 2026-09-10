#include "celeg/model/position.hpp"
#include "metal_inference_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>

namespace {

struct Case {
    celeg::RopePositionSpec spec;
    uint32_t pair;
    uint32_t rotary_dimension;
    uint32_t position;
};

struct Abi {
    uint32_t mode = 0;
    float factor = 1.0f;
    float beta_fast = 32.0f;
    float beta_slow = 1.0f;
    uint32_t original_context = 0;
    float low_frequency_factor = 1.0f;
    float high_frequency_factor = 1.0f;
    float attention_factor = 1.0f;
    std::vector<float> short_factors{1.0f};
    std::vector<float> long_factors{1.0f};
};

Abi lower(const celeg::RopeScalingSpec& scaling) {
    Abi abi;
    std::visit([&](const auto& value) {
        using Scaling = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Scaling, celeg::NoRopeScaling>) {
            abi.mode = 0;
        } else if constexpr (std::is_same_v<Scaling, celeg::LinearRopeScaling>) {
            abi.mode = 1;
            abi.factor = static_cast<float>(value.factor);
        } else if constexpr (std::is_same_v<Scaling, celeg::DynamicNtkRopeScaling>) {
            abi.mode = 2;
            abi.factor = static_cast<float>(value.factor);
            abi.original_context = static_cast<uint32_t>(value.original_context);
        } else if constexpr (std::is_same_v<Scaling, celeg::YarnRopeScaling>) {
            abi.mode = 3;
            abi.factor = static_cast<float>(value.factor);
            abi.attention_factor = static_cast<float>(value.attention_factor);
            abi.beta_fast = static_cast<float>(value.beta_fast);
            abi.beta_slow = static_cast<float>(value.beta_slow);
            abi.original_context = static_cast<uint32_t>(value.original_context);
        } else if constexpr (std::is_same_v<Scaling, celeg::LongRopeScaling>) {
            abi.mode = 4;
            abi.original_context = static_cast<uint32_t>(value.original_context);
            abi.short_factors = value.short_factors;
            abi.long_factors = value.long_factors;
        } else if constexpr (std::is_same_v<Scaling, celeg::Llama3FrequencyScaling>) {
            abi.mode = 5;
            abi.factor = static_cast<float>(value.factor);
            abi.original_context = static_cast<uint32_t>(value.original_context);
            abi.low_frequency_factor = static_cast<float>(value.low_frequency_factor);
            abi.high_frequency_factor = static_cast<float>(value.high_frequency_factor);
        } else if constexpr (std::is_same_v<Scaling, celeg::ProportionalRopeScaling>) {
            abi.mode = 6;
            abi.factor = static_cast<float>(value.factor);
        }
    }, scaling);
    return abi;
}

bool close_enough(float actual, double expected) {
    const double error = std::abs(static_cast<double>(actual) - expected);
    const double tolerance = std::max(2.0e-6, std::abs(expected) * 8.0e-5);
    return error <= tolerance;
}

void run_case(id<MTLDevice> device,
              id<MTLComputePipelineState> state,
              id<MTLCommandQueue> queue,
              const Case& test_case) {
    Abi abi = lower(test_case.spec.scaling);
    const float theta = static_cast<float>(test_case.spec.theta);
    const float rotary_fraction = static_cast<float>(test_case.spec.rotary_fraction);

    id<MTLBuffer> output = [device newBufferWithLength:2 * sizeof(float)
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> short_factors = [device newBufferWithBytes:abi.short_factors.data()
                                                       length:abi.short_factors.size() * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
    id<MTLBuffer> long_factors = [device newBufferWithBytes:abi.long_factors.data()
                                                      length:abi.long_factors.size() * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    if (!output || !short_factors || !long_factors) {
        throw std::runtime_error("failed to allocate Metal RoPE scaling buffers");
    }

    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setBuffer:output offset:0 atIndex:0];
    [encoder setBuffer:short_factors offset:0 atIndex:1];
    [encoder setBuffer:long_factors offset:0 atIndex:2];
    [encoder setBytes:&theta length:sizeof(theta) atIndex:3];
    [encoder setBytes:&rotary_fraction length:sizeof(rotary_fraction) atIndex:4];
    [encoder setBytes:&test_case.pair length:sizeof(test_case.pair) atIndex:5];
    [encoder setBytes:&test_case.rotary_dimension length:sizeof(test_case.rotary_dimension) atIndex:6];
    [encoder setBytes:&test_case.position length:sizeof(test_case.position) atIndex:7];
    [encoder setBytes:&abi.mode length:sizeof(abi.mode) atIndex:8];
    [encoder setBytes:&abi.factor length:sizeof(abi.factor) atIndex:9];
    [encoder setBytes:&abi.beta_fast length:sizeof(abi.beta_fast) atIndex:10];
    [encoder setBytes:&abi.beta_slow length:sizeof(abi.beta_slow) atIndex:11];
    [encoder setBytes:&abi.original_context length:sizeof(abi.original_context) atIndex:12];
    [encoder setBytes:&abi.low_frequency_factor length:sizeof(abi.low_frequency_factor) atIndex:13];
    [encoder setBytes:&abi.high_frequency_factor length:sizeof(abi.high_frequency_factor) atIndex:14];
    [encoder setBytes:&abi.attention_factor length:sizeof(abi.attention_factor) atIndex:15];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal RoPE scaling dispatch failed");
    }

    const auto* actual = static_cast<const float*>(output.contents);
    const double expected_frequency = celeg::rope_frequency(
        test_case.spec,
        static_cast<int>(test_case.pair),
        static_cast<int>(test_case.rotary_dimension),
        static_cast<int>(test_case.position));
    const double expected_attention_scale = celeg::rope_attention_scale(
        test_case.spec, static_cast<int>(test_case.position));
    if (!close_enough(actual[0], expected_frequency) ||
        !close_enough(actual[1], expected_attention_scale)) {
        throw std::runtime_error("Metal RoPE scaling semantics drifted from host contract");
    }
}

celeg::RopePositionSpec rope(double fraction, celeg::RopeScalingSpec scaling) {
    celeg::RopePositionSpec spec;
    spec.theta = 10000.0;
    spec.rotary_fraction = fraction;
    spec.scaling = std::move(scaling);
    return spec;
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
            const char* message = error ? error.localizedDescription.UTF8String : "unknown";
            throw std::runtime_error(std::string("Metal RoPE scaling shader compilation failed: ") + message);
        }
        id<MTLFunction> function =
            [library newFunctionWithName:@"celeg_rope_scaling_semantics_probe"];
        if (!function) throw std::runtime_error("missing Metal RoPE scaling semantics probe");
        id<MTLComputePipelineState> state =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!state) throw std::runtime_error("failed Metal RoPE scaling probe pipeline");
        id<MTLCommandQueue> queue = [device newCommandQueue];

        celeg::LongRopeScaling long_rope;
        long_rope.original_context = 64;
        long_rope.short_factors = {1.0f, 1.25f, 1.5f, 1.75f, 2.0f, 2.25f, 2.5f, 2.75f};
        long_rope.long_factors = {2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f};

        const std::array<Case, 14> cases{{
            {rope(1.0, celeg::NoRopeScaling{}), 0, 16, 0},
            {rope(1.0, celeg::NoRopeScaling{}), 7, 16, 127},
            {rope(1.0, celeg::LinearRopeScaling{4.0}), 3, 16, 127},
            {rope(1.0, celeg::DynamicNtkRopeScaling{4.0, 64}), 3, 16, 32},
            {rope(1.0, celeg::DynamicNtkRopeScaling{4.0, 64}), 1, 16, 256},
            {rope(1.0, celeg::DynamicNtkRopeScaling{4.0, 64}), 7, 16, 256},
            {rope(1.0, celeg::YarnRopeScaling{8.0, 1.2, 32.0, 1.0, 8192}), 0, 16, 4096},
            {rope(1.0, celeg::YarnRopeScaling{8.0, 1.2, 32.0, 1.0, 8192}), 4, 16, 4096},
            {rope(1.0, celeg::YarnRopeScaling{8.0, 1.2, 32.0, 1.0, 8192}), 7, 16, 4096},
            {rope(1.0, long_rope), 5, 16, 32},
            {rope(1.0, long_rope), 5, 16, 128},
            {rope(1.0, celeg::Llama3FrequencyScaling{8.0, 8192, 1.0, 4.0}), 1, 16, 4096},
            {rope(1.0, celeg::Llama3FrequencyScaling{8.0, 8192, 1.0, 4.0}), 7, 16, 4096},
            {rope(0.5, celeg::ProportionalRopeScaling{2.0}), 3, 8, 127},
        }};
        for (const Case& test_case : cases) run_case(device, state, queue, test_case);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
