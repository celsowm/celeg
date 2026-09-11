/// @file metal_gated_gelu_test.mm
/// @brief Differential test: Metal gated GELU-tanh kernels vs the CPU oracle.
///
/// Covers the decode kernel (`celeg_gated_gelu_tanh`) and both prefill batch
/// kernels (`celeg_gated_gelu_tanh_batch_2d`, strict and relaxed) against
/// `cpu_gated_gelu_tanh`, which implements the same tanh approximation the
/// loader maps `gelu_pytorch_tanh` to.

#include "celeg/backend/cpu/elementwise.hpp"
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

constexpr uint32_t kWidth = 96;
constexpr uint32_t kRows = 3;
constexpr float kStrictTolerance = 1.0e-5f;
constexpr float kRelaxedTolerance = 5.0e-3f;

std::string ns_string(NSString* value) {
    return value ? std::string(value.UTF8String) : std::string{};
}

id<MTLBuffer> float_buffer(id<MTLDevice> device, const std::vector<float>& values) {
    id<MTLBuffer> result = [device newBufferWithBytes:values.data()
                                                  length:values.size() * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    if (!result) throw std::runtime_error("Metal gated GELU buffer allocation failed");
    return result;
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                      const char* name) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:
        [NSString stringWithUTF8String:name]];
    if (!function) throw std::runtime_error(std::string("missing Metal gated GELU kernel: ") + name);
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!result) {
        throw std::runtime_error("Metal gated GELU pipeline failed: " +
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

/// @brief Gate/up inputs spanning the saturating and linear GELU regimes.
std::vector<float> gate_up_values(float phase, size_t count) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = std::sin(phase + static_cast<float>(i) * 0.61f) * 5.0f +
                    std::cos(phase * 1.7f + static_cast<float>(i) * 0.23f) * 1.5f;
    }
    return values;
}

void check_close(const float* actual, const std::vector<float>& expected,
                 const char* label, float tolerance) {
    float worst = 0.0f;
    size_t worst_index = 0;
    for (size_t index = 0; index < expected.size(); ++index) {
        const float delta = std::abs(actual[index] - expected[index]);
        if (delta > worst) {
            worst = delta;
            worst_index = index;
        }
    }
    std::cout << label << " worst delta=" << worst << " at " << worst_index
              << " actual=" << actual[worst_index]
              << " expected=" << expected[worst_index]
              << " tolerance=" << tolerance << '\n';
    if (worst > tolerance) {
        throw std::runtime_error(std::string(label) + " exceeds tolerance: worst=" +
            std::to_string(worst));
    }
}

void run_decode_case(id<MTLDevice> device, id<MTLLibrary> library,
                     id<MTLCommandQueue> queue) {
    const std::vector<float> gate_up = gate_up_values(0.3f, 2 * kWidth);
    std::vector<float> expected(kWidth, 0.0f);
    celeg::cpu_gated_gelu_tanh(gate_up.data(), expected.data(), kWidth);

    id<MTLBuffer> input = float_buffer(device, gate_up);
    id<MTLBuffer> output = float_buffer(device, std::vector<float>(kWidth, 0.0f));
    id<MTLComputePipelineState> state = pipeline(device, library, "celeg_gated_gelu_tanh");

    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBytes:&kWidth length:sizeof(kWidth) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(kWidth, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min(NSUInteger{kWidth},
            state.maxTotalThreadsPerThreadgroup), 1, 1)];
    [encoder endEncoding];
    wait(command, "gated gelu decode");

    check_close(static_cast<const float*>(output.contents), expected,
                "gated gelu decode", kStrictTolerance);
}

void run_batch_case(id<MTLDevice> device, id<MTLLibrary> library,
                    id<MTLCommandQueue> queue, const char* kernel, float tolerance) {
    const std::vector<float> gate_up = gate_up_values(1.1f, 2 * kRows * kWidth);
    std::vector<float> expected(0);
    expected.reserve(kRows * kWidth);
    for (uint32_t row = 0; row < kRows; ++row) {
        std::vector<float> row_gate_up(
            gate_up.begin() + static_cast<ptrdiff_t>(2 * row * kWidth),
            gate_up.begin() + static_cast<ptrdiff_t>(2 * (row + 1) * kWidth));
        celeg::cpu_gated_gelu_tanh(row_gate_up.data(), row_gate_up.data(), kWidth);
        expected.insert(expected.end(), row_gate_up.begin(),
                        row_gate_up.begin() + static_cast<ptrdiff_t>(kWidth));
    }

    id<MTLBuffer> input = float_buffer(device, gate_up);
    id<MTLBuffer> output = float_buffer(device, std::vector<float>(kRows * kWidth, 0.0f));
    id<MTLComputePipelineState> state = pipeline(device, library, kernel);

    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBytes:&kRows length:sizeof(kRows) atIndex:2];
    [encoder setBytes:&kWidth length:sizeof(kWidth) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(kWidth, kRows, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min(NSUInteger{kWidth},
            state.maxTotalThreadsPerThreadgroup), 1, 1)];
    [encoder endEncoding];
    wait(command, kernel);

    check_close(static_cast<const float*>(output.contents), expected, kernel, tolerance);
}

}  // namespace

int main() {
    try {
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("no default Metal device is available");
            NSError* error = nil;
            NSString* source = [NSString stringWithUTF8String:
                celeg::metal_detail::kInferenceShader];
            id<MTLLibrary> library = [device newLibraryWithSource:source
                                                          options:nil error:&error];
            if (!library) {
                throw std::runtime_error("Metal shader compilation failed: " +
                    (error ? ns_string(error.localizedDescription) : "unknown error"));
            }
            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (!queue) throw std::runtime_error("Metal command queue creation failed");
            run_decode_case(device, library, queue);
            run_batch_case(device, library, queue,
                           "celeg_gated_gelu_tanh_batch_2d", kStrictTolerance);
            run_batch_case(device, library, queue,
                           "celeg_gated_gelu_tanh_batch_2d_relaxed", kRelaxedTolerance);
        }
        std::cout << "metal gated GELU CPU parity passed\n";
        return 0;
    } catch (const std::exception& failure) {
        std::cerr << "error: " << failure.what() << '\n';
        return 1;
    }
}
