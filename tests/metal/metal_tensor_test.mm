#include "metal_tensor_source.hpp"
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

float half_value(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16u;
    const uint32_t exponent = (value >> 10u) & 0x1fu;
    const uint32_t fraction = value & 0x3ffu;
    uint32_t bits = sign;
    if (exponent == 0) {
        if (fraction != 0) {
            uint32_t normalized = fraction;
            uint32_t shift = 0;
            while ((normalized & 0x400u) == 0) {
                normalized <<= 1u;
                ++shift;
            }
            bits |= (127u - 14u - shift) << 23u;
            bits |= (normalized & 0x3ffu) << 13u;
        }
    } else if (exponent == 0x1fu) {
        bits |= 0x7f800000u | (fraction << 13u);
    } else {
        bits |= (exponent + 112u) << 23u;
        bits |= fraction << 13u;
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t weight_bits(uint32_t row, uint32_t col) {
    constexpr uint16_t values[]{0x3c00u, 0x3800u, 0xbc00u};
    return values[(row + col) % 3u];
}

}

int main() {
    try {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("no default Metal device is available");
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:celeg::metal_detail::kTensorShader];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            throw std::runtime_error("tensor shader compilation failed: " +
                                     (error ? ns_string(error.localizedDescription) : "unknown error"));
        }
        id<MTLFunction> function = [library newFunctionWithName:@"celeg_matmul_tensor_f16"];
        if (!function) throw std::runtime_error("tensor function is missing");
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                         error:&error];
        if (!pipeline) throw std::runtime_error("tensor pipeline creation failed");

        constexpr uint32_t rows = 3;
        constexpr uint32_t cols = 37;
        constexpr uint32_t output_rows = 5;
        constexpr uint32_t output_stride = 7;
        std::vector<uint16_t> weights(output_rows * cols);
        std::vector<float> input(rows * cols);
        std::vector<float> output(rows * output_stride, 0.0f);
        for (uint32_t row = 0; row < output_rows; ++row) {
            for (uint32_t col = 0; col < cols; ++col) {
                weights[row * cols + col] = weight_bits(row, col);
            }
        }
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t col = 0; col < cols; ++col) {
                input[row * cols + col] = std::sin(static_cast<float>((row + 1) * (col + 3)));
            }
        }
        id<MTLBuffer> weights_buffer = [device newBufferWithBytes:weights.data()
                                                               length:weights.size() * sizeof(uint16_t)
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> input_buffer = [device newBufferWithBytes:input.data()
                                                             length:input.size() * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buffer = [device newBufferWithBytes:output.data()
                                                              length:output.size() * sizeof(float)
                                                             options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:weights_buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        [encoder setBytes:&output_rows length:sizeof(output_rows) atIndex:5];
        [encoder setBytes:&output_stride length:sizeof(output_stride) atIndex:6];
        [encoder setThreadgroupMemoryLength:64u * 32u * sizeof(uint16_t) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("tensor dispatch failed");
        }
        std::copy_n(static_cast<const float*>(output_buffer.contents), output.size(), output.data());
        float maximum = 0.0f;
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t out = 0; out < output_rows; ++out) {
                float expected = 0.0f;
                for (uint32_t col = 0; col < cols; ++col) {
                    expected += half_value(weight_bits(out, col)) * input[row * cols + col];
                }
                maximum = std::max(maximum,
                    std::abs(expected - output[row * output_stride + out]));
            }
            if (output[row * output_stride + output_rows] != 0.0f ||
                output[row * output_stride + output_rows + 1] != 0.0f) {
                throw std::runtime_error("tensor kernel overwrote output padding");
            }
        }
        std::cout << "max_error=" << maximum << '\n';
        if (!(maximum < 1.0e-3f)) throw std::runtime_error("tensor result differs from reference");

        NSString* inference_source =
            [NSString stringWithUTF8String:celeg::metal_detail::kInferenceShader];
        id<MTLLibrary> inference_library =
            [device newLibraryWithSource:inference_source options:nil error:&error];
        if (!inference_library) throw std::runtime_error("inference shader compilation failed");
        id<MTLFunction> matvec_function =
            [inference_library newFunctionWithName:@"celeg_matvec_f16"];
        if (!matvec_function) throw std::runtime_error("half matvec function is missing");
        id<MTLComputePipelineState> matvec_pipeline =
            [device newComputePipelineStateWithFunction:matvec_function error:&error];
        if (!matvec_pipeline) throw std::runtime_error("half matvec pipeline creation failed");
        std::vector<float> matvec_output(rows, 0.0f);
        id<MTLBuffer> matvec_output_buffer = [device newBufferWithBytes:matvec_output.data()
                                                                    length:matvec_output.size() * sizeof(float)
                                                                   options:MTLResourceStorageModeShared];
        command_buffer = [queue commandBuffer];
        encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:matvec_pipeline];
        [encoder setBuffer:weights_buffer offset:0 atIndex:0];
        [encoder setBuffer:input_buffer offset:0 atIndex:1];
        [encoder setBuffer:matvec_output_buffer offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        [encoder setThreadgroupMemoryLength:8u * sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake((rows + 1u) / 2u, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("half matvec dispatch failed");
        }
        std::copy_n(static_cast<const float*>(matvec_output_buffer.contents),
                    matvec_output.size(), matvec_output.data());
        maximum = 0.0f;
        for (uint32_t row = 0; row < rows; ++row) {
            float expected = 0.0f;
            for (uint32_t col = 0; col < cols; ++col) {
                expected += half_value(weight_bits(row, col)) * input[col];
            }
            maximum = std::max(maximum, std::abs(expected - matvec_output[row]));
        }
        std::cout << "matvec_max_error=" << maximum << '\n';
        if (!(maximum < 1.0e-3f)) throw std::runtime_error("half matvec differs from reference");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
