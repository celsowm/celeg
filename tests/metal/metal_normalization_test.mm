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

std::string ns_string(NSString* value) {
    return value ? std::string(value.UTF8String) : std::string{};
}

id<MTLBuffer> make_buffer(id<MTLDevice> device, const std::vector<float>& values) {
    id<MTLBuffer> result = [device newBufferWithBytes:values.data()
                                                length:values.size() * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    if (!result) throw std::runtime_error("Metal normalization buffer allocation failed");
    return result;
}

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                          const char* name) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:
        [NSString stringWithUTF8String:name]];
    if (!function) throw std::runtime_error(std::string("missing normalization kernel: ") + name);
    id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function
                                                                                error:&error];
    if (!state) {
        throw std::runtime_error("normalization pipeline failed: " +
            (error ? ns_string(error.localizedDescription) : "unknown error"));
    }
    return state;
}

void dispatch(id<MTLCommandQueue> queue, id<MTLComputePipelineState> state,
              id<MTLBuffer> input, id<MTLBuffer> residual, id<MTLBuffer> weight,
              id<MTLBuffer> output, id<MTLBuffer> normed,
              uint32_t rows, uint32_t width, float multiplier, float epsilon,
              bool cached) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:residual offset:0 atIndex:1];
    [encoder setBuffer:weight offset:0 atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBuffer:normed offset:0 atIndex:4];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:5];
    [encoder setBytes:&width length:sizeof(width) atIndex:6];
    [encoder setBytes:&multiplier length:sizeof(multiplier) atIndex:7];
    [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:8];
    if (cached) {
        [encoder setThreadgroupMemoryLength:static_cast<NSUInteger>(width) * sizeof(float)
                                    atIndex:0];
    }
    [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal normalization dispatch failed");
    }
}

void check_equal(const float* actual, const float* expected, size_t count,
                 const char* label) {
    for (size_t index = 0; index < count; ++index) {
        if (std::abs(actual[index] - expected[index]) > 1.0e-6f) {
            throw std::runtime_error(std::string(label) + " differs at " +
                                     std::to_string(index) + ": actual=" +
                                     std::to_string(actual[index]) + " expected=" +
                                     std::to_string(expected[index]));
        }
    }
}

void check_cached_residual_norm(id<MTLDevice> device, id<MTLCommandQueue> queue,
                                id<MTLLibrary> library) {
    constexpr uint32_t rows = 3;
    constexpr uint32_t width = 1024;
    constexpr float multiplier = 0.75f;
    constexpr float epsilon = 1.0e-5f;
    const size_t count = static_cast<size_t>(rows) * width;

    std::vector<float> input(count);
    std::vector<float> residual(count);
    std::vector<float> weight(width);
    for (size_t index = 0; index < count; ++index) {
        input[index] = std::sin(static_cast<float>(index) * 0.013f) * 0.7f;
        residual[index] = std::cos(static_cast<float>(index) * 0.017f) * 0.4f;
    }
    for (uint32_t index = 0; index < width; ++index) {
        weight[index] = 0.8f + 0.0002f * static_cast<float>(index);
    }

    id<MTLBuffer> input_reference = make_buffer(device, input);
    id<MTLBuffer> residual_reference = make_buffer(device, residual);
    id<MTLBuffer> weight_buffer = make_buffer(device, weight);
    id<MTLBuffer> output_reference = make_buffer(device, std::vector<float>(count, 0.0f));
    id<MTLBuffer> normed_reference = make_buffer(device, std::vector<float>(count, 0.0f));
    dispatch(queue, make_pipeline(device, library, "celeg_residual_rmsnorm_batch"),
             input_reference, residual_reference, weight_buffer,
             output_reference, normed_reference, rows, width, multiplier, epsilon, false);

    id<MTLBuffer> input_cached = make_buffer(device, input);
    id<MTLBuffer> residual_cached = make_buffer(device, residual);
    id<MTLBuffer> output_cached = make_buffer(device, std::vector<float>(count, 0.0f));
    id<MTLBuffer> normed_cached = make_buffer(device, std::vector<float>(count, 0.0f));
    dispatch(queue, make_pipeline(device, library, "celeg_residual_rmsnorm_batch_cached"),
             input_cached, residual_cached, weight_buffer,
             output_cached, normed_cached, rows, width, multiplier, epsilon, true);

    check_equal(static_cast<const float*>(output_cached.contents),
                static_cast<const float*>(output_reference.contents), count,
                "cached residual output");
    check_equal(static_cast<const float*>(normed_cached.contents),
                static_cast<const float*>(normed_reference.contents), count,
                "cached residual norm");

    id<MTLBuffer> input_alias = make_buffer(device, input);
    id<MTLBuffer> residual_alias = make_buffer(device, residual);
    id<MTLBuffer> normed_alias = make_buffer(device, std::vector<float>(count, 0.0f));
    dispatch(queue, make_pipeline(device, library, "celeg_residual_rmsnorm_batch_cached"),
             input_alias, residual_alias, weight_buffer,
             residual_alias, normed_alias, rows, width, multiplier, epsilon, true);
    check_equal(static_cast<const float*>(residual_alias.contents),
                static_cast<const float*>(output_reference.contents), count,
                "aliased residual output");
    check_equal(static_cast<const float*>(normed_alias.contents),
                static_cast<const float*>(normed_reference.contents), count,
                "aliased residual norm");
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
            throw std::runtime_error("Metal normalization shader compilation failed: " +
                (error ? ns_string(error.localizedDescription) : "unknown error"));
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) throw std::runtime_error("Metal normalization command queue failed");
        check_cached_residual_norm(device, queue, library);
        std::cout << "metal normalization kernels passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
