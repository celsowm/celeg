#include "metal_inference_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

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
    if (!result) throw std::runtime_error("Metal recurrent buffer allocation failed");
    return result;
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                     const char* name) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:
        [NSString stringWithUTF8String:name]];
    if (!function) throw std::runtime_error(std::string("Metal recurrent function missing: ") + name);
    id<MTLComputePipelineState> result = [device newComputePipelineStateWithFunction:function
                                                                                  error:&error];
    if (!result) {
        throw std::runtime_error("Metal recurrent pipeline failed: " +
                                 (error ? ns_string(error.localizedDescription) : "unknown error"));
    }
    return result;
}

void run(id<MTLCommandQueue> queue, id<MTLComputePipelineState> state,
         id<MTLBuffer>* buffers, size_t buffer_count,
         const std::vector<std::pair<const void*, size_t>>& constants) {
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state];
    for (size_t index = 0; index < buffer_count; ++index) {
        [encoder setBuffer:buffers[index] offset:0 atIndex:index];
    }
    for (size_t index = 0; index < constants.size(); ++index) {
        [encoder setBytes:constants[index].first length:constants[index].second
                atIndex:buffer_count + index];
    }
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal recurrent dispatch failed");
    }
}

void check_close(float actual, float expected, const char* label) {
    if (std::abs(actual - expected) > 1.0e-4f) {
        throw std::runtime_error(std::string(label) + " differs from reference: actual=" +
                                 std::to_string(actual) + " expected=" +
                                 std::to_string(expected));
    }
}

void check_gated_delta(id<MTLDevice> device, id<MTLCommandQueue> queue,
                       id<MTLLibrary> library) {
    const std::vector<float> projected_values{1.0f, 2.0f, 3.0f};
    const std::vector<float> zero{0.0f};
    const std::vector<float> one{1.0f};
    id<MTLBuffer> projected = buffer(device, projected_values);
    id<MTLBuffer> z = buffer(device, one);
    id<MTLBuffer> beta = buffer(device, zero);
    id<MTLBuffer> decay = buffer(device, zero);
    id<MTLBuffer> convolution = buffer(device, one);
    id<MTLBuffer> dt_bias = buffer(device, zero);
    id<MTLBuffer> a_log = buffer(device, zero);
    id<MTLBuffer> norm = buffer(device, one);
    id<MTLBuffer> conv_state = buffer(device, zero);
    id<MTLBuffer> recurrent_state = buffer(device, zero);
    id<MTLBuffer> output = buffer(device, zero);
    id<MTLBuffer> buffers[] = {projected, z, beta, decay, convolution, dt_bias,
                               a_log, norm, conv_state, recurrent_state, output};
    const uint32_t conv_kernel = 1;
    const uint32_t key_head_dim = 1;
    const uint32_t value_head_dim = 1;
    const uint32_t key_heads = 1;
    const uint32_t value_heads = 1;
    const float epsilon = 1.0e-5f;
    const uint32_t vector_decay = 0;
    const uint32_t safe_decay = 0;
    const float decay_lower_bound = -5.0f;
    const uint32_t sigmoid_output_gate = 0;
    const uint32_t a_log_needs_exp = 0;
    const std::vector<std::pair<const void*, size_t>> constants = {
        {&conv_kernel, sizeof(conv_kernel)}, {&key_head_dim, sizeof(key_head_dim)},
        {&value_head_dim, sizeof(value_head_dim)}, {&key_heads, sizeof(key_heads)},
        {&value_heads, sizeof(value_heads)}, {&epsilon, sizeof(epsilon)},
        {&vector_decay, sizeof(vector_decay)}, {&safe_decay, sizeof(safe_decay)},
        {&decay_lower_bound, sizeof(decay_lower_bound)},
        {&sigmoid_output_gate, sizeof(sigmoid_output_gate)},
        {&a_log_needs_exp, sizeof(a_log_needs_exp)}};
    run(queue, pipeline(device, library, "celeg_gated_delta"), buffers, 11, constants);
    const float* actual_output = static_cast<const float*>(output.contents);
    const float filtered_key = 2.0f / (1.0f + std::exp(-2.0f));
    const float filtered_value = 3.0f / (1.0f + std::exp(-3.0f));
    const float normalized_key = filtered_key /
        std::sqrt(filtered_key * filtered_key + epsilon);
    const float state = normalized_key * filtered_value * 0.5f;
    const float expected = state / std::sqrt(state * state + epsilon) *
        (1.0f / (1.0f + std::exp(-1.0f)));
    std::cerr << "gdn projected=" << static_cast<const float*>(projected.contents)[0] << "," <<
        static_cast<const float*>(projected.contents)[1] << "," <<
        static_cast<const float*>(projected.contents)[2] << " actual=" << actual_output[0] << " state=" <<
        static_cast<const float*>(recurrent_state.contents)[0] << " expected=" <<
        expected << " ref_state=" << state << '\n';
    check_close(actual_output[0], expected, "gated-delta output");
    check_close(static_cast<const float*>(recurrent_state.contents)[0], state,
                "gated-delta state");
}

void check_mamba(id<MTLDevice> device, id<MTLCommandQueue> queue,
                 id<MTLLibrary> library) {
    const std::vector<float> projected_values{1.0f, 2.0f, 3.0f, 4.0f, 0.0f};
    const std::vector<float> one{1.0f};
    const std::vector<float> zero{0.0f};
    id<MTLBuffer> projected = buffer(device, projected_values);
    id<MTLBuffer> convolution = buffer(device, one);
    id<MTLBuffer> conv_bias = buffer(device, zero);
    id<MTLBuffer> dt_bias = buffer(device, zero);
    id<MTLBuffer> a_log = buffer(device, zero);
    id<MTLBuffer> d = buffer(device, one);
    id<MTLBuffer> norm = buffer(device, one);
    id<MTLBuffer> conv_state = buffer(device, zero);
    id<MTLBuffer> ssm_state = buffer(device, zero);
    id<MTLBuffer> output = buffer(device, zero);
    id<MTLBuffer> buffers[] = {projected, convolution, conv_bias, dt_bias, a_log,
                               d, norm, conv_state, ssm_state, output};
    const uint32_t inner = 1;
    const uint32_t state_size = 1;
    const uint32_t num_heads = 1;
    const uint32_t head_dim = 1;
    const uint32_t group_count = 1;
    const uint32_t conv_kernel = 1;
    const float epsilon = 1.0e-5f;
    const uint32_t a_log_needs_exp = 0;
    const std::vector<std::pair<const void*, size_t>> constants = {
        {&inner, sizeof(inner)}, {&state_size, sizeof(state_size)},
        {&num_heads, sizeof(num_heads)}, {&head_dim, sizeof(head_dim)},
        {&group_count, sizeof(group_count)}, {&conv_kernel, sizeof(conv_kernel)},
        {&epsilon, sizeof(epsilon)}, {&a_log_needs_exp, sizeof(a_log_needs_exp)}};
    run(queue, pipeline(device, library, "celeg_mamba2"), buffers, 10, constants);
    const float x = 2.0f / (1.0f + std::exp(-2.0f));
    const float state = std::log(2.0f) * 3.0f * x;
    const float raw = state * 4.0f + x;
    const float gated = raw * (1.0f / (1.0f + std::exp(-raw))) *
        (1.0f / std::sqrt(raw * raw + epsilon));
    check_close(static_cast<const float*>(output.contents)[0], gated, "Mamba-2 output");
    check_close(static_cast<const float*>(ssm_state.contents)[0], state, "Mamba-2 state");
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
            throw std::runtime_error("Metal recurrent shader compilation failed: " +
                (error ? ns_string(error.localizedDescription) : "unknown error"));
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) throw std::runtime_error("Metal recurrent command queue failed");
        check_gated_delta(device, queue, library);
        check_mamba(device, queue, library);
        std::cout << "metal recurrent kernels passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
