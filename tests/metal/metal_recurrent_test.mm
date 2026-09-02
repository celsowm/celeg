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

void check_shortconv_ring(id<MTLDevice> device, id<MTLCommandQueue> queue,
                          id<MTLLibrary> library) {
    constexpr uint32_t width = 3;
    constexpr uint32_t cache_length = 4;
    constexpr uint32_t base_position = 7;
    constexpr uint32_t rows = 6;
    const std::vector<float> taps{
        0.25f, -0.5f, 0.75f,
        -1.0f, 0.5f, 0.125f,
        0.75f, 0.25f, -0.25f,
        0.5f, -0.75f, 1.0f};
    std::vector<float> projected(static_cast<size_t>(rows) * 3 * width);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t column = 0; column < width; ++column) {
            const size_t base = static_cast<size_t>(row) * 3 * width;
            projected[base + column] = 0.5f + 0.1f * static_cast<float>(row + column);
            projected[base + width + column] = 1.0f + 0.05f * static_cast<float>(row + 2 * column);
            projected[base + 2 * width + column] = -0.3f + 0.07f * static_cast<float>(2 * row + column);
        }
    }

    std::vector<float> expected_state(static_cast<size_t>(cache_length) * width, 0.0f);
    std::vector<float> expected_output(static_cast<size_t>(rows) * width, 0.0f);
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t cursor = (base_position + row) % cache_length;
        const size_t projected_base = static_cast<size_t>(row) * 3 * width;
        for (uint32_t column = 0; column < width; ++column) {
            expected_state[static_cast<size_t>(cursor) * width + column] =
                projected[projected_base + column] *
                projected[projected_base + 2 * width + column];
        }
        for (uint32_t column = 0; column < width; ++column) {
            float sum = 0.0f;
            for (uint32_t tap = 0; tap < cache_length; ++tap) {
                const uint32_t slot = (cursor + 1 + tap) % cache_length;
                sum += expected_state[static_cast<size_t>(slot) * width + column] *
                       taps[static_cast<size_t>(tap) * width + column];
            }
            expected_output[static_cast<size_t>(row) * width + column] =
                projected[projected_base + width + column] * sum;
        }
    }

    id<MTLBuffer> projected_buffer = buffer(device, projected);
    id<MTLBuffer> taps_buffer = buffer(device, taps);
    id<MTLBuffer> state_buffer = buffer(
        device, std::vector<float>(static_cast<size_t>(cache_length) * width, 0.0f));
    id<MTLBuffer> output_buffer = buffer(
        device, std::vector<float>(static_cast<size_t>(rows) * width, 0.0f));
    id<MTLBuffer> buffers[] = {projected_buffer, taps_buffer, state_buffer, output_buffer};
    const uint32_t initial_cursor = base_position % cache_length;

    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    id<MTLComputePipelineState> state = pipeline(
        device, library, "celeg_shortconv_batch_ring");
    [encoder setComputePipelineState:state];
    for (NSUInteger index = 0; index < 4; ++index) {
        [encoder setBuffer:buffers[index] offset:0 atIndex:index];
    }
    [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
    [encoder setBytes:&width length:sizeof(width) atIndex:5];
    [encoder setBytes:&cache_length length:sizeof(cache_length) atIndex:6];
    [encoder setBytes:&initial_cursor length:sizeof(initial_cursor) atIndex:7];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal short-convolution batch dispatch failed");
    }

    const float* actual_output = static_cast<const float*>(output_buffer.contents);
    const float* actual_state = static_cast<const float*>(state_buffer.contents);
    for (size_t index = 0; index < expected_output.size(); ++index) {
        check_close(actual_output[index], expected_output[index], "short-convolution batch output");
    }
    for (size_t index = 0; index < expected_state.size(); ++index) {
        check_close(actual_state[index], expected_state[index], "short-convolution batch state");
    }

    const std::vector<float> decode_projected(projected.begin(), projected.begin() + 3 * width);
    std::vector<float> decode_expected_state(static_cast<size_t>(cache_length) * width, 0.0f);
    std::vector<float> decode_expected_output(width, 0.0f);
    const uint32_t decode_cursor = base_position % cache_length;
    for (uint32_t column = 0; column < width; ++column) {
        decode_expected_state[static_cast<size_t>(decode_cursor) * width + column] =
            decode_projected[column] * decode_projected[2 * width + column];
        float sum = 0.0f;
        for (uint32_t tap = 0; tap < cache_length; ++tap) {
            const uint32_t slot = (decode_cursor + 1 + tap) % cache_length;
            sum += decode_expected_state[static_cast<size_t>(slot) * width + column] *
                   taps[static_cast<size_t>(tap) * width + column];
        }
        decode_expected_output[column] = decode_projected[width + column] * sum;
    }

    id<MTLBuffer> decode_projected_buffer = buffer(device, decode_projected);
    id<MTLBuffer> decode_state_buffer = buffer(device, std::vector<float>(
        static_cast<size_t>(cache_length) * width, 0.0f));
    id<MTLBuffer> decode_output_buffer = buffer(device, std::vector<float>(width, 0.0f));
    id<MTLBuffer> decode_buffers[] = {
        decode_projected_buffer, taps_buffer, decode_state_buffer, decode_output_buffer};
    id<MTLCommandBuffer> decode_command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> decode_encoder = [decode_command computeCommandEncoder];
    id<MTLComputePipelineState> decode_pipeline = pipeline(
        device, library, "celeg_shortconv_ring");
    [decode_encoder setComputePipelineState:decode_pipeline];
    for (NSUInteger index = 0; index < 4; ++index) {
        [decode_encoder setBuffer:decode_buffers[index] offset:0 atIndex:index];
    }
    [decode_encoder setBytes:&width length:sizeof(width) atIndex:4];
    [decode_encoder setBytes:&cache_length length:sizeof(cache_length) atIndex:5];
    [decode_encoder setBytes:&decode_cursor length:sizeof(decode_cursor) atIndex:6];
    [decode_encoder dispatchThreads:MTLSizeMake(width, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [decode_encoder endEncoding];
    [decode_command commit];
    [decode_command waitUntilCompleted];
    if (decode_command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal short-convolution decode dispatch failed");
    }
    const float* decode_actual = static_cast<const float*>(decode_output_buffer.contents);
    const float* decode_state = static_cast<const float*>(decode_state_buffer.contents);
    for (uint32_t index = 0; index < width; ++index) {
        check_close(decode_actual[index], decode_expected_output[index],
                    "short-convolution decode output");
    }
    for (size_t index = 0; index < decode_expected_state.size(); ++index) {
        check_close(decode_state[index], decode_expected_state[index],
                    "short-convolution decode state");
    }
}

void check_gated_delta(id<MTLDevice> device, id<MTLCommandQueue> queue,
                       id<MTLLibrary> library) {
    const std::vector<float> projected_values{1.0f, 2.0f, 3.0f};
    const std::vector<float> zero{0.0f};
    const std::vector<float> one{1.0f};
    const std::vector<float> convolution_values{1.0f, 1.0f, 1.0f};
    const std::vector<float> convolution_state_values{0.0f, 0.0f, 0.0f};
    id<MTLBuffer> projected = buffer(device, projected_values);
    id<MTLBuffer> z = buffer(device, one);
    id<MTLBuffer> beta = buffer(device, zero);
    id<MTLBuffer> decay = buffer(device, zero);
    id<MTLBuffer> convolution = buffer(device, convolution_values);
    id<MTLBuffer> dt_bias = buffer(device, zero);
    id<MTLBuffer> a_log = buffer(device, zero);
    id<MTLBuffer> norm = buffer(device, one);
    id<MTLBuffer> conv_state = buffer(device, convolution_state_values);
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
    check_close(actual_output[0], expected, "gated-delta output");
    check_close(static_cast<const float*>(recurrent_state.contents)[0], state,
                "gated-delta state");
}

void check_mamba(id<MTLDevice> device, id<MTLCommandQueue> queue,
                 id<MTLLibrary> library) {
    const std::vector<float> projected_values{1.0f, 2.0f, 3.0f, 4.0f, 0.0f};
    const std::vector<float> one{1.0f};
    const std::vector<float> convolution_values{1.0f, 1.0f, 1.0f};
    const std::vector<float> convolution_state_values{0.0f, 0.0f, 0.0f};
    const std::vector<float> convolution_bias_values{0.0f, 0.0f, 0.0f};
    const std::vector<float> zero{0.0f};
    id<MTLBuffer> projected = buffer(device, projected_values);
    id<MTLBuffer> convolution = buffer(device, convolution_values);
    id<MTLBuffer> conv_bias = buffer(device, convolution_bias_values);
    id<MTLBuffer> dt_bias = buffer(device, zero);
    id<MTLBuffer> a_log = buffer(device, zero);
    id<MTLBuffer> d = buffer(device, one);
    id<MTLBuffer> norm = buffer(device, one);
    id<MTLBuffer> conv_state = buffer(device, convolution_state_values);
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
    const float b = 3.0f / (1.0f + std::exp(-3.0f));
    const float c = 4.0f / (1.0f + std::exp(-4.0f));
    const float state = std::log(2.0f) * b * x;
    const float raw = state * c + x;
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
        check_shortconv_ring(device, queue, library);
        check_gated_delta(device, queue, library);
        check_mamba(device, queue, library);
        std::cout << "metal recurrent kernels passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
