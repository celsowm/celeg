#include "celeg/checkpoint/formats/gguf.hpp"
#include "celeg/quantization/ggml.hpp"

#include "metal_inference_source.hpp"
#include "metal_tensor_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
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

std::vector<float> run_embedding(id<MTLDevice> device,
                                 const celeg::GgufTensorView& tensor) {
    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:celeg::metal_detail::kInferenceShader];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        throw std::runtime_error("Metal quantization shader compilation failed: " +
                                 (error ? ns_string(error.localizedDescription) : "unknown error"));
    }
    const celeg::GgmlType type = tensor.type;
    const char* kernel = nullptr;
    switch (type) {
        case celeg::GgmlType::Q4_0: kernel = "celeg_embedding_q4_0"; break;
        case celeg::GgmlType::Q4_K: kernel = "celeg_embedding_q4k"; break;
        case celeg::GgmlType::Q5_K: kernel = "celeg_embedding_q5k"; break;
        case celeg::GgmlType::Q6_K: kernel = "celeg_embedding_q6k"; break;
        case celeg::GgmlType::Q8_0: kernel = "celeg_embedding_q8_0"; break;
        default: throw std::runtime_error("unsupported Metal quantization test type");
    }
    NSString* function_name = [NSString stringWithUTF8String:kernel];
    id<MTLFunction> function = [library newFunctionWithName:function_name];
    if (!function) throw std::runtime_error("Metal quantization function is missing");
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                      error:&error];
    if (!pipeline) throw std::runtime_error("Metal quantization pipeline creation failed");
    id<MTLBuffer> weights = [device newBufferWithBytes:tensor.data
                                                 length:tensor.bytes
                                                options:MTLResourceStorageModeShared];
    const uint32_t width = static_cast<uint32_t>(tensor.shape.at(1));
    const uint32_t token = 0;
    id<MTLBuffer> output = [device newBufferWithLength:width * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:weights offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBytes:&width length:sizeof(width) atIndex:2];
    [encoder setBytes:&token length:sizeof(token) atIndex:3];
    const NSUInteger threads = std::min<NSUInteger>(width, pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(width, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal quantization dispatch failed");
    }
    std::vector<float> values(width);
    std::memcpy(values.data(), output.contents, values.size() * sizeof(float));
    return values;
}

std::vector<float> run_matvec(id<MTLDevice> device,
                              const celeg::GgufTensorView& tensor,
                              uint32_t rows,
                              const std::vector<float>& input) {
    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:celeg::metal_detail::kInferenceShader];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (!library) throw std::runtime_error("Metal matvec shader compilation failed");
    const char* kernel = nullptr;
    switch (tensor.type) {
        case celeg::GgmlType::Q4_0: kernel = "celeg_matvec_q4_0"; break;
        case celeg::GgmlType::Q4_K: kernel = "celeg_matvec_q4k"; break;
        case celeg::GgmlType::Q5_K: kernel = "celeg_matvec_q5k"; break;
        case celeg::GgmlType::Q6_K: kernel = "celeg_matvec_q6k"; break;
        case celeg::GgmlType::Q8_0: kernel = "celeg_matvec_q8_0"; break;
        default: throw std::runtime_error("unsupported Metal matvec test type");
    }
    id<MTLFunction> function = [library newFunctionWithName:
        [NSString stringWithUTF8String:kernel]];
    if (!function) throw std::runtime_error("Metal matvec function is missing");
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                      error:&error];
    if (!pipeline) throw std::runtime_error("Metal matvec pipeline creation failed");
    id<MTLBuffer> weights = [device newBufferWithBytes:tensor.data
                                                 length:tensor.bytes
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> input_buffer = [device newBufferWithBytes:input.data()
                                                       length:input.size() * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
    std::vector<float> output(rows + 1, 0.0f);
    id<MTLBuffer> output_buffer = [device newBufferWithBytes:output.data()
                                                        length:output.size() * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    const uint32_t cols = static_cast<uint32_t>(tensor.shape.at(1));
    const uint32_t row_bytes = static_cast<uint32_t>(tensor.bytes / tensor.shape.at(0));
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:weights offset:0 atIndex:0];
    [encoder setBuffer:input_buffer offset:sizeof(float) atIndex:1];
    [encoder setBuffer:output_buffer offset:sizeof(float) atIndex:2];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
    [encoder setBytes:&row_bytes length:sizeof(row_bytes) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake((rows + 15u) / 16u, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal matvec dispatch failed");
    }
    std::memcpy(output.data(), output_buffer.contents, output.size() * sizeof(float));
    return std::vector<float>(output.begin() + 1, output.end());
}

std::vector<float> run_matmul_quantized(id<MTLDevice> device,
                                        const celeg::GgufTensorView& tensor,
                                        uint32_t input_rows, uint32_t output_rows,
                                        const std::vector<float>& input,
                                        const char* kernel_name) {
    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:celeg::metal_detail::kTensorShader];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (!library) throw std::runtime_error("Metal tensor shader compilation failed");
    id<MTLFunction> function = [library newFunctionWithName:
        [NSString stringWithUTF8String:kernel_name]];
    if (!function) throw std::runtime_error("Metal quantized tensor function is missing");
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                      error:&error];
    if (!pipeline) throw std::runtime_error("Metal quantized tensor pipeline creation failed");
    id<MTLBuffer> weights = [device newBufferWithBytes:tensor.data
                                                 length:tensor.bytes
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> input_buffer = [device newBufferWithBytes:input.data()
                                                       length:input.size() * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    std::vector<float> output(static_cast<size_t>(input_rows) * output_rows, 0.0f);
    id<MTLBuffer> output_buffer = [device newBufferWithBytes:output.data()
                                                        length:output.size() * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    const uint32_t cols = static_cast<uint32_t>(tensor.shape.at(1));
    const uint32_t row_bytes = static_cast<uint32_t>(tensor.bytes / tensor.shape.at(0));
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:weights offset:0 atIndex:0];
    [encoder setBuffer:input_buffer offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBytes:&input_rows length:sizeof(input_rows) atIndex:3];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
    [encoder setBytes:&output_rows length:sizeof(output_rows) atIndex:5];
    [encoder setBytes:&output_rows length:sizeof(output_rows) atIndex:6];
    [encoder setBytes:&row_bytes length:sizeof(row_bytes) atIndex:7];
    [encoder setThreadgroupMemoryLength:64u * 32u * sizeof(uint16_t) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake((output_rows + 63u) / 64u,
                                              (input_rows + 127u) / 128u, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal quantized tensor dispatch failed");
    }
    std::memcpy(output.data(), output_buffer.contents, output.size() * sizeof(float));
    return output;
}

std::vector<float> run_matmul_pair_quantized(id<MTLDevice> device,
                                             const celeg::GgufTensorView& tensor,
                                             uint32_t input_rows, uint32_t output_rows,
                                             const std::vector<float>& input,
                                             const char* kernel_name) {
    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:celeg::metal_detail::kTensorShader];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (!library) throw std::runtime_error("Metal paired tensor shader compilation failed");
    id<MTLFunction> function = [library newFunctionWithName:
        [NSString stringWithUTF8String:kernel_name]];
    if (!function) throw std::runtime_error("Metal paired tensor function is missing");
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                      error:&error];
    if (!pipeline) throw std::runtime_error("Metal paired tensor pipeline creation failed");
    id<MTLBuffer> weights = [device newBufferWithBytes:tensor.data
                                                 length:tensor.bytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> input_buffer = [device newBufferWithBytes:input.data()
                                                       length:input.size() * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    const uint32_t output_stride = output_rows * 2;
    std::vector<float> output(static_cast<size_t>(input_rows) * output_stride, 0.0f);
    id<MTLBuffer> output_buffer = [device newBufferWithBytes:output.data()
                                                        length:output.size() * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    const uint32_t cols = static_cast<uint32_t>(tensor.shape.at(1));
    const uint32_t row_bytes = static_cast<uint32_t>(tensor.bytes / tensor.shape.at(0));
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:weights offset:0 atIndex:0];
    [encoder setBuffer:weights offset:0 atIndex:1];
    [encoder setBuffer:input_buffer offset:0 atIndex:2];
    [encoder setBuffer:output_buffer offset:0 atIndex:3];
    [encoder setBytes:&input_rows length:sizeof(input_rows) atIndex:4];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:5];
    [encoder setBytes:&output_rows length:sizeof(output_rows) atIndex:6];
    [encoder setBytes:&output_stride length:sizeof(output_stride) atIndex:7];
    [encoder setBytes:&row_bytes length:sizeof(row_bytes) atIndex:8];
    [encoder setThreadgroupMemoryLength:2u * 64u * 32u * sizeof(uint16_t) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake((output_rows + 63u) / 64u,
                                              (input_rows + 127u) / 128u, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal paired tensor dispatch failed");
    }
    std::memcpy(output.data(), output_buffer.contents, output.size() * sizeof(float));
    return output;
}

/// Mirrors `MetalModel::Impl::swiglu_matvec_kernel` so the test exercises the
/// same launch geometry the runtime uses.
struct Geometry {
    uint32_t rows_per_threadgroup;
    uint32_t threads;
    uint32_t threadgroup_floats;
};

Geometry swiglu_geometry(const char* kernel_name) {
    if (std::strcmp(kernel_name, "celeg_swiglu_matvec_q4k") == 0) return {16, 128, 0};
    if (std::strcmp(kernel_name, "celeg_swiglu_matvec_q5k") == 0) return {16, 128, 0};
    if (std::strcmp(kernel_name, "celeg_swiglu_matvec_q6k") == 0) return {16, 128, 0};
    return {2, 128, 8};
}

std::vector<float> run_swiglu_matvec(id<MTLDevice> device,
                                     const celeg::GgufTensorView& tensor,
                                     uint32_t rows,
                                     const std::vector<float>& gate_up,
                                     const char* kernel_name) {
    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:celeg::metal_detail::kInferenceShader];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (!library) throw std::runtime_error("Metal SwiGLU shader compilation failed");
    id<MTLFunction> function = [library newFunctionWithName:
        [NSString stringWithUTF8String:kernel_name]];
    if (!function) throw std::runtime_error("Metal quantized SwiGLU function is missing");
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                      error:&error];
    if (!pipeline) throw std::runtime_error("Metal quantized SwiGLU pipeline creation failed");
    id<MTLBuffer> weights = [device newBufferWithBytes:tensor.data
                                                 length:tensor.bytes
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> gate_up_buffer = [device newBufferWithBytes:gate_up.data()
                                                         length:gate_up.size() * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    std::vector<float> output(rows, 0.0f);
    id<MTLBuffer> output_buffer = [device newBufferWithBytes:output.data()
                                                        length:output.size() * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    const uint32_t cols = static_cast<uint32_t>(tensor.shape.at(1));
    const uint32_t row_bytes = static_cast<uint32_t>(tensor.bytes / tensor.shape.at(0));
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:weights offset:0 atIndex:0];
    [encoder setBuffer:gate_up_buffer offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
    [encoder setBytes:&row_bytes length:sizeof(row_bytes) atIndex:5];
    const Geometry geometry = swiglu_geometry(kernel_name);
    if (geometry.threadgroup_floats != 0) {
        [encoder setThreadgroupMemoryLength:geometry.threadgroup_floats * sizeof(float)
                                    atIndex:0];
    }
    const uint32_t groups =
        (rows + geometry.rows_per_threadgroup - 1u) / geometry.rows_per_threadgroup;
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(geometry.threads, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal quantized SwiGLU dispatch failed");
    }
    std::memcpy(output.data(), output_buffer.contents, output.size() * sizeof(float));
    return output;
}

bool check_type(const celeg::GgufFile& file, celeg::GgmlType type,
                id<MTLDevice> device) {
    for (const std::string& name : file.tensor_names()) {
        const celeg::GgufTensorView tensor = file.tensor(name);
        const celeg::GgmlTypeTrait trait = celeg::ggml_type_trait(type);
        if (tensor.type != type || tensor.shape.size() != 2 || tensor.shape[0] == 0 ||
            tensor.shape[1] == 0 || tensor.shape[1] % trait.block_size != 0) {
            continue;
        }
        celeg::GgmlMatrixView matrix;
        matrix.type = type;
        matrix.rows = static_cast<uint32_t>(tensor.shape[0]);
        matrix.cols = static_cast<uint32_t>(tensor.shape[1]);
        matrix.data = tensor.data;
        matrix.bytes = tensor.bytes;
        matrix.validate();
        std::vector<float> expected(matrix.cols);
        celeg::ggml_decode_row(matrix, 0, expected.data());
        const std::vector<float> actual = run_embedding(device, tensor);
        float maximum = 0.0f;
        for (size_t index = 0; index < expected.size(); ++index) {
            const float difference = std::abs(expected[index] - actual[index]);
            maximum = std::max(maximum, difference);
        }
        std::cout << celeg::ggml_type_name(type) << " tensor=" << name
                  << " max_error=" << maximum << '\n';
        if (!(maximum < 1.0e-5f)) {
            throw std::runtime_error("Metal quantized embedding differs from CPU reference");
        }
        return true;
    }
    return false;
}

bool check_matvec(const celeg::GgufFile& file, celeg::GgmlType type,
                  id<MTLDevice> device) {
    for (const std::string& name : file.tensor_names()) {
        const celeg::GgufTensorView tensor = file.tensor(name);
        const celeg::GgmlTypeTrait trait = celeg::ggml_type_trait(type);
        if (tensor.type != type || tensor.shape.size() != 2 || tensor.shape[0] == 0 ||
            tensor.shape[1] == 0 || tensor.shape[1] % trait.block_size != 0) {
            continue;
        }
        const uint32_t rows = 3;
        const uint32_t cols = static_cast<uint32_t>(tensor.shape.at(1));
        std::vector<float> input(cols + 1);
        for (uint32_t index = 0; index < cols; ++index) {
            input[index + 1] = std::sin(static_cast<float>(index + 1) * 0.017f);
        }
        const std::vector<float> actual = run_matvec(device, tensor, rows, input);
        celeg::GgmlMatrixView matrix;
        matrix.type = type;
        matrix.rows = static_cast<uint32_t>(tensor.shape.at(0));
        matrix.cols = cols;
        matrix.data = tensor.data;
        matrix.bytes = tensor.bytes;
        matrix.validate();
        std::vector<float> expected(cols);
        float maximum = 0.0f;
        for (uint32_t row = 0; row < rows; ++row) {
            celeg::ggml_decode_row(matrix, row, expected.data());
            float reference = 0.0f;
            for (uint32_t index = 0; index < cols; ++index) {
                reference += expected[index] * input[index + 1];
            }
            maximum = std::max(maximum, std::abs(reference - actual[row]));
        }
        std::cout << celeg::ggml_type_name(type) << " matvec=" << name
                  << " max_error=" << maximum << '\n';
        if (!(maximum < 1.0e-3f)) {
            throw std::runtime_error("Metal quantized matvec differs from FP32 reference");
        }
        return true;
    }
    return false;
}

bool check_matmul_quantized(const celeg::GgufFile& file, celeg::GgmlType type,
                            const char* kernel_name, id<MTLDevice> device) {
    for (const std::string& name : file.tensor_names()) {
        const celeg::GgufTensorView tensor = file.tensor(name);
        const celeg::GgmlTypeTrait trait = celeg::ggml_type_trait(type);
        if (tensor.type != type || tensor.shape.size() != 2 || tensor.shape[0] < 65 ||
            tensor.shape[1] == 0 || tensor.shape[1] % trait.block_size != 0) {
            continue;
        }
        constexpr uint32_t input_rows = 129;
        constexpr uint32_t output_rows = 65;
        const uint32_t cols = static_cast<uint32_t>(tensor.shape.at(1));
        std::vector<float> input(static_cast<size_t>(input_rows) * cols);
        for (uint32_t row = 0; row < input_rows; ++row) {
            for (uint32_t column = 0; column < cols; ++column) {
                input[static_cast<size_t>(row) * cols + column] = std::sin(
                    static_cast<float>((row + 1) * (column + 3)) * 0.00037f);
            }
        }
        const std::vector<float> actual = run_matmul_quantized(
            device, tensor, input_rows, output_rows, input, kernel_name);
        celeg::GgmlMatrixView matrix;
        matrix.type = type;
        matrix.rows = static_cast<uint32_t>(tensor.shape.at(0));
        matrix.cols = cols;
        matrix.data = tensor.data;
        matrix.bytes = tensor.bytes;
        matrix.validate();
        std::vector<float> decoded(cols);
        float maximum = 0.0f;
        for (uint32_t row = 0; row < output_rows; ++row) {
            celeg::ggml_decode_row(matrix, row, decoded.data());
            for (uint32_t input_row = 0; input_row < input_rows; ++input_row) {
                float reference = 0.0f;
                for (uint32_t column = 0; column < cols; ++column) {
                    reference += decoded[column] *
                        input[static_cast<size_t>(input_row) * cols + column];
                }
                const float observed = actual[static_cast<size_t>(input_row) * output_rows + row];
                maximum = std::max(maximum, std::abs(reference - observed));
            }
        }
        std::cout << celeg::ggml_type_name(type) << " matmul=" << name
                  << " max_error=" << maximum << '\n';
        if (!(maximum < 0.05f)) {
            throw std::runtime_error("Metal quantized tensor matmul differs from FP32 reference");
        }
        return true;
    }
    return false;
}

bool check_matmul_pair_quantized(const celeg::GgufFile& file, celeg::GgmlType type,
                                 const char* kernel_name, id<MTLDevice> device) {
    for (const std::string& name : file.tensor_names()) {
        const celeg::GgufTensorView tensor = file.tensor(name);
        const celeg::GgmlTypeTrait trait = celeg::ggml_type_trait(type);
        if (tensor.type != type || tensor.shape.size() != 2 || tensor.shape[0] < 65 ||
            tensor.shape[1] == 0 || tensor.shape[1] % trait.block_size != 0) {
            continue;
        }
        constexpr uint32_t input_rows = 129;
        constexpr uint32_t output_rows = 65;
        const uint32_t cols = static_cast<uint32_t>(tensor.shape.at(1));
        std::vector<float> input(static_cast<size_t>(input_rows) * cols);
        for (uint32_t row = 0; row < input_rows; ++row) {
            for (uint32_t column = 0; column < cols; ++column) {
                input[static_cast<size_t>(row) * cols + column] = std::sin(
                    static_cast<float>((row + 1) * (column + 3)) * 0.00037f);
            }
        }
        const std::vector<float> actual = run_matmul_pair_quantized(
            device, tensor, input_rows, output_rows, input, kernel_name);
        celeg::GgmlMatrixView matrix;
        matrix.type = type;
        matrix.rows = static_cast<uint32_t>(tensor.shape.at(0));
        matrix.cols = cols;
        matrix.data = tensor.data;
        matrix.bytes = tensor.bytes;
        matrix.validate();
        std::vector<float> decoded(cols);
        float maximum = 0.0f;
        for (uint32_t row = 0; row < output_rows; ++row) {
            celeg::ggml_decode_row(matrix, row, decoded.data());
            for (uint32_t input_row = 0; input_row < input_rows; ++input_row) {
                float reference = 0.0f;
                for (uint32_t column = 0; column < cols; ++column) {
                    reference += decoded[column] *
                        input[static_cast<size_t>(input_row) * cols + column];
                }
                const size_t base = static_cast<size_t>(input_row) * output_rows * 2 + row;
                maximum = std::max(maximum, std::abs(reference - actual[base]));
                maximum = std::max(maximum,
                                   std::abs(reference - actual[base + output_rows]));
            }
        }
        std::cout << celeg::ggml_type_name(type) << " matmul_pair=" << name
                  << " max_error=" << maximum << '\n';
        if (!(maximum < 0.05f)) {
            throw std::runtime_error("Metal paired quantized tensor matmul differs from FP32 reference");
        }
        return true;
    }
    return false;
}

bool check_swiglu_matvec(const celeg::GgufFile& file, celeg::GgmlType type,
                         const char* kernel_name, id<MTLDevice> device) {
    for (const std::string& name : file.tensor_names()) {
        const celeg::GgufTensorView tensor = file.tensor(name);
        const celeg::GgmlTypeTrait trait = celeg::ggml_type_trait(type);
        if (tensor.type != type || tensor.shape.size() != 2 || tensor.shape[0] < 3 ||
            tensor.shape[1] == 0 || tensor.shape[1] % trait.block_size != 0) {
            continue;
        }
        constexpr uint32_t rows = 3;
        const uint32_t cols = static_cast<uint32_t>(tensor.shape.at(1));
        std::vector<float> gate_up(static_cast<size_t>(cols) * 2);
        for (uint32_t index = 0; index < cols; ++index) {
            gate_up[index] = std::sin(static_cast<float>(index + 1) * 0.013f);
            gate_up[cols + index] = std::cos(static_cast<float>(index + 1) * 0.019f);
        }
        const std::vector<float> actual = run_swiglu_matvec(
            device, tensor, rows, gate_up, kernel_name);
        celeg::GgmlMatrixView matrix;
        matrix.type = type;
        matrix.rows = static_cast<uint32_t>(tensor.shape.at(0));
        matrix.cols = cols;
        matrix.data = tensor.data;
        matrix.bytes = tensor.bytes;
        matrix.validate();
        std::vector<float> decoded(cols);
        float maximum = 0.0f;
        for (uint32_t row = 0; row < rows; ++row) {
            celeg::ggml_decode_row(matrix, row, decoded.data());
            float reference = 0.0f;
            for (uint32_t index = 0; index < cols; ++index) {
                const float gate = gate_up[index];
                reference += decoded[index] * gate / (1.0f + std::exp(-gate)) *
                    gate_up[cols + index];
            }
            maximum = std::max(maximum, std::abs(reference - actual[row]));
        }
        std::cout << celeg::ggml_type_name(type) << " swiglu_matvec=" << name
                  << " max_error=" << maximum << '\n';
        if (!(maximum < 1.0e-3f)) {
            throw std::runtime_error("Metal quantized SwiGLU matvec differs from FP32 reference");
        }
        return true;
    }
    return false;
}


}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: metal_quantization_test GGUF");
        const celeg::GgufFile file(argv[1]);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("no default Metal device is available");
        constexpr std::array types{
            celeg::GgmlType::Q4_0,
            celeg::GgmlType::Q4_K,
            celeg::GgmlType::Q5_K,
            celeg::GgmlType::Q6_K,
            celeg::GgmlType::Q8_0};
        int checked = 0;
        int matvec_checked = 0;
        bool q4_0_matmul_checked = false;
        bool q4k_matmul_checked = false;
        bool q5k_matmul_checked = false;
        bool q6k_matmul_checked = false;
        bool q8_0_matmul_checked = false;
        bool q5k_matmul_pair_checked = false;
        bool q6k_matmul_pair_checked = false;
        bool q4_0_checked = false;
        bool q4k_checked = false;
        bool q5k_checked = false;
        bool q6k_checked = false;
        bool q8_0_checked = false;
        int swiglu_checked = 0;
        for (const celeg::GgmlType type : types) {
            const bool type_checked = check_type(file, type, device);
            checked += type_checked ? 1 : 0;
            q4_0_checked = q4_0_checked ||
                (type == celeg::GgmlType::Q4_0 && type_checked);
            q4k_checked = q4k_checked ||
                (type == celeg::GgmlType::Q4_K && type_checked);
            q5k_checked = q5k_checked ||
                (type == celeg::GgmlType::Q5_K && type_checked);
            q6k_checked = q6k_checked ||
                (type == celeg::GgmlType::Q6_K && type_checked);
            q8_0_checked = q8_0_checked ||
                (type == celeg::GgmlType::Q8_0 && type_checked);
            matvec_checked += check_matvec(file, type, device) ? 1 : 0;
        }
        q4_0_matmul_checked = check_matmul_quantized(
            file, celeg::GgmlType::Q4_0, "celeg_matmul_tensor_q4_0", device);
        q4k_matmul_checked = check_matmul_quantized(
            file, celeg::GgmlType::Q4_K, "celeg_matmul_tensor_q4k", device);
        q5k_matmul_checked = check_matmul_quantized(
            file, celeg::GgmlType::Q5_K, "celeg_matmul_tensor_q5k", device);
        q6k_matmul_checked = check_matmul_quantized(
            file, celeg::GgmlType::Q6_K, "celeg_matmul_tensor_q6k", device);
        q8_0_matmul_checked = check_matmul_quantized(
            file, celeg::GgmlType::Q8_0, "celeg_matmul_tensor_q8_0", device);
        q5k_matmul_pair_checked = check_matmul_pair_quantized(
            file, celeg::GgmlType::Q5_K, "celeg_matmul_tensor_pair_q5k", device);
        q6k_matmul_pair_checked = check_matmul_pair_quantized(
            file, celeg::GgmlType::Q6_K, "celeg_matmul_tensor_pair_q6k", device);
        swiglu_checked += check_swiglu_matvec(
            file, celeg::GgmlType::Q4_K, "celeg_swiglu_matvec_q4k", device) ? 1 : 0;
        swiglu_checked += check_swiglu_matvec(
            file, celeg::GgmlType::Q5_K, "celeg_swiglu_matvec_q5k", device) ? 1 : 0;
        swiglu_checked += check_swiglu_matvec(
            file, celeg::GgmlType::Q6_K, "celeg_swiglu_matvec_q6k", device) ? 1 : 0;
        if (checked == 0) throw std::runtime_error("cached GGUF has no native Metal test tensor");
        if (matvec_checked == 0) {
            throw std::runtime_error("cached GGUF has no native Metal matvec test tensor");
        }
        if ((q4_0_checked && !q4_0_matmul_checked) ||
            (q4k_checked && !q4k_matmul_checked) ||
            (q5k_checked && !q5k_matmul_checked) ||
            (q6k_checked && !q6k_matmul_checked) ||
            (q5k_checked && !q5k_matmul_pair_checked) ||
            (q6k_checked && !q6k_matmul_pair_checked) ||
            (q8_0_checked && !q8_0_matmul_checked)) {
            throw std::runtime_error(
                "cached GGUF has no native Metal quantized matmul test tensor");
        }
        if ((q4k_checked || q5k_checked || q6k_checked) && swiglu_checked == 0) {
            throw std::runtime_error(
                "cached GGUF has no native Metal Q4_K/Q5_K/Q6_K SwiGLU test tensor");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
