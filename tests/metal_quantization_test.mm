#include "celeg/backend/cpu/gguf.hpp"
#include "celeg/checkpoint/formats/gguf.hpp"

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
    NSString* function_name = [NSString stringWithUTF8String:
        type == celeg::GgmlType::Q4_K ? "celeg_embedding_q4k" : "celeg_embedding_q6k"];
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

void check_type(const celeg::GgufFile& file, celeg::GgmlType type,
                id<MTLDevice> device) {
    for (const std::string& name : file.tensor_names()) {
        const celeg::GgufTensorInfo& info = file.tensor_info(name);
        if (info.type != type || info.dims.size() != 2 || info.dims[0] == 0 ||
            info.dims[1] == 0 || info.dims[1] % 256 != 0) {
            continue;
        }
        const celeg::GgufTensorView tensor = file.tensor(name);
        celeg::CpuGgufMatrix matrix;
        matrix.type = type;
        matrix.rows = static_cast<uint32_t>(tensor.shape[0]);
        matrix.cols = static_cast<uint32_t>(tensor.shape[1]);
        matrix.data = tensor.data;
        matrix.bytes = tensor.bytes;
        matrix.validate();
        std::vector<float> expected(matrix.cols);
        celeg::cpu_gguf_dequantize_row(matrix, 0, expected.data());
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
        return;
    }
    throw std::runtime_error(std::string("cached GGUF has no test tensor for ") +
                             celeg::ggml_type_name(type));
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: metal_quantization_test GGUF");
        const celeg::GgufFile file(argv[1]);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("no default Metal device is available");
        check_type(file, celeg::GgmlType::Q4_K, device);
        check_type(file, celeg::GgmlType::Q6_K, device);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
