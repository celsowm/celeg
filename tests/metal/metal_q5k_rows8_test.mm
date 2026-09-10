#include "celeg/quantization/ggml.hpp"
#include "celeg/quantization/gguf_blocks.hpp"

#include "metal_inference_source.hpp"

#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr uint32_t kRows = 33;
constexpr uint32_t kCols = 256;
constexpr uint32_t kRowsPerThreadgroup = 32;
constexpr uint32_t kThreads = 128;

std::vector<celeg::ggml_detail::BlockQ5K> make_weights() {
    std::vector<celeg::ggml_detail::BlockQ5K> blocks(kRows);
    for (uint32_t row = 0; row < kRows; ++row) {
        auto& block = blocks[row];
        block.d = 0x3c00u;     // fp16(1.0)
        block.dmin = 0x3800u;  // fp16(0.5)
        for (uint32_t i = 0; i < 12; ++i) {
            block.scales[i] = static_cast<uint8_t>(
                (17u * i + 29u * row + 3u) & 0xffu);
        }
        for (uint32_t i = 0; i < 32; ++i) {
            block.qh[i] = static_cast<uint8_t>(
                ((i * 37u) ^ (row * 53u) ^ 0xa5u) & 0xffu);
        }
        for (uint32_t i = 0; i < 128; ++i) {
            block.qs[i] = static_cast<uint8_t>(
                ((i * 11u) + (row * 23u) + (i >> 2)) & 0xffu);
        }
    }
    return blocks;
}

std::vector<float> host_reference(
    const std::vector<celeg::ggml_detail::BlockQ5K>& blocks,
    const std::vector<float>& input) {
    celeg::GgmlMatrixView matrix;
    matrix.type = celeg::GgmlType::Q5_K;
    matrix.rows = kRows;
    matrix.cols = kCols;
    matrix.data = reinterpret_cast<const std::byte*>(blocks.data());
    matrix.bytes = blocks.size() * sizeof(blocks.front());
    matrix.validate();

    std::vector<float> decoded(kCols);
    std::vector<float> output(kRows, 0.0f);
    for (uint32_t row = 0; row < kRows; ++row) {
        celeg::ggml_decode_row(matrix, row, decoded.data());
        float sum = 0.0f;
        for (uint32_t column = 0; column < kCols; ++column) {
            sum += decoded[column] * input[column];
        }
        output[row] = sum;
    }
    return output;
}

std::vector<float> metal_rows8(
    id<MTLDevice> device,
    const std::vector<celeg::ggml_detail::BlockQ5K>& blocks,
    const std::vector<float>& input) {
    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:
        celeg::metal_detail::kInferenceShader];
    id<MTLLibrary> library = [device newLibraryWithSource:source
                                                  options:nil
                                                    error:&error];
    if (!library) {
        throw std::runtime_error("failed to compile Metal inference shader");
    }
    id<MTLFunction> function = [library newFunctionWithName:
        @"celeg_matvec_q5k_rows8"];
    if (!function) {
        throw std::runtime_error("celeg_matvec_q5k_rows8 is missing");
    }
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        throw std::runtime_error("failed to create Q5_K rows8 pipeline");
    }

    id<MTLBuffer> weights = [device newBufferWithBytes:blocks.data()
        length:blocks.size() * sizeof(blocks.front())
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> input_buffer = [device newBufferWithBytes:input.data()
        length:input.size() * sizeof(float)
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer = [device newBufferWithLength:kRows * sizeof(float)
        options:MTLResourceStorageModeShared];

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    const uint32_t row_bytes = sizeof(celeg::ggml_detail::BlockQ5K);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:weights offset:0 atIndex:0];
    [encoder setBuffer:input_buffer offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBytes:&kRows length:sizeof(kRows) atIndex:3];
    [encoder setBytes:&kCols length:sizeof(kCols) atIndex:4];
    [encoder setBytes:&row_bytes length:sizeof(row_bytes) atIndex:5];
    [encoder dispatchThreadgroups:
        MTLSizeMake((kRows + kRowsPerThreadgroup - 1u) / kRowsPerThreadgroup, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Q5_K rows8 dispatch failed");
    }

    std::vector<float> output(kRows);
    std::memcpy(output.data(), output_buffer.contents, output.size() * sizeof(float));
    return output;
}

}

int main() {
    try {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("no default Metal device is available");

        const auto blocks = make_weights();
        std::vector<float> input(kCols);
        for (uint32_t column = 0; column < kCols; ++column) {
            input[column] = std::sin(static_cast<float>(column + 1u) * 0.017f) +
                            0.25f * std::cos(static_cast<float>(column + 3u) * 0.031f);
        }

        const auto expected = host_reference(blocks, input);
        const auto actual = metal_rows8(device, blocks, input);
        float maximum = 0.0f;
        for (uint32_t row = 0; row < kRows; ++row) {
            if (!std::isfinite(actual[row])) {
                throw std::runtime_error("Q5_K rows8 produced a non-finite value");
            }
            maximum = std::max(maximum, std::abs(expected[row] - actual[row]));
        }
        std::cout << "Q5_K rows8 max_error=" << maximum << '\n';
        if (!(maximum < 1.0e-3f)) {
            throw std::runtime_error("Metal Q5_K rows8 differs from host reference");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
