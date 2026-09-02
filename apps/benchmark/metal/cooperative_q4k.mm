/// @file
/// Benchmark-only A/B for direct Q4_K dequantization into a TensorOps
/// cooperative right-input tensor. The candidate must match the production
/// threadgroup-staged kernel bit-for-bit before timing.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr NSUInteger kTileRows = 64;
constexpr NSUInteger kTileTokens = 128;
constexpr NSUInteger kTileK = 64;
constexpr NSUInteger kThreads = 128;
constexpr NSUInteger kBaselineWeightTileBytes = kTileRows * kTileK * sizeof(uint16_t);
constexpr uint32_t kQ4KBlockValues = 256;
constexpr uint32_t kQ4KBlockBytes = 144;

struct Shape {
    const char* label;
    uint32_t output_rows;
    uint32_t cols;
};

constexpr Shape kShapes[] = {
    {"proj_1024x1024", 1024, 1024},
    {"ffn_up_4608x1024", 4608, 1024},
    {"ffn_down_1024x4608", 1024, 4608},
};
constexpr uint32_t kTokenCounts[] = {128, 256, 512};

std::string ns_string(NSString* value) {
    return value ? std::string(value.UTF8String) : std::string{};
}

std::string read_text(const char* path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error(std::string("cannot read ") + path);
    std::ostringstream output;
    output << input.rdbuf();
    return output.str();
}

id<MTLBuffer> zero_buffer(id<MTLDevice> device, size_t bytes) {
    id<MTLBuffer> result = [device newBufferWithLength:bytes
                                               options:MTLResourceStorageModeShared];
    if (!result) throw std::runtime_error("Metal buffer allocation failed");
    std::memset(result.contents, 0, bytes);
    return result;
}

id<MTLBuffer> make_input(id<MTLDevice> device, size_t elements, uint32_t seed) {
    id<MTLBuffer> result = zero_buffer(device, elements * sizeof(float));
    auto* values = static_cast<float*>(result.contents);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (size_t index = 0; index < elements; ++index) {
        values[index] = distribution(generator);
    }
    return result;
}

void fill_q4k_block(uint8_t* block, uint32_t seed) {
    block[0] = 0x00;
    block[1] = 0x3c;  // half(1.0)
    block[2] = 0x00;
    block[3] = 0x00;  // zero minimum term
    for (uint32_t index = 4; index < 16; ++index) {
        block[index] = static_cast<uint8_t>(1u + ((seed + index) & 7u));
    }
    for (uint32_t index = 16; index < kQ4KBlockBytes; ++index) {
        const uint8_t low = static_cast<uint8_t>((seed + index) & 0x0fu);
        const uint8_t high = static_cast<uint8_t>((seed + index * 3u + 5u) & 0x0fu);
        block[index] = static_cast<uint8_t>(low | (high << 4));
    }
}

id<MTLBuffer> make_weights(id<MTLDevice> device, const Shape& shape,
                           uint32_t& row_bytes) {
    if (shape.cols % kQ4KBlockValues != 0 || shape.cols % kTileK != 0) {
        throw std::runtime_error(std::string(shape.label) + " is not Q4_K/K64 aligned");
    }
    row_bytes = shape.cols / kQ4KBlockValues * kQ4KBlockBytes;
    id<MTLBuffer> result = zero_buffer(
        device, static_cast<size_t>(shape.output_rows) * row_bytes);
    auto* data = static_cast<uint8_t*>(result.contents);
    const uint32_t blocks_per_row = shape.cols / kQ4KBlockValues;
    for (uint32_t row = 0; row < shape.output_rows; ++row) {
        for (uint32_t block_index = 0; block_index < blocks_per_row; ++block_index) {
            fill_q4k_block(
                data + static_cast<size_t>(row) * row_bytes +
                    static_cast<size_t>(block_index) * kQ4KBlockBytes,
                row * 17u + block_index * 29u);
        }
    }
    return result;
}

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device,
                                          id<MTLLibrary> library,
                                          const char* name) {
    id<MTLFunction> function = [library newFunctionWithName:
        [NSString stringWithUTF8String:name]];
    if (!function) throw std::runtime_error(std::string("missing Metal kernel: ") + name);
    NSError* error = nil;
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!result) {
        throw std::runtime_error(std::string("pipeline failed for ") + name + ": " +
            (error ? ns_string(error.localizedDescription) : "unknown"));
    }
    if (result.maxTotalThreadsPerThreadgroup < kThreads) {
        throw std::runtime_error(std::string(name) + " cannot launch 128 threads");
    }
    return result;
}

void encode_matmul(id<MTLComputeCommandEncoder> encoder,
                   id<MTLComputePipelineState> state,
                   id<MTLBuffer> weights, id<MTLBuffer> input, id<MTLBuffer> output,
                   uint32_t tokens, const Shape& shape, uint32_t row_bytes,
                   bool baseline) {
    const uint32_t output_stride = shape.output_rows;
    [encoder setComputePipelineState:state];
    [encoder setBuffer:weights offset:0 atIndex:0];
    [encoder setBuffer:input offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&tokens length:sizeof(tokens) atIndex:3];
    [encoder setBytes:&shape.cols length:sizeof(shape.cols) atIndex:4];
    [encoder setBytes:&shape.output_rows length:sizeof(shape.output_rows) atIndex:5];
    [encoder setBytes:&output_stride length:sizeof(output_stride) atIndex:6];
    [encoder setBytes:&row_bytes length:sizeof(row_bytes) atIndex:7];
    if (baseline) {
        [encoder setThreadgroupMemoryLength:kBaselineWeightTileBytes atIndex:0];
    }
    const NSUInteger row_groups =
        (static_cast<NSUInteger>(shape.output_rows) + kTileRows - 1u) / kTileRows;
    const NSUInteger token_groups =
        (static_cast<NSUInteger>(tokens) + kTileTokens - 1u) / kTileTokens;
    [encoder dispatchThreadgroups:MTLSizeMake(row_groups, token_groups, 1)
            threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
}

void require_bit_exact(id<MTLDevice> device, id<MTLCommandQueue> queue,
                       id<MTLComputePipelineState> baseline,
                       id<MTLComputePipelineState> cooperative,
                       id<MTLBuffer> weights, id<MTLBuffer> input,
                       uint32_t tokens, const Shape& shape, uint32_t row_bytes) {
    const size_t elements = static_cast<size_t>(tokens) * shape.output_rows;
    id<MTLBuffer> left = zero_buffer(device, elements * sizeof(float));
    id<MTLBuffer> right = zero_buffer(device, elements * sizeof(float));
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    encode_matmul(encoder, baseline, weights, input, left,
                  tokens, shape, row_bytes, true);
    encode_matmul(encoder, cooperative, weights, input, right,
                  tokens, shape, row_bytes, false);
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("correctness command failed: " +
            (command_buffer.error ? ns_string(command_buffer.error.localizedDescription)
                                  : std::string("unknown Metal error")));
    }
    if (std::memcmp(left.contents, right.contents, elements * sizeof(float)) == 0) return;
    const auto* a = static_cast<const float*>(left.contents);
    const auto* b = static_cast<const float*>(right.contents);
    for (size_t index = 0; index < elements; ++index) {
        if (std::memcmp(a + index, b + index, sizeof(float)) != 0) {
            throw std::runtime_error(
                std::string("cooperative Q4_K is not bit-exact for ") + shape.label +
                " pp" + std::to_string(tokens) + " at output index " +
                std::to_string(index) + ": " + std::to_string(a[index]) +
                " vs " + std::to_string(b[index]));
        }
    }
    throw std::runtime_error("bit-exact comparison failed without differing float");
}

double time_kernel(id<MTLCommandQueue> queue, int repetitions, int iterations,
                   void (^encode)(id<MTLComputeCommandEncoder>)) {
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(repetitions));
    for (int repetition = 0; repetition <= repetitions; ++repetition) {
        @autoreleasepool {
            id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            for (int iteration = 0; iteration < iterations; ++iteration) encode(encoder);
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.status != MTLCommandBufferStatusCompleted) {
                throw std::runtime_error("benchmark command failed");
            }
            if (repetition == 0) continue;
            samples.push_back(
                (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0 /
                static_cast<double>(iterations));
        }
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

}  // namespace

int main() {
    try {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("no default Metal device is available");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) throw std::runtime_error("Metal command queue creation failed");

        const std::string source =
            read_text("src/backend/metal/kernels/tensor.metal") + "\n" +
            read_text("apps/benchmark/metal/tensor_cooperative_q4k.metal");
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:
            [NSString stringWithUTF8String:source.c_str()] options:nil error:&error];
        if (!library) {
            throw std::runtime_error("Metal cooperative Q4_K shader failed: " +
                (error ? ns_string(error.localizedDescription) : "unknown"));
        }

        id<MTLComputePipelineState> baseline =
            make_pipeline(device, library, "celeg_matmul_tensor_q4k");
        id<MTLComputePipelineState> cooperative =
            make_pipeline(device, library, "celeg_matmul_tensor_q4k_cooperative");

        std::cout << "Metal Q4_K cooperative-input A/B on " << ns_string(device.name) << '\n';
        std::cout << "geometry=64x128xK64 strict_precision bit_exact=required\n";
        std::cout << "candidate_threadgroup_weight_bytes=0 baseline=8192\n\n";
        std::cout << std::left << std::setw(24) << "shape"
                  << std::right << std::setw(7) << "pp"
                  << std::setw(13) << "base ms"
                  << std::setw(13) << "coop ms"
                  << std::setw(11) << "speedup" << '\n';

        for (const Shape& shape : kShapes) {
            uint32_t row_bytes = 0;
            id<MTLBuffer> weights = make_weights(device, shape, row_bytes);
            for (uint32_t tokens : kTokenCounts) {
                id<MTLBuffer> input = make_input(
                    device, static_cast<size_t>(tokens) * shape.cols,
                    0xc004u + tokens + shape.cols);
                require_bit_exact(device, queue, baseline, cooperative, weights, input,
                                  tokens, shape, row_bytes);
                const size_t output_bytes =
                    static_cast<size_t>(tokens) * shape.output_rows * sizeof(float);
                id<MTLBuffer> output = zero_buffer(device, output_bytes);
                const int iterations = shape.output_rows >= 4096 ? 3 : 8;
                const double baseline_ms = time_kernel(queue, 5, iterations,
                    ^(id<MTLComputeCommandEncoder> encoder) {
                        encode_matmul(encoder, baseline, weights, input, output,
                                      tokens, shape, row_bytes, true);
                    });
                const double cooperative_ms = time_kernel(queue, 5, iterations,
                    ^(id<MTLComputeCommandEncoder> encoder) {
                        encode_matmul(encoder, cooperative, weights, input, output,
                                      tokens, shape, row_bytes, false);
                    });
                std::cout << std::left << std::setw(24) << shape.label
                          << std::right << std::setw(7) << tokens
                          << std::fixed << std::setprecision(3)
                          << std::setw(13) << baseline_ms
                          << std::setw(13) << cooperative_ms
                          << std::setw(10) << (baseline_ms / cooperative_ms) << "x\n";
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
