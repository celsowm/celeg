/// @file
/// Benchmark-only A/B for caching Q4_K weights as their exact F16
/// dequantization for prefill. The one-time predecode is measured separately
/// and excluded from steady-state matmul timing. Native Q4_K and cached F16
/// outputs must be bit-identical for every tested shape and token count.

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
constexpr NSUInteger kWeightTileBytes = kTileRows * kTileK * sizeof(uint16_t);
constexpr uint32_t kQ4KBlockValues = 256;
constexpr uint32_t kQ4KBlockBytes = 144;

struct Shape {
    const char* label;
    uint32_t output_rows;
    uint32_t cols;
};

constexpr Shape kShapes[] = {
    {"proj_1024x1024", 1024, 1024},
    {"ffn_up_6656x1024", 6656, 1024},
    {"ffn_down_1024x6656", 1024, 6656},
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
    for (size_t index = 0; index < elements; ++index) values[index] = distribution(generator);
    return result;
}

void fill_q4k_block(uint8_t* block, uint32_t seed) {
    block[0] = 0x00;
    block[1] = 0x3c;  // half(1.0)
    block[2] = 0x00;
    block[3] = 0x38;  // half(0.5), exercises the Q4_K minimum/bias term
    for (uint32_t index = 4; index < 16; ++index) {
        block[index] = static_cast<uint8_t>(1u + ((seed + index * 5u) & 0x3fu));
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
    return result;
}

void encode_predecode(id<MTLComputeCommandEncoder> encoder,
                      id<MTLComputePipelineState> state,
                      id<MTLBuffer> compressed, id<MTLBuffer> dense,
                      const Shape& shape, uint32_t row_bytes) {
    [encoder setComputePipelineState:state];
    [encoder setBuffer:compressed offset:0 atIndex:0];
    [encoder setBuffer:dense offset:0 atIndex:1];
    [encoder setBytes:&shape.output_rows length:sizeof(shape.output_rows) atIndex:2];
    [encoder setBytes:&shape.cols length:sizeof(shape.cols) atIndex:3];
    [encoder setBytes:&row_bytes length:sizeof(row_bytes) atIndex:4];
    const NSUInteger count =
        static_cast<NSUInteger>(shape.output_rows) * (shape.cols / 32u);
    const NSUInteger threads = std::min<NSUInteger>(
        256u, state.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

void encode_matmul(id<MTLComputeCommandEncoder> encoder,
                   id<MTLComputePipelineState> state,
                   id<MTLBuffer> weights, id<MTLBuffer> input, id<MTLBuffer> output,
                   uint32_t tokens, const Shape& shape, uint32_t row_bytes,
                   bool quantized) {
    const uint32_t output_stride = shape.output_rows;
    [encoder setComputePipelineState:state];
    [encoder setBuffer:weights offset:0 atIndex:0];
    [encoder setBuffer:input offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&tokens length:sizeof(tokens) atIndex:3];
    [encoder setBytes:&shape.cols length:sizeof(shape.cols) atIndex:4];
    [encoder setBytes:&shape.output_rows length:sizeof(shape.output_rows) atIndex:5];
    [encoder setBytes:&output_stride length:sizeof(output_stride) atIndex:6];
    if (quantized) [encoder setBytes:&row_bytes length:sizeof(row_bytes) atIndex:7];
    [encoder setThreadgroupMemoryLength:kWeightTileBytes atIndex:0];
    const NSUInteger row_groups =
        (static_cast<NSUInteger>(shape.output_rows) + kTileRows - 1u) / kTileRows;
    const NSUInteger token_groups =
        (static_cast<NSUInteger>(tokens) + kTileTokens - 1u) / kTileTokens;
    [encoder dispatchThreadgroups:MTLSizeMake(row_groups, token_groups, 1)
            threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
}

double time_command(id<MTLCommandQueue> queue,
                    void (^encode)(id<MTLComputeCommandEncoder>)) {
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    encode(encoder);
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal command failed: " +
            (command_buffer.error ? ns_string(command_buffer.error.localizedDescription)
                                  : std::string("unknown Metal error")));
    }
    return (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
}

void require_bit_exact(id<MTLDevice> device, id<MTLCommandQueue> queue,
                       id<MTLComputePipelineState> q4k,
                       id<MTLComputePipelineState> f16,
                       id<MTLBuffer> compressed, id<MTLBuffer> dense,
                       id<MTLBuffer> input, uint32_t tokens,
                       const Shape& shape, uint32_t row_bytes) {
    const size_t elements = static_cast<size_t>(tokens) * shape.output_rows;
    id<MTLBuffer> left = zero_buffer(device, elements * sizeof(float));
    id<MTLBuffer> right = zero_buffer(device, elements * sizeof(float));
    time_command(queue, ^(id<MTLComputeCommandEncoder> encoder) {
        encode_matmul(encoder, q4k, compressed, input, left,
                      tokens, shape, row_bytes, true);
        encode_matmul(encoder, f16, dense, input, right,
                      tokens, shape, row_bytes, false);
    });
    if (std::memcmp(left.contents, right.contents, elements * sizeof(float)) == 0) return;
    const auto* a = static_cast<const float*>(left.contents);
    const auto* b = static_cast<const float*>(right.contents);
    for (size_t index = 0; index < elements; ++index) {
        if (std::memcmp(a + index, b + index, sizeof(float)) != 0) {
            throw std::runtime_error(
                std::string("predecoded F16 is not bit-exact for ") + shape.label +
                " pp" + std::to_string(tokens) + " at output index " +
                std::to_string(index) + ": " + std::to_string(a[index]) +
                " vs " + std::to_string(b[index]));
        }
    }
}

double time_kernel(id<MTLCommandQueue> queue, int repetitions, int iterations,
                   void (^encode)(id<MTLComputeCommandEncoder>)) {
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(repetitions));
    for (int repetition = 0; repetition <= repetitions; ++repetition) {
        const double ms = time_command(queue, ^(id<MTLComputeCommandEncoder> encoder) {
            for (int iteration = 0; iteration < iterations; ++iteration) encode(encoder);
        }) / static_cast<double>(iterations);
        if (repetition != 0) samples.push_back(ms);
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
            read_text("apps/benchmark/metal/tensor_q4k_predecoded.metal");
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:
            [NSString stringWithUTF8String:source.c_str()] options:nil error:&error];
        if (!library) {
            throw std::runtime_error("Metal predecoded Q4_K shader failed: " +
                (error ? ns_string(error.localizedDescription) : "unknown"));
        }

        id<MTLComputePipelineState> predecode =
            make_pipeline(device, library, "celeg_predecode_q4k_f16");
        id<MTLComputePipelineState> q4k =
            make_pipeline(device, library, "celeg_matmul_tensor_q4k");
        id<MTLComputePipelineState> f16 =
            make_pipeline(device, library, "celeg_matmul_tensor_f16");

        std::cout << "Metal Q4_K predecoded-F16 A/B on " << ns_string(device.name) << '\n';
        std::cout << "geometry=64x128xK64 strict_precision bit_exact=required\n";
        std::cout << "predecode_cost=one_time_and_excluded_from_matmul_timing\n\n";
        std::cout << std::left << std::setw(24) << "shape"
                  << std::right << std::setw(7) << "pp"
                  << std::setw(13) << "Q4_K ms"
                  << std::setw(13) << "F16 ms"
                  << std::setw(11) << "speedup"
                  << std::setw(13) << "cache MiB"
                  << std::setw(15) << "predec ms" << '\n';

        for (const Shape& shape : kShapes) {
            uint32_t row_bytes = 0;
            id<MTLBuffer> compressed = make_weights(device, shape, row_bytes);
            const size_t dense_bytes =
                static_cast<size_t>(shape.output_rows) * shape.cols * sizeof(uint16_t);
            id<MTLBuffer> dense = zero_buffer(device, dense_bytes);
            const double predecode_ms = time_command(
                queue, ^(id<MTLComputeCommandEncoder> encoder) {
                    encode_predecode(encoder, predecode, compressed, dense, shape, row_bytes);
                });

            for (uint32_t tokens : kTokenCounts) {
                id<MTLBuffer> input = make_input(
                    device, static_cast<size_t>(tokens) * shape.cols,
                    0xf16u + tokens + shape.cols);
                require_bit_exact(device, queue, q4k, f16, compressed, dense,
                                  input, tokens, shape, row_bytes);
                const size_t output_bytes =
                    static_cast<size_t>(tokens) * shape.output_rows * sizeof(float);
                id<MTLBuffer> output = zero_buffer(device, output_bytes);
                const int iterations = shape.output_rows >= 4096 ? 3 : 8;
                const double q4k_ms = time_kernel(queue, 5, iterations,
                    ^(id<MTLComputeCommandEncoder> encoder) {
                        encode_matmul(encoder, q4k, compressed, input, output,
                                      tokens, shape, row_bytes, true);
                    });
                const double f16_ms = time_kernel(queue, 5, iterations,
                    ^(id<MTLComputeCommandEncoder> encoder) {
                        encode_matmul(encoder, f16, dense, input, output,
                                      tokens, shape, row_bytes, false);
                    });
                std::cout << std::left << std::setw(24) << shape.label
                          << std::right << std::setw(7) << tokens
                          << std::fixed << std::setprecision(3)
                          << std::setw(13) << q4k_ms
                          << std::setw(13) << f16_ms
                          << std::setw(10) << (q4k_ms / f16_ms) << "x"
                          << std::setw(13)
                          << (static_cast<double>(dense_bytes) / (1024.0 * 1024.0))
                          << std::setw(15) << predecode_ms << '\n';
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
