/// @file
/// Benchmark-only A/B for dense F16/BF16 Metal TensorOps prompt geometry.
/// Production uses 64 output rows x 128 tokens x K64. Candidate keeps the
/// accumulation/staging order and doubles only the token tile to 256.

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
constexpr NSUInteger kBaselineTileTokens = 128;
constexpr NSUInteger kCandidateTileTokens = 256;
constexpr NSUInteger kTileK = 64;
constexpr NSUInteger kThreads = 128;
constexpr NSUInteger kWeightTileBytes = kTileRows * kTileK * sizeof(uint16_t);

struct Shape {
    const char* label;
    uint32_t output_rows;
    uint32_t cols;
};

struct Storage {
    const char* label;
    const char* baseline_kernel;
    const char* candidate_kernel;
    const uint16_t* patterns;
    size_t pattern_count;
};

constexpr Shape kShapes[] = {
    {"proj_1024x1024", 1024, 1024},
    {"ffn_up_6656x1024", 6656, 1024},
    {"ffn_down_1024x6656", 1024, 6656},
};
constexpr uint32_t kTokenCounts[] = {128, 256, 512};
constexpr uint16_t kF16Patterns[] = {
    0x3c00, 0xbc00, 0x3800, 0xb800, 0x3400, 0xb400, 0x4000, 0xc000,
};
constexpr uint16_t kBF16Patterns[] = {
    0x3f80, 0xbf80, 0x3f00, 0xbf00, 0x3e80, 0xbe80, 0x4000, 0xc000,
};
constexpr Storage kStorages[] = {
    {"F16", "celeg_matmul_tensor_f16_fast", "celeg_matmul_tensor_f16_fast_64x256",
     kF16Patterns, std::size(kF16Patterns)},
    {"BF16", "celeg_matmul_tensor_bf16_fast", "celeg_matmul_tensor_bf16_fast_64x256",
     kBF16Patterns, std::size(kBF16Patterns)},
};

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
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes
                                               options:MTLResourceStorageModeShared];
    if (!buffer) throw std::runtime_error("Metal buffer allocation failed");
    std::memset(buffer.contents, 0, bytes);
    return buffer;
}

id<MTLBuffer> input_buffer(id<MTLDevice> device, size_t elements) {
    id<MTLBuffer> buffer = zero_buffer(device, elements * sizeof(float));
    auto* values = static_cast<float*>(buffer.contents);
    std::mt19937 generator(0xd3256u);
    std::uniform_real_distribution<float> distribution(-0.125f, 0.125f);
    for (size_t index = 0; index < elements; ++index) values[index] = distribution(generator);
    return buffer;
}

id<MTLBuffer> weight_buffer(id<MTLDevice> device, size_t elements, const Storage& storage) {
    id<MTLBuffer> buffer = zero_buffer(device, elements * sizeof(uint16_t));
    auto* values = static_cast<uint16_t*>(buffer.contents);
    for (size_t index = 0; index < elements; ++index) {
        values[index] = storage.patterns[(index * 13u + index / 17u) % storage.pattern_count];
    }
    return buffer;
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                     const char* name) {
    id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!function) throw std::runtime_error(std::string("missing Metal kernel: ") + name);
    NSError* error = nil;
    id<MTLComputePipelineState> state =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!state) {
        throw std::runtime_error(std::string("pipeline failed for ") + name + ": " +
            (error ? ns_string(error.localizedDescription) : "unknown"));
    }
    if (state.maxTotalThreadsPerThreadgroup < kThreads) {
        throw std::runtime_error(std::string(name) + " cannot launch 128 threads");
    }
    return state;
}

void encode_matmul(id<MTLComputeCommandEncoder> encoder,
                   id<MTLComputePipelineState> state,
                   id<MTLBuffer> weights, id<MTLBuffer> input, id<MTLBuffer> output,
                   uint32_t tokens, const Shape& shape, NSUInteger tile_tokens) {
    const uint32_t output_stride = shape.output_rows;
    [encoder setComputePipelineState:state];
    [encoder setBuffer:weights offset:0 atIndex:0];
    [encoder setBuffer:input offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&tokens length:sizeof(tokens) atIndex:3];
    [encoder setBytes:&shape.cols length:sizeof(shape.cols) atIndex:4];
    [encoder setBytes:&shape.output_rows length:sizeof(shape.output_rows) atIndex:5];
    [encoder setBytes:&output_stride length:sizeof(output_stride) atIndex:6];
    [encoder setThreadgroupMemoryLength:kWeightTileBytes atIndex:0];
    const NSUInteger row_groups =
        (static_cast<NSUInteger>(shape.output_rows) + kTileRows - 1u) / kTileRows;
    const NSUInteger token_groups =
        (static_cast<NSUInteger>(tokens) + tile_tokens - 1u) / tile_tokens;
    [encoder dispatchThreadgroups:MTLSizeMake(row_groups, token_groups, 1)
            threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
}

void check_identical(id<MTLCommandQueue> queue,
                     id<MTLComputePipelineState> baseline,
                     id<MTLComputePipelineState> candidate,
                     id<MTLBuffer> weights, id<MTLBuffer> input,
                     id<MTLBuffer> baseline_output, id<MTLBuffer> candidate_output,
                     uint32_t tokens, const Shape& shape, const Storage& storage) {
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    encode_matmul(encoder, baseline, weights, input, baseline_output,
                  tokens, shape, kBaselineTileTokens);
    encode_matmul(encoder, candidate, weights, input, candidate_output,
                  tokens, shape, kCandidateTileTokens);
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        const std::string message = command_buffer.error
            ? ns_string(command_buffer.error.localizedDescription) : "unknown";
        throw std::runtime_error("correctness command failed: " + message);
    }
    const size_t elements = static_cast<size_t>(tokens) * shape.output_rows;
    const size_t bytes = elements * sizeof(float);
    if (std::memcmp(baseline_output.contents, candidate_output.contents, bytes) == 0) return;
    const auto* left = static_cast<const float*>(baseline_output.contents);
    const auto* right = static_cast<const float*>(candidate_output.contents);
    for (size_t index = 0; index < elements; ++index) {
        if (std::memcmp(left + index, right + index, sizeof(float)) != 0) {
            throw std::runtime_error(
                std::string(storage.label) + " 64x256 is not bit-exact for " + shape.label +
                " pp" + std::to_string(tokens) + " at output index " +
                std::to_string(index) + ": " + std::to_string(left[index]) +
                " vs " + std::to_string(right[index]));
        }
    }
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
                const std::string message = command_buffer.error
                    ? ns_string(command_buffer.error.localizedDescription) : "unknown";
                throw std::runtime_error("benchmark command failed: " + message);
            }
            if (repetition == 0) continue;
            const double elapsed =
                (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
            samples.push_back(elapsed / static_cast<double>(iterations));
        }
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

uint64_t staging_count(const Shape& shape, uint32_t tokens, NSUInteger tile_tokens) {
    const uint64_t row_tiles = (shape.output_rows + kTileRows - 1u) / kTileRows;
    const uint64_t token_tiles = (tokens + tile_tokens - 1u) / tile_tokens;
    const uint64_t k_tiles = (shape.cols + kTileK - 1u) / kTileK;
    return row_tiles * token_tiles * k_tiles;
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
            read_text("src/backend/metal/kernels/tensor_dense_relaxed.metal") + "\n" +
            read_text("apps/benchmark/metal/tensor_dense_64x256.metal");
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:
            [NSString stringWithUTF8String:source.c_str()] options:nil error:&error];
        if (!library) {
            throw std::runtime_error("dense TensorOps benchmark shader failed: " +
                (error ? ns_string(error.localizedDescription) : "unknown"));
        }

        std::cout << "Metal dense TensorOps tile A/B on " << ns_string(device.name) << "\n";
        std::cout << "baseline=64x128xK64 candidate=64x256xK64 bit_exact=required\n\n";
        std::cout << std::left << std::setw(7) << "type"
                  << std::setw(24) << "shape"
                  << std::right << std::setw(7) << "pp"
                  << std::setw(12) << "base ms"
                  << std::setw(12) << "cand ms"
                  << std::setw(11) << "speedup"
                  << std::setw(13) << "base stages"
                  << std::setw(13) << "cand stages" << '\n';

        for (const Storage& storage : kStorages) {
            id<MTLComputePipelineState> baseline =
                pipeline(device, library, storage.baseline_kernel);
            id<MTLComputePipelineState> candidate =
                pipeline(device, library, storage.candidate_kernel);
            for (const Shape& shape : kShapes) {
                id<MTLBuffer> weights = weight_buffer(
                    device, static_cast<size_t>(shape.output_rows) * shape.cols, storage);
                for (uint32_t tokens : kTokenCounts) {
                    id<MTLBuffer> input = input_buffer(
                        device, static_cast<size_t>(tokens) * shape.cols);
                    const size_t output_elements =
                        static_cast<size_t>(tokens) * shape.output_rows;
                    id<MTLBuffer> baseline_output = zero_buffer(
                        device, output_elements * sizeof(float));
                    id<MTLBuffer> candidate_output = zero_buffer(
                        device, output_elements * sizeof(float));

                    check_identical(queue, baseline, candidate, weights, input,
                                    baseline_output, candidate_output,
                                    tokens, shape, storage);

                    const int iterations = tokens == 512 ? 4 : 8;
                    const double baseline_ms = time_kernel(queue, 5, iterations,
                        ^(id<MTLComputeCommandEncoder> encoder) {
                            encode_matmul(encoder, baseline, weights, input,
                                          baseline_output, tokens, shape,
                                          kBaselineTileTokens);
                        });
                    const double candidate_ms = time_kernel(queue, 5, iterations,
                        ^(id<MTLComputeCommandEncoder> encoder) {
                            encode_matmul(encoder, candidate, weights, input,
                                          candidate_output, tokens, shape,
                                          kCandidateTileTokens);
                        });
                    const double speedup = baseline_ms / candidate_ms;
                    std::cout << std::left << std::setw(7) << storage.label
                              << std::setw(24) << shape.label
                              << std::right << std::setw(7) << tokens
                              << std::fixed << std::setprecision(3)
                              << std::setw(12) << baseline_ms
                              << std::setw(12) << candidate_ms
                              << std::setw(10) << speedup << "x"
                              << std::setw(13) << staging_count(shape, tokens, kBaselineTileTokens)
                              << std::setw(13) << staging_count(shape, tokens, kCandidateTileTokens)
                              << '\n';
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
