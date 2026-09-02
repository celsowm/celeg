/// @file
/// Benchmark-only A/B for the Metal tensor prefill token tile.
///
/// Baseline is the production 64 output rows x 128 tokens x K64 geometry.
/// Candidate changes only the token tile to 256. Quantization decoders are
/// shared with production by concatenating tensor.metal before the experiment
/// shader at runtime. Every shape must remain bit-exact before it is timed.

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
    uint32_t block_size;
    uint32_t type_size;
    const char* baseline_kernel;
    const char* candidate_kernel;
};

// LFM2.5-350M linear shapes. The tied LM head is Q6_K; the remaining Q4_K_M
// linear weights use Q4_K for this experiment.
constexpr Shape kShapes[] = {
    {"proj_1024x1024", 1024, 1024, 256, 144,
     "celeg_matmul_tensor_q4k", "celeg_matmul_tensor_q4k_64x256"},
    {"ffn_up_4608x1024", 4608, 1024, 256, 144,
     "celeg_matmul_tensor_q4k", "celeg_matmul_tensor_q4k_64x256"},
    {"ffn_down_1024x4608", 1024, 4608, 256, 144,
     "celeg_matmul_tensor_q4k", "celeg_matmul_tensor_q4k_64x256"},
    {"lm_head_65536x1024", 65536, 1024, 256, 210,
     "celeg_matmul_tensor_q6k", "celeg_matmul_tensor_q6k_64x256"},
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
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes
                                               options:MTLResourceStorageModeShared];
    if (!buffer) throw std::runtime_error("Metal buffer allocation failed");
    std::memset(buffer.contents, 0, bytes);
    return buffer;
}

id<MTLBuffer> input_buffer(id<MTLDevice> device, size_t elements) {
    id<MTLBuffer> buffer = zero_buffer(device, elements * sizeof(float));
    auto* values = static_cast<float*>(buffer.contents);
    std::mt19937 generator(0x64256u);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (size_t index = 0; index < elements; ++index) {
        values[index] = distribution(generator);
    }
    return buffer;
}

void fill_q4k_block(uint8_t* block, uint32_t seed) {
    block[0] = 0x00;
    block[1] = 0x3c;  // half(1.0)
    block[2] = 0x00;
    block[3] = 0x00;  // half(0.0) minimum scale
    for (uint32_t index = 4; index < 16; ++index) {
        block[index] = static_cast<uint8_t>(1u + ((seed + index) & 7u));
    }
    for (uint32_t index = 16; index < 144; ++index) {
        const uint8_t low = static_cast<uint8_t>((seed + index) & 0x0fu);
        const uint8_t high = static_cast<uint8_t>((seed + index * 3u + 5u) & 0x0fu);
        block[index] = static_cast<uint8_t>(low | (high << 4));
    }
}

void fill_q6k_block(uint8_t* block, uint32_t seed) {
    for (uint32_t index = 0; index < 128; ++index) {
        const uint8_t low = static_cast<uint8_t>((seed + index) & 0x0fu);
        const uint8_t high = static_cast<uint8_t>((seed + index * 5u + 3u) & 0x0fu);
        block[index] = static_cast<uint8_t>(low | (high << 4));
    }
    for (uint32_t index = 128; index < 192; ++index) {
        block[index] = static_cast<uint8_t>(0x55u ^ ((seed + index) & 0x0fu));
    }
    for (uint32_t index = 192; index < 208; ++index) {
        block[index] = static_cast<uint8_t>(1u + ((seed + index) & 7u));
    }
    block[208] = 0x00;
    block[209] = 0x2c;  // half(0.0625)
}

id<MTLBuffer> quantized_weight_buffer(id<MTLDevice> device, const Shape& shape,
                                      uint32_t row_bytes) {
    const size_t bytes = static_cast<size_t>(shape.output_rows) * row_bytes;
    id<MTLBuffer> buffer = zero_buffer(device, bytes);
    auto* data = static_cast<uint8_t*>(buffer.contents);
    const uint32_t blocks_per_row = shape.cols / shape.block_size;
    for (uint32_t row = 0; row < shape.output_rows; ++row) {
        for (uint32_t block_index = 0; block_index < blocks_per_row; ++block_index) {
            uint8_t* block = data + static_cast<size_t>(row) * row_bytes +
                static_cast<size_t>(block_index) * shape.type_size;
            const uint32_t seed = row * 17u + block_index * 29u;
            if (shape.type_size == 144) {
                fill_q4k_block(block, seed);
            } else if (shape.type_size == 210) {
                fill_q6k_block(block, seed);
            } else {
                throw std::runtime_error("unsupported benchmark quantized block");
            }
        }
    }
    return buffer;
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                     const char* name) {
    id<MTLFunction> function = [library newFunctionWithName:
        [NSString stringWithUTF8String:name]];
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
                   uint32_t tokens, const Shape& shape, uint32_t row_bytes,
                   NSUInteger tile_tokens) {
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
    [encoder setThreadgroupMemoryLength:kWeightTileBytes atIndex:0];
    const NSUInteger row_groups =
        (static_cast<NSUInteger>(shape.output_rows) + kTileRows - 1u) / kTileRows;
    const NSUInteger token_groups =
        (static_cast<NSUInteger>(tokens) + tile_tokens - 1u) / tile_tokens;
    [encoder dispatchThreadgroups:MTLSizeMake(row_groups, token_groups, 1)
            threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
}

void run_pair(id<MTLCommandQueue> queue,
              id<MTLComputePipelineState> baseline,
              id<MTLComputePipelineState> candidate,
              id<MTLBuffer> weights, id<MTLBuffer> input,
              id<MTLBuffer> baseline_output, id<MTLBuffer> candidate_output,
              uint32_t tokens, const Shape& shape, uint32_t row_bytes) {
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    encode_matmul(encoder, baseline, weights, input, baseline_output,
                  tokens, shape, row_bytes, kBaselineTileTokens);
    encode_matmul(encoder, candidate, weights, input, candidate_output,
                  tokens, shape, row_bytes, kCandidateTileTokens);
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        const std::string message = command_buffer.error
            ? ns_string(command_buffer.error.localizedDescription)
            : "unknown Metal command-buffer error";
        throw std::runtime_error("correctness command failed: " + message);
    }
}

void check_identical(id<MTLBuffer> baseline, id<MTLBuffer> candidate,
                     size_t elements, const Shape& shape, uint32_t tokens) {
    const size_t bytes = elements * sizeof(float);
    if (std::memcmp(baseline.contents, candidate.contents, bytes) == 0) return;
    const auto* left = static_cast<const float*>(baseline.contents);
    const auto* right = static_cast<const float*>(candidate.contents);
    for (size_t index = 0; index < elements; ++index) {
        if (std::memcmp(left + index, right + index, sizeof(float)) != 0) {
            throw std::runtime_error(
                std::string("64x256 is not bit-exact for ") + shape.label +
                " pp" + std::to_string(tokens) + " at output index " +
                std::to_string(index) + ": " + std::to_string(left[index]) +
                " vs " + std::to_string(right[index]));
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
                const std::string message = command_buffer.error
                    ? ns_string(command_buffer.error.localizedDescription)
                    : "unknown Metal command-buffer error";
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
            read_text("apps/benchmark/metal/tensor_tile_64x256.metal");
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:
            [NSString stringWithUTF8String:source.c_str()] options:nil error:&error];
        if (!library) {
            throw std::runtime_error("Metal tensor benchmark shader failed: " +
                (error ? ns_string(error.localizedDescription) : "unknown"));
        }

        std::cout << "Metal tensor tile A/B on " << ns_string(device.name) << "\n";
        std::cout << "baseline=64x128xK64 candidate=64x256xK64 bit_exact=required\n\n";
        std::cout << std::left << std::setw(24) << "shape"
                  << std::right << std::setw(7) << "pp"
                  << std::setw(13) << "base ms"
                  << std::setw(13) << "cand ms"
                  << std::setw(11) << "speedup"
                  << std::setw(15) << "base stages"
                  << std::setw(15) << "cand stages" << '\n';

        for (const Shape& shape : kShapes) {
            if (shape.cols % shape.block_size != 0) {
                throw std::runtime_error(std::string(shape.label) +
                                         " has a partial quantization block");
            }
            const uint32_t row_bytes =
                shape.cols / shape.block_size * shape.type_size;
            const size_t weight_bytes =
                static_cast<size_t>(shape.output_rows) * row_bytes;
            id<MTLBuffer> weights = quantized_weight_buffer(device, shape, row_bytes);
            id<MTLComputePipelineState> baseline =
                pipeline(device, library, shape.baseline_kernel);
            id<MTLComputePipelineState> candidate =
                pipeline(device, library, shape.candidate_kernel);

            for (uint32_t tokens : kTokenCounts) {
                id<MTLBuffer> input = input_buffer(
                    device, static_cast<size_t>(tokens) * shape.cols);
                const size_t output_elements =
                    static_cast<size_t>(tokens) * shape.output_rows;
                id<MTLBuffer> baseline_output = zero_buffer(
                    device, output_elements * sizeof(float));
                id<MTLBuffer> candidate_output = zero_buffer(
                    device, output_elements * sizeof(float));

                run_pair(queue, baseline, candidate, weights, input,
                         baseline_output, candidate_output,
                         tokens, shape, row_bytes);
                check_identical(baseline_output, candidate_output,
                                output_elements, shape, tokens);

                const int iterations =
                    weight_bytes > (16u << 20) || tokens == 512 ? 1 : 4;
                const double baseline_ms = time_kernel(
                    queue, 5, iterations, ^(id<MTLComputeCommandEncoder> encoder) {
                        encode_matmul(encoder, baseline, weights, input, baseline_output,
                                      tokens, shape, row_bytes, kBaselineTileTokens);
                    });
                const double candidate_ms = time_kernel(
                    queue, 5, iterations, ^(id<MTLComputeCommandEncoder> encoder) {
                        encode_matmul(encoder, candidate, weights, input, candidate_output,
                                      tokens, shape, row_bytes, kCandidateTileTokens);
                    });

                std::cout << std::left << std::setw(24) << shape.label
                          << std::right << std::setw(7) << tokens
                          << std::fixed << std::setprecision(3)
                          << std::setw(13) << baseline_ms
                          << std::setw(13) << candidate_ms
                          << std::setprecision(2)
                          << std::setw(10) << (baseline_ms / candidate_ms) << "x"
                          << std::setw(15)
                          << staging_count(shape, tokens, kBaselineTileTokens)
                          << std::setw(15)
                          << staging_count(shape, tokens, kCandidateTileTokens)
                          << '\n';
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
