/// @file
/// A/B benchmark for Celeg's production causal batch attention versus a
/// bit-exact one-exp online-softmax recurrence. The geometry matches
/// LFM2.5-350M: 16 query heads, 8 KV heads, head_dim=64.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
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

constexpr uint32_t kQueryHeads = 16;
constexpr uint32_t kKeyHeads = 8;
constexpr uint32_t kHeadDim = 64;
constexpr uint32_t kBasePosition = 0;
constexpr uint32_t kPageTokens = 16;
constexpr uint32_t kRows[] = {128, 256, 512};
constexpr NSUInteger kSimdgroups = 8;
constexpr NSUInteger kThreads = 32 * kSimdgroups;
constexpr NSUInteger kSharedFloats =
    2 * kSimdgroups + kSimdgroups * kHeadDim;
constexpr float kScale = 1.0f / 8.0f;

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

id<MTLBuffer> random_buffer(id<MTLDevice> device, size_t elements,
                            uint32_t seed, float low, float high) {
    id<MTLBuffer> result = zero_buffer(device, elements * sizeof(float));
    auto* values = static_cast<float*>(result.contents);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(low, high);
    for (size_t index = 0; index < elements; ++index) {
        values[index] = distribution(generator);
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
        throw std::runtime_error(std::string(name) + " cannot launch 256 threads");
    }
    return result;
}

void encode_attention(id<MTLComputeCommandEncoder> encoder,
                      id<MTLComputePipelineState> state,
                      id<MTLBuffer> query,
                      id<MTLBuffer> key_cache,
                      id<MTLBuffer> value_cache,
                      id<MTLBuffer> output,
                      uint32_t rows) {
    [encoder setComputePipelineState:state];
    [encoder setBuffer:query offset:0 atIndex:0];
    [encoder setBuffer:key_cache offset:0 atIndex:1];
    [encoder setBuffer:value_cache offset:0 atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
    [encoder setBytes:&kBasePosition length:sizeof(kBasePosition) atIndex:5];
    [encoder setBytes:&kQueryHeads length:sizeof(kQueryHeads) atIndex:6];
    [encoder setBytes:&kKeyHeads length:sizeof(kKeyHeads) atIndex:7];
    [encoder setBytes:&kHeadDim length:sizeof(kHeadDim) atIndex:8];
    [encoder setBytes:&kScale length:sizeof(kScale) atIndex:9];
    [encoder setBytes:&kPageTokens length:sizeof(kPageTokens) atIndex:10];
    [encoder setThreadgroupMemoryLength:kSharedFloats * sizeof(float) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(kQueryHeads, rows, 1)
            threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
}

void wait_completed(id<MTLCommandBuffer> command_buffer, const char* phase) {
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error(std::string(phase) + " command failed: " +
            (command_buffer.error ? ns_string(command_buffer.error.localizedDescription)
                                  : std::string("unknown Metal error")));
    }
}

void require_bit_exact(id<MTLDevice> device, id<MTLCommandQueue> queue,
                       id<MTLComputePipelineState> baseline,
                       id<MTLComputePipelineState> candidate,
                       id<MTLBuffer> query, id<MTLBuffer> key_cache,
                       id<MTLBuffer> value_cache, uint32_t rows) {
    const size_t output_elements =
        static_cast<size_t>(rows) * kQueryHeads * kHeadDim;
    id<MTLBuffer> left = zero_buffer(device, output_elements * sizeof(float));
    id<MTLBuffer> right = zero_buffer(device, output_elements * sizeof(float));

    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    encode_attention(encoder, baseline, query, key_cache, value_cache, left, rows);
    encode_attention(encoder, candidate, query, key_cache, value_cache, right, rows);
    [encoder endEncoding];
    wait_completed(command_buffer, "correctness");

    if (std::memcmp(left.contents, right.contents,
                    output_elements * sizeof(float)) == 0) {
        return;
    }
    const auto* a = static_cast<const float*>(left.contents);
    const auto* b = static_cast<const float*>(right.contents);
    for (size_t index = 0; index < output_elements; ++index) {
        if (std::memcmp(a + index, b + index, sizeof(float)) != 0) {
            throw std::runtime_error(
                "one-exp attention mismatch at index " + std::to_string(index) +
                ": " + std::to_string(a[index]) + " vs " +
                std::to_string(b[index]));
        }
    }
}

template <typename Encode>
double time_kernel(id<MTLCommandQueue> queue, int repetitions, int iterations,
                   Encode encode) {
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(repetitions));
    for (int repetition = 0; repetition <= repetitions; ++repetition) {
        @autoreleasepool {
            id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            for (int iteration = 0; iteration < iterations; ++iteration) {
                encode(encoder);
            }
            [encoder endEncoding];
            wait_completed(command_buffer, "benchmark");
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
            read_text("src/backend/metal/kernels/inference/common.metal") + "\n" +
            read_text("src/backend/metal/kernels/inference/batch.metal") + "\n" +
            read_text("apps/benchmark/metal/attention_one_exp.metal");
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:
            [NSString stringWithUTF8String:source.c_str()] options:nil error:&error];
        if (!library) {
            throw std::runtime_error("Metal attention benchmark shader failed: " +
                (error ? ns_string(error.localizedDescription) : "unknown"));
        }

        id<MTLComputePipelineState> baseline =
            make_pipeline(device, library, "celeg_attention_batch");
        id<MTLComputePipelineState> candidate =
            make_pipeline(device, library, "celeg_attention_batch_one_exp");

        std::cout << "Metal LFM2.5 causal attention one-exp A/B on "
                  << ns_string(device.name) << '\n';
        std::cout << "heads=16 kv_heads=8 head_dim=64 bit_exact=required\n\n";
        std::cout << std::left << std::setw(10) << "rows"
                  << std::right << std::setw(15) << "baseline ms"
                  << std::setw(15) << "one-exp ms"
                  << std::setw(12) << "speedup" << '\n';

        for (uint32_t rows : kRows) {
            const size_t query_elements =
                static_cast<size_t>(rows) * kQueryHeads * kHeadDim;
            const size_t kv_elements =
                static_cast<size_t>(rows) * kKeyHeads * kHeadDim;
            id<MTLBuffer> query = random_buffer(
                device, query_elements, 0x1000u + rows, -0.4f, 0.4f);
            id<MTLBuffer> key_cache = random_buffer(
                device, kv_elements, 0x2000u + rows, -0.4f, 0.4f);
            id<MTLBuffer> value_cache = random_buffer(
                device, kv_elements, 0x3000u + rows, -1.0f, 1.0f);

            require_bit_exact(device, queue, baseline, candidate,
                              query, key_cache, value_cache, rows);

            id<MTLBuffer> output = zero_buffer(
                device, query_elements * sizeof(float));
            const int iterations = rows >= 512 ? 2 : 3;
            const double baseline_ms = time_kernel(queue, 5, iterations,
                ^(id<MTLComputeCommandEncoder> encoder) {
                    encode_attention(encoder, baseline, query, key_cache,
                                     value_cache, output, rows);
                });
            const double candidate_ms = time_kernel(queue, 5, iterations,
                ^(id<MTLComputeCommandEncoder> encoder) {
                    encode_attention(encoder, candidate, query, key_cache,
                                     value_cache, output, rows);
                });

            std::cout << std::left << std::setw(10) << rows
                      << std::right << std::fixed << std::setprecision(3)
                      << std::setw(15) << baseline_ms
                      << std::setw(15) << candidate_ms
                      << std::setw(11) << (baseline_ms / candidate_ms) << "x\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
