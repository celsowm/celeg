/// @file
/// A/B benchmark for the production one-exp causal batch attention versus a
/// materialized float simdgroup-matrix implementation. The candidate is a
/// fast-mode experiment: numerical error is reported, not rejected.

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
constexpr NSUInteger kBaselineSimdgroups = 8;
constexpr NSUInteger kBaselineThreads = 32 * kBaselineSimdgroups;
constexpr NSUInteger kBaselineSharedFloats =
    2 * kBaselineSimdgroups + kBaselineSimdgroups * kHeadDim;
constexpr NSUInteger kMatrixThreads = 128;
constexpr float kScale = 1.0f / 8.0f;

struct ErrorStats {
    double mean_abs = 0.0;
    double rmse = 0.0;
    float max_abs = 0.0f;
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
    return result;
}

void encode_baseline(id<MTLComputeCommandEncoder> encoder,
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
    [encoder setThreadgroupMemoryLength:kBaselineSharedFloats * sizeof(float) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(kQueryHeads, rows, 1)
            threadsPerThreadgroup:MTLSizeMake(kBaselineThreads, 1, 1)];
}

void encode_candidate(id<MTLComputeCommandEncoder> encoder,
                      id<MTLComputePipelineState> scores_state,
                      id<MTLComputePipelineState> softmax_state,
                      id<MTLComputePipelineState> values_state,
                      id<MTLBuffer> query,
                      id<MTLBuffer> key_cache,
                      id<MTLBuffer> value_cache,
                      id<MTLBuffer> scores,
                      id<MTLBuffer> output,
                      uint32_t rows) {
    [encoder setComputePipelineState:scores_state];
    [encoder setBuffer:query offset:0 atIndex:0];
    [encoder setBuffer:key_cache offset:0 atIndex:1];
    [encoder setBuffer:scores offset:0 atIndex:2];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
    [encoder setBytes:&kQueryHeads length:sizeof(kQueryHeads) atIndex:4];
    [encoder setBytes:&kKeyHeads length:sizeof(kKeyHeads) atIndex:5];
    [encoder setBytes:&kHeadDim length:sizeof(kHeadDim) atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake(
        kQueryHeads, (rows + 31u) / 32u, (rows + 7u) / 8u)
            threadsPerThreadgroup:MTLSizeMake(kMatrixThreads, 1, 1)];

    [encoder setComputePipelineState:softmax_state];
    [encoder setBuffer:scores offset:0 atIndex:0];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:1];
    [encoder setBytes:&kQueryHeads length:sizeof(kQueryHeads) atIndex:2];
    [encoder setBytes:&kScale length:sizeof(kScale) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(kQueryHeads, (rows + 3u) / 4u, 1)
            threadsPerThreadgroup:MTLSizeMake(kMatrixThreads, 1, 1)];

    [encoder setComputePipelineState:values_state];
    [encoder setBuffer:scores offset:0 atIndex:0];
    [encoder setBuffer:value_cache offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
    [encoder setBytes:&kQueryHeads length:sizeof(kQueryHeads) atIndex:4];
    [encoder setBytes:&kKeyHeads length:sizeof(kKeyHeads) atIndex:5];
    [encoder setBytes:&kHeadDim length:sizeof(kHeadDim) atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake(
        kQueryHeads, (rows + 31u) / 32u, (kHeadDim + 7u) / 8u)
            threadsPerThreadgroup:MTLSizeMake(kMatrixThreads, 1, 1)];
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

ErrorStats compare_outputs(id<MTLBuffer> baseline,
                           id<MTLBuffer> candidate,
                           size_t elements) {
    const auto* left = static_cast<const float*>(baseline.contents);
    const auto* right = static_cast<const float*>(candidate.contents);
    ErrorStats stats;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    for (size_t index = 0; index < elements; ++index) {
        const float error = std::abs(left[index] - right[index]);
        stats.max_abs = std::max(stats.max_abs, error);
        sum_abs += static_cast<double>(error);
        sum_sq += static_cast<double>(error) * static_cast<double>(error);
    }
    stats.mean_abs = sum_abs / static_cast<double>(elements);
    stats.rmse = std::sqrt(sum_sq / static_cast<double>(elements));
    return stats;
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
            read_text("src/backend/metal/kernels/inference/attention_one_exp.metal") + "\n" +
            read_text("apps/benchmark/metal/attention_materialized_simdgroup.metal");
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:
            [NSString stringWithUTF8String:source.c_str()] options:nil error:&error];
        if (!library) {
            throw std::runtime_error("Metal materialized attention shader failed: " +
                (error ? ns_string(error.localizedDescription) : "unknown"));
        }

        id<MTLComputePipelineState> baseline =
            make_pipeline(device, library, "celeg_attention_batch");
        id<MTLComputePipelineState> scores_state =
            make_pipeline(device, library, "celeg_attention_materialized_scores");
        id<MTLComputePipelineState> softmax_state =
            make_pipeline(device, library, "celeg_attention_materialized_softmax");
        id<MTLComputePipelineState> values_state =
            make_pipeline(device, library, "celeg_attention_materialized_values");

        std::cout << "Metal LFM2.5 materialized simdgroup attention A/B on "
                  << ns_string(device.name) << '\n';
        std::cout << "heads=16 kv_heads=8 head_dim=64 candidate=float8x8 materialized fast-mode\n\n";
        std::cout << std::left << std::setw(8) << "rows"
                  << std::right << std::setw(14) << "one-exp ms"
                  << std::setw(16) << "matrix ms"
                  << std::setw(11) << "speedup"
                  << std::setw(14) << "mean abs"
                  << std::setw(14) << "max abs"
                  << std::setw(14) << "rmse" << '\n';

        for (uint32_t rows : kRows) {
            const size_t query_elements =
                static_cast<size_t>(rows) * kQueryHeads * kHeadDim;
            const size_t kv_elements =
                static_cast<size_t>(rows) * kKeyHeads * kHeadDim;
            const size_t score_elements =
                static_cast<size_t>(kQueryHeads) * rows * rows;

            id<MTLBuffer> query = random_buffer(
                device, query_elements, 0x4100u + rows, -0.4f, 0.4f);
            id<MTLBuffer> key_cache = random_buffer(
                device, kv_elements, 0x4200u + rows, -0.4f, 0.4f);
            id<MTLBuffer> value_cache = random_buffer(
                device, kv_elements, 0x4300u + rows, -1.0f, 1.0f);
            id<MTLBuffer> scores = zero_buffer(
                device, score_elements * sizeof(float));
            id<MTLBuffer> baseline_output = zero_buffer(
                device, query_elements * sizeof(float));
            id<MTLBuffer> candidate_output = zero_buffer(
                device, query_elements * sizeof(float));

            id<MTLCommandBuffer> correctness = [queue commandBuffer];
            id<MTLComputeCommandEncoder> correctness_encoder =
                [correctness computeCommandEncoder];
            encode_baseline(correctness_encoder, baseline, query, key_cache,
                            value_cache, baseline_output, rows);
            encode_candidate(correctness_encoder, scores_state, softmax_state,
                             values_state, query, key_cache, value_cache, scores,
                             candidate_output, rows);
            [correctness_encoder endEncoding];
            wait_completed(correctness, "correctness");

            const ErrorStats stats = compare_outputs(
                baseline_output, candidate_output, query_elements);

            const int iterations = rows >= 512 ? 1 : 2;
            const double baseline_ms = time_kernel(queue, 5, iterations,
                ^(id<MTLComputeCommandEncoder> encoder) {
                    encode_baseline(encoder, baseline, query, key_cache,
                                    value_cache, baseline_output, rows);
                });
            const double candidate_ms = time_kernel(queue, 5, iterations,
                ^(id<MTLComputeCommandEncoder> encoder) {
                    encode_candidate(encoder, scores_state, softmax_state,
                                     values_state, query, key_cache, value_cache,
                                     scores, candidate_output, rows);
                });

            std::cout << std::left << std::setw(8) << rows
                      << std::right << std::fixed << std::setprecision(3)
                      << std::setw(14) << baseline_ms
                      << std::setw(16) << candidate_ms
                      << std::setw(10) << (baseline_ms / candidate_ms) << "x"
                      << std::scientific << std::setprecision(3)
                      << std::setw(14) << stats.mean_abs
                      << std::setw(14) << stats.max_abs
                      << std::setw(14) << stats.rmse << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
