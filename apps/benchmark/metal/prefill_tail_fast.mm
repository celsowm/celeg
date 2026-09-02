#include "metal_inference_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kRows = 512;
constexpr uint32_t kHidden = 1024;
constexpr uint32_t kIntermediate = 6656;
constexpr uint32_t kResidualDispatches = 32;
constexpr uint32_t kSwigluDispatches = 16;
constexpr float kMultiplier = 1.0f;
constexpr float kEpsilon = 1.0e-5f;

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

id<MTLLibrary> compile(id<MTLDevice> device, NSString* source, const char* label) {
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        throw std::runtime_error(std::string(label) + " compilation failed: " +
            (error ? ns_string(error.localizedDescription) : "unknown error"));
    }
    return library;
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                     const char* name) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!function) throw std::runtime_error(std::string("missing kernel: ") + name);
    id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function
                                                                                error:&error];
    if (!state) {
        throw std::runtime_error(std::string("pipeline failed: ") + name + ": " +
            (error ? ns_string(error.localizedDescription) : "unknown error"));
    }
    return state;
}

id<MTLBuffer> floats(id<MTLDevice> device, const std::vector<float>& values) {
    return [device newBufferWithBytes:values.data()
                               length:values.size() * sizeof(float)
                              options:MTLResourceStorageModeShared];
}

id<MTLBuffer> zeros(id<MTLDevice> device, size_t count) {
    return [device newBufferWithLength:count * sizeof(float)
                               options:MTLResourceStorageModeShared];
}

std::vector<float> random_values(size_t count, uint32_t seed, float low, float high) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(low, high);
    std::vector<float> values(count);
    for (float& value : values) value = distribution(generator);
    return values;
}

double max_abs(const float* left, const float* right, size_t count) {
    double result = 0.0;
    for (size_t index = 0; index < count; ++index) {
        result = std::max(result, std::abs(static_cast<double>(left[index]) - right[index]));
    }
    return result;
}

double time(id<MTLCommandQueue> queue, int repetitions, int iterations,
            void (^encode)(id<MTLComputeCommandEncoder>)) {
    std::vector<double> samples;
    for (int repetition = 0; repetition <= repetitions; ++repetition) {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        for (int iteration = 0; iteration < iterations; ++iteration) encode(encoder);
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("Metal tail benchmark command failed");
        }
        if (repetition == 0) continue;
        samples.push_back((command.GPUEndTime - command.GPUStartTime) * 1000.0 /
                          static_cast<double>(iterations));
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void run_once(id<MTLCommandQueue> queue,
              void (^encode)(id<MTLComputeCommandEncoder>)) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    encode(encoder);
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal tail validation command failed");
    }
}

void encode_rmsnorm(id<MTLComputeCommandEncoder> encoder,
                    id<MTLComputePipelineState> state,
                    NSUInteger threads,
                    id<MTLBuffer> input,
                    id<MTLBuffer> residual,
                    id<MTLBuffer> weight,
                    id<MTLBuffer> output,
                    id<MTLBuffer> normed) {
    [encoder setComputePipelineState:state];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:residual offset:0 atIndex:1];
    [encoder setBuffer:weight offset:0 atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBuffer:normed offset:0 atIndex:4];
    [encoder setBytes:&kRows length:sizeof(kRows) atIndex:5];
    [encoder setBytes:&kHidden length:sizeof(kHidden) atIndex:6];
    [encoder setBytes:&kMultiplier length:sizeof(kMultiplier) atIndex:7];
    [encoder setBytes:&kEpsilon length:sizeof(kEpsilon) atIndex:8];
    [encoder setThreadgroupMemoryLength:kHidden * sizeof(float) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(kRows, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

void encode_swiglu(id<MTLComputeCommandEncoder> encoder,
                   id<MTLComputePipelineState> state,
                   id<MTLBuffer> gate_up,
                   id<MTLBuffer> output) {
    [encoder setComputePipelineState:state];
    [encoder setBuffer:gate_up offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBytes:&kRows length:sizeof(kRows) atIndex:2];
    [encoder setBytes:&kIntermediate length:sizeof(kIntermediate) atIndex:3];
    const NSUInteger threads_x = std::min<NSUInteger>(
        kIntermediate, state.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(kIntermediate, kRows, 1)
       threadsPerThreadgroup:MTLSizeMake(threads_x, 1, 1)];
}

}

int main() {
    try {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("no Metal device");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) throw std::runtime_error("no Metal command queue");

        id<MTLLibrary> baseline_library = compile(
            device,
            [NSString stringWithUTF8String:celeg::metal_detail::kInferenceShader],
            "runtime shader");
        const std::string candidate_source = read_text(
            "apps/benchmark/metal/prefill_tail_fast.metal");
        id<MTLLibrary> candidate_library = compile(
            device, [NSString stringWithUTF8String:candidate_source.c_str()],
            "tail candidate shader");

        id<MTLComputePipelineState> rms_baseline = pipeline(
            device, baseline_library, "celeg_residual_rmsnorm_batch_cached");
        id<MTLComputePipelineState> rms_candidate = pipeline(
            device, candidate_library, "celeg_residual_rmsnorm_batch_cached_128_fast");
        id<MTLComputePipelineState> swiglu_baseline = pipeline(
            device, baseline_library, "celeg_swiglu_batch_2d");
        id<MTLComputePipelineState> swiglu_candidate = pipeline(
            device, candidate_library, "celeg_swiglu_batch_2d_fast");

        const size_t hidden_count = static_cast<size_t>(kRows) * kHidden;
        const size_t intermediate_count = static_cast<size_t>(kRows) * kIntermediate;
        id<MTLBuffer> input = floats(device, random_values(hidden_count, 11, -1.0f, 1.0f));
        id<MTLBuffer> residual = floats(device, random_values(hidden_count, 12, -1.0f, 1.0f));
        id<MTLBuffer> weight = floats(device, random_values(kHidden, 13, 0.8f, 1.2f));
        id<MTLBuffer> rms_base_output = zeros(device, hidden_count);
        id<MTLBuffer> rms_base_normed = zeros(device, hidden_count);
        id<MTLBuffer> rms_fast_output = zeros(device, hidden_count);
        id<MTLBuffer> rms_fast_normed = zeros(device, hidden_count);

        run_once(queue, ^(id<MTLComputeCommandEncoder> encoder) {
            encode_rmsnorm(encoder, rms_baseline, 256, input, residual, weight,
                           rms_base_output, rms_base_normed);
            encode_rmsnorm(encoder, rms_candidate, 128, input, residual, weight,
                           rms_fast_output, rms_fast_normed);
        });
        const double rms_output_error = max_abs(
            static_cast<const float*>(rms_base_output.contents),
            static_cast<const float*>(rms_fast_output.contents), hidden_count);
        const double rms_normed_error = max_abs(
            static_cast<const float*>(rms_base_normed.contents),
            static_cast<const float*>(rms_fast_normed.contents), hidden_count);

        const double rms_baseline_ms = time(queue, 5, 4,
            ^(id<MTLComputeCommandEncoder> encoder) {
                encode_rmsnorm(encoder, rms_baseline, 256, input, residual, weight,
                               rms_base_output, rms_base_normed);
            });
        const double rms_candidate_ms = time(queue, 5, 4,
            ^(id<MTLComputeCommandEncoder> encoder) {
                encode_rmsnorm(encoder, rms_candidate, 128, input, residual, weight,
                               rms_fast_output, rms_fast_normed);
            });

        id<MTLBuffer> gate_up = floats(
            device, random_values(intermediate_count * 2, 21, -4.0f, 4.0f));
        id<MTLBuffer> swiglu_base_output = zeros(device, intermediate_count);
        id<MTLBuffer> swiglu_fast_output = zeros(device, intermediate_count);
        run_once(queue, ^(id<MTLComputeCommandEncoder> encoder) {
            encode_swiglu(encoder, swiglu_baseline, gate_up, swiglu_base_output);
            encode_swiglu(encoder, swiglu_candidate, gate_up, swiglu_fast_output);
        });
        const double swiglu_error = max_abs(
            static_cast<const float*>(swiglu_base_output.contents),
            static_cast<const float*>(swiglu_fast_output.contents), intermediate_count);
        const double swiglu_baseline_ms = time(queue, 5, 4,
            ^(id<MTLComputeCommandEncoder> encoder) {
                encode_swiglu(encoder, swiglu_baseline, gate_up, swiglu_base_output);
            });
        const double swiglu_candidate_ms = time(queue, 5, 4,
            ^(id<MTLComputeCommandEncoder> encoder) {
                encode_swiglu(encoder, swiglu_candidate, gate_up, swiglu_fast_output);
            });

        const double rms_base_aggregate = rms_baseline_ms * kResidualDispatches;
        const double rms_fast_aggregate = rms_candidate_ms * kResidualDispatches;
        const double swiglu_base_aggregate = swiglu_baseline_ms * kSwigluDispatches;
        const double swiglu_fast_aggregate = swiglu_candidate_ms * kSwigluDispatches;

        std::cout << "Metal LFM2.5 pp512 tail A/B on " << ns_string(device.name) << "\n\n"
                  << std::fixed << std::setprecision(6)
                  << "residual_rmsnorm baseline_ms=" << rms_baseline_ms
                  << " candidate128_ms=" << rms_candidate_ms
                  << " speedup=" << (rms_baseline_ms / rms_candidate_ms) << "x"
                  << " output_max_abs=" << rms_output_error
                  << " normed_max_abs=" << rms_normed_error << "\n"
                  << "residual_rmsnorm aggregate32 baseline_ms=" << rms_base_aggregate
                  << " candidate_ms=" << rms_fast_aggregate
                  << " savings_ms=" << (rms_base_aggregate - rms_fast_aggregate) << "\n\n"
                  << "swiglu baseline_ms=" << swiglu_baseline_ms
                  << " fast_exp_ms=" << swiglu_candidate_ms
                  << " speedup=" << (swiglu_baseline_ms / swiglu_candidate_ms) << "x"
                  << " max_abs=" << swiglu_error << "\n"
                  << "swiglu aggregate16 baseline_ms=" << swiglu_base_aggregate
                  << " candidate_ms=" << swiglu_fast_aggregate
                  << " savings_ms=" << (swiglu_base_aggregate - swiglu_fast_aggregate) << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
