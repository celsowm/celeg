/// @file
/// A/B benchmark for the production serial-per-channel shortconv batch kernel
/// versus a three-pass rows*width-parallel implementation. Output and final
/// ring state must both remain bit-exact before timing starts.

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

constexpr uint32_t kWidth = 1024;
constexpr uint32_t kCacheLength = 3;
constexpr uint32_t kInitialCursor = 2;
constexpr NSUInteger kThreads = 256;
constexpr uint32_t kRows[] = {128, 256, 512};

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

id<MTLBuffer> make_buffer(id<MTLDevice> device, size_t bytes) {
    id<MTLBuffer> result = [device newBufferWithLength:bytes
                                               options:MTLResourceStorageModeShared];
    if (!result) throw std::runtime_error("Metal buffer allocation failed");
    return result;
}

id<MTLBuffer> zero_buffer(id<MTLDevice> device, size_t bytes) {
    id<MTLBuffer> result = make_buffer(device, bytes);
    std::memset(result.contents, 0, bytes);
    return result;
}

id<MTLBuffer> random_buffer(id<MTLDevice> device, size_t elements,
                            uint32_t seed, float low, float high) {
    id<MTLBuffer> result = make_buffer(device, elements * sizeof(float));
    auto* values = static_cast<float*>(result.contents);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(low, high);
    for (size_t index = 0; index < elements; ++index) {
        values[index] = distribution(generator);
    }
    return result;
}

id<MTLBuffer> clone_buffer(id<MTLDevice> device, id<MTLBuffer> source) {
    id<MTLBuffer> result = make_buffer(device, source.length);
    std::memcpy(result.contents, source.contents, source.length);
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

void dispatch_threads(id<MTLComputeCommandEncoder> encoder,
                      id<MTLComputePipelineState> state,
                      NSUInteger count) {
    [encoder setComputePipelineState:state];
    const NSUInteger threads = std::min(kThreads, state.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

void encode_baseline(id<MTLComputeCommandEncoder> encoder,
                     id<MTLComputePipelineState> state,
                     id<MTLBuffer> projected, id<MTLBuffer> taps,
                     id<MTLBuffer> ring_state, id<MTLBuffer> output,
                     uint32_t rows) {
    [encoder setComputePipelineState:state];
    [encoder setBuffer:projected offset:0 atIndex:0];
    [encoder setBuffer:taps offset:0 atIndex:1];
    [encoder setBuffer:ring_state offset:0 atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
    [encoder setBytes:&kWidth length:sizeof(kWidth) atIndex:5];
    [encoder setBytes:&kCacheLength length:sizeof(kCacheLength) atIndex:6];
    [encoder setBytes:&kInitialCursor length:sizeof(kInitialCursor) atIndex:7];
    [encoder dispatchThreadgroups:MTLSizeMake((kWidth + 255u) / 256u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void encode_candidate(id<MTLComputeCommandEncoder> encoder,
                      id<MTLComputePipelineState> gate,
                      id<MTLComputePipelineState> convolve,
                      id<MTLComputePipelineState> publish,
                      id<MTLBuffer> projected, id<MTLBuffer> taps,
                      id<MTLBuffer> ring_state, id<MTLBuffer> gated,
                      id<MTLBuffer> output, uint32_t rows) {
    const uint32_t count = rows * kWidth;

    [encoder setBuffer:projected offset:0 atIndex:0];
    [encoder setBuffer:gated offset:0 atIndex:1];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
    [encoder setBytes:&kWidth length:sizeof(kWidth) atIndex:3];
    dispatch_threads(encoder, gate, count);

    [encoder setBuffer:projected offset:0 atIndex:0];
    [encoder setBuffer:taps offset:0 atIndex:1];
    [encoder setBuffer:ring_state offset:0 atIndex:2];
    [encoder setBuffer:gated offset:0 atIndex:3];
    [encoder setBuffer:output offset:0 atIndex:4];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:5];
    [encoder setBytes:&kWidth length:sizeof(kWidth) atIndex:6];
    [encoder setBytes:&kCacheLength length:sizeof(kCacheLength) atIndex:7];
    [encoder setBytes:&kInitialCursor length:sizeof(kInitialCursor) atIndex:8];
    dispatch_threads(encoder, convolve, count);

    const uint32_t tail_rows = std::min(rows, kCacheLength);
    const uint32_t tail_count = tail_rows * kWidth;
    [encoder setBuffer:gated offset:0 atIndex:0];
    [encoder setBuffer:ring_state offset:0 atIndex:1];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
    [encoder setBytes:&kWidth length:sizeof(kWidth) atIndex:3];
    [encoder setBytes:&kCacheLength length:sizeof(kCacheLength) atIndex:4];
    [encoder setBytes:&kInitialCursor length:sizeof(kInitialCursor) atIndex:5];
    dispatch_threads(encoder, publish, tail_count);
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
                       id<MTLComputePipelineState> gate,
                       id<MTLComputePipelineState> convolve,
                       id<MTLComputePipelineState> publish,
                       id<MTLBuffer> projected, id<MTLBuffer> taps,
                       id<MTLBuffer> initial_state, uint32_t rows) {
    const size_t output_elements = static_cast<size_t>(rows) * kWidth;
    id<MTLBuffer> baseline_state = clone_buffer(device, initial_state);
    id<MTLBuffer> candidate_state = clone_buffer(device, initial_state);
    id<MTLBuffer> baseline_output = zero_buffer(device, output_elements * sizeof(float));
    id<MTLBuffer> candidate_output = zero_buffer(device, output_elements * sizeof(float));
    id<MTLBuffer> gated = zero_buffer(device, output_elements * sizeof(float));

    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    encode_baseline(encoder, baseline, projected, taps, baseline_state,
                    baseline_output, rows);
    encode_candidate(encoder, gate, convolve, publish, projected, taps,
                     candidate_state, gated, candidate_output, rows);
    [encoder endEncoding];
    wait_completed(command_buffer, "correctness");

    if (std::memcmp(baseline_output.contents, candidate_output.contents,
                    output_elements * sizeof(float)) != 0) {
        const auto* left = static_cast<const float*>(baseline_output.contents);
        const auto* right = static_cast<const float*>(candidate_output.contents);
        for (size_t index = 0; index < output_elements; ++index) {
            if (std::memcmp(left + index, right + index, sizeof(float)) != 0) {
                throw std::runtime_error(
                    "parallel shortconv output mismatch at index " +
                    std::to_string(index) + ": " + std::to_string(left[index]) +
                    " vs " + std::to_string(right[index]));
            }
        }
    }

    const size_t state_elements = static_cast<size_t>(kCacheLength) * kWidth;
    if (std::memcmp(baseline_state.contents, candidate_state.contents,
                    state_elements * sizeof(float)) != 0) {
        const auto* left = static_cast<const float*>(baseline_state.contents);
        const auto* right = static_cast<const float*>(candidate_state.contents);
        for (size_t index = 0; index < state_elements; ++index) {
            if (std::memcmp(left + index, right + index, sizeof(float)) != 0) {
                throw std::runtime_error(
                    "parallel shortconv state mismatch at index " +
                    std::to_string(index) + ": " + std::to_string(left[index]) +
                    " vs " + std::to_string(right[index]));
            }
        }
    }
}

template <typename Encode>
double time_sequence(id<MTLCommandQueue> queue, int repetitions, int iterations,
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
            std::string("#include <metal_stdlib>\nusing namespace metal;\n") +
            read_text("src/backend/metal/kernels/inference/convolution.metal") + "\n" +
            read_text("apps/benchmark/metal/shortconv_batch_parallel.metal");
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:
            [NSString stringWithUTF8String:source.c_str()] options:nil error:&error];
        if (!library) {
            throw std::runtime_error("Metal shortconv benchmark shader failed: " +
                (error ? ns_string(error.localizedDescription) : "unknown"));
        }

        id<MTLComputePipelineState> baseline =
            make_pipeline(device, library, "celeg_shortconv_batch_ring");
        id<MTLComputePipelineState> gate =
            make_pipeline(device, library, "celeg_shortconv_batch_gate_parallel");
        id<MTLComputePipelineState> convolve =
            make_pipeline(device, library, "celeg_shortconv_batch_convolve_parallel");
        id<MTLComputePipelineState> publish =
            make_pipeline(device, library, "celeg_shortconv_batch_publish_state_parallel");

        id<MTLBuffer> taps = random_buffer(
            device, static_cast<size_t>(kCacheLength) * kWidth, 0x51c0u, -0.5f, 0.5f);
        id<MTLBuffer> initial_state = random_buffer(
            device, static_cast<size_t>(kCacheLength) * kWidth, 0x7a7eu, -1.0f, 1.0f);

        std::cout << "Metal LFM2.5 shortconv batch A/B on "
                  << ns_string(device.name) << '\n';
        std::cout << "width=1024 cache_length=3 output_and_state_bit_exact=required\n\n";
        std::cout << std::left << std::setw(10) << "rows"
                  << std::right << std::setw(15) << "serial ms"
                  << std::setw(15) << "parallel ms"
                  << std::setw(12) << "speedup" << '\n';

        for (uint32_t rows : kRows) {
            const size_t output_elements = static_cast<size_t>(rows) * kWidth;
            id<MTLBuffer> projected = random_buffer(
                device, static_cast<size_t>(rows) * 3 * kWidth,
                0x1234u + rows, -1.0f, 1.0f);

            require_bit_exact(device, queue, baseline, gate, convolve, publish,
                              projected, taps, initial_state, rows);

            id<MTLBuffer> baseline_state = clone_buffer(device, initial_state);
            id<MTLBuffer> candidate_state = clone_buffer(device, initial_state);
            id<MTLBuffer> baseline_output = zero_buffer(
                device, output_elements * sizeof(float));
            id<MTLBuffer> candidate_output = zero_buffer(
                device, output_elements * sizeof(float));
            id<MTLBuffer> gated = zero_buffer(
                device, output_elements * sizeof(float));

            const int iterations = rows >= 512 ? 8 : 12;
            const double serial_ms = time_sequence(queue, 5, iterations,
                ^(id<MTLComputeCommandEncoder> encoder) {
                    encode_baseline(encoder, baseline, projected, taps,
                                    baseline_state, baseline_output, rows);
                });
            const double parallel_ms = time_sequence(queue, 5, iterations,
                ^(id<MTLComputeCommandEncoder> encoder) {
                    encode_candidate(encoder, gate, convolve, publish,
                                     projected, taps, candidate_state, gated,
                                     candidate_output, rows);
                });

            std::cout << std::left << std::setw(10) << rows
                      << std::right << std::fixed << std::setprecision(3)
                      << std::setw(15) << serial_ms
                      << std::setw(15) << parallel_ms
                      << std::setw(11) << (serial_ms / parallel_ms) << "x\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
