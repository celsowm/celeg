/// @file
/// Benchmark-only A/B for F16/BF16 decode matvec row reuse.

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

constexpr NSUInteger kThreads = 128;

struct Shape {
    const char* label;
    uint32_t rows;
    uint32_t cols;
    uint32_t uses;
};

struct Storage {
    const char* label;
    const char* rows2_kernel;
    const char* rows4_kernel;
    const char* rows8_kernel;
    const uint16_t* patterns;
    size_t pattern_count;
};

constexpr Shape kShapes[] = {
    {"attn_kv_512x1024", 512, 1024, 12},
    {"proj_1024x1024", 1024, 1024, 22},
    {"conv_in_3072x1024", 3072, 1024, 10},
    {"ffn_up_4608x1024", 4608, 1024, 32},
    {"ffn_down_1024x4608", 1024, 4608, 16},
    {"lm_head_65536x1024", 65536, 1024, 1},
};

constexpr uint16_t kF16Patterns[] = {
    0x3c00, 0xbc00, 0x3800, 0xb800, 0x3400, 0xb400, 0x4000, 0xc000,
};
constexpr uint16_t kBF16Patterns[] = {
    0x3f80, 0xbf80, 0x3f00, 0xbf00, 0x3e80, 0xbe80, 0x4000, 0xc000,
};
constexpr Storage kStorages[] = {
    {"F16", "celeg_matvec_f16_rows2_bench", "celeg_matvec_f16_rows4_bench",
     "celeg_matvec_f16_rows8_bench", kF16Patterns,
     sizeof(kF16Patterns) / sizeof(kF16Patterns[0])},
    {"BF16", "celeg_matvec_bf16_rows2_bench", "celeg_matvec_bf16_rows4_bench",
     "celeg_matvec_bf16_rows8_bench", kBF16Patterns,
     sizeof(kBF16Patterns) / sizeof(kBF16Patterns[0])},
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

id<MTLBuffer> input_buffer(id<MTLDevice> device, uint32_t cols) {
    id<MTLBuffer> buffer = zero_buffer(device, static_cast<size_t>(cols) * sizeof(float));
    auto* values = static_cast<float*>(buffer.contents);
    std::mt19937 generator(0x726f7773u + cols);
    std::uniform_real_distribution<float> distribution(-0.125f, 0.125f);
    for (uint32_t index = 0; index < cols; ++index) values[index] = distribution(generator);
    return buffer;
}

id<MTLBuffer> weight_buffer(id<MTLDevice> device, const Shape& shape,
                            const Storage& storage) {
    const size_t elements = static_cast<size_t>(shape.rows) * shape.cols;
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

void encode_matvec(id<MTLComputeCommandEncoder> encoder,
                   id<MTLComputePipelineState> state,
                   id<MTLBuffer> weights, id<MTLBuffer> input, id<MTLBuffer> output,
                   const Shape& shape, NSUInteger rows_per_threadgroup) {
    [encoder setComputePipelineState:state];
    [encoder setBuffer:weights offset:0 atIndex:0];
    [encoder setBuffer:input offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&shape.rows length:sizeof(shape.rows) atIndex:3];
    [encoder setBytes:&shape.cols length:sizeof(shape.cols) atIndex:4];
    [encoder setThreadgroupMemoryLength:4u * rows_per_threadgroup * sizeof(float) atIndex:0];
    const NSUInteger groups =
        (static_cast<NSUInteger>(shape.rows) + rows_per_threadgroup - 1u) /
        rows_per_threadgroup;
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
}

struct ErrorStats {
    double max_abs = 0.0;
    double rms = 0.0;
    bool bit_exact = true;
};

ErrorStats compare_outputs(id<MTLBuffer> baseline, id<MTLBuffer> candidate, uint32_t rows) {
    const auto* left = static_cast<const float*>(baseline.contents);
    const auto* right = static_cast<const float*>(candidate.contents);
    ErrorStats result;
    long double squared = 0.0;
    for (uint32_t index = 0; index < rows; ++index) {
        if (!std::isfinite(left[index]) || !std::isfinite(right[index])) {
            throw std::runtime_error("non-finite dense matvec output");
        }
        if (std::memcmp(left + index, right + index, sizeof(float)) != 0) {
            result.bit_exact = false;
        }
        const double delta = std::abs(static_cast<double>(left[index]) - right[index]);
        result.max_abs = std::max(result.max_abs, delta);
        squared += delta * delta;
    }
    result.rms = std::sqrt(static_cast<double>(squared / rows));
    return result;
}

void run_correctness(id<MTLCommandQueue> queue,
                     id<MTLComputePipelineState> rows2,
                     id<MTLComputePipelineState> rows4,
                     id<MTLComputePipelineState> rows8,
                     id<MTLBuffer> weights, id<MTLBuffer> input,
                     id<MTLBuffer> output2, id<MTLBuffer> output4,
                     id<MTLBuffer> output8, const Shape& shape) {
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    encode_matvec(encoder, rows2, weights, input, output2, shape, 2);
    encode_matvec(encoder, rows4, weights, input, output4, shape, 4);
    encode_matvec(encoder, rows8, weights, input, output8, shape, 8);
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        const std::string message = command_buffer.error
            ? ns_string(command_buffer.error.localizedDescription) : "unknown";
        throw std::runtime_error("correctness command failed: " + message);
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

        const std::string source = read_text("apps/benchmark/metal/dense_matvec_rows.metal");
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:
            [NSString stringWithUTF8String:source.c_str()] options:nil error:&error];
        if (!library) {
            throw std::runtime_error("dense matvec rows benchmark shader failed: " +
                (error ? ns_string(error.localizedDescription) : "unknown"));
        }

        std::cout << "Metal dense decode matvec row-reuse A/B on " << ns_string(device.name) << "\n";
        std::cout << "same K partition/reduction order; 2 vs 4 vs 8 output rows/TG\n\n";
        std::cout << std::left << std::setw(7) << "type"
                  << std::setw(26) << "shape"
                  << std::right << std::setw(7) << "uses"
                  << std::setw(10) << "rows2"
                  << std::setw(10) << "rows4"
                  << std::setw(9) << "r4 x"
                  << std::setw(11) << "r4 max"
                  << std::setw(11) << "r4 RMS"
                  << std::setw(8) << "r4 eq"
                  << std::setw(10) << "rows8"
                  << std::setw(9) << "r8 x"
                  << std::setw(11) << "r8 max"
                  << std::setw(11) << "r8 RMS"
                  << std::setw(8) << "r8 eq" << '\n';

        for (const Storage& storage : kStorages) {
            id<MTLComputePipelineState> rows2 = pipeline(device, library, storage.rows2_kernel);
            id<MTLComputePipelineState> rows4 = pipeline(device, library, storage.rows4_kernel);
            id<MTLComputePipelineState> rows8 = pipeline(device, library, storage.rows8_kernel);
            for (const Shape& shape : kShapes) {
                id<MTLBuffer> weights = weight_buffer(device, shape, storage);
                id<MTLBuffer> input = input_buffer(device, shape.cols);
                const size_t output_bytes = static_cast<size_t>(shape.rows) * sizeof(float);
                id<MTLBuffer> output2 = zero_buffer(device, output_bytes);
                id<MTLBuffer> output4 = zero_buffer(device, output_bytes);
                id<MTLBuffer> output8 = zero_buffer(device, output_bytes);

                run_correctness(queue, rows2, rows4, rows8, weights, input,
                                output2, output4, output8, shape);
                const ErrorStats error4 = compare_outputs(output2, output4, shape.rows);
                const ErrorStats error8 = compare_outputs(output2, output8, shape.rows);

                constexpr int repetitions = 7;
                constexpr int iterations = 64;
                const double rows2_ms = time_kernel(queue, repetitions, iterations,
                    ^(id<MTLComputeCommandEncoder> encoder) {
                        encode_matvec(encoder, rows2, weights, input, output2, shape, 2);
                    });
                const double rows4_ms = time_kernel(queue, repetitions, iterations,
                    ^(id<MTLComputeCommandEncoder> encoder) {
                        encode_matvec(encoder, rows4, weights, input, output4, shape, 4);
                    });
                const double rows8_ms = time_kernel(queue, repetitions, iterations,
                    ^(id<MTLComputeCommandEncoder> encoder) {
                        encode_matvec(encoder, rows8, weights, input, output8, shape, 8);
                    });

                std::cout << std::left << std::setw(7) << storage.label
                          << std::setw(26) << shape.label
                          << std::right << std::setw(7) << shape.uses
                          << std::fixed << std::setprecision(3)
                          << std::setw(10) << rows2_ms
                          << std::setw(10) << rows4_ms
                          << std::setw(8) << (rows2_ms / rows4_ms) << "x"
                          << std::scientific << std::setprecision(2)
                          << std::setw(11) << error4.max_abs
                          << std::setw(11) << error4.rms
                          << std::fixed << std::setw(8) << (error4.bit_exact ? "yes" : "no")
                          << std::setprecision(3)
                          << std::setw(10) << rows8_ms
                          << std::setw(8) << (rows2_ms / rows8_ms) << "x"
                          << std::scientific << std::setprecision(2)
                          << std::setw(11) << error8.max_abs
                          << std::setw(11) << error8.rms
                          << std::fixed << std::setw(8) << (error8.bit_exact ? "yes" : "no")
                          << '\n';
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
