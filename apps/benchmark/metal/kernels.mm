/// @file
/// Roofline microbenchmark for the individual Metal inference kernels.
///
/// The end-to-end `celeg-metal-bench` reports one number per phase, which is
/// not enough to tell a slow kernel from a slow schedule.  This tool times each
/// linear kernel and the decode attention kernel in isolation, at the shapes
/// the supported checkpoints actually use, and reports achieved bandwidth
/// against a measured device-copy roofline rather than a datasheet figure.
///
/// Weight contents are irrelevant to timing (no kernel branches on decoded
/// values), so buffers are synthesized instead of loaded from a checkpoint.

#include "celeg/quantization/ggml.hpp"

#include "metal_inference_source.hpp"
#include "metal_tensor_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string ns_string(NSString* value) {
    return value ? std::string(value.UTF8String) : std::string{};
}

struct Shape {
    const char* label;
    uint32_t rows;
    uint32_t cols;
};

/// LFM2.5-350M shapes: mixer/attention projections, both feed-forward
/// orientations, and the tied Q6_K language-model head.
constexpr Shape kShapes[] = {
    {"proj_1024x1024", 1024, 1024},
    {"ffn_up_4608x1024", 4608, 1024},
    {"ffn_down_1024x4608", 1024, 4608},
    {"lm_head_65536x1024", 65536, 1024},
};

/// Mirrors the launch geometry of `MetalModel::Impl::matvec_kernel`.
struct Binding {
    const char* label;
    celeg::GgmlType type;
    const char* matvec;
    uint32_t rows_per_threadgroup;
    uint32_t threads;
    uint32_t threadgroup_floats;
};

constexpr Binding kBindings[] = {
    {"bf16", celeg::GgmlType::BF16, "celeg_matvec_bf16", 2, 128, 8},
    {"f16", celeg::GgmlType::F16, "celeg_matvec_f16", 2, 128, 8},
    {"q8_0", celeg::GgmlType::Q8_0, "celeg_matvec_q8_0", 16, 128, 0},
    {"q6_k", celeg::GgmlType::Q6_K, "celeg_matvec_q6k", 16, 128, 0},
    {"q5_k", celeg::GgmlType::Q5_K, "celeg_matvec_q5k", 16, 128, 0},
    {"q4_k", celeg::GgmlType::Q4_K, "celeg_matvec_q4k", 16, 128, 0},
    {"q4_0", celeg::GgmlType::Q4_0, "celeg_matvec_q4_0", 16, 128, 0},
};

constexpr uint32_t kAttentionDepths[] = {96, 960, 1984, 4096};

constexpr NSUInteger kTensorTileRows = 64;
constexpr NSUInteger kTensorTileTokens = 128;
constexpr NSUInteger kTensorTileK = 64;
constexpr NSUInteger kTensorTileThreads = 128;

/// Mirrors `MetalModel::Impl::tensor_pipeline` bindings for the batched
/// (prefill) matmul path.
struct TensorBinding {
    const char* label;
    celeg::GgmlType type;
    const char* kernel;
    bool dense;
};

constexpr TensorBinding kTensorBindings[] = {
    {"bf16", celeg::GgmlType::BF16, "celeg_matmul_tensor_bf16", true},
    {"f16", celeg::GgmlType::F16, "celeg_matmul_tensor_f16", true},
    {"q8_0", celeg::GgmlType::Q8_0, "celeg_matmul_tensor_q8_0", false},
    {"q6_k", celeg::GgmlType::Q6_K, "celeg_matmul_tensor_q6k", false},
    {"q5_k", celeg::GgmlType::Q5_K, "celeg_matmul_tensor_q5k", false},
    {"q4_k", celeg::GgmlType::Q4_K, "celeg_matmul_tensor_q4k", false},
    {"q4_0", celeg::GgmlType::Q4_0, "celeg_matmul_tensor_q4_0", false},
};

constexpr uint32_t kPrefillTokens = 512;

class Harness {
public:
    Harness() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("no Metal device");
        queue = [device newCommandQueue];
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:celeg::metal_detail::kInferenceShader];
        library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            throw std::runtime_error("Metal shader compilation failed: " +
                                     (error ? ns_string(error.localizedDescription) : "unknown"));
        }
        NSString* tensor_source =
            [NSString stringWithUTF8String:celeg::metal_detail::kTensorShader];
        tensor_library = [device newLibraryWithSource:tensor_source options:nil error:&error];
        if (!tensor_library) {
            throw std::runtime_error("Metal tensor shader compilation failed: " +
                                     (error ? ns_string(error.localizedDescription) : "unknown"));
        }
    }

    id<MTLComputePipelineState> pipeline_from(id<MTLLibrary> source_library, const char* name) {
        NSError* error = nil;
        id<MTLFunction> function =
            [source_library newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (!function) throw std::runtime_error(std::string("missing kernel: ") + name);
        id<MTLComputePipelineState> state =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!state) {
            throw std::runtime_error(std::string("pipeline failed for ") + name + ": " +
                                     (error ? ns_string(error.localizedDescription) : "unknown"));
        }
        return state;
    }

    id<MTLComputePipelineState> pipeline(const char* name) {
        return pipeline_from(library, name);
    }

    id<MTLComputePipelineState> tensor_pipeline(const char* name) {
        return pipeline_from(tensor_library, name);
    }

    id<MTLBuffer> random_buffer(size_t bytes) {
        id<MTLBuffer> buffer = [device newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
        auto* bytes_out = static_cast<uint8_t*>(buffer.contents);
        std::mt19937 generator(12345);
        for (size_t index = 0; index < bytes; ++index) {
            bytes_out[index] = static_cast<uint8_t>(generator() & 0xffu);
        }
        return buffer;
    }

    id<MTLBuffer> float_buffer(size_t count) {
        id<MTLBuffer> buffer = [device newBufferWithLength:count * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        auto* values = static_cast<float*>(buffer.contents);
        std::mt19937 generator(6789);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        for (size_t index = 0; index < count; ++index) values[index] = distribution(generator);
        return buffer;
    }

    id<MTLBuffer> zero_buffer(size_t bytes) {
        return [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    }

    /// Encodes @p iterations back-to-back dispatches and returns the median GPU
    /// milliseconds per iteration over @p repetitions command buffers.
    double time_dispatches(int repetitions, int iterations,
                           void (^encode)(id<MTLComputeCommandEncoder>)) {
        std::vector<double> samples;
        samples.reserve(static_cast<size_t>(repetitions));
        for (int repetition = 0; repetition < repetitions + 1; ++repetition) {
            id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            for (int index = 0; index < iterations; ++index) encode(encoder);
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.status != MTLCommandBufferStatusCompleted) {
                throw std::runtime_error("command buffer failed");
            }
            if (repetition == 0) continue;
            samples.push_back((command_buffer.GPUEndTime - command_buffer.GPUStartTime) *
                              1000.0 / static_cast<double>(iterations));
        }
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2];
    }

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLLibrary> tensor_library = nil;
};

struct Row {
    std::string kernel;
    std::string shape;
    size_t bytes = 0;
    double milliseconds = 0.0;
    double gb_per_second = 0.0;
};

double roofline(Harness& harness) {
    constexpr size_t kElements = 64u * 1024u * 1024u;
    id<MTLBuffer> source = harness.zero_buffer(kElements * sizeof(float));
    id<MTLBuffer> destination = harness.zero_buffer(kElements * sizeof(float));
    id<MTLComputePipelineState> state = harness.pipeline("celeg_copy_batch");
    const uint32_t count = static_cast<uint32_t>(kElements);
    const double milliseconds = harness.time_dispatches(5, 1, ^(id<MTLComputeCommandEncoder> encoder) {
        [encoder setComputePipelineState:state];
        [encoder setBuffer:source offset:0 atIndex:0];
        [encoder setBuffer:destination offset:0 atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(state.maxTotalThreadsPerThreadgroup, 1, 1)];
    });
    return static_cast<double>(kElements) * sizeof(float) * 2.0 / (milliseconds * 1.0e6);
}

std::vector<Row> measure_matvec(Harness& harness) {
    std::vector<Row> rows;
    for (const Binding& binding : kBindings) {
        const celeg::GgmlTypeTrait trait = celeg::ggml_type_trait(binding.type);
        if (trait.block_size == 0) continue;
        id<MTLComputePipelineState> state = harness.pipeline(binding.matvec);
        for (const Shape& shape : kShapes) {
            const uint32_t row_bytes =
                shape.cols / trait.block_size * trait.type_size;
            const size_t weight_bytes = static_cast<size_t>(shape.rows) * row_bytes;
            id<MTLBuffer> weights = harness.random_buffer(weight_bytes);
            id<MTLBuffer> input = harness.float_buffer(shape.cols);
            id<MTLBuffer> output = harness.zero_buffer(shape.rows * sizeof(float));
            const uint32_t rows_value = shape.rows;
            const uint32_t cols_value = shape.cols;
            const uint32_t row_bytes_value = row_bytes;
            const int iterations = weight_bytes > (32u << 20) ? 4 : 32;
            const double milliseconds = harness.time_dispatches(
                5, iterations, ^(id<MTLComputeCommandEncoder> encoder) {
                    [encoder setComputePipelineState:state];
                    [encoder setBuffer:weights offset:0 atIndex:0];
                    [encoder setBuffer:input offset:0 atIndex:1];
                    [encoder setBuffer:output offset:0 atIndex:2];
                    [encoder setBytes:&rows_value length:sizeof(rows_value) atIndex:3];
                    [encoder setBytes:&cols_value length:sizeof(cols_value) atIndex:4];
                    [encoder setBytes:&row_bytes_value length:sizeof(row_bytes_value) atIndex:5];
                    if (binding.threadgroup_floats != 0) {
                        [encoder setThreadgroupMemoryLength:binding.threadgroup_floats *
                                                           sizeof(float)
                                                    atIndex:0];
                    }
                    [encoder dispatchThreadgroups:MTLSizeMake(
                        (rows_value + binding.rows_per_threadgroup - 1u) /
                            binding.rows_per_threadgroup, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(binding.threads, 1, 1)];
                });
            rows.push_back({binding.matvec, shape.label, weight_bytes, milliseconds,
                            static_cast<double>(weight_bytes) / (milliseconds * 1.0e6)});
        }
    }
    return rows;
}

/// Times the batched (prefill) tensor matmul kernels at `kPrefillTokens`
/// tokens per shape. Reports weight bytes only, so `gb_per_second` /
/// `percent_of_roofline` read low by design once the kernel is dominated by
/// re-reading the activation tile once per row-tile rather than once total;
/// `ms` is the number to compare across tile-geometry experiments.
std::vector<Row> measure_matmul(Harness& harness) {
    std::vector<Row> rows;
    for (const TensorBinding& binding : kTensorBindings) {
        const celeg::GgmlTypeTrait trait = celeg::ggml_type_trait(binding.type);
        id<MTLComputePipelineState> state = harness.tensor_pipeline(binding.kernel);
        for (const Shape& shape : kShapes) {
            id<MTLBuffer> weights;
            uint32_t row_bytes = 0;
            size_t weight_bytes = 0;
            if (binding.dense) {
                weight_bytes = static_cast<size_t>(shape.rows) * shape.cols * sizeof(uint16_t);
                weights = harness.random_buffer(weight_bytes);
            } else {
                row_bytes = shape.cols / trait.block_size * trait.type_size;
                weight_bytes = static_cast<size_t>(shape.rows) * row_bytes;
                weights = harness.random_buffer(weight_bytes);
            }
            id<MTLBuffer> input = harness.float_buffer(
                static_cast<size_t>(kPrefillTokens) * shape.cols);
            id<MTLBuffer> output = harness.zero_buffer(
                static_cast<size_t>(kPrefillTokens) * shape.rows * sizeof(float));
            const uint32_t tokens_value = kPrefillTokens;
            const uint32_t cols_value = shape.cols;
            const uint32_t output_rows_value = shape.rows;
            const uint32_t output_stride_value = shape.rows;
            const NSUInteger threadgroup_bytes = kTensorTileRows * kTensorTileK * sizeof(uint16_t);
            const MTLSize groups = MTLSizeMake(
                (shape.rows + kTensorTileRows - 1u) / kTensorTileRows,
                (kPrefillTokens + kTensorTileTokens - 1u) / kTensorTileTokens, 1);
            const double milliseconds = harness.time_dispatches(
                5, 2, ^(id<MTLComputeCommandEncoder> encoder) {
                    [encoder setComputePipelineState:state];
                    [encoder setBuffer:weights offset:0 atIndex:0];
                    [encoder setBuffer:input offset:0 atIndex:1];
                    [encoder setBuffer:output offset:0 atIndex:2];
                    [encoder setBytes:&tokens_value length:sizeof(tokens_value) atIndex:3];
                    [encoder setBytes:&cols_value length:sizeof(cols_value) atIndex:4];
                    [encoder setBytes:&output_rows_value length:sizeof(output_rows_value) atIndex:5];
                    [encoder setBytes:&output_stride_value length:sizeof(output_stride_value) atIndex:6];
                    if (!binding.dense) {
                        [encoder setBytes:&row_bytes length:sizeof(row_bytes) atIndex:7];
                    }
                    [encoder setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
                    [encoder dispatchThreadgroups:groups
                            threadsPerThreadgroup:MTLSizeMake(kTensorTileThreads, 1, 1)];
                });
            rows.push_back({binding.kernel, shape.label, weight_bytes, milliseconds,
                            static_cast<double>(weight_bytes) / (milliseconds * 1.0e6)});
        }
    }
    return rows;
}

std::vector<Row> measure_attention(Harness& harness) {
    constexpr uint32_t kQueryHeads = 16;
    constexpr uint32_t kKeyHeads = 8;
    constexpr uint32_t kHeadDim = 64;
    constexpr uint32_t kPageTokens = 16;
    constexpr uint32_t kSimdgroups = 8;
    std::vector<Row> rows;
    id<MTLComputePipelineState> state = harness.pipeline("celeg_attention");
    for (const uint32_t depth : kAttentionDepths) {
        const size_t cache_elements =
            static_cast<size_t>(depth) * kKeyHeads * kHeadDim;
        id<MTLBuffer> query = harness.float_buffer(kQueryHeads * kHeadDim);
        id<MTLBuffer> keys = harness.float_buffer(cache_elements);
        id<MTLBuffer> values = harness.float_buffer(cache_elements);
        id<MTLBuffer> output = harness.zero_buffer(kQueryHeads * kHeadDim * sizeof(float));
        const uint32_t sequence_length = depth;
        const uint32_t query_heads = kQueryHeads;
        const uint32_t key_heads = kKeyHeads;
        const uint32_t head_dim = kHeadDim;
        const float scale = 0.125f;
        const uint32_t page_tokens = kPageTokens;
        const size_t traffic = cache_elements * 2 * sizeof(float);
        const NSUInteger shared_bytes =
            (2u * kSimdgroups + kSimdgroups * kHeadDim) * sizeof(float);
        const double milliseconds = harness.time_dispatches(
            3, 4, ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setComputePipelineState:state];
                [encoder setBuffer:query offset:0 atIndex:0];
                [encoder setBuffer:keys offset:0 atIndex:1];
                [encoder setBuffer:values offset:0 atIndex:2];
                [encoder setBuffer:output offset:0 atIndex:3];
                [encoder setBytes:&sequence_length length:sizeof(sequence_length) atIndex:4];
                [encoder setBytes:&query_heads length:sizeof(query_heads) atIndex:5];
                [encoder setBytes:&key_heads length:sizeof(key_heads) atIndex:6];
                [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:7];
                [encoder setBytes:&scale length:sizeof(scale) atIndex:8];
                [encoder setBytes:&page_tokens length:sizeof(page_tokens) atIndex:9];
                [encoder setThreadgroupMemoryLength:shared_bytes atIndex:0];
                [encoder dispatchThreadgroups:MTLSizeMake(query_heads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32 * kSimdgroups, 1, 1)];
            });
        std::ostringstream label;
        label << "depth_" << depth;
        rows.push_back({"celeg_attention", label.str(), traffic, milliseconds,
                        static_cast<double>(traffic) / (milliseconds * 1.0e6)});
    }
    return rows;
}

void emit(const std::vector<Row>& rows, double peak, const char* section, bool& first) {
    for (const Row& row : rows) {
        if (!first) std::cout << ",\n";
        first = false;
        std::cout << "    {\"section\": \"" << section << "\", \"kernel\": \"" << row.kernel
                  << "\", \"shape\": \"" << row.shape << "\", \"bytes\": " << row.bytes
                  << ", \"ms\": " << std::setprecision(6) << row.milliseconds
                  << ", \"gb_per_second\": " << row.gb_per_second
                  << ", \"percent_of_roofline\": "
                  << (peak > 0.0 ? row.gb_per_second * 100.0 / peak : 0.0) << "}";
    }
}

}

int main() {
    try {
        Harness harness;
        const double peak = roofline(harness);
        const std::vector<Row> matvec = measure_matvec(harness);
        const std::vector<Row> matmul = measure_matmul(harness);
        const std::vector<Row> attention = measure_attention(harness);
        std::cout << "{\n  \"device\": \"" << ns_string(harness.device.name) << "\",\n"
                  << "  \"copy_roofline_gb_per_second\": " << std::setprecision(6) << peak
                  << ",\n  \"rows\": [\n";
        bool first = true;
        emit(matvec, peak, "matvec", first);
        emit(matmul, peak, "matmul", first);
        emit(attention, peak, "attention", first);
        std::cout << "\n  ]\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
