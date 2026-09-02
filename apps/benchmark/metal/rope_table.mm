#include "metal_inference_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kRows = 512;
constexpr uint32_t kQueryHeads = 16;
constexpr uint32_t kKeyHeads = 8;
constexpr uint32_t kHeadDim = 64;
constexpr uint32_t kPairs = kHeadDim / 2;
constexpr uint32_t kBasePosition = 0;
constexpr float kTheta = 10000.0f;
constexpr float kQueryScale = 1.0f;
constexpr float kEpsilon = 1.0e-5f;

constexpr const char* kTableShader = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void celeg_rope_table_batch(
    device float2* table [[buffer(0)]],
    constant uint& rows [[buffer(1)]],
    constant uint& pairs [[buffer(2)]],
    constant uint& base_position [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant float& theta [[buffer(5)]],
    uint2 index [[thread_position_in_grid]]) {
    const uint pair = index.x;
    const uint token = index.y;
    if (pair >= pairs || token >= rows) return;
    const float frequency = pow(theta, -2.0f * static_cast<float>(pair) /
                                      static_cast<float>(head_dim));
    const float angle = static_cast<float>(base_position + token) * frequency;
    table[static_cast<size_t>(token) * pairs + pair] = float2(cos(angle), sin(angle));
}

kernel void celeg_qk_norm_rope_batch_split_table(
    device float* query [[buffer(0)]],
    device const float* query_weight [[buffer(1)]],
    device float* key [[buffer(2)]],
    device const float* key_weight [[buffer(3)]],
    device const float2* table [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& query_heads [[buffer(6)]],
    constant uint& key_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant float& query_scale [[buffer(9)]],
    constant float& query_epsilon [[buffer(10)]],
    constant float& key_epsilon [[buffer(11)]],
    uint2 index [[thread_position_in_grid]]) {
    const uint head = index.x;
    const uint token = index.y;
    if (token >= rows) return;
    const uint pairs = head_dim / 2;
    const device float2* angles = table + static_cast<size_t>(token) * pairs;

    if (head < query_heads) {
        const size_t base = static_cast<size_t>(token) * query_heads * head_dim +
            static_cast<size_t>(head) * head_dim;
        float sum = 0.0f;
        for (uint d = 0; d < head_dim; ++d) sum += query[base + d] * query[base + d];
        const float inverse = rsqrt(sum / static_cast<float>(head_dim) + query_epsilon);
        for (uint pair = 0; pair < pairs; ++pair) {
            const float2 cs = angles[pair];
            const size_t first = base + pair;
            const size_t second = base + pairs + pair;
            const float x = query[first] * (inverse * query_weight[pair]);
            const float y = query[second] * (inverse * query_weight[pairs + pair]);
            query[first] = (x * cs.x - y * cs.y) * query_scale;
            query[second] = (y * cs.x + x * cs.y) * query_scale;
        }
    }
    if (head < key_heads) {
        const size_t base = static_cast<size_t>(token) * key_heads * head_dim +
            static_cast<size_t>(head) * head_dim;
        float sum = 0.0f;
        for (uint d = 0; d < head_dim; ++d) sum += key[base + d] * key[base + d];
        const float inverse = rsqrt(sum / static_cast<float>(head_dim) + key_epsilon);
        for (uint pair = 0; pair < pairs; ++pair) {
            const float2 cs = angles[pair];
            const size_t first = base + pair;
            const size_t second = base + pairs + pair;
            const float x = key[first] * (inverse * key_weight[pair]);
            const float y = key[second] * (inverse * key_weight[pairs + pair]);
            key[first] = x * cs.x - y * cs.y;
            key[second] = y * cs.x + x * cs.y;
        }
    }
}
)METAL";

std::string ns_string(NSString* value) {
    return value ? std::string(value.UTF8String) : std::string{};
}

id<MTLLibrary> compile(id<MTLDevice> device, const char* source, const char* label) {
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source]
                                                   options:nil error:&error];
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

id<MTLBuffer> zeros(id<MTLDevice> device, size_t bytes) {
    return [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

struct Inputs {
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> query_weight;
    std::vector<float> key_weight;
};

Inputs make_inputs() {
    std::mt19937 generator(12345);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    Inputs result;
    result.query.resize(static_cast<size_t>(kRows) * kQueryHeads * kHeadDim);
    result.key.resize(static_cast<size_t>(kRows) * kKeyHeads * kHeadDim);
    result.query_weight.resize(kHeadDim);
    result.key_weight.resize(kHeadDim);
    for (float& value : result.query) value = distribution(generator);
    for (float& value : result.key) value = distribution(generator);
    for (float& value : result.query_weight) value = 0.8f + 0.2f * distribution(generator);
    for (float& value : result.key_weight) value = 0.8f + 0.2f * distribution(generator);
    return result;
}

void encode_baseline(id<MTLComputeCommandEncoder> encoder,
                     id<MTLComputePipelineState> state,
                     id<MTLBuffer> query, id<MTLBuffer> query_weight,
                     id<MTLBuffer> key, id<MTLBuffer> key_weight) {
    [encoder setComputePipelineState:state];
    [encoder setBuffer:query offset:0 atIndex:0];
    [encoder setBuffer:query_weight offset:0 atIndex:1];
    [encoder setBuffer:key offset:0 atIndex:2];
    [encoder setBuffer:key_weight offset:0 atIndex:3];
    [encoder setBytes:&kRows length:sizeof(kRows) atIndex:4];
    [encoder setBytes:&kQueryHeads length:sizeof(kQueryHeads) atIndex:5];
    [encoder setBytes:&kKeyHeads length:sizeof(kKeyHeads) atIndex:6];
    [encoder setBytes:&kHeadDim length:sizeof(kHeadDim) atIndex:7];
    [encoder setBytes:&kBasePosition length:sizeof(kBasePosition) atIndex:8];
    [encoder setBytes:&kTheta length:sizeof(kTheta) atIndex:9];
    [encoder setBytes:&kQueryScale length:sizeof(kQueryScale) atIndex:10];
    [encoder setBytes:&kEpsilon length:sizeof(kEpsilon) atIndex:11];
    [encoder setBytes:&kEpsilon length:sizeof(kEpsilon) atIndex:12];
    const uint32_t count = kRows * std::max(kQueryHeads, kKeyHeads);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(
           count, state.maxTotalThreadsPerThreadgroup), 1, 1)];
}

void encode_table(id<MTLComputeCommandEncoder> encoder,
                  id<MTLComputePipelineState> table_state,
                  id<MTLBuffer> table) {
    [encoder setComputePipelineState:table_state];
    [encoder setBuffer:table offset:0 atIndex:0];
    [encoder setBytes:&kRows length:sizeof(kRows) atIndex:1];
    [encoder setBytes:&kPairs length:sizeof(kPairs) atIndex:2];
    [encoder setBytes:&kBasePosition length:sizeof(kBasePosition) atIndex:3];
    [encoder setBytes:&kHeadDim length:sizeof(kHeadDim) atIndex:4];
    [encoder setBytes:&kTheta length:sizeof(kTheta) atIndex:5];
    [encoder dispatchThreads:MTLSizeMake(kPairs, kRows, 1)
       threadsPerThreadgroup:MTLSizeMake(kPairs, 1, 1)];
}

void encode_table_qk(id<MTLComputeCommandEncoder> encoder,
                     id<MTLComputePipelineState> state,
                     id<MTLBuffer> query, id<MTLBuffer> query_weight,
                     id<MTLBuffer> key, id<MTLBuffer> key_weight,
                     id<MTLBuffer> table) {
    [encoder setComputePipelineState:state];
    [encoder setBuffer:query offset:0 atIndex:0];
    [encoder setBuffer:query_weight offset:0 atIndex:1];
    [encoder setBuffer:key offset:0 atIndex:2];
    [encoder setBuffer:key_weight offset:0 atIndex:3];
    [encoder setBuffer:table offset:0 atIndex:4];
    [encoder setBytes:&kRows length:sizeof(kRows) atIndex:5];
    [encoder setBytes:&kQueryHeads length:sizeof(kQueryHeads) atIndex:6];
    [encoder setBytes:&kKeyHeads length:sizeof(kKeyHeads) atIndex:7];
    [encoder setBytes:&kHeadDim length:sizeof(kHeadDim) atIndex:8];
    [encoder setBytes:&kQueryScale length:sizeof(kQueryScale) atIndex:9];
    [encoder setBytes:&kEpsilon length:sizeof(kEpsilon) atIndex:10];
    [encoder setBytes:&kEpsilon length:sizeof(kEpsilon) atIndex:11];
    const uint32_t heads = std::max(kQueryHeads, kKeyHeads);
    [encoder dispatchThreads:MTLSizeMake(heads, kRows, 1)
       threadsPerThreadgroup:MTLSizeMake(heads, 1, 1)];
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
            throw std::runtime_error("RoPE benchmark command failed");
        }
        if (repetition == 0) continue;
        samples.push_back((command.GPUEndTime - command.GPUStartTime) * 1000.0 /
                          static_cast<double>(iterations));
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

double max_abs(const float* a, const float* b, size_t count) {
    double result = 0.0;
    for (size_t index = 0; index < count; ++index) {
        result = std::max(result, std::abs(static_cast<double>(a[index]) - b[index]));
    }
    return result;
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
        throw std::runtime_error("RoPE validation command failed");
    }
}

}

int main() {
    try {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("no Metal device");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) throw std::runtime_error("no Metal command queue");
        id<MTLLibrary> baseline_library = compile(
            device, celeg::metal_detail::kInferenceShader, "runtime shader");
        id<MTLLibrary> table_library = compile(device, kTableShader, "table shader");
        id<MTLComputePipelineState> baseline = pipeline(
            device, baseline_library, "celeg_qk_norm_rope_batch_split");
        id<MTLComputePipelineState> table_state = pipeline(
            device, table_library, "celeg_rope_table_batch");
        id<MTLComputePipelineState> table_qk = pipeline(
            device, table_library, "celeg_qk_norm_rope_batch_split_table");

        const Inputs source = make_inputs();
        id<MTLBuffer> query_weight = floats(device, source.query_weight);
        id<MTLBuffer> key_weight = floats(device, source.key_weight);
        id<MTLBuffer> baseline_query = floats(device, source.query);
        id<MTLBuffer> baseline_key = floats(device, source.key);
        id<MTLBuffer> table_query = floats(device, source.query);
        id<MTLBuffer> table_key = floats(device, source.key);
        id<MTLBuffer> table = zeros(device,
            static_cast<size_t>(kRows) * kPairs * sizeof(float) * 2);

        run_once(queue, ^(id<MTLComputeCommandEncoder> encoder) {
            encode_baseline(encoder, baseline, baseline_query, query_weight,
                            baseline_key, key_weight);
        });
        run_once(queue, ^(id<MTLComputeCommandEncoder> encoder) {
            encode_table(encoder, table_state, table);
            encode_table_qk(encoder, table_qk, table_query, query_weight,
                            table_key, key_weight, table);
        });
        const double query_error = max_abs(
            static_cast<const float*>(baseline_query.contents),
            static_cast<const float*>(table_query.contents), source.query.size());
        const double key_error = max_abs(
            static_cast<const float*>(baseline_key.contents),
            static_cast<const float*>(table_key.contents), source.key.size());

        id<MTLBuffer> timed_query = floats(device, source.query);
        id<MTLBuffer> timed_key = floats(device, source.key);
        const double baseline_ms = time(queue, 5, 4, ^(id<MTLComputeCommandEncoder> encoder) {
            encode_baseline(encoder, baseline, timed_query, query_weight, timed_key, key_weight);
        });
        const double table_ms = time(queue, 5, 4, ^(id<MTLComputeCommandEncoder> encoder) {
            encode_table(encoder, table_state, table);
        });
        const double table_qk_ms = time(queue, 5, 4, ^(id<MTLComputeCommandEncoder> encoder) {
            encode_table_qk(encoder, table_qk, timed_query, query_weight,
                            timed_key, key_weight, table);
        });
        const double combined_ms = time(queue, 5, 4, ^(id<MTLComputeCommandEncoder> encoder) {
            encode_table(encoder, table_state, table);
            encode_table_qk(encoder, table_qk, timed_query, query_weight,
                            timed_key, key_weight, table);
        });

        std::cout << "{\n"
                  << "  \"device\": \"" << ns_string(device.name) << "\",\n"
                  << "  \"shape\": \"pp512_q16_kv8_hd64_split\",\n"
                  << "  \"query_max_abs_error\": " << std::setprecision(9) << query_error << ",\n"
                  << "  \"key_max_abs_error\": " << key_error << ",\n"
                  << "  \"baseline_ms\": " << baseline_ms << ",\n"
                  << "  \"table_generation_ms\": " << table_ms << ",\n"
                  << "  \"table_qk_ms\": " << table_qk_ms << ",\n"
                  << "  \"table_combined_ms\": " << combined_ms << ",\n"
                  << "  \"combined_speedup\": "
                  << (combined_ms > 0.0 ? baseline_ms / combined_ms : 0.0) << "\n"
                  << "}\n";
        return (query_error <= 1.0e-6 && key_error <= 1.0e-6) ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
