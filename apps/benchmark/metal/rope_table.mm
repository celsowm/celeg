#include "metal_inference_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
constexpr uint32_t kAttentionLayers = 6;

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

inline void celeg_qk_norm_rope_table_head(
    device float* data,
    device const float* weight,
    device const float2* angles,
    size_t base,
    uint head_dim,
    float epsilon,
    float output_scale) {
    const uint pairs = head_dim / 2;
    float sum = 0.0f;
    for (uint d = 0; d < head_dim; ++d) sum += data[base + d] * data[base + d];
    const float inverse = rsqrt(sum / static_cast<float>(head_dim) + epsilon);
    for (uint pair = 0; pair < pairs; ++pair) {
        const float2 cs = angles[pair];
        const size_t first = base + pair;
        const size_t second = base + pairs + pair;
        const float x = data[first] * (inverse * weight[pair]);
        const float y = data[second] * (inverse * weight[pairs + pair]);
        data[first] = (x * cs.x - y * cs.y) * output_scale;
        data[second] = (y * cs.x + x * cs.y) * output_scale;
    }
}

kernel void celeg_qk_norm_rope_batch_split_table_store_kv(
    device float* query [[buffer(0)]],
    device const float* query_weight [[buffer(1)]],
    device float* key [[buffer(2)]],
    device const float* key_weight [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& query_heads [[buffer(5)]],
    constant uint& key_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& base_position [[buffer(8)]],
    constant float& theta [[buffer(9)]],
    constant float& query_scale [[buffer(10)]],
    constant float& query_epsilon [[buffer(11)]],
    constant float& key_epsilon [[buffer(12)]],
    device const float* value [[buffer(13)]],
    device float* key_cache [[buffer(14)]],
    device float* value_cache [[buffer(15)]],
    device const float2* table [[buffer(16)]],
    uint2 index [[thread_position_in_grid]]) {
    const uint head = index.x;
    const uint token = index.y;
    if (token >= rows) return;
    const uint pairs = head_dim / 2;
    const device float2* angles = table + static_cast<size_t>(token) * pairs;
    const uint position = base_position + token;

    if (head < query_heads) {
        const size_t base = static_cast<size_t>(token) * query_heads * head_dim +
            static_cast<size_t>(head) * head_dim;
        celeg_qk_norm_rope_table_head(query, query_weight, angles, base, head_dim,
                                      query_epsilon, query_scale);
    }
    if (head < key_heads) {
        const size_t source_base = static_cast<size_t>(token) * key_heads * head_dim +
            static_cast<size_t>(head) * head_dim;
        celeg_qk_norm_rope_table_head(key, key_weight, angles, source_base, head_dim,
                                      key_epsilon, 1.0f);
        const size_t cache_base = static_cast<size_t>(position) * key_heads * head_dim +
            static_cast<size_t>(head) * head_dim;
        for (uint d = 0; d < head_dim; ++d) {
            key_cache[cache_base + d] = key[source_base + d];
            value_cache[cache_base + d] = value[source_base + d];
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
    id<MTLBuffer> result = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (result) std::memset(result.contents, 0, bytes);
    return result;
}

struct Inputs {
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    std::vector<float> query_weight;
    std::vector<float> key_weight;
};

Inputs make_inputs() {
    std::mt19937 generator(12345);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    Inputs result;
    result.query.resize(static_cast<size_t>(kRows) * kQueryHeads * kHeadDim);
    result.key.resize(static_cast<size_t>(kRows) * kKeyHeads * kHeadDim);
    result.value.resize(static_cast<size_t>(kRows) * kKeyHeads * kHeadDim);
    result.query_weight.resize(kHeadDim);
    result.key_weight.resize(kHeadDim);
    for (float& value : result.query) value = distribution(generator);
    for (float& value : result.key) value = distribution(generator);
    for (float& value : result.value) value = distribution(generator);
    for (float& value : result.query_weight) value = 0.8f + 0.2f * distribution(generator);
    for (float& value : result.key_weight) value = 0.8f + 0.2f * distribution(generator);
    return result;
}

void encode_baseline(id<MTLComputeCommandEncoder> encoder,
                     id<MTLComputePipelineState> state,
                     id<MTLBuffer> query, id<MTLBuffer> query_weight,
                     id<MTLBuffer> key, id<MTLBuffer> key_weight,
                     id<MTLBuffer> value, id<MTLBuffer> key_cache,
                     id<MTLBuffer> value_cache) {
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
    [encoder setBuffer:value offset:0 atIndex:13];
    [encoder setBuffer:key_cache offset:0 atIndex:14];
    [encoder setBuffer:value_cache offset:0 atIndex:15];
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
                     id<MTLBuffer> value, id<MTLBuffer> key_cache,
                     id<MTLBuffer> value_cache, id<MTLBuffer> table) {
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
    [encoder setBuffer:value offset:0 atIndex:13];
    [encoder setBuffer:key_cache offset:0 atIndex:14];
    [encoder setBuffer:value_cache offset:0 atIndex:15];
    [encoder setBuffer:table offset:0 atIndex:16];
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

bool bit_exact(const float* a, const float* b, size_t count) {
    return std::memcmp(a, b, count * sizeof(float)) == 0;
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
            device, baseline_library, "celeg_qk_norm_rope_batch_split_store_kv");
        id<MTLComputePipelineState> table_state = pipeline(
            device, table_library, "celeg_rope_table_batch");
        id<MTLComputePipelineState> table_qk = pipeline(
            device, table_library, "celeg_qk_norm_rope_batch_split_table_store_kv");

        const Inputs source = make_inputs();
        const size_t query_count = source.query.size();
        const size_t kv_count = source.key.size();
        const size_t kv_bytes = kv_count * sizeof(float);
        id<MTLBuffer> query_weight = floats(device, source.query_weight);
        id<MTLBuffer> key_weight = floats(device, source.key_weight);
        id<MTLBuffer> value = floats(device, source.value);
        id<MTLBuffer> baseline_query = floats(device, source.query);
        id<MTLBuffer> baseline_key = floats(device, source.key);
        id<MTLBuffer> baseline_key_cache = zeros(device, kv_bytes);
        id<MTLBuffer> baseline_value_cache = zeros(device, kv_bytes);
        id<MTLBuffer> table_query = floats(device, source.query);
        id<MTLBuffer> table_key = floats(device, source.key);
        id<MTLBuffer> table_key_cache = zeros(device, kv_bytes);
        id<MTLBuffer> table_value_cache = zeros(device, kv_bytes);
        id<MTLBuffer> table = zeros(device,
            static_cast<size_t>(kRows) * kPairs * sizeof(float) * 2);

        run_once(queue, ^(id<MTLComputeCommandEncoder> encoder) {
            encode_baseline(encoder, baseline, baseline_query, query_weight,
                            baseline_key, key_weight, value,
                            baseline_key_cache, baseline_value_cache);
        });
        run_once(queue, ^(id<MTLComputeCommandEncoder> encoder) {
            encode_table(encoder, table_state, table);
            encode_table_qk(encoder, table_qk, table_query, query_weight,
                            table_key, key_weight, value,
                            table_key_cache, table_value_cache, table);
        });

        const auto* baseline_query_values = static_cast<const float*>(baseline_query.contents);
        const auto* table_query_values = static_cast<const float*>(table_query.contents);
        const auto* baseline_key_values = static_cast<const float*>(baseline_key.contents);
        const auto* table_key_values = static_cast<const float*>(table_key.contents);
        const auto* baseline_key_cache_values =
            static_cast<const float*>(baseline_key_cache.contents);
        const auto* table_key_cache_values = static_cast<const float*>(table_key_cache.contents);
        const auto* baseline_value_cache_values =
            static_cast<const float*>(baseline_value_cache.contents);
        const auto* table_value_cache_values = static_cast<const float*>(table_value_cache.contents);

        const double query_error = max_abs(baseline_query_values, table_query_values, query_count);
        const double key_error = max_abs(baseline_key_values, table_key_values, kv_count);
        const double key_cache_error = max_abs(
            baseline_key_cache_values, table_key_cache_values, kv_count);
        const double value_cache_error = max_abs(
            baseline_value_cache_values, table_value_cache_values, kv_count);
        const bool query_exact = bit_exact(baseline_query_values, table_query_values, query_count);
        const bool key_exact = bit_exact(baseline_key_values, table_key_values, kv_count);
        const bool key_cache_exact = bit_exact(
            baseline_key_cache_values, table_key_cache_values, kv_count);
        const bool value_cache_exact = bit_exact(
            baseline_value_cache_values, table_value_cache_values, kv_count);

        id<MTLBuffer> timed_query = floats(device, source.query);
        id<MTLBuffer> timed_key = floats(device, source.key);
        id<MTLBuffer> timed_key_cache = zeros(device, kv_bytes);
        id<MTLBuffer> timed_value_cache = zeros(device, kv_bytes);
        const double baseline_ms = time(queue, 5, 4, ^(id<MTLComputeCommandEncoder> encoder) {
            encode_baseline(encoder, baseline, timed_query, query_weight,
                            timed_key, key_weight, value,
                            timed_key_cache, timed_value_cache);
        });
        const double table_ms = time(queue, 5, 4, ^(id<MTLComputeCommandEncoder> encoder) {
            encode_table(encoder, table_state, table);
        });
        const double table_qk_ms = time(queue, 5, 4, ^(id<MTLComputeCommandEncoder> encoder) {
            encode_table_qk(encoder, table_qk, timed_query, query_weight,
                            timed_key, key_weight, value,
                            timed_key_cache, timed_value_cache, table);
        });
        const double combined_ms = time(queue, 5, 4, ^(id<MTLComputeCommandEncoder> encoder) {
            encode_table(encoder, table_state, table);
            encode_table_qk(encoder, table_qk, timed_query, query_weight,
                            timed_key, key_weight, value,
                            timed_key_cache, timed_value_cache, table);
        });
        const double baseline_six_ms = baseline_ms * kAttentionLayers;
        const double reused_six_ms = table_ms + table_qk_ms * kAttentionLayers;

        std::cout << "Metal fused RoPE-table QK+KV A/B on " << ns_string(device.name) << "\n"
                  << "rows=512 q_heads=16 kv_heads=8 head_dim=64 split_half layers=6\n\n"
                  << std::fixed << std::setprecision(6)
                  << "query_max_abs=" << query_error
                  << " key_max_abs=" << key_error
                  << " key_cache_max_abs=" << key_cache_error
                  << " value_cache_max_abs=" << value_cache_error << "\n"
                  << "bit_exact query=" << (query_exact ? "yes" : "no")
                  << " key=" << (key_exact ? "yes" : "no")
                  << " key_cache=" << (key_cache_exact ? "yes" : "no")
                  << " value_cache=" << (value_cache_exact ? "yes" : "no") << "\n\n"
                  << "baseline_store_kv_ms=" << baseline_ms << "\n"
                  << "table_generation_ms=" << table_ms << "\n"
                  << "table_qk_store_kv_ms=" << table_qk_ms << "\n"
                  << "table_combined_one_layer_ms=" << combined_ms << "\n"
                  << "one_layer_speedup=" << (baseline_ms / combined_ms) << "x\n"
                  << "baseline_6_layers_ms=" << baseline_six_ms << "\n"
                  << "table_reused_6_layers_ms=" << reused_six_ms << "\n"
                  << "six_layer_reuse_speedup=" << (baseline_six_ms / reused_six_ms) << "x\n";

        const double tolerance = 1.0e-6;
        return (query_error <= tolerance && key_error <= tolerance &&
                key_cache_error <= tolerance && value_cache_error == 0.0) ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
