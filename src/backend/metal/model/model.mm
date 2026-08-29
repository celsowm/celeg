#include "celeg/backend/metal/model.hpp"

#include "celeg/backend/cpu/gguf.hpp"
#include "celeg/backend/cpu/sampler.hpp"
#include "celeg/detail/checkpoint/bootstrap.hpp"
#include "celeg/model/program.hpp"
#include "celeg/model/weights/roles.hpp"

#include "metal_inference_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace celeg {

namespace {

std::string ns_string(NSString* value) {
    return value ? std::string(value.UTF8String) : std::string{};
}

float bf16_to_float(uint16_t bits) {
    return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
}

float fp16_to_float(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1fu;
    const uint32_t fraction = bits & 0x3ffu;
    uint32_t value = 0;
    if (exponent == 0) {
        if (fraction == 0) return std::bit_cast<float>(sign);
        uint32_t normalized = fraction;
        int shift = 0;
        while ((normalized & 0x400u) == 0) {
            normalized <<= 1;
            ++shift;
        }
        normalized &= 0x3ffu;
        value = sign | static_cast<uint32_t>(127 - 15 - shift) << 23 |
                normalized << 13;
    } else if (exponent == 0x1fu) {
        value = sign | 0x7f800000u | fraction << 13;
    } else {
        value = sign | (exponent + 112u) << 23 | fraction << 13;
    }
    return std::bit_cast<float>(value);
}

size_t tensor_elements(const HostTensorView& view) {
    size_t count = 1;
    for (const int64_t dimension : view.shape) {
        if (dimension <= 0 || count > std::numeric_limits<size_t>::max() /
                                  static_cast<size_t>(dimension)) {
            throw std::runtime_error("invalid Metal tensor shape");
        }
        count *= static_cast<size_t>(dimension);
    }
    return count;
}

std::vector<float> tensor_values(const HostTensorView& view,
                                 std::span<const int64_t> expected,
                                 const std::string& name) {
    if (view.shape.size() != expected.size() ||
        !std::equal(view.shape.begin(), view.shape.end(), expected.begin())) {
        throw std::runtime_error("unexpected Metal tensor shape: " + name);
    }
    const size_t count = tensor_elements(view);
    std::vector<float> values(count);
    if (view.dtype == TensorDType::F32) {
        if (view.bytes != count * sizeof(float)) {
            throw std::runtime_error("invalid Metal F32 tensor: " + name);
        }
        std::memcpy(values.data(), view.data, view.bytes);
        return values;
    }
    if (view.dtype == TensorDType::BF16 || view.dtype == TensorDType::F16) {
        if (view.bytes != count * sizeof(uint16_t)) {
            throw std::runtime_error("invalid Metal 16-bit tensor: " + name);
        }
        for (size_t index = 0; index < count; ++index) {
            uint16_t bits = 0;
            std::memcpy(&bits, view.data + index * sizeof(bits), sizeof(bits));
            values[index] = view.dtype == TensorDType::BF16
                ? bf16_to_float(bits) : fp16_to_float(bits);
        }
        return values;
    }
    if (view.dtype == TensorDType::Quantized && expected.size() == 2) {
        CpuGgufMatrix matrix;
        matrix.type = ggml_type_from_block_encoding(view.block_encoding);
        matrix.rows = static_cast<uint32_t>(expected[0]);
        matrix.cols = static_cast<uint32_t>(expected[1]);
        matrix.data = view.data;
        matrix.bytes = view.bytes;
        matrix.validate();
        for (size_t row = 0; row < matrix.rows; ++row) {
            cpu_gguf_dequantize_row(matrix, row,
                                    values.data() + row * matrix.cols);
        }
        return values;
    }
    throw std::runtime_error("unsupported Metal tensor type: " + name);
}

const TensorRequest& request_for(std::span<const TensorRequest> requests,
                                 TensorRole role, int layer = -1,
                                 int expert = -1) {
    for (const TensorRequest& request : requests) {
        if (request.role == role && request.layer == layer && request.expert == expert) {
            return request;
        }
    }
    throw std::runtime_error("Metal weight request is missing: " +
                             std::string(tensor_role_name(role)));
}

std::string request_name(std::span<const TensorRequest> requests,
                         TensorRole role, int layer = -1, int expert = -1) {
    const TensorRequest& request = request_for(requests, role, layer, expert);
    if (!request.source_name) {
        throw std::runtime_error("Metal weight request was not resolved: " +
                                 std::string(tensor_role_name(role)));
    }
    return *request.source_name;
}

struct MetalMoeRoute {
    std::vector<int> experts;
    std::vector<float> weights;
};

float metal_moe_sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

MetalMoeRoute route_metal_moe(const RouterProgram& program,
                              std::span<const float> logits,
                              std::span<const float> expert_bias) {
    if (logits.size() != static_cast<size_t>(program.expert_count)) {
        throw std::invalid_argument("Metal MoE router width does not match program");
    }
    std::vector<float> probabilities(logits.size());
    const float maximum = program.score == MoeRouterScoreKind::SoftmaxLogits
        ? *std::max_element(logits.begin(), logits.end()) : 0.0f;
    float probability_sum = 0.0f;
    for (size_t index = 0; index < logits.size(); ++index) {
        probabilities[index] = program.score == MoeRouterScoreKind::SoftmaxLogits
            ? std::exp(logits[index] - maximum) : metal_moe_sigmoid(logits[index]);
        probability_sum += probabilities[index];
    }
    if (program.score == MoeRouterScoreKind::SoftmaxLogits) {
        for (float& probability : probabilities) probability /= probability_sum;
    }

    std::vector<std::pair<float, int>> scored(logits.size());
    for (size_t index = 0; index < logits.size(); ++index) {
        const float bias = program.has_expert_bias && index < expert_bias.size()
            ? expert_bias[index] : 0.0f;
        scored[index] = {probabilities[index] + bias, static_cast<int>(index)};
    }
    if (const auto* grouped =
            std::get_if<MoeGroupedTopKSelectionSpec>(&program.selection)) {
        std::vector<std::pair<float, int>> groups;
        groups.reserve(static_cast<size_t>(grouped->group_count));
        for (int group = 0; group < grouped->group_count; ++group) {
            std::vector<float> group_scores;
            group_scores.reserve(static_cast<size_t>(grouped->experts_per_group));
            const int first = group * grouped->experts_per_group;
            for (int offset = 0; offset < grouped->experts_per_group; ++offset) {
                group_scores.push_back(probabilities[static_cast<size_t>(first + offset)]);
            }
            const int score_count = std::min(grouped->group_score_top_k,
                                             grouped->experts_per_group);
            std::partial_sort(group_scores.begin(), group_scores.begin() + score_count,
                              group_scores.end(), std::greater<float>());
            float score = 0.0f;
            for (int index = 0; index < score_count; ++index) score += group_scores[index];
            groups.emplace_back(score, group);
        }
        std::partial_sort(groups.begin(),
                          groups.begin() + grouped->groups_per_token,
                          groups.end(), [](const auto& left, const auto& right) {
            return left.first == right.first ? left.second < right.second
                                             : left.first > right.first;
        });
        std::vector<bool> selected(static_cast<size_t>(grouped->group_count), false);
        for (int index = 0; index < grouped->groups_per_token; ++index) {
            selected[static_cast<size_t>(groups[static_cast<size_t>(index)].second)] = true;
        }
        for (auto& entry : scored) {
            if (!selected[static_cast<size_t>(
                    entry.second / grouped->experts_per_group)]) {
                entry.first = -std::numeric_limits<float>::infinity();
            }
        }
    }
    std::partial_sort(scored.begin(), scored.begin() + program.experts_per_token,
                      scored.end(), [](const auto& left, const auto& right) {
        return left.first == right.first ? left.second < right.second
                                         : left.first > right.first;
    });

    MetalMoeRoute result;
    result.experts.resize(static_cast<size_t>(program.experts_per_token));
    result.weights.resize(result.experts.size());
    float selected_sum = 0.0f;
    for (int route = 0; route < program.experts_per_token; ++route) {
        const int expert = scored[static_cast<size_t>(route)].second;
        result.experts[static_cast<size_t>(route)] = expert;
        result.weights[static_cast<size_t>(route)] = probabilities[static_cast<size_t>(expert)];
        selected_sum += result.weights[static_cast<size_t>(route)];
    }
    if (program.normalization == MoeNormalizationKind::SumSelected) {
        const float inverse = 1.0f / (selected_sum + 1.0e-6f);
        for (float& weight : result.weights) weight *= inverse;
    }
    for (float& weight : result.weights) weight *= program.routed_scaling;
    return result;
}

} 

struct MetalModel::Impl {
    enum class LinearStorage {
        Float32,
        Q4K,
        Q6K,
    };

    struct Linear {
        id<MTLBuffer> buffer = nil;
        uint32_t rows = 0;
        uint32_t cols = 0;
        uint32_t row_bytes = 0;
        LinearStorage storage = LinearStorage::Float32;
    };

    struct Layer {
        struct Expert {
            std::string gate_name;
            std::string up_name;
            std::string down_name;
        };

        struct Moe {
            RouterProgram router;
            std::vector<float> router_weight;
            std::vector<float> router_bias;
            std::vector<Expert> experts;
        };

        id<MTLBuffer> operator_norm = nil;
        id<MTLBuffer> ffn_norm = nil;
        bool convolution = false;
        bool gated_delta = false;
        bool mamba2 = false;
        int cache_length = 0;
        int page_tokens = 16;
        Linear mixer_in;
        Linear mixer_out;
        id<MTLBuffer> convolution_taps = nil;
        id<MTLBuffer> recurrent_conv_weight = nil;
        id<MTLBuffer> recurrent_conv_bias = nil;
        id<MTLBuffer> recurrent_dt_bias = nil;
        id<MTLBuffer> recurrent_a_log = nil;
        id<MTLBuffer> recurrent_d = nil;
        id<MTLBuffer> recurrent_norm = nil;
        id<MTLBuffer> recurrent_conv_state = nil;
        id<MTLBuffer> recurrent_state = nil;
        int recurrent_conv_kernel = 0;
        int recurrent_key_head_dim = 0;
        int recurrent_value_head_dim = 0;
        int recurrent_key_heads = 0;
        int recurrent_value_heads = 0;
        int recurrent_inner = 0;
        int recurrent_state_size = 0;
        int recurrent_group_count = 0;
        bool recurrent_vector_decay = false;
        bool recurrent_safe_decay = false;
        float recurrent_decay_lower_bound = -5.0f;
        bool recurrent_sigmoid_output_gate = false;
        bool recurrent_a_log_needs_exp = true;
        Linear recurrent_in;
        Linear recurrent_qkv;
        Linear recurrent_q;
        Linear recurrent_k;
        Linear recurrent_v;
        Linear recurrent_z_weight;
        Linear recurrent_b;
        Linear recurrent_a;
        Linear recurrent_out;
        Linear query;
        Linear key;
        Linear value;
        Linear attention_out;
        id<MTLBuffer> query_norm = nil;
        id<MTLBuffer> key_norm = nil;
        id<MTLBuffer> key_cache = nil;
        id<MTLBuffer> value_cache = nil;
        int query_heads = 0;
        int key_value_heads = 0;
        int head_dim = 0;
        float query_scale = 1.0f;
        float rope_theta = 10000.0f;
        float query_norm_epsilon = 1.0e-5f;
        float key_norm_epsilon = 1.0e-5f;
        Linear ffn_gate;
        Linear ffn_up;
        Linear ffn_down;
        int intermediate = 0;
        std::optional<Moe> moe;
    };

    std::string model_path;
    int max_context = 0;
    MetalModelOptions options;
    std::shared_ptr<const RuntimeContext> runtime;
    ResolvedModel model;
    CompiledModelProgram program;
    std::shared_ptr<IWeightRepository> repository;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
    Linear embedding;
    id<MTLBuffer> final_norm = nil;
    id<MTLBuffer> hidden = nil;
    id<MTLBuffer> residual = nil;
    id<MTLBuffer> normed = nil;
    id<MTLBuffer> projected = nil;
    id<MTLBuffer> operation = nil;
    id<MTLBuffer> query_buffer = nil;
    id<MTLBuffer> key_buffer = nil;
    id<MTLBuffer> value_buffer = nil;
    id<MTLBuffer> gate_up = nil;
    id<MTLBuffer> activated = nil;
    id<MTLBuffer> moe_output = nil;
    id<MTLBuffer> recurrent_z = nil;
    id<MTLBuffer> recurrent_b = nil;
    id<MTLBuffer> recurrent_a = nil;
    id<MTLBuffer> recurrent_output = nil;
    id<MTLBuffer> logits = nil;
    std::vector<Layer> layers;
    std::vector<uint8_t> seen;
    GenerationConfig generation;
    uint64_t rng_state = 1;
    int position = 0;
    bool ready = false;
    RuntimeMetrics metrics;

    id<MTLBuffer> buffer(const std::vector<float>& values) const {
        id<MTLBuffer> result = [device newBufferWithBytes:values.data()
                                                    length:values.size() * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        if (!result) throw std::runtime_error("Metal buffer allocation failed");
        return result;
    }

    id<MTLBuffer> zero_buffer(size_t bytes) const {
        id<MTLBuffer> result = [device newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
        if (!result) throw std::runtime_error("Metal state allocation failed");
        std::memset(result.contents, 0, bytes);
        return result;
    }

    id<MTLBuffer> raw_buffer(const std::byte* data, size_t bytes) const {
        id<MTLBuffer> result = [device newBufferWithBytes:data
                                                    length:bytes
                                                   options:MTLResourceStorageModeShared];
        if (!result) throw std::runtime_error("Metal raw buffer allocation failed");
        return result;
    }

    std::vector<float> load_vector_values(TensorRole role, int layer, int width,
                                          bool allow_missing = false) const {
        const TensorRequest* selected = nullptr;
        for (const TensorRequest& request : model.weight_plan.requests) {
            if (request.role == role && request.layer == layer && request.expert < 0) {
                selected = &request;
                break;
            }
        }
        if (!selected) {
            if (!allow_missing) throw std::runtime_error("Metal vector request is missing");
            return std::vector<float>(static_cast<size_t>(width), 1.0f);
        }
        if (!selected->source_name) {
            throw std::runtime_error("Metal vector request was not resolved");
        }
        const std::string& name = *selected->source_name;
        std::vector<float> values = tensor_values(repository->tensor(name),
            std::span<const int64_t>(selected->expected_shape.data(),
                                     selected->expected_shape.size()), name);
        if (values.size() != static_cast<size_t>(width)) {
            throw std::runtime_error("Metal vector width mismatch: " + name);
        }
        if (selected->norm_weight_kind == NormWeightKind::OnePlusScale) {
            for (float& value : values) value += 1.0f;
        }
        return values;
    }

    id<MTLBuffer> load_vector(TensorRole role, int layer, int width,
                              bool allow_missing = false) const {
        return buffer(load_vector_values(role, layer, width, allow_missing));
    }

    Linear load_linear(TensorRole role, int layer, int rows, int cols) const {
        const TensorRequest& request = request_for(model.weight_plan.requests, role, layer);
        if (!request.source_name) throw std::runtime_error("Metal matrix request was not resolved");
        const std::string& name = *request.source_name;
        return load_linear_source(name, rows, cols);
    }

    Linear load_linear_source(const std::string& name, int rows, int cols) const {
        const HostTensorView view = repository->tensor(name);
        const std::vector<int64_t> expected_values{rows, cols};
        const std::span<const int64_t> expected(expected_values.data(),
                                                 expected_values.size());
        if (view.shape.size() != expected.size() ||
            !std::equal(view.shape.begin(), view.shape.end(), expected.begin())) {
            throw std::runtime_error("unexpected Metal tensor shape: " + name);
        }
        if (view.dtype == TensorDType::Quantized) {
            const GgmlType type = ggml_type_from_block_encoding(view.block_encoding);
            if ((type == GgmlType::Q4_K || type == GgmlType::Q6_K) &&
                cols % 256 == 0) {
                const GgmlTypeTrait trait = ggml_type_trait(type);
                const size_t row_bytes = static_cast<size_t>(cols) /
                    static_cast<size_t>(trait.block_size) *
                    static_cast<size_t>(trait.type_size);
                if (view.bytes != static_cast<size_t>(rows) * row_bytes) {
                    throw std::runtime_error("invalid Metal quantized tensor bytes: " + name);
                }
                return {raw_buffer(view.data, view.bytes),
                        static_cast<uint32_t>(rows), static_cast<uint32_t>(cols),
                        static_cast<uint32_t>(row_bytes),
                        type == GgmlType::Q4_K ? LinearStorage::Q4K
                                               : LinearStorage::Q6K};
            }
        }
        std::vector<float> values = tensor_values(view, expected, name);
        if (values.size() != static_cast<size_t>(rows) * cols) {
            throw std::runtime_error("Metal matrix dimensions mismatch: " + name);
        }
        return {buffer(values), static_cast<uint32_t>(rows), static_cast<uint32_t>(cols),
                0, LinearStorage::Float32};
    }

    id<MTLBuffer> load_conv_kernel(int layer, int width, int cache_length) const {
        const TensorRequest& request = request_for(model.weight_plan.requests,
                                                   TensorRole::ShortConvKernel, layer);
        if (!request.source_name) throw std::runtime_error("Metal convolution request was not resolved");
        const std::string& name = *request.source_name;
        std::vector<float> channel_major = tensor_values(repository->tensor(name),
            std::span<const int64_t>(request.expected_shape.data(), request.expected_shape.size()), name);
        std::vector<float> tap_major(channel_major.size());
        for (int tap = 0; tap < cache_length; ++tap) {
            for (int channel = 0; channel < width; ++channel) {
                tap_major[static_cast<size_t>(tap) * width + channel] =
                    channel_major[static_cast<size_t>(channel) * cache_length + tap];
            }
        }
        return buffer(tap_major);
    }

    id<MTLComputePipelineState> pipeline(std::string_view name) {
        const auto found = pipelines.find(std::string(name));
        if (found != pipelines.end()) return found->second;
        NSString* function_name = [NSString stringWithUTF8String:std::string(name).c_str()];
        id<MTLFunction> function = [library newFunctionWithName:function_name];
        if (!function) throw std::runtime_error("Metal inference function is missing: " + std::string(name));
        NSError* error = nil;
        id<MTLComputePipelineState> result =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!result) {
            const std::string message = error ? ns_string(error.localizedDescription)
                                              : "unknown Metal pipeline error";
            throw std::runtime_error("Metal inference pipeline failed: " + message);
        }
        pipelines.emplace(std::string(name), result);
        return result;
    }

    void dispatch(id<MTLComputeCommandEncoder> encoder, std::string_view name,
                  NSUInteger count) {
        if (count == 0) return;
        id<MTLComputePipelineState> state = pipeline(name);
        [encoder setComputePipelineState:state];
        const NSUInteger threads = std::min(count, state.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    }

    void set_buffer(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> value,
                    NSUInteger index, NSUInteger offset = 0) {
        [encoder setBuffer:value offset:offset atIndex:index];
    }

    void set_bytes(id<MTLComputeCommandEncoder> encoder, const void* value,
                   NSUInteger length, NSUInteger index) {
        [encoder setBytes:value length:length atIndex:index];
    }

    void encode_matvec(id<MTLComputeCommandEncoder> encoder, const Linear& weight,
                       id<MTLBuffer> input, id<MTLBuffer> output,
                       NSUInteger output_offset = 0) {
        set_buffer(encoder, weight.buffer, 0);
        set_buffer(encoder, input, 1);
        set_buffer(encoder, output, 2, output_offset);
        set_bytes(encoder, &weight.rows, sizeof(weight.rows), 3);
        set_bytes(encoder, &weight.cols, sizeof(weight.cols), 4);
        if (weight.storage == LinearStorage::Float32) {
            dispatch(encoder, "celeg_matvec", weight.rows);
            return;
        }
        set_bytes(encoder, &weight.row_bytes, sizeof(weight.row_bytes), 5);
        dispatch(encoder, weight.storage == LinearStorage::Q4K
                           ? "celeg_matvec_q4k" : "celeg_matvec_q6k",
                 weight.rows);
    }

    void encode_embedding(id<MTLComputeCommandEncoder> encoder, uint32_t width,
                          uint32_t token) {
        if (embedding.storage == LinearStorage::Float32) {
            set_buffer(encoder, embedding.buffer, 0);
            set_buffer(encoder, hidden, 1);
            set_bytes(encoder, &width, sizeof(width), 2);
            set_bytes(encoder, &token, sizeof(token), 3);
            dispatch(encoder, "celeg_embedding", width);
            return;
        }
        set_buffer(encoder, embedding.buffer, 0);
        set_buffer(encoder, hidden, 1);
        set_bytes(encoder, &width, sizeof(width), 2);
        set_bytes(encoder, &token, sizeof(token), 3);
        dispatch(encoder, embedding.storage == LinearStorage::Q4K
                           ? "celeg_embedding_q4k" : "celeg_embedding_q6k",
                 width);
    }

    void encode_rmsnorm(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> input,
                        id<MTLBuffer> weight, id<MTLBuffer> output, uint32_t width,
                        float epsilon) {
        set_buffer(encoder, input, 0);
        set_buffer(encoder, weight, 1);
        set_buffer(encoder, output, 2);
        set_bytes(encoder, &width, sizeof(width), 3);
        set_bytes(encoder, &epsilon, sizeof(epsilon), 4);
        dispatch(encoder, "celeg_rmsnorm", 1);
    }

    void encode_weighted_add(id<MTLComputeCommandEncoder> encoder,
                             id<MTLBuffer> input, id<MTLBuffer> output,
                             uint32_t count, float weight) {
        set_buffer(encoder, input, 0);
        set_buffer(encoder, output, 1);
        set_bytes(encoder, &count, sizeof(count), 2);
        set_bytes(encoder, &weight, sizeof(weight), 3);
        dispatch(encoder, "celeg_weighted_add", count);
    }

    void encode_gated_delta(id<MTLComputeCommandEncoder> encoder, Layer& layer) {
        const uint32_t conv_kernel = static_cast<uint32_t>(layer.recurrent_conv_kernel);
        const uint32_t key_head_dim = static_cast<uint32_t>(layer.recurrent_key_head_dim);
        const uint32_t value_head_dim = static_cast<uint32_t>(layer.recurrent_value_head_dim);
        const uint32_t key_heads = static_cast<uint32_t>(layer.recurrent_key_heads);
        const uint32_t value_heads = static_cast<uint32_t>(layer.recurrent_value_heads);
        const float epsilon = program.final_norm.epsilon;
        const uint32_t vector_decay = layer.recurrent_vector_decay ? 1 : 0;
        const uint32_t safe_decay = layer.recurrent_safe_decay ? 1 : 0;
        const uint32_t sigmoid_output_gate = layer.recurrent_sigmoid_output_gate ? 1 : 0;
        const uint32_t a_log_needs_exp = layer.recurrent_a_log_needs_exp ? 1 : 0;
        set_buffer(encoder, projected, 0);
        set_buffer(encoder, recurrent_z, 1);
        set_buffer(encoder, recurrent_b, 2);
        set_buffer(encoder, recurrent_a, 3);
        set_buffer(encoder, layer.recurrent_conv_weight, 4);
        set_buffer(encoder, layer.recurrent_dt_bias, 5);
        set_buffer(encoder, layer.recurrent_a_log, 6);
        set_buffer(encoder, layer.recurrent_norm, 7);
        set_buffer(encoder, layer.recurrent_conv_state, 8);
        set_buffer(encoder, layer.recurrent_state, 9);
        set_buffer(encoder, recurrent_output, 10);
        set_bytes(encoder, &conv_kernel, sizeof(conv_kernel), 11);
        set_bytes(encoder, &key_head_dim, sizeof(key_head_dim), 12);
        set_bytes(encoder, &value_head_dim, sizeof(value_head_dim), 13);
        set_bytes(encoder, &key_heads, sizeof(key_heads), 14);
        set_bytes(encoder, &value_heads, sizeof(value_heads), 15);
        set_bytes(encoder, &epsilon, sizeof(epsilon), 16);
        set_bytes(encoder, &vector_decay, sizeof(vector_decay), 17);
        set_bytes(encoder, &safe_decay, sizeof(safe_decay), 18);
        set_bytes(encoder, &layer.recurrent_decay_lower_bound,
                  sizeof(layer.recurrent_decay_lower_bound), 19);
        set_bytes(encoder, &sigmoid_output_gate, sizeof(sigmoid_output_gate), 20);
        set_bytes(encoder, &a_log_needs_exp, sizeof(a_log_needs_exp), 21);
        dispatch(encoder, "celeg_gated_delta", 1);
    }

    void encode_mamba2(id<MTLComputeCommandEncoder> encoder, Layer& layer) {
        const uint32_t inner = static_cast<uint32_t>(layer.recurrent_inner);
        const uint32_t state_size = static_cast<uint32_t>(layer.recurrent_state_size);
        const uint32_t num_heads = static_cast<uint32_t>(layer.recurrent_value_heads);
        const uint32_t head_dim = static_cast<uint32_t>(layer.recurrent_key_head_dim);
        const uint32_t group_count = static_cast<uint32_t>(layer.recurrent_group_count);
        const uint32_t conv_kernel = static_cast<uint32_t>(layer.recurrent_conv_kernel);
        const float epsilon = program.final_norm.epsilon;
        const uint32_t a_log_needs_exp = layer.recurrent_a_log_needs_exp ? 1 : 0;
        set_buffer(encoder, projected, 0);
        set_buffer(encoder, layer.recurrent_conv_weight, 1);
        set_buffer(encoder, layer.recurrent_conv_bias, 2);
        set_buffer(encoder, layer.recurrent_dt_bias, 3);
        set_buffer(encoder, layer.recurrent_a_log, 4);
        set_buffer(encoder, layer.recurrent_d, 5);
        set_buffer(encoder, layer.recurrent_norm, 6);
        set_buffer(encoder, layer.recurrent_conv_state, 7);
        set_buffer(encoder, layer.recurrent_state, 8);
        set_buffer(encoder, recurrent_output, 9);
        set_bytes(encoder, &inner, sizeof(inner), 10);
        set_bytes(encoder, &state_size, sizeof(state_size), 11);
        set_bytes(encoder, &num_heads, sizeof(num_heads), 12);
        set_bytes(encoder, &head_dim, sizeof(head_dim), 13);
        set_bytes(encoder, &group_count, sizeof(group_count), 14);
        set_bytes(encoder, &conv_kernel, sizeof(conv_kernel), 15);
        set_bytes(encoder, &epsilon, sizeof(epsilon), 16);
        set_bytes(encoder, &a_log_needs_exp, sizeof(a_log_needs_exp), 17);
        dispatch(encoder, "celeg_mamba2", 1);
    }

    void run_token(int32_t token) {
        if (position >= max_context) throw std::runtime_error("Metal context limit reached");
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        auto begin_commands = [&]() {
            command_buffer = [queue commandBuffer];
            if (!command_buffer) {
                throw std::runtime_error("Metal command buffer creation failed");
            }
            encoder = [command_buffer computeCommandEncoder];
            if (!encoder) {
                throw std::runtime_error("Metal compute encoder creation failed");
            }
        };
        auto finish_commands = [&]() {
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.status != MTLCommandBufferStatusCompleted) {
                const std::string message = command_buffer.error
                    ? ns_string(command_buffer.error.localizedDescription)
                    : "unknown Metal command-buffer error";
                throw std::runtime_error("Metal inference dispatch failed: " + message);
            }
            encoder = nil;
            command_buffer = nil;
        };
        begin_commands();

        const uint32_t hidden_width = static_cast<uint32_t>(model.graph.hidden);
        const uint32_t token_value = static_cast<uint32_t>(token);
        encode_embedding(encoder, hidden_width, token_value);
        if (model.graph.embedding_transform.multiplier != 1.0f) {
            const uint32_t count = hidden_width;
            const float multiplier = model.graph.embedding_transform.multiplier;
            set_buffer(encoder, hidden, 0);
            set_bytes(encoder, &count, sizeof(count), 1);
            set_bytes(encoder, &multiplier, sizeof(multiplier), 2);
            dispatch(encoder, "celeg_scale", count);
        }
        if (model.graph.embedding_transform.post_norm) {
            encode_rmsnorm(encoder, hidden, final_norm, operation, hidden_width,
                           model.graph.embedding_transform.post_norm->epsilon);
            set_buffer(encoder, operation, 0);
            set_buffer(encoder, hidden, 1);
            set_bytes(encoder, &hidden_width, sizeof(hidden_width), 2);
            dispatch(encoder, "celeg_copy", hidden_width);
        }

        for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            Layer& layer = layers[layer_index];
            set_buffer(encoder, hidden, 0);
            set_buffer(encoder, residual, 1);
            set_bytes(encoder, &hidden_width, sizeof(hidden_width), 2);
            dispatch(encoder, "celeg_copy", hidden_width);
            encode_rmsnorm(encoder, hidden, layer.operator_norm, normed,
                           hidden_width, model.graph.final_norm.epsilon);

            if (layer.convolution) {
                encode_matvec(encoder, layer.mixer_in, normed, projected);
                const uint32_t cache_length = static_cast<uint32_t>(layer.cache_length);
                const uint32_t position_value = static_cast<uint32_t>(position);
                set_buffer(encoder, projected, 0);
                set_buffer(encoder, layer.convolution_taps, 1);
                set_buffer(encoder, layer.key_cache, 2);
                set_buffer(encoder, operation, 3);
                set_bytes(encoder, &hidden_width, sizeof(hidden_width), 4);
                set_bytes(encoder, &cache_length, sizeof(cache_length), 5);
                set_bytes(encoder, &position_value, sizeof(position_value), 6);
                dispatch(encoder, "celeg_shortconv", hidden_width);
                encode_matvec(encoder, layer.mixer_out, operation, hidden);
            } else if (layer.gated_delta) {
                const uint32_t key_width = static_cast<uint32_t>(
                    layer.recurrent_key_heads * layer.recurrent_key_head_dim);
                const uint32_t value_width = static_cast<uint32_t>(
                    layer.recurrent_value_heads * layer.recurrent_value_head_dim);
                if (layer.recurrent_qkv.buffer) {
                    encode_matvec(encoder, layer.recurrent_qkv, normed, projected);
                } else {
                    encode_matvec(encoder, layer.recurrent_q, normed, projected, 0);
                    encode_matvec(encoder, layer.recurrent_k, normed, projected,
                                  static_cast<NSUInteger>(key_width) * sizeof(float));
                    encode_matvec(encoder, layer.recurrent_v, normed, projected,
                                  static_cast<NSUInteger>(2 * key_width) * sizeof(float));
                }
                encode_matvec(encoder, layer.recurrent_z_weight, normed, recurrent_z);
                encode_matvec(encoder, layer.recurrent_b, normed, recurrent_b);
                encode_matvec(encoder, layer.recurrent_a, normed, recurrent_a);
                encode_gated_delta(encoder, layer);
                encode_matvec(encoder, layer.recurrent_out, recurrent_output, hidden);
            } else if (layer.mamba2) {
                encode_matvec(encoder, layer.recurrent_in, normed, projected);
                encode_mamba2(encoder, layer);
                encode_matvec(encoder, layer.recurrent_out, recurrent_output, hidden);
            } else {
                encode_matvec(encoder, layer.query, normed, query_buffer);
                encode_matvec(encoder, layer.key, normed, key_buffer);
                encode_matvec(encoder, layer.value, normed, value_buffer);
                const uint32_t query_heads = static_cast<uint32_t>(layer.query_heads);
                const uint32_t key_heads = static_cast<uint32_t>(layer.key_value_heads);
                const uint32_t head_dim = static_cast<uint32_t>(layer.head_dim);
                const uint32_t position_value = static_cast<uint32_t>(position);
                const float query_scale = layer.query_scale /
                    (1.0f / std::sqrt(static_cast<float>(layer.head_dim)));
                set_buffer(encoder, query_buffer, 0);
                set_buffer(encoder, layer.query_norm, 1);
                set_buffer(encoder, key_buffer, 2);
                set_buffer(encoder, layer.key_norm, 3);
                set_bytes(encoder, &query_heads, sizeof(query_heads), 4);
                set_bytes(encoder, &key_heads, sizeof(key_heads), 5);
                set_bytes(encoder, &head_dim, sizeof(head_dim), 6);
                set_bytes(encoder, &position_value, sizeof(position_value), 7);
                set_bytes(encoder, &layer.rope_theta, sizeof(layer.rope_theta), 8);
                set_bytes(encoder, &query_scale, sizeof(query_scale), 9);
                set_bytes(encoder, &layer.query_norm_epsilon,
                          sizeof(layer.query_norm_epsilon), 10);
                set_bytes(encoder, &layer.key_norm_epsilon,
                          sizeof(layer.key_norm_epsilon), 11);
                dispatch(encoder, "celeg_qk_norm_rope",
                         std::max(query_heads, key_heads));
                const uint32_t kv_width = key_heads * head_dim;
                set_buffer(encoder, key_buffer, 0);
                set_buffer(encoder, value_buffer, 1);
                set_buffer(encoder, layer.key_cache, 2);
                set_buffer(encoder, layer.value_cache, 3);
                set_bytes(encoder, &position_value, sizeof(position_value), 4);
                set_bytes(encoder, &kv_width, sizeof(kv_width), 5);
                const uint32_t page_tokens = static_cast<uint32_t>(layer.page_tokens);
                set_bytes(encoder, &page_tokens, sizeof(page_tokens), 6);
                dispatch(encoder, "celeg_store_kv", kv_width);
                const float attention_scale = 1.0f /
                    std::sqrt(static_cast<float>(layer.head_dim));
                set_buffer(encoder, query_buffer, 0);
                set_buffer(encoder, layer.key_cache, 1);
                set_buffer(encoder, layer.value_cache, 2);
                set_buffer(encoder, operation, 3);
                const uint32_t sequence_length = position_value + 1;
                set_bytes(encoder, &sequence_length, sizeof(sequence_length), 4);
                set_bytes(encoder, &query_heads, sizeof(query_heads), 5);
                set_bytes(encoder, &key_heads, sizeof(key_heads), 6);
                set_bytes(encoder, &head_dim, sizeof(head_dim), 7);
                set_bytes(encoder, &attention_scale, sizeof(attention_scale), 8);
                set_bytes(encoder, &page_tokens, sizeof(page_tokens), 9);
                dispatch(encoder, "celeg_attention", query_heads * head_dim);
                encode_matvec(encoder, layer.attention_out, operation, hidden);
            }

            const uint32_t count = hidden_width;
            const float mixer_multiplier = 1.0f;
            set_buffer(encoder, hidden, 0);
            set_buffer(encoder, residual, 1);
            set_buffer(encoder, hidden, 2);
            set_bytes(encoder, &count, sizeof(count), 3);
            set_bytes(encoder, &mixer_multiplier, sizeof(mixer_multiplier), 4);
            dispatch(encoder, "celeg_residual", count);

            if (std::holds_alternative<std::monostate>(
                    program.layers[layer_index].feed_forward)) {
                continue;
            }

            encode_rmsnorm(encoder, hidden, layer.ffn_norm, normed, hidden_width,
                           program.final_norm.epsilon);
            if (layer.moe) {
                finish_commands();
                std::vector<float> router_logits(
                    static_cast<size_t>(layer.moe->router.expert_count));
                for (int expert = 0; expert < layer.moe->router.expert_count; ++expert) {
                    float sum = 0.0f;
                    const size_t base = static_cast<size_t>(expert) * hidden_width;
                    for (uint32_t index = 0; index < hidden_width; ++index) {
                        sum += layer.moe->router_weight[base + index] *
                               static_cast<const float*>(normed.contents)[index];
                    }
                    router_logits[static_cast<size_t>(expert)] = sum;
                }
                const MetalMoeRoute route = route_metal_moe(
                    layer.moe->router, router_logits, layer.moe->router_bias);
                std::memset(moe_output.contents, 0,
                            static_cast<size_t>(hidden_width) * sizeof(float));
                begin_commands();
                const uint32_t intermediate = static_cast<uint32_t>(layer.intermediate);
                for (size_t route_index = 0; route_index < route.experts.size(); ++route_index) {
                    const int expert = route.experts[route_index];
                    const Impl::Layer::Expert& names =
                        layer.moe->experts[static_cast<size_t>(expert)];
                    const Linear gate = load_linear_source(
                        names.gate_name, layer.intermediate, hidden_width);
                    const Linear up = load_linear_source(
                        names.up_name, layer.intermediate, hidden_width);
                    const Linear down = load_linear_source(
                        names.down_name, hidden_width, layer.intermediate);
                    encode_matvec(encoder, gate, normed, gate_up, 0);
                    encode_matvec(encoder, up, normed, gate_up,
                                  static_cast<NSUInteger>(layer.intermediate) * sizeof(float));
                    set_buffer(encoder, gate_up, 0);
                    set_buffer(encoder, activated, 1);
                    set_bytes(encoder, &intermediate, sizeof(intermediate), 2);
                    dispatch(encoder, "celeg_swiglu", intermediate);
                    encode_matvec(encoder, down, activated, operation);
                    encode_weighted_add(encoder, operation, moe_output, hidden_width,
                                        route.weights[route_index]);
                }
            } else {
                encode_matvec(encoder, layer.ffn_gate, normed, gate_up, 0);
                encode_matvec(encoder, layer.ffn_up, normed, gate_up,
                              static_cast<NSUInteger>(layer.intermediate) * sizeof(float));
                const uint32_t intermediate = static_cast<uint32_t>(layer.intermediate);
                set_buffer(encoder, gate_up, 0);
                set_buffer(encoder, activated, 1);
                set_bytes(encoder, &intermediate, sizeof(intermediate), 2);
                dispatch(encoder, "celeg_swiglu", intermediate);
                encode_matvec(encoder, layer.ffn_down, activated, operation);
            }
            set_buffer(encoder, layer.moe ? moe_output : operation, 0);
            set_buffer(encoder, hidden, 1);
            set_buffer(encoder, hidden, 2);
            set_bytes(encoder, &count, sizeof(count), 3);
            set_bytes(encoder, &mixer_multiplier, sizeof(mixer_multiplier), 4);
            dispatch(encoder, "celeg_residual", count);
        }

        encode_rmsnorm(encoder, hidden, final_norm, normed, hidden_width,
                       program.final_norm.epsilon);
        encode_matvec(encoder, embedding, normed, logits);
        finish_commands();
        ++position;
    }

    void reset() {
        position = 0;
        ready = false;
        metrics = {};
        std::fill(seen.begin(), seen.end(), 0);
        std::memset(hidden.contents, 0, model.graph.hidden * sizeof(float));
        std::memset(residual.contents, 0, model.graph.hidden * sizeof(float));
        for (Layer& layer : layers) {
            if (layer.convolution) {
                std::memset(layer.key_cache.contents, 0,
                            static_cast<size_t>(layer.cache_length) *
                                model.graph.hidden * sizeof(float));
            } else if (layer.gated_delta || layer.mamba2) {
                const size_t conv_elements = layer.gated_delta
                    ? static_cast<size_t>(2 * layer.recurrent_key_heads *
                        layer.recurrent_key_head_dim +
                        layer.recurrent_value_heads * layer.recurrent_value_head_dim) *
                        layer.recurrent_conv_kernel
                    : static_cast<size_t>(layer.recurrent_inner +
                        2 * layer.recurrent_group_count * layer.recurrent_state_size) *
                        layer.recurrent_conv_kernel;
                const size_t recurrent_elements = layer.gated_delta
                    ? static_cast<size_t>(layer.recurrent_value_heads) *
                        layer.recurrent_key_head_dim * layer.recurrent_value_head_dim
                    : static_cast<size_t>(layer.recurrent_inner) *
                        layer.recurrent_state_size;
                std::memset(layer.recurrent_conv_state.contents, 0,
                            conv_elements * sizeof(float));
                std::memset(layer.recurrent_state.contents, 0,
                            recurrent_elements * sizeof(float));
            } else {
                const size_t pages =
                    (static_cast<size_t>(max_context) +
                     static_cast<size_t>(layer.page_tokens) - 1) /
                    static_cast<size_t>(layer.page_tokens);
                const size_t elements = pages *
                    static_cast<size_t>(layer.page_tokens) *
                    static_cast<size_t>(layer.key_value_heads) *
                    static_cast<size_t>(layer.head_dim);
                std::memset(layer.key_cache.contents, 0, elements * sizeof(float));
                std::memset(layer.value_cache.contents, 0, elements * sizeof(float));
            }
        }
    }

    static std::vector<float> copy_buffer(id<MTLBuffer> source, size_t elements) {
        std::vector<float> result(elements);
        if (elements != 0) {
            std::memcpy(result.data(), source.contents, elements * sizeof(float));
        }
        return result;
    }

    MetalSessionSnapshot export_snapshot() const {
        MetalSessionSnapshot snapshot;
        snapshot.position = position;
        snapshot.ready = ready;
        snapshot.generation = generation;
        snapshot.rng_state = rng_state;
        snapshot.metrics = metrics;
        snapshot.seen = seen;
        snapshot.hidden = copy_buffer(hidden, static_cast<size_t>(model.graph.hidden));
        snapshot.residual = copy_buffer(residual, static_cast<size_t>(model.graph.hidden));
        snapshot.logits = copy_buffer(logits,
                                      static_cast<size_t>(model.topology.dims.vocab_size));
        snapshot.key_state.reserve(layers.size());
        snapshot.value_state.reserve(layers.size());
        snapshot.mixer_state.reserve(layers.size());
        snapshot.recurrent_state.reserve(layers.size());
        for (const Layer& layer : layers) {
            if (layer.convolution) {
                snapshot.key_state.push_back(copy_buffer(
                    layer.key_cache, static_cast<size_t>(layer.cache_length) *
                        static_cast<size_t>(model.graph.hidden)));
                snapshot.value_state.emplace_back();
            } else if (!layer.gated_delta && !layer.mamba2) {
                const size_t pages =
                    (static_cast<size_t>(max_context) +
                     static_cast<size_t>(layer.page_tokens) - 1) /
                    static_cast<size_t>(layer.page_tokens);
                const size_t elements = pages *
                    static_cast<size_t>(layer.page_tokens) *
                    static_cast<size_t>(layer.key_value_heads) *
                    static_cast<size_t>(layer.head_dim);
                snapshot.key_state.push_back(copy_buffer(layer.key_cache, elements));
                snapshot.value_state.push_back(copy_buffer(layer.value_cache, elements));
            } else {
                snapshot.key_state.emplace_back();
                snapshot.value_state.emplace_back();
            }
            if (layer.gated_delta || layer.mamba2) {
                const size_t conv_elements = layer.gated_delta
                    ? static_cast<size_t>(2 * layer.recurrent_key_heads *
                        layer.recurrent_key_head_dim +
                        layer.recurrent_value_heads * layer.recurrent_value_head_dim) *
                        layer.recurrent_conv_kernel
                    : static_cast<size_t>(layer.recurrent_inner +
                        2 * layer.recurrent_group_count * layer.recurrent_state_size) *
                        layer.recurrent_conv_kernel;
                const size_t recurrent_elements = layer.gated_delta
                    ? static_cast<size_t>(layer.recurrent_value_heads) *
                        layer.recurrent_key_head_dim * layer.recurrent_value_head_dim
                    : static_cast<size_t>(layer.recurrent_inner) *
                        layer.recurrent_state_size;
                snapshot.mixer_state.push_back(copy_buffer(
                    layer.recurrent_conv_state, conv_elements));
                snapshot.recurrent_state.push_back(copy_buffer(
                    layer.recurrent_state, recurrent_elements));
            } else {
                snapshot.mixer_state.emplace_back();
                snapshot.recurrent_state.emplace_back();
            }
        }
        return snapshot;
    }

    void restore_snapshot(MetalSessionSnapshot snapshot) {
        snapshot.generation.validate();
        if (snapshot.position < 0 || snapshot.position > max_context ||
            snapshot.seen.size() != static_cast<size_t>(model.topology.dims.vocab_size) ||
            snapshot.hidden.size() != static_cast<size_t>(model.graph.hidden) ||
            snapshot.residual.size() != static_cast<size_t>(model.graph.hidden) ||
            snapshot.logits.size() != static_cast<size_t>(model.topology.dims.vocab_size) ||
            snapshot.key_state.size() != layers.size() ||
            snapshot.value_state.size() != layers.size() ||
            snapshot.mixer_state.size() != layers.size() ||
            snapshot.recurrent_state.size() != layers.size()) {
            throw std::invalid_argument("invalid Metal session snapshot dimensions");
        }
        position = snapshot.position;
        ready = snapshot.ready;
        generation = std::move(snapshot.generation);
        rng_state = snapshot.rng_state;
        metrics = snapshot.metrics;
        seen = std::move(snapshot.seen);
        std::memcpy(hidden.contents, snapshot.hidden.data(),
                    snapshot.hidden.size() * sizeof(float));
        std::memcpy(residual.contents, snapshot.residual.data(),
                    snapshot.residual.size() * sizeof(float));
        std::memcpy(logits.contents, snapshot.logits.data(),
                    snapshot.logits.size() * sizeof(float));
        for (size_t index = 0; index < layers.size(); ++index) {
            const Layer& layer = layers[index];
            if (layer.convolution) {
                const size_t elements = static_cast<size_t>(layer.cache_length) *
                    static_cast<size_t>(model.graph.hidden);
                if (snapshot.key_state[index].size() != elements ||
                    !snapshot.value_state[index].empty() ||
                    !snapshot.mixer_state[index].empty() ||
                    !snapshot.recurrent_state[index].empty()) {
                    throw std::invalid_argument("invalid Metal convolution snapshot state");
                }
                std::memcpy(layer.key_cache.contents, snapshot.key_state[index].data(),
                            elements * sizeof(float));
            } else if (layer.gated_delta || layer.mamba2) {
                const size_t conv_elements = layer.gated_delta
                    ? static_cast<size_t>(2 * layer.recurrent_key_heads *
                        layer.recurrent_key_head_dim +
                        layer.recurrent_value_heads * layer.recurrent_value_head_dim) *
                        layer.recurrent_conv_kernel
                    : static_cast<size_t>(layer.recurrent_inner +
                        2 * layer.recurrent_group_count * layer.recurrent_state_size) *
                        layer.recurrent_conv_kernel;
                const size_t recurrent_elements = layer.gated_delta
                    ? static_cast<size_t>(layer.recurrent_value_heads) *
                        layer.recurrent_key_head_dim * layer.recurrent_value_head_dim
                    : static_cast<size_t>(layer.recurrent_inner) *
                        layer.recurrent_state_size;
                if (!snapshot.key_state[index].empty() ||
                    !snapshot.value_state[index].empty() ||
                    snapshot.mixer_state[index].size() != conv_elements ||
                    snapshot.recurrent_state[index].size() != recurrent_elements) {
                    throw std::invalid_argument("invalid Metal recurrent snapshot state");
                }
                std::memcpy(layer.recurrent_conv_state.contents,
                            snapshot.mixer_state[index].data(),
                            conv_elements * sizeof(float));
                std::memcpy(layer.recurrent_state.contents,
                            snapshot.recurrent_state[index].data(),
                            recurrent_elements * sizeof(float));
            } else {
                const size_t elements = ((static_cast<size_t>(max_context) +
                    static_cast<size_t>(layer.page_tokens) - 1) /
                    static_cast<size_t>(layer.page_tokens)) *
                    static_cast<size_t>(layer.page_tokens) *
                    static_cast<size_t>(layer.key_value_heads) *
                    static_cast<size_t>(layer.head_dim);
                if (snapshot.key_state[index].size() != elements ||
                    snapshot.value_state[index].size() != elements ||
                    !snapshot.mixer_state[index].empty() ||
                    !snapshot.recurrent_state[index].empty()) {
                    throw std::invalid_argument("invalid Metal attention snapshot state");
                }
                std::memcpy(layer.key_cache.contents, snapshot.key_state[index].data(),
                            elements * sizeof(float));
                std::memcpy(layer.value_cache.contents, snapshot.value_state[index].data(),
                            elements * sizeof(float));
            }
        }
    }
};

MetalModel::MetalModel(const std::string& path, int context,
                       MetalModelOptions model_options,
                       GenerationConfig generation,
                       std::shared_ptr<const RuntimeContext> runtime)
    : impl_(std::make_unique<Impl>()) {
    if (context <= 0) throw std::invalid_argument("Metal context must be positive");
    if (model_options.kv_page_tokens <= 0) {
        throw std::invalid_argument("Metal KV page size must be positive");
    }
    generation.validate();
    (*impl_).model_path = path;
    (*impl_).max_context = context;
    (*impl_).options = model_options;
    (*impl_).generation = std::move(generation);
    (*impl_).rng_state = (*impl_).generation.seed;
    (*impl_).runtime = runtime ? std::move(runtime) : create_builtin_runtime_context();
    const detail::ModelBootstrap bootstrap =
        detail::load_model_bootstrap(std::filesystem::path(path), *(*impl_).runtime);
    (*impl_).model = bootstrap.model;
    (*impl_).repository = bootstrap.checkpoint.repository;
    (*impl_).model.validate();
    (*impl_).program = build_model_program((*impl_).model);
    (*impl_).program.validate();
    if ((*impl_).program.layers.empty()) throw std::runtime_error("Metal model has no layers");
    for (const CompiledLayerProgram& layer : (*impl_).program.layers) {
        if (!std::holds_alternative<ShortConvolutionSpec>(layer.mixer) &&
            !std::holds_alternative<CompiledAttentionProgram>(layer.mixer) &&
            !std::holds_alternative<GatedDeltaNetSpec>(layer.mixer) &&
            !std::holds_alternative<Mamba2Spec>(layer.mixer)) {
            throw std::runtime_error("Metal native path does not support this mixer");
        }
        if (const auto* dense = std::get_if<CompiledDenseFeedForwardProgram>(
                &layer.feed_forward)) {
            if (dense->activation != ActivationKind::SwiGLU) {
                throw std::runtime_error("Metal native path requires SwiGLU feed-forward layers");
            }
        } else if (const auto* moe = std::get_if<MoeLayerProgram>(&layer.feed_forward)) {
            if (moe->routed.mlp.activation != MoeActivation::SwiGLU) {
                throw std::runtime_error("Metal native path requires SwiGLU MoE experts");
            }
            if (moe->shared) {
                throw std::runtime_error("Metal native path does not yet support shared MoE experts");
            }
        } else if (!std::holds_alternative<std::monostate>(layer.feed_forward)) {
            throw std::runtime_error("Metal native path requires dense or MoE feed-forward layers");
        }
    }

    (*impl_).device = MTLCreateSystemDefaultDevice();
    if (!(*impl_).device) throw std::runtime_error("no default Metal device is available");
    (*impl_).queue = [(*impl_).device newCommandQueue];
    if (!(*impl_).queue) throw std::runtime_error("Metal command queue creation failed");
    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:metal_detail::kInferenceShader];
    (*impl_).library = [(*impl_).device newLibraryWithSource:source options:nil error:&error];
    if (!(*impl_).library) {
        const std::string message = error ? ns_string(error.localizedDescription)
                                          : "unknown Metal shader compilation error";
        throw std::runtime_error("Metal inference shader compilation failed: " + message);
    }

    const int hidden = (*impl_).program.hidden;
    const int vocab = (*impl_).model.topology.dims.vocab_size;
    size_t max_projection = static_cast<size_t>(3 * hidden);
    size_t max_recurrent_aux = static_cast<size_t>(hidden);
    size_t max_recurrent_output = static_cast<size_t>(hidden);
    for (const CompiledLayerProgram& layer : (*impl_).program.layers) {
        if (const auto* gated_delta = std::get_if<GatedDeltaNetSpec>(&layer.mixer)) {
            const size_t key_width = static_cast<size_t>(gated_delta->key_heads) *
                static_cast<size_t>(gated_delta->key_head_dim);
            const size_t value_width = static_cast<size_t>(gated_delta->value_heads) *
                static_cast<size_t>(gated_delta->value_head_dim);
            max_projection = std::max(max_projection, 2 * key_width + value_width);
            max_recurrent_aux = std::max(max_recurrent_aux,
                                         static_cast<size_t>(value_width));
            max_recurrent_aux = std::max(max_recurrent_aux,
                                         static_cast<size_t>(gated_delta->decay_width()));
            max_recurrent_output = std::max(max_recurrent_output, value_width);
        } else if (const auto* mamba = std::get_if<Mamba2Spec>(&layer.mixer)) {
            const size_t projection = static_cast<size_t>(2 * mamba->intermediate_size) +
                static_cast<size_t>(2 * mamba->group_count * mamba->state_size) +
                static_cast<size_t>(mamba->num_heads);
            max_projection = std::max(max_projection, projection);
            max_recurrent_output = std::max(max_recurrent_output,
                                            static_cast<size_t>(mamba->intermediate_size));
        }
    }
    (*impl_).embedding = (*impl_).load_linear(TensorRole::TokenEmbedding, -1, vocab, hidden);
    (*impl_).final_norm = (*impl_).load_vector(TensorRole::FinalNorm, -1, hidden);
    (*impl_).hidden = (*impl_).zero_buffer(static_cast<size_t>(hidden) * sizeof(float));
    (*impl_).residual = (*impl_).zero_buffer(static_cast<size_t>(hidden) * sizeof(float));
    (*impl_).normed = (*impl_).zero_buffer(static_cast<size_t>(hidden) * sizeof(float));
    (*impl_).projected = (*impl_).zero_buffer(max_projection * sizeof(float));
    (*impl_).operation = (*impl_).zero_buffer(static_cast<size_t>(hidden) * sizeof(float));
    (*impl_).query_buffer = (*impl_).zero_buffer(static_cast<size_t>(hidden) * sizeof(float));
    (*impl_).key_buffer = (*impl_).zero_buffer(static_cast<size_t>(hidden) * sizeof(float));
    (*impl_).value_buffer = (*impl_).zero_buffer(static_cast<size_t>(hidden) * sizeof(float));
    (*impl_).gate_up = (*impl_).zero_buffer(static_cast<size_t>(2 * hidden * 8) * sizeof(float));
    (*impl_).activated = (*impl_).zero_buffer(static_cast<size_t>(hidden * 8) * sizeof(float));
    (*impl_).moe_output = (*impl_).zero_buffer(static_cast<size_t>(hidden) * sizeof(float));
    (*impl_).recurrent_z = (*impl_).zero_buffer(max_recurrent_aux * sizeof(float));
    (*impl_).recurrent_b = (*impl_).zero_buffer(max_recurrent_aux * sizeof(float));
    (*impl_).recurrent_a = (*impl_).zero_buffer(max_recurrent_aux * sizeof(float));
    (*impl_).recurrent_output = (*impl_).zero_buffer(max_recurrent_output * sizeof(float));
    (*impl_).logits = (*impl_).zero_buffer(static_cast<size_t>(vocab) * sizeof(float));

    (*impl_).layers.reserve((*impl_).model.graph.layers.size());
    for (size_t index = 0; index < (*impl_).program.layers.size(); ++index) {
        const CompiledLayerProgram& program_layer = (*impl_).program.layers[index];
        Impl::Layer layer;
        layer.operator_norm = (*impl_).load_vector(TensorRole::AttentionInputNorm,
                                                    static_cast<int>(index), hidden, true);
        layer.ffn_norm = (*impl_).load_vector(TensorRole::FfnInputNorm,
                                              static_cast<int>(index), hidden, true);
        if (const auto* convolution = std::get_if<ShortConvolutionSpec>(&program_layer.mixer)) {
            layer.convolution = true;
            layer.cache_length = convolution->cache_length;
            layer.mixer_in = (*impl_).load_linear(TensorRole::ShortConvInput,
                                                  static_cast<int>(index), 3 * hidden, hidden);
            layer.mixer_out = (*impl_).load_linear(TensorRole::ShortConvOutput,
                                                   static_cast<int>(index), hidden, hidden);
            layer.convolution_taps = (*impl_).load_conv_kernel(
                static_cast<int>(index), hidden, convolution->cache_length);
            layer.key_cache = (*impl_).zero_buffer(static_cast<size_t>(hidden) *
                                                    convolution->cache_length * sizeof(float));
            layer.value_cache = layer.key_cache;
        } else if (const auto* gated_delta =
                       std::get_if<GatedDeltaNetSpec>(&program_layer.mixer)) {
            layer.gated_delta = true;
            layer.recurrent_conv_kernel = gated_delta->conv_kernel;
            layer.recurrent_key_head_dim = gated_delta->key_head_dim;
            layer.recurrent_value_head_dim = gated_delta->value_head_dim;
            layer.recurrent_key_heads = gated_delta->key_heads;
            layer.recurrent_value_heads = gated_delta->value_heads;
            layer.recurrent_vector_decay = gated_delta->vector_decay;
            layer.recurrent_safe_decay = gated_delta->safe_decay;
            layer.recurrent_decay_lower_bound = gated_delta->decay_lower_bound;
            layer.recurrent_sigmoid_output_gate = gated_delta->sigmoid_output_gate;
            layer.recurrent_a_log_needs_exp = gated_delta->a_log_needs_exp;
            const int key_width = gated_delta->key_heads * gated_delta->key_head_dim;
            const int value_width = gated_delta->value_heads * gated_delta->value_head_dim;
            const int qkv_width = 2 * key_width + value_width;
            if (gated_delta->factorized_projections) {
                layer.recurrent_q = (*impl_).load_linear(
                    TensorRole::GatedDeltaNetQuery, static_cast<int>(index),
                    key_width, hidden);
                layer.recurrent_k = (*impl_).load_linear(
                    TensorRole::GatedDeltaNetKey, static_cast<int>(index),
                    key_width, hidden);
                layer.recurrent_v = (*impl_).load_linear(
                    TensorRole::GatedDeltaNetValue, static_cast<int>(index),
                    value_width, hidden);
                layer.recurrent_z_weight = (*impl_).load_linear(
                    TensorRole::GatedDeltaNetOutputGate, static_cast<int>(index),
                    value_width, hidden);
                std::vector<float> convolution;
                const std::vector<float> query = (*impl_).load_vector_values(
                    TensorRole::GatedDeltaNetQueryConv, static_cast<int>(index),
                    key_width * gated_delta->conv_kernel);
                const std::vector<float> key = (*impl_).load_vector_values(
                    TensorRole::GatedDeltaNetKeyConv, static_cast<int>(index),
                    key_width * gated_delta->conv_kernel);
                const std::vector<float> value = (*impl_).load_vector_values(
                    TensorRole::GatedDeltaNetValueConv, static_cast<int>(index),
                    value_width * gated_delta->conv_kernel);
                convolution.reserve(static_cast<size_t>(qkv_width) *
                                     gated_delta->conv_kernel);
                convolution.insert(convolution.end(), query.begin(), query.end());
                convolution.insert(convolution.end(), key.begin(), key.end());
                convolution.insert(convolution.end(), value.begin(), value.end());
                layer.recurrent_conv_weight = (*impl_).buffer(convolution);
            } else {
                layer.recurrent_qkv = (*impl_).load_linear(
                    TensorRole::GatedDeltaNetQkv, static_cast<int>(index),
                    qkv_width, hidden);
                layer.recurrent_z_weight = (*impl_).load_linear(
                    TensorRole::GatedDeltaNetZ, static_cast<int>(index),
                    value_width, hidden);
                layer.recurrent_conv_weight = (*impl_).load_vector(
                    TensorRole::GatedDeltaNetConv, static_cast<int>(index),
                    qkv_width * gated_delta->conv_kernel);
            }
            layer.recurrent_b = (*impl_).load_linear(
                TensorRole::GatedDeltaNetBeta, static_cast<int>(index),
                gated_delta->value_heads, hidden);
            layer.recurrent_a = (*impl_).load_linear(
                gated_delta->factorized_projections
                    ? TensorRole::GatedDeltaNetDecay : TensorRole::GatedDeltaNetAlpha,
                static_cast<int>(index), gated_delta->decay_width(), hidden);
            layer.recurrent_dt_bias = (*impl_).load_vector(
                TensorRole::GatedDeltaNetDtBias, static_cast<int>(index),
                gated_delta->decay_width());
            layer.recurrent_a_log = (*impl_).load_vector(
                TensorRole::GatedDeltaNetALog, static_cast<int>(index),
                gated_delta->value_heads);
            layer.recurrent_norm = (*impl_).load_vector(
                TensorRole::GatedDeltaNetNorm, static_cast<int>(index),
                gated_delta->value_head_dim);
            layer.recurrent_out = (*impl_).load_linear(
                TensorRole::GatedDeltaNetOutput, static_cast<int>(index),
                hidden, value_width);
            layer.recurrent_conv_state = (*impl_).zero_buffer(
                static_cast<size_t>(qkv_width) * gated_delta->conv_kernel * sizeof(float));
            layer.recurrent_state = (*impl_).zero_buffer(
                static_cast<size_t>(value_width) * gated_delta->key_head_dim * sizeof(float));
        } else if (const auto* mamba =
                       std::get_if<Mamba2Spec>(&program_layer.mixer)) {
            layer.mamba2 = true;
            layer.recurrent_conv_kernel = mamba->conv_kernel;
            layer.recurrent_inner = mamba->intermediate_size;
            layer.recurrent_state_size = mamba->state_size;
            layer.recurrent_value_heads = mamba->num_heads;
            layer.recurrent_key_head_dim = mamba->head_dim;
            layer.recurrent_group_count = mamba->group_count;
            layer.recurrent_a_log_needs_exp = mamba->a_log_needs_exp;
            const int conv_width = mamba->intermediate_size +
                2 * mamba->group_count * mamba->state_size;
            const int projection_width = 2 * mamba->intermediate_size +
                2 * mamba->group_count * mamba->state_size + mamba->num_heads;
            layer.recurrent_in = (*impl_).load_linear(
                TensorRole::Mamba2Input, static_cast<int>(index),
                projection_width, hidden);
            layer.recurrent_conv_weight = (*impl_).load_vector(
                TensorRole::Mamba2Conv, static_cast<int>(index),
                conv_width * mamba->conv_kernel);
            layer.recurrent_conv_bias = (*impl_).load_vector(
                TensorRole::Mamba2ConvBias, static_cast<int>(index), conv_width);
            layer.recurrent_dt_bias = (*impl_).load_vector(
                TensorRole::Mamba2DtBias, static_cast<int>(index), mamba->num_heads);
            layer.recurrent_a_log = (*impl_).load_vector(
                TensorRole::Mamba2ALog, static_cast<int>(index), mamba->num_heads);
            layer.recurrent_d = (*impl_).load_vector(
                TensorRole::Mamba2D, static_cast<int>(index), mamba->num_heads);
            layer.recurrent_norm = (*impl_).load_vector(
                TensorRole::Mamba2Norm, static_cast<int>(index), mamba->intermediate_size);
            layer.recurrent_out = (*impl_).load_linear(
                TensorRole::Mamba2Output, static_cast<int>(index),
                hidden, mamba->intermediate_size);
            layer.recurrent_conv_state = (*impl_).zero_buffer(
                static_cast<size_t>(conv_width) * mamba->conv_kernel * sizeof(float));
            layer.recurrent_state = (*impl_).zero_buffer(
                static_cast<size_t>(mamba->intermediate_size) * mamba->state_size *
                sizeof(float));
        } else {
            const auto& attention = std::get<CompiledAttentionProgram>(program_layer.mixer);
            if (!std::holds_alternative<OrdinaryKvStateSpec>(attention.semantics.state)) {
                throw std::runtime_error("Metal native path requires ordinary KV attention");
            }
            const auto& spec = attention.semantics;
            layer.query_heads = spec.query_heads;
            layer.key_value_heads = spec.key_value_heads;
            layer.head_dim = spec.head_dim;
            layer.page_tokens = (*impl_).options.kv_page_tokens;
            layer.query_scale = spec.query_scale;
            if (const RopePositionSpec* rope = spec.rope_position()) layer.rope_theta = rope->theta;
            layer.query = (*impl_).load_linear(TensorRole::AttentionQuery,
                                               static_cast<int>(index), spec.query_projection_width(), hidden);
            layer.key = (*impl_).load_linear(TensorRole::AttentionKey,
                                             static_cast<int>(index), spec.key_value_width(), hidden);
            layer.value = (*impl_).load_linear(TensorRole::AttentionValue,
                                               static_cast<int>(index), spec.key_value_width(), hidden);
            layer.attention_out = (*impl_).load_linear(TensorRole::AttentionOutput,
                                                       static_cast<int>(index), hidden, spec.query_width());
            layer.query_norm = (*impl_).load_vector(TensorRole::AttentionQueryNorm,
                                                     static_cast<int>(index), spec.head_dim, true);
            layer.key_norm = (*impl_).load_vector(TensorRole::AttentionKeyNorm,
                                                  static_cast<int>(index), spec.head_dim, true);
            const size_t page_count =
                (static_cast<size_t>(context) + static_cast<size_t>(layer.page_tokens) - 1) /
                static_cast<size_t>(layer.page_tokens);
            const size_t cache_elements = page_count *
                static_cast<size_t>(layer.page_tokens) *
                static_cast<size_t>(spec.key_value_width());
            layer.key_cache = (*impl_).zero_buffer(cache_elements * sizeof(float));
            layer.value_cache = (*impl_).zero_buffer(cache_elements * sizeof(float));
            if (spec.query_norm) layer.query_norm_epsilon = spec.query_norm->epsilon;
            if (spec.key_norm) layer.key_norm_epsilon = spec.key_norm->epsilon;
        }
        if (const auto* dense = std::get_if<CompiledDenseFeedForwardProgram>(
                &program_layer.feed_forward)) {
            layer.intermediate = dense->intermediate_size;
            layer.ffn_gate = (*impl_).load_linear(TensorRole::FfnGate,
                                                  static_cast<int>(index), layer.intermediate, hidden);
            layer.ffn_up = (*impl_).load_linear(TensorRole::FfnUp,
                                                static_cast<int>(index), layer.intermediate, hidden);
            layer.ffn_down = (*impl_).load_linear(TensorRole::FfnDown,
                                                  static_cast<int>(index), hidden, layer.intermediate);
        } else if (const auto* moe_program = std::get_if<MoeLayerProgram>(
                       &program_layer.feed_forward)) {
            layer.intermediate = moe_program->routed.mlp.intermediate_size;
            Impl::Layer::Moe moe;
            moe.router = moe_program->router;
            const std::string router_name = request_name(
                (*impl_).model.weight_plan.requests, TensorRole::MoeRouter,
                static_cast<int>(index));
            const TensorRequest& router_request = request_for(
                (*impl_).model.weight_plan.requests, TensorRole::MoeRouter,
                static_cast<int>(index));
            moe.router_weight = tensor_values(
                (*impl_).repository->tensor(router_name),
                std::span<const int64_t>(router_request.expected_shape.data(),
                                         router_request.expected_shape.size()),
                router_name);
            if (const auto bias = std::find_if(
                    (*impl_).model.weight_plan.requests.begin(),
                    (*impl_).model.weight_plan.requests.end(),
                    [index](const TensorRequest& request) {
                        return request.role == TensorRole::MoeRouterBias &&
                               request.layer == static_cast<int>(index) &&
                               request.expert < 0;
                    });
                bias != (*impl_).model.weight_plan.requests.end()) {
                if (!bias->source_name) {
                    throw std::runtime_error("Metal MoE router bias was not resolved");
                }
                moe.router_bias = tensor_values(
                    (*impl_).repository->tensor(*bias->source_name),
                    std::span<const int64_t>(bias->expected_shape.data(),
                                             bias->expected_shape.size()),
                    *bias->source_name);
            }
            moe.experts.resize(static_cast<size_t>(moe.router.expert_count));
            for (int expert = 0; expert < moe.router.expert_count; ++expert) {
                Impl::Layer::Expert& names = moe.experts[static_cast<size_t>(expert)];
                names.gate_name = request_name(
                    (*impl_).model.weight_plan.requests, TensorRole::MoeExpertGate,
                    static_cast<int>(index), expert);
                names.up_name = request_name(
                    (*impl_).model.weight_plan.requests, TensorRole::MoeExpertUp,
                    static_cast<int>(index), expert);
                names.down_name = request_name(
                    (*impl_).model.weight_plan.requests, TensorRole::MoeExpertDown,
                    static_cast<int>(index), expert);
            }
            layer.moe = std::move(moe);
        } else if (std::holds_alternative<std::monostate>(program_layer.feed_forward)) {
            layer.intermediate = 0;
        } else {
            throw std::runtime_error("Metal layer has no supported feed-forward program");
        }
        (*impl_).layers.push_back(std::move(layer));
    }
    int max_intermediate = 0;
    for (const Impl::Layer& layer : (*impl_).layers) {
        max_intermediate = std::max(max_intermediate, layer.intermediate);
    }
    (*impl_).gate_up = (*impl_).zero_buffer(static_cast<size_t>(2 * std::max(1, max_intermediate)) * sizeof(float));
    (*impl_).activated = (*impl_).zero_buffer(static_cast<size_t>(std::max(1, max_intermediate)) * sizeof(float));
    (*impl_).seen.resize(static_cast<size_t>(vocab));
    (*impl_).reset();
}

MetalModel::~MetalModel() = default;
MetalModel::MetalModel(MetalModel&&) noexcept = default;
MetalModel& MetalModel::operator=(MetalModel&&) noexcept = default;

void MetalModel::reset_session() { (*impl_).reset(); }

void MetalModel::prefill_session(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) throw std::invalid_argument("Metal prefill needs at least one token");
    if (tokens.size() > static_cast<size_t>((*impl_).max_context)) {
        throw std::invalid_argument("Metal prefill exceeds context");
    }
    (*impl_).reset();
    const auto started = std::chrono::steady_clock::now();
    for (const int32_t token : tokens) {
        if (token < 0 || token >= (*impl_).model.topology.dims.vocab_size) {
            throw std::invalid_argument("Metal token out of range");
        }
        (*impl_).seen[static_cast<size_t>(token)] = 1;
        (*impl_).run_token(token);
    }
    (*impl_).metrics.last_prefill_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
    (*impl_).metrics.prefill_tokens = tokens.size();
    (*impl_).ready = true;
}

int32_t MetalModel::decode_session() {
    if (!(*impl_).ready) throw std::runtime_error("Metal model is not ready for decode");
    const auto started = std::chrono::steady_clock::now();
    std::vector<float> values = session_logits();
    const int32_t token = CpuSampler::sample(values, vocab_size(), (*impl_).generation,
                                             (*impl_).seen, (*impl_).rng_state);
    (*impl_).seen[static_cast<size_t>(token)] = 1;
    (*impl_).run_token(token);
    (*impl_).metrics.cumulative_decode_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    ++(*impl_).metrics.decoded_tokens;
    return token;
}

void MetalModel::eval_token_session(int32_t token) {
    if (!(*impl_).ready) throw std::runtime_error("Metal model is not ready for token evaluation");
    if (token < 0 || token >= vocab_size()) throw std::invalid_argument("Metal token out of range");
    (*impl_).seen[static_cast<size_t>(token)] = 1;
    (*impl_).run_token(token);
    ++(*impl_).metrics.decoded_tokens;
}

void MetalModel::set_session_generation(GenerationConfig generation) {
    generation.validate();
    (*impl_).generation = std::move(generation);
    (*impl_).rng_state = (*impl_).generation.seed;
}

std::vector<float> MetalModel::session_logits() const {
    std::vector<float> values(static_cast<size_t>(vocab_size()));
    std::memcpy(values.data(), (*impl_).logits.contents,
                values.size() * sizeof(float));
    return values;
}

int MetalModel::vocab_size() const { return (*impl_).model.topology.dims.vocab_size; }
const std::string& MetalModel::model_identity() const { return (*impl_).model.provenance.identity; }
std::string MetalModel::backend_description() const {
    return std::string("metal-native device=") + (*impl_).device.name.UTF8String;
}
RuntimeMetrics MetalModel::metrics() const { return (*impl_).metrics; }
MetalSessionSnapshot MetalModel::export_session_snapshot() const {
    return (*impl_).export_snapshot();
}
void MetalModel::restore_session_snapshot(MetalSessionSnapshot snapshot) {
    (*impl_).restore_snapshot(std::move(snapshot));
}

int MetalModel::session_position() const { return (*impl_).position; }
bool MetalModel::session_ready_for_decode() const { return (*impl_).ready; }

void MetalInferenceSession::reset() { owner_->reset_session(); }
void MetalInferenceSession::prefill(const std::vector<int32_t>& tokens) { owner_->prefill_session(tokens); }
int32_t MetalInferenceSession::decode() { return owner_->decode_session(); }
void MetalInferenceSession::eval_token(int32_t token) { owner_->eval_token_session(token); }
void MetalInferenceSession::set_generation_config(GenerationConfig generation) {
    owner_->set_session_generation(std::move(generation));
}
std::vector<float> MetalInferenceSession::copy_logits() const { return owner_->session_logits(); }
int MetalInferenceSession::position() const { return owner_->session_position(); }
bool MetalInferenceSession::ready_for_decode() const { return owner_->session_ready_for_decode(); }

}
