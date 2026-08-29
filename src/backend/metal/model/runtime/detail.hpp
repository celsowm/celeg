#pragma once

#include "celeg/backend/metal/model.hpp"
#include "celeg/detail/checkpoint/bootstrap.hpp"
#include "celeg/model/program.hpp"
#include "celeg/model/weights/roles.hpp"

#import <Metal/Metal.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace celeg::metal_model_detail {

std::string ns_string(NSString* value);
std::vector<float> tensor_values(const HostTensorView& view,
                                 std::span<const int64_t> expected,
                                 const std::string& name);
const TensorRequest& request_for(std::span<const TensorRequest> requests,
                                 TensorRole role, int layer = -1,
                                 int expert = -1);
std::string request_name(std::span<const TensorRequest> requests,
                         TensorRole role, int layer = -1,
                         int expert = -1);

}

namespace celeg {

struct MetalModel::Impl {
    enum class LinearStorage {
        Float32,
        Float16,
        BFloat16,
        Q4_0,
        Q4K,
        Q5K,
        Q6K,
        Q8_0,
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
    MetalExecutionMetrics execution_metrics;
    std::chrono::steady_clock::time_point command_started;
    uint64_t command_dispatches = 0;


    id<MTLBuffer> buffer(const std::vector<float>& values) const;
    id<MTLBuffer> zero_buffer(size_t bytes) const;
    id<MTLBuffer> raw_buffer(const std::byte* data, size_t bytes) const;

    std::vector<float> load_vector_values(TensorRole role, int layer, int width,
                                          bool allow_missing = false) const;
    id<MTLBuffer> load_vector(TensorRole role, int layer, int width,
                              bool allow_missing = false) const;
    Linear load_linear(TensorRole role, int layer, int rows, int cols) const;
    Linear load_linear_source(const std::string& name, int rows, int cols) const;
    id<MTLBuffer> load_conv_kernel(int layer, int width, int cache_length) const;

    id<MTLComputePipelineState> pipeline(std::string_view name);
    void begin_commands(id<MTLCommandBuffer>& command_buffer,
                        id<MTLComputeCommandEncoder>& encoder);
    void finish_commands(id<MTLCommandBuffer>& command_buffer,
                         id<MTLComputeCommandEncoder>& encoder);
    void dispatch(id<MTLComputeCommandEncoder> encoder, std::string_view name,
                  NSUInteger count);
    void dispatch_cooperative(id<MTLComputeCommandEncoder> encoder, std::string_view name,
                              NSUInteger groups);
    void set_buffer(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> value,
                    NSUInteger index, NSUInteger offset = 0);
    void set_bytes(id<MTLComputeCommandEncoder> encoder, const void* value,
                   NSUInteger length, NSUInteger index);
    void encode_matvec(id<MTLComputeCommandEncoder> encoder, const Linear& weight,
                       id<MTLBuffer> input, id<MTLBuffer> output,
                       NSUInteger output_offset = 0);
    void encode_embedding(id<MTLComputeCommandEncoder> encoder, uint32_t width,
                          uint32_t token);
    void encode_rmsnorm(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> input,
                        id<MTLBuffer> weight, id<MTLBuffer> output, uint32_t width,
                        float epsilon);
    void encode_weighted_add(id<MTLComputeCommandEncoder> encoder,
                             id<MTLBuffer> input, id<MTLBuffer> output,
                             uint32_t count, float weight);

    void encode_short_convolution(id<MTLComputeCommandEncoder> encoder,
                                  Layer& layer);
    void encode_gated_delta(id<MTLComputeCommandEncoder> encoder, Layer& layer);
    void encode_gated_delta_layer(id<MTLComputeCommandEncoder> encoder,
                                  Layer& layer);
    void encode_mamba2(id<MTLComputeCommandEncoder> encoder, Layer& layer);
    void encode_mamba2_layer(id<MTLComputeCommandEncoder> encoder, Layer& layer);
    void encode_attention(id<MTLComputeCommandEncoder> encoder, Layer& layer);
    void encode_dense_feed_forward(id<MTLComputeCommandEncoder> encoder,
                                   Layer& layer);
    void encode_moe(id<MTLCommandBuffer>& command_buffer,
                    id<MTLComputeCommandEncoder>& encoder, Layer& layer);

    void encode_token(id<MTLCommandBuffer>& command_buffer,
                      id<MTLComputeCommandEncoder>& encoder, int32_t token);
    void run_token(int32_t token);
    void reset();
    static std::vector<float> copy_buffer(id<MTLBuffer> source, size_t elements);
    MetalSessionSnapshot export_snapshot() const;
    void restore_snapshot(MetalSessionSnapshot snapshot);

};

}
