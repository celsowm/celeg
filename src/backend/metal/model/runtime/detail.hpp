#pragma once

#include "../detail.hpp"

#include <Metal/Metal.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

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

    struct Moe {
        Linear gate;
        std::vector<Linear> gate_up;
        std::vector<Linear> down;
        uint32_t experts = 0;
        uint32_t experts_per_token = 0;
        uint32_t intermediate = 0;
        bool normalize_topk = true;
        float routed_scaling_factor = 1.0f;
        MoeScoreKind score_kind = MoeScoreKind::Softmax;
        MoeRouteScaleOrder route_scale_order = MoeRouteScaleOrder::AfterNormalize;
        bool add_shared_expert = false;
        Linear shared_gate;
        Linear shared_up;
        Linear shared_down;
        Linear shared_router;
    };

    struct Layer {
        enum class MixerKind {
            ShortConvolution,
            GatedDelta,
            Mamba2,
            Attention,
        };

        MixerKind mixer_kind = MixerKind::ShortConvolution;
        Linear operator_norm;
        Linear ffn_norm;
        Linear mixer_in;
        Linear mixer_out;
        id<MTLBuffer> convolution_taps = nil;
        int cache_length = 0;
        int page_tokens = 0;
        int recurrent_key_heads = 0;
        int recurrent_value_heads = 0;
        int recurrent_key_head_dim = 0;
        int recurrent_value_head_dim = 0;
        int recurrent_inner = 0;
        int recurrent_state_size = 0;
        int recurrent_group_count = 0;
        int recurrent_conv_kernel = 0;
        bool recurrent_vector_decay = false;
        bool recurrent_safe_decay = false;
        float recurrent_decay_lower_bound = -10.0f;
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
        Linear attention_gate;
        Linear attention_out;
        id<MTLBuffer> query_norm = nil;
        id<MTLBuffer> key_norm = nil;
        id<MTLBuffer> key_cache = nil;
        id<MTLBuffer> value_cache = nil;
        id<MTLBuffer> alibi_slopes = nil;
        id<MTLBuffer> relative_bias = nil;
        int kv_owner_layer = -1;
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
    MetalPipelineCache pipeline_cache;
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
    id<MTLBuffer> batch_tokens = nil;
    id<MTLBuffer> batch_rope_positions = nil;
    id<MTLBuffer> batch_hidden = nil;
    id<MTLBuffer> batch_residual = nil;
    id<MTLBuffer> batch_normed = nil;
    id<MTLBuffer> batch_projected = nil;
    id<MTLBuffer> batch_operation = nil;
    id<MTLBuffer> batch_query = nil;
    id<MTLBuffer> batch_key = nil;
    id<MTLBuffer> batch_value = nil;
    id<MTLBuffer> batch_gate_up = nil;
    id<MTLBuffer> batch_activated = nil;
    int batch_capacity = 0;
    std::vector<Layer> layers;
    std::vector<uint8_t> seen;
    GenerationConfig generation;
    uint64_t rng_state = 1;
    int position = 0;
    std::array<int32_t, 3> next_rope_position{0, 0, 0};
    bool ready = false;
    RuntimeMetrics metrics;
    MetalExecutionMetrics execution_metrics;
    uint64_t command_dispatches = 0;
    std::unordered_map<std::string, uint64_t> dispatch_histogram;
    std::chrono::steady_clock::time_point command_started;

    Impl(std::string model_path, int max_context, MetalModelOptions options,
         std::shared_ptr<const RuntimeContext> runtime);

    void ensure_batch_capacity(uint32_t rows);
    void reset_state();
    id<MTLBuffer> buffer(size_t bytes);
    id<MTLBuffer> buffer(const std::vector<float>& values);
    id<MTLBuffer> zero_buffer(size_t bytes);
    id<MTLBuffer> load_vector(TensorRole role, int layer, int64_t expected_count);
    Linear load_linear(TensorRole role, int layer, int64_t rows, int64_t cols);
    Linear load_linear_any(TensorRole role, int layer, int64_t rows, int64_t cols,
                           std::span<const TensorRole> alternatives = {});
    bool load_linear_optional(Linear& target, TensorRole role, int layer,
                              int64_t rows, int64_t cols,
                              std::span<const TensorRole> alternatives = {});
    void load_layer(size_t layer_index, Layer& layer);
    std::optional<std::string_view> linear_kernel(
        LinearStorage storage, LinearOperationKind operation) const;
    MetalMatvecKernel matvec_kernel(LinearStorage storage) const;
    MetalMatvecKernel swiglu_matvec_kernel(LinearStorage storage) const;
    bool tensor_matmul_available(LinearStorage storage, uint32_t rows) const;

    id<MTLComputePipelineState> pipeline(std::string_view name);
    id<MTLComputePipelineState> tensor_pipeline(std::string_view name);
    void begin_commands(id<MTLCommandBuffer>& command_buffer,
                        id<MTLComputeCommandEncoder>& encoder);
    void finish_commands(id<MTLCommandBuffer>& command_buffer,
                         id<MTLComputeCommandEncoder>& encoder);
    void dispatch(id<MTLComputeCommandEncoder> encoder, std::string_view name,
                  NSUInteger count);
    void dispatch_cooperative(id<MTLComputeCommandEncoder> encoder, std::string_view name,
                              NSUInteger groups);
    void record_dispatch(std::string_view name);
    void set_buffer(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> value,
                    NSUInteger index, NSUInteger offset = 0);
    void set_bytes(id<MTLComputeCommandEncoder> encoder, const void* value,
                   NSUInteger length, NSUInteger index);
    void encode_matvec(id<MTLComputeCommandEncoder> encoder, const Linear& weight,
                       id<MTLBuffer> input, id<MTLBuffer> output,
                       NSUInteger output_offset = 0, NSUInteger input_offset = 0);
    bool encode_swiglu_matvec(id<MTLComputeCommandEncoder> encoder,
                              const Linear& weight, id<MTLBuffer> gate_up,
                              id<MTLBuffer> output);
    void encode_matmul(id<MTLComputeCommandEncoder> encoder, const Linear& weight,
                       id<MTLBuffer> input, id<MTLBuffer> output, uint32_t rows,
                       NSUInteger input_offset = 0, NSUInteger output_offset = 0,
                       uint32_t output_stride = 0);
    void encode_embedding(id<MTLComputeCommandEncoder> encoder, uint32_t width,
                          uint32_t token);
    void encode_embedding_batch(id<MTLComputeCommandEncoder> encoder,
                                uint32_t rows, const std::vector<int32_t>& tokens);
    void encode_rmsnorm(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> input,
                        id<MTLBuffer> weight, id<MTLBuffer> output, uint32_t width,
                        float epsilon);
    void encode_rmsnorm_save(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> input,
                             id<MTLBuffer> residual, id<MTLBuffer> weight,
                             id<MTLBuffer> output, uint32_t width, float epsilon);
    void encode_residual_rmsnorm(id<MTLComputeCommandEncoder> encoder,
                                 id<MTLBuffer> input, id<MTLBuffer> residual,
                                 id<MTLBuffer> weight, id<MTLBuffer> output,
                                 id<MTLBuffer> normed, uint32_t width,
                                 float multiplier, float epsilon);
    void encode_residual_rmsnorm_save(id<MTLComputeCommandEncoder> encoder,
                                      id<MTLBuffer> input, id<MTLBuffer> residual,
                                      id<MTLBuffer> weight, id<MTLBuffer> output,
                                      id<MTLBuffer> next_residual,
                                      id<MTLBuffer> normed, uint32_t width,
                                      float multiplier, float epsilon);
    void encode_rmsnorm_batch(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> input,
                              id<MTLBuffer> weight, id<MTLBuffer> output,
                              uint32_t rows, uint32_t width, float epsilon);
    void encode_residual_batch(id<MTLComputeCommandEncoder> encoder,
                               id<MTLBuffer> input, id<MTLBuffer> residual,
                               id<MTLBuffer> output, uint32_t count, float multiplier);
    void encode_swiglu_batch(id<MTLComputeCommandEncoder> encoder,
                             id<MTLBuffer> input, id<MTLBuffer> output,
                             uint32_t rows, uint32_t width);
    void encode_weighted_add(id<MTLComputeCommandEncoder> encoder,
                             id<MTLBuffer> input, id<MTLBuffer> output,
                             uint32_t count, float weight);

    void encode_short_convolution(id<MTLComputeCommandEncoder> encoder,
                                  Layer& layer);
    void encode_short_convolution_batch(id<MTLComputeCommandEncoder> encoder,
                                        Layer& layer, uint32_t rows,
                                        uint32_t base_position);
    void encode_gated_delta(id<MTLComputeCommandEncoder> encoder, Layer& layer);
    void encode_gated_delta_layer(id<MTLComputeCommandEncoder> encoder,
                                  Layer& layer);
    void encode_mamba2(id<MTLComputeCommandEncoder> encoder, Layer& layer);
    void encode_mamba2_layer(id<MTLComputeCommandEncoder> encoder, Layer& layer);
    void encode_attention_span(id<MTLComputeCommandEncoder> encoder,
                               std::string_view name, uint32_t query_heads,
                               uint32_t rows, uint32_t head_dim);
    void encode_attention(id<MTLComputeCommandEncoder> encoder, Layer& layer,
                          const CompiledAttentionProgram& attention,
                          const std::array<int32_t, 3>* rope_position = nullptr);
    void encode_attention_batch(id<MTLComputeCommandEncoder> encoder,
                                Layer& layer,
                                const CompiledAttentionProgram& attention,
                                uint32_t rows, uint32_t base_position);
    void encode_dense_feed_forward(id<MTLComputeCommandEncoder> encoder,
                                   Layer& layer, bool gate_up_ready = false);
    void encode_dense_feed_forward_batch(id<MTLComputeCommandEncoder> encoder,
                                         Layer& layer, uint32_t rows);
    void encode_moe(id<MTLCommandBuffer>& command_buffer,
                    id<MTLComputeCommandEncoder>& encoder, Layer& layer);
    void encode_prefill_batch(
        id<MTLComputeCommandEncoder>& encoder,
        const std::vector<int32_t>& tokens,
        std::span<const std::array<int32_t, 3>> rope_positions = {});
    bool supports_prefill_batch() const;

    void encode_token(id<MTLCommandBuffer>& command_buffer,
                      id<MTLComputeCommandEncoder>& encoder, int32_t token,
                      const std::array<int32_t, 3>* rope_position = nullptr);
    void run_token(int32_t token,
                   const std::array<int32_t, 3>* rope_position = nullptr);
    void run_prefill(std::span<const int32_t> tokens,
                     std::span<const std::array<int32_t, 3>> rope_positions = {});
    void sample();
    void apply_logits_transforms();
    void read_logits(std::vector<float>& target);
};

}
