#include "detail.hpp"

#include "metal_inference_source.hpp"
#include "metal_tensor_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <utility>
#include <vector>

namespace celeg {

using metal_model_detail::ns_string;
using metal_model_detail::request_for;
using metal_model_detail::request_name;
using metal_model_detail::tensor_values;

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
    (*impl_).pipeline_cache.device = (*impl_).device;
    (*impl_).pipeline_cache.library = [(*impl_).device newLibraryWithSource:source
                                                                       options:nil error:&error];
    if (!(*impl_).pipeline_cache.library) {
        const std::string message = error ? ns_string(error.localizedDescription)
                                          : "unknown Metal shader compilation error";
        throw std::runtime_error("Metal inference shader compilation failed: " + message);
    }
    NSString* tensor_source = [NSString stringWithUTF8String:metal_detail::kTensorShader];
    if (std::getenv("CELEG_METAL_TENSOR_DISABLE")) {
        (*impl_).pipeline_cache.tensor_compile_error = "disabled";
    } else {
        (*impl_).pipeline_cache.tensor_library = [(*impl_).device
            newLibraryWithSource:tensor_source options:nil error:&error];
        if (!(*impl_).pipeline_cache.tensor_library) {
            (*impl_).pipeline_cache.tensor_compile_error = error
                ? ns_string(error.localizedDescription) : "unknown tensor shader error";
        }
    }
    if ((*impl_).pipeline_cache.tensor_library) {
        id<MTLFunction> f16_function = [(*impl_).pipeline_cache.tensor_library
            newFunctionWithName:@"celeg_matmul_tensor_f16"];
        id<MTLFunction> bf16_function = [(*impl_).pipeline_cache.tensor_library
            newFunctionWithName:@"celeg_matmul_tensor_bf16"];
        NSError* f16_error = nil;
        NSError* bf16_error = nil;
        if (f16_function) {
            id<MTLComputePipelineState> state =
                [(*impl_).device newComputePipelineStateWithFunction:f16_function
                                                                  error:&f16_error];
            (*impl_).pipeline_cache.tensor_matmul_f16 = state != nil;
            if (!state && f16_error) {
                (*impl_).pipeline_cache.tensor_compile_error = ns_string(f16_error.localizedDescription);
            }
        }
        if (bf16_function) {
            id<MTLComputePipelineState> state =
                [(*impl_).device newComputePipelineStateWithFunction:bf16_function
                                                                  error:&bf16_error];
            (*impl_).pipeline_cache.tensor_matmul_bf16 = state != nil;
            if (!state && bf16_error) {
                (*impl_).pipeline_cache.tensor_compile_error = ns_string(bf16_error.localizedDescription);
            }
        }
        if (!(*impl_).pipeline_cache.tensor_matmul_f16 &&
            !(*impl_).pipeline_cache.tensor_matmul_bf16) {
            (*impl_).pipeline_cache.tensor_library = nil;
        }
        if ((*impl_).pipeline_cache.tensor_matmul_f16 ||
            (*impl_).pipeline_cache.tensor_matmul_bf16) {
            (*impl_).pipeline_cache.tensor_compile_error.clear();
        }
    }

    const int hidden = (*impl_).program.hidden;
    const int vocab = (*impl_).model.topology.dims.vocab_size;
    size_t max_projection = static_cast<size_t>(3 * hidden);
    size_t max_recurrent_aux = static_cast<size_t>(hidden);
    size_t max_recurrent_output = static_cast<size_t>(hidden);
    size_t max_batch_intermediate = static_cast<size_t>(hidden) * 8;
    for (const CompiledLayerProgram& layer : (*impl_).program.layers) {
        if (const auto* dense = std::get_if<CompiledDenseFeedForwardProgram>(
                &layer.feed_forward)) {
            max_batch_intermediate = std::max(max_batch_intermediate,
                                        static_cast<size_t>(dense->intermediate_size));
        }
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
    (*impl_).batch_capacity = context;
    (*impl_).batch_tokens = (*impl_).zero_buffer(
        static_cast<size_t>(context) * sizeof(uint32_t));
    (*impl_).batch_hidden = (*impl_).zero_buffer(
        static_cast<size_t>(context) * hidden * sizeof(float));
    (*impl_).batch_residual = (*impl_).zero_buffer(
        static_cast<size_t>(context) * hidden * sizeof(float));
    (*impl_).batch_normed = (*impl_).zero_buffer(
        static_cast<size_t>(context) * hidden * sizeof(float));
    (*impl_).batch_projected = (*impl_).zero_buffer(
        static_cast<size_t>(context) * max_projection * sizeof(float));
    (*impl_).batch_operation = (*impl_).zero_buffer(
        static_cast<size_t>(context) * hidden * sizeof(float));
    (*impl_).batch_query = (*impl_).zero_buffer(
        static_cast<size_t>(context) * hidden * sizeof(float));
    (*impl_).batch_key = (*impl_).zero_buffer(
        static_cast<size_t>(context) * hidden * sizeof(float));
    (*impl_).batch_value = (*impl_).zero_buffer(
        static_cast<size_t>(context) * hidden * sizeof(float));
    (*impl_).batch_gate_up = (*impl_).zero_buffer(
        static_cast<size_t>(context) * max_batch_intermediate * 2 * sizeof(float));
    (*impl_).batch_activated = (*impl_).zero_buffer(
        static_cast<size_t>(context) * max_batch_intermediate * sizeof(float));

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


}
