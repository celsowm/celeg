#include "lfm/detail/model_impl.hpp"
#include "lfm/kernels.cuh"
#include "lfm/quantization.hpp"
#include "lfm/paged_kv.hpp"
#include "lfm/model_variant.hpp"
#include "lfm/weight_layout.hpp"
#include "lfm/weight_loader.hpp"
#include "lfm/moe.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <utility>

namespace lfm {
namespace {

size_t checked_element_count(const std::vector<int64_t>& shape) {
    size_t count = 1;
    for (int64_t dim : shape) {
        if (dim < 0) throw std::runtime_error("negative tensor dimension");
        if (dim == 0) return 0;
        const size_t value = static_cast<size_t>(dim);
        if (count > std::numeric_limits<size_t>::max() / value) {
            throw std::overflow_error("tensor element count overflow");
        }
        count *= value;
    }
    return count;
}

std::string layer_name(int index, const std::string& suffix) {
    return "model.layers." + std::to_string(index) + "." + suffix;
}

// Layout for the new session-file header. The previous format is rejected;
// there is no migration path because the on-disk layout now depends on the
// variant fingerprint.
struct SessionHeader {
    std::array<char, 8> magic{{'L', 'F', 'M', 'S', 'E', 'S', 'S', '2'}};
    uint32_t version = 2;
    uint32_t kv_cache_mode = 0;
    int32_t position = 0;
    int32_t max_context = 0;
    int32_t layers = 0;
    int32_t kv_width = 0;
    int32_t kv_heads = 0;
    int32_t vocab = 0;
    int32_t attention_layers = 0;
    uint64_t rng_state = 0;
    char variant_id[32] = {};
};

template <typename T>
void write_scalar(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("failed writing session");
}

template <typename T>
void read_scalar(std::ifstream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("truncated session file");
}

std::filesystem::path config_path_for(const std::string& safetensors_path) {
    const std::filesystem::path p(safetensors_path);
    return p.parent_path() / "config.json";
}

} // namespace

struct LtPlan;  // defined in include/lfm/detail/model_types.hpp

void LinearWeight::validate_storage() const {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("linear weight dimensions must be positive");
    }
    switch (kind) {
        case LinearStorageKind::Bf16:
            if (!bf16 || int8 || int4 || scales) {
                throw std::runtime_error("invalid BF16 linear weight storage");
            }
            break;
        case LinearStorageKind::Int8:
            if (bf16 || !int8 || int4 || !scales) {
                throw std::runtime_error("invalid INT8 linear weight storage");
            }
            break;
        case LinearStorageKind::Int4:
            if (bf16 || int8 || !int4 || !scales) {
                throw std::runtime_error("invalid INT4 linear weight storage");
            }
            break;
    }
}

size_t SharedModelWeights::memory_bytes() const {
    size_t total = 0;
    for (const auto& [unused_name, weight] : tensors) {
        (void)unused_name;
        total += weight.bf16_storage.bytes() +
                 weight.int8_storage.bytes() +
                 weight.int4_storage.bytes() +
                 weight.scales_storage.bytes();
    }
    return total;
}

LfmModel::Impl::Impl(const std::string& safetensors_path,
                   int max_context,
                   ModelOptions options,
                   GenerationConfig generation)
    : plan_(ExecutionPlan::compile(options, max_context)),
      options_(plan_.options()),
      generation_(generation),
      stream_(),
      max_context_(max_context) {
    generation_.validate();
    if (max_context_ <= 0) {
        throw std::invalid_argument("max_context must be positive");
    }
    // Load the model topology from config.json next to the safetensors file.
    // The runtime no longer hard-codes the 230M shape.
    const std::filesystem::path config_path = config_path_for(safetensors_path);
    if (!std::filesystem::exists(config_path)) {
        throw std::runtime_error("config.json not found alongside checkpoint: " + config_path.string());
    }
    ModelConfig config = ModelConfig::load(config_path.string());
    shape_ = ModelShape::from_config(config);
    const IModelVariant& variant = ModelVariantRegistry::instance().select(shape_, config.repo_hint);
    variant_ = &variant;
    shape_ = variant.resolve_shape(shape_);
    (void)variant_;  // retained for future per-variant dispatch

    // Size all per-session device buffers from the runtime shape.
    seen_tokens_.reset(static_cast<size_t>(shape_.vocab_size));
    sampling_scores_.reset(static_cast<size_t>(shape_.vocab_size));
    hidden_.reset(static_cast<size_t>(shape_.hidden));
    residual_.reset(static_cast<size_t>(shape_.hidden));
    normed_.reset(static_cast<size_t>(shape_.hidden));
    op_output_.reset(static_cast<size_t>(shape_.hidden));
    qkv_output_.reset(static_cast<size_t>(shape_.qkv_width));
    conv_projected_.reset(static_cast<size_t>(3 * shape_.hidden));
    gate_up_.reset(static_cast<size_t>(2 * shape_.intermediate));
    activated_.reset(static_cast<size_t>(shape_.intermediate));
    mlp_output_.reset(static_cast<size_t>(shape_.hidden));
    logits_.reset(static_cast<size_t>(shape_.vocab_size));

    // MoE scratch (decode path: one token). Sized from the MoE topology when
    // present; harmless (tiny) for dense-only models.
    {
        const int E = shape_.num_experts > 0 ? shape_.num_experts : 1;
        const int K = shape_.experts_per_token > 0 ? shape_.experts_per_token : 1;
        const int inter = shape_.moe_intermediate > 0 ? shape_.moe_intermediate : 1;
        moe_hidden_float_.reset(static_cast<size_t>(shape_.hidden));
        moe_sel_.reset(static_cast<size_t>(K));
        moe_routing_w_.reset(static_cast<size_t>(K));
        moe_router_scratch_.reset(static_cast<size_t>(E));
        moe_output_.reset(static_cast<size_t>(shape_.hidden));
        moe_gu_scratch_.reset(static_cast<size_t>(K) * 2 * inter);
        moe_act_scratch_.reset(static_cast<size_t>(K) * inter);
        moe_router_float_.resize(static_cast<size_t>(shape_.num_hidden_layers));
    }

    attention_chunks_ = plan_.attention_chunks();
    if (attention_chunks_ > 0) {
        const size_t partials =
            static_cast<size_t>(shape_.num_attention_heads) * attention_chunks_;
        attention_partial_max_.reset(partials);
        attention_partial_denom_.reset(partials);
        attention_partial_accum_.reset(partials * shape_.head_dim);
    }
    if (options_.gemm_backend == GemmBackend::CublasLt &&
        options_.lt_workspace_bytes > 0) {
        // GemmDispatcher owns the workspace internally.
    }
    gemm_ = std::make_unique<GemmDispatcher>(stream_.get(), options_);
    initialize_rope_tables();

    // Immutable device weights are shared across all request lanes that use
    // the same checkpoint, device and quantization mode. The WeightLoader
    // owns the process-wide cache and the SafeTensor I/O + quantization;
    // the Impl only retains the resulting shared_ptr + the layer views.
    weights_ = WeightLoader::acquire(safetensors_path, options_.weight_mode);
    weight_loader_ = std::make_unique<WeightLoader>(weights_, options_.weight_mode);

    // Only one constructor populates a shared checkpoint at a time. Other
    // sessions wait, then reuse the immutable device buffers.
    std::unique_lock<std::mutex> shared_weights_lock(weights_->mutex);
    SafeTensorRepository repo(safetensors_path);
    embedding_ = weight_loader_->load_linear_weight(
        repo, "model.embed_tokens.weight",
        {shape_.vocab_size, shape_.hidden});
    final_norm_ = weight_loader_->load_weight(
        repo, "model.embedding_norm.weight", {shape_.hidden});
    {
        const void* table = nullptr;
        if (options_.weight_mode == WeightMode::Int8) {
            table = embedding_->int8;
        } else if (options_.weight_mode == WeightMode::Int4) {
            table = embedding_->int4;
        } else {
            table = embedding_->bf16;
        }
        weight_layout_ = make_weight_layout(
            options_.weight_mode, table, embedding_->scales);
    }

    layers_.reserve(static_cast<size_t>(shape_.num_hidden_layers));
    for (int i = 0; i < shape_.num_hidden_layers; ++i) {
        LayerCommon common_layer;
        common_layer.operator_norm = weight_loader_->load_weight(
            repo, layer_name(i, "operator_norm.weight"), {shape_.hidden});
        common_layer.ffn_norm = weight_loader_->load_weight(
            repo, layer_name(i, "ffn_norm.weight"), {shape_.hidden});
        if (shape_.layer_uses_moe(i)) {
            // Mixture-of-experts feed-forward for this layer.
            const int E = shape_.num_experts;
            const int inter = shape_.moe_intermediate;
            const float* expert_bias = nullptr;
            if (shape_.use_expert_bias) {
                expert_bias = weight_loader_->load_f32_weight(
                    repo, layer_name(i, "feed_forward.expert_bias.weight"),
                    {static_cast<int64_t>(E)});
            }
            const LinearWeight* router = weight_loader_->load_router(
                repo, i, E, shape_.hidden);
            const ExpertLinearWeight* gate_up =
                weight_loader_->load_moe_gate_up(repo, i, E, inter, shape_.hidden);
            const ExpertLinearWeight* down =
                weight_loader_->load_moe_down(repo, i, E, inter, shape_.hidden);

            // Cache a device float copy of the router weight for the CUDA
            // router kernel (it consumes float, the loaded weight is BF16).
            DeviceBuffer<float>& router_float = moe_router_float_[static_cast<size_t>(i)];
            router_float.reset(static_cast<size_t>(E) * shape_.hidden);
            launch_cast_bf16_to_float(
                router->bf16, router_float.data(),
                static_cast<int>(E) * shape_.hidden, stream_.get());

            common_layer.feed_forward = MoeFfnWeights{
                router, expert_bias, gate_up, down, router_float.data()};
        } else {
            const LinearWeight* w13 = weight_loader_->load_concat_linear_weight(
                repo, layer_name(i, "feed_forward.w13.weight"),
                {
                    {layer_name(i, "feed_forward.w1.weight"),
                     {shape_.intermediate, shape_.hidden}},
                    {layer_name(i, "feed_forward.w3.weight"),
                     {shape_.intermediate, shape_.hidden}},
                });
            const LinearWeight* w2 = weight_loader_->load_linear_weight(
                repo, layer_name(i, "feed_forward.w2.weight"),
                {shape_.hidden, shape_.intermediate});
            common_layer.feed_forward = DenseFfnWeights{w13, w2};
        }

        const LayerType layer_type =
            shape_.layer_types[static_cast<size_t>(i)];
        if (layer_type == LayerType::FullAttention) {
            AttentionLayer attention_layer;
            attention_layer.common = common_layer;
            attention_layer.qkv = weight_loader_->load_concat_linear_weight(
                repo, layer_name(i, "self_attn.qkv.weight"),
                {
                    {layer_name(i, "self_attn.q_proj.weight"),
                     {shape_.q_width, shape_.hidden}},
                    {layer_name(i, "self_attn.k_proj.weight"),
                     {shape_.kv_width, shape_.hidden}},
                    {layer_name(i, "self_attn.v_proj.weight"),
                     {shape_.kv_width, shape_.hidden}},
                });
            attention_layer.out = weight_loader_->load_linear_weight(
                repo, layer_name(i, "self_attn.out_proj.weight"),
                {shape_.hidden, shape_.hidden});
            attention_layer.q_norm = weight_loader_->load_weight(
                repo, layer_name(i, "self_attn.q_layernorm.weight"),
                {shape_.head_dim});
            attention_layer.k_norm = weight_loader_->load_weight(
                repo, layer_name(i, "self_attn.k_layernorm.weight"),
                {shape_.head_dim});

            if (options_.allocate_local_kv_cache) {
                const size_t cache_elements =
                    static_cast<size_t>(max_context_) * shape_.kv_width;
                if (options_.kv_cache_mode == KvCacheMode::Int8) {
                    attention_layer.key_cache_int8.reset(cache_elements);
                    attention_layer.value_cache_int8.reset(cache_elements);
                    const size_t scale_elements =
                        static_cast<size_t>(max_context_) * shape_.num_key_value_heads;
                    attention_layer.key_cache_scales.reset(scale_elements);
                    attention_layer.value_cache_scales.reset(scale_elements);
                } else {
                    attention_layer.key_cache.reset(cache_elements);
                    attention_layer.value_cache.reset(cache_elements);
                }
            }
            layers_.emplace_back(std::move(attention_layer));
        } else {
            ConvolutionLayer convolution_layer;
            convolution_layer.common = common_layer;
            convolution_layer.conv_in = weight_loader_->load_linear_weight(
                repo, layer_name(i, "conv.in_proj.weight"),
                {3 * shape_.hidden, shape_.hidden});
            convolution_layer.conv_weight = weight_loader_->load_weight(
                repo, layer_name(i, "conv.conv.weight"),
                {shape_.hidden, 1, shape_.conv_cache});
            convolution_layer.conv_out = weight_loader_->load_linear_weight(
                repo, layer_name(i, "conv.out_proj.weight"),
                {shape_.hidden, shape_.hidden});
            convolution_layer.conv_state.reset(
                static_cast<size_t>(shape_.conv_cache) * shape_.hidden);
            layers_.emplace_back(std::move(convolution_layer));
        }
    }

    shared_weights_lock.unlock();

    if (options_.cuda_graph ||
        options_.gemm_backend == GemmBackend::CublasLt) {
        warmup_decode_gemms();
    }
    local_kv_cache_available_ = options_.allocate_local_kv_cache;
    reset(options_.allocate_local_kv_cache);
}

LfmModel::Impl::~Impl() = default;


void LfmModel::Impl::initialize_rope_tables() {
    const size_t table_elements =
        static_cast<size_t>(max_context_) * shape_.rope_pairs;
    std::vector<__nv_bfloat16> cos_table(table_elements);
    std::vector<__nv_bfloat16> sin_table(table_elements);

    for (int position = 0; position < max_context_; ++position) {
        for (int pair = 0; pair < shape_.rope_pairs; ++pair) {
            const float exponent =
                -2.0f * static_cast<float>(pair) /
                static_cast<float>(shape_.head_dim);
            const float frequency = std::pow(shape_.rope_theta, exponent);
            const float angle = static_cast<float>(position) * frequency;
            const size_t index =
                static_cast<size_t>(position) * shape_.rope_pairs + pair;
            cos_table[index] = __float2bfloat16(std::cos(angle));
            sin_table[index] = __float2bfloat16(std::sin(angle));
        }
    }

    rope_cos_.reset(table_elements);
    rope_sin_.reset(table_elements);
    LFM_CUDA(cudaMemcpy(rope_cos_.data(), cos_table.data(),
                        cos_table.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
    LFM_CUDA(cudaMemcpy(rope_sin_.data(), sin_table.data(),
                        sin_table.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
}

void LfmModel::Impl::warmup_decode_gemms() {
    hidden_.zero_async(stream_.get());
    normed_.zero_async(stream_.get());
    op_output_.zero_async(stream_.get());
    qkv_output_.zero_async(stream_.get());
    conv_projected_.zero_async(stream_.get());
    gate_up_.zero_async(stream_.get());
    activated_.zero_async(stream_.get());
    logits_.zero_async(stream_.get());

    const AttentionLayer* attention_layer = nullptr;
    const ConvolutionLayer* convolution_layer = nullptr;
    for (const Layer& layer : layers_) {
        if (!attention_layer) attention_layer = as_attention(layer);
        if (!convolution_layer) convolution_layer = as_convolution(layer);
    }
    if (!attention_layer || !convolution_layer || layers_.empty()) {
        throw std::runtime_error("compiled layer map is incomplete");
    }
    const LayerCommon& first_common = common(layers_.front());

    if (options_.fused_projections) {
        linear(normed_.data(), *attention_layer->qkv, qkv_output_.data(),
               1, shape_.qkv_width, shape_.hidden);
        linear(normed_.data(), *as_dense_ffn(first_common.feed_forward)->w13, gate_up_.data(),
               1, 2 * shape_.intermediate, shape_.hidden);
    } else {
        const LinearWeight q_weight =
            slice_rows(*attention_layer->qkv, 0, shape_.q_width);
        const LinearWeight k_weight = slice_rows(
            *attention_layer->qkv, shape_.q_width, shape_.kv_width);
        const LinearWeight v_weight = slice_rows(
            *attention_layer->qkv, shape_.q_width + shape_.kv_width,
            shape_.kv_width);
        linear(normed_.data(), q_weight, qkv_output_.data(),
               1, shape_.q_width, shape_.hidden);
        linear(normed_.data(), k_weight, qkv_output_.data() + shape_.q_width,
               1, shape_.kv_width, shape_.hidden);
        linear(normed_.data(), v_weight,
               qkv_output_.data() + shape_.q_width + shape_.kv_width,
               1, shape_.kv_width, shape_.hidden);

        const LinearWeight w1 =
            slice_rows(*as_dense_ffn(first_common.feed_forward)->w13, 0, shape_.intermediate);
        const LinearWeight w3 = slice_rows(
            *as_dense_ffn(first_common.feed_forward)->w13, shape_.intermediate,
            shape_.intermediate);
        linear(normed_.data(), w1, gate_up_.data(),
               1, shape_.intermediate, shape_.hidden);
        linear(normed_.data(), w3, gate_up_.data() + shape_.intermediate,
               1, shape_.intermediate, shape_.hidden);
    }

    linear(op_output_.data(), *attention_layer->out, hidden_.data(),
           1, shape_.hidden, shape_.hidden,
           options_.fused_residuals ? 1.0f : 0.0f);
    linear(normed_.data(), *convolution_layer->conv_in, conv_projected_.data(),
           1, 3 * shape_.hidden, shape_.hidden);
    linear(activated_.data(), *as_dense_ffn(first_common.feed_forward)->w2, hidden_.data(),
           1, shape_.hidden, shape_.intermediate,
           options_.fused_residuals ? 1.0f : 0.0f);
    linear(normed_.data(), *embedding_, logits_.data(),
           1, shape_.vocab_size, shape_.hidden);
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
}

void LfmModel::Impl::reset(bool allocate_local_kv) {
    allocate_local_kv = allocate_local_kv && options_.allocate_local_kv_cache;
    if (allocate_local_kv && !local_kv_cache_available_) {
        const size_t cache_elements =
            static_cast<size_t>(max_context_) * shape_.kv_width;
        const size_t scale_elements =
            static_cast<size_t>(max_context_) * shape_.num_key_value_heads;
        for (Layer& layer : layers_) {
            AttentionLayer* attention = as_attention(layer);
            if (!attention) continue;
            if (options_.kv_cache_mode == KvCacheMode::Int8) {
                attention->key_cache_int8.reset(cache_elements);
                attention->value_cache_int8.reset(cache_elements);
                attention->key_cache_scales.reset(scale_elements);
                attention->value_cache_scales.reset(scale_elements);
            } else {
                attention->key_cache.reset(cache_elements);
                attention->value_cache.reset(cache_elements);
            }
        }
        local_kv_cache_available_ = true;
    }
    position_ = 0;
    phase_ = SessionPhase::Empty;
    const int32_t zero = 0;
    uint64_t seed = generation_.seed;
    LFM_CUDA(cudaMemcpyAsync(position_device_.data(), &zero, sizeof(zero),
                             cudaMemcpyHostToDevice, stream_.get()));
    LFM_CUDA(cudaMemcpyAsync(rng_state_.data(), &seed, sizeof(seed),
                             cudaMemcpyHostToDevice, stream_.get()));
    seen_tokens_.zero_async(stream_.get());
    for (Layer& layer : layers_) {
        if (AttentionLayer* attention = as_attention(layer)) {
            if (!local_kv_cache_available_) continue;
            if (options_.kv_cache_mode == KvCacheMode::Int8) {
                attention->key_cache_int8.zero_async(stream_.get());
                attention->value_cache_int8.zero_async(stream_.get());
                attention->key_cache_scales.zero_async(stream_.get());
                attention->value_cache_scales.zero_async(stream_.get());
            } else {
                attention->key_cache.zero_async(stream_.get());
                attention->value_cache.zero_async(stream_.get());
            }
        } else {
            as_convolution(layer)->conv_state.zero_async(stream_.get());
        }
    }
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
}


void LfmModel::Impl::allocate_prefill_workspace(int rows) {
    const size_t r = static_cast<size_t>(rows);
    prefill_tokens_.reset(r);
    prefill_hidden_.reset(r * shape_.hidden);
    prefill_residual_.reset(r * shape_.hidden);
    prefill_normed_.reset(r * shape_.hidden);
    prefill_op_output_.reset(r * shape_.hidden);
    prefill_q_.reset(r * shape_.q_width);
    prefill_k_.reset(r * shape_.kv_width);
    prefill_v_.reset(r * shape_.kv_width);
    prefill_conv_projected_.reset(r * 3 * shape_.hidden);
    prefill_gate_up_.reset(r * 2 * shape_.intermediate);
    prefill_activated_.reset(r * shape_.intermediate);
    prefill_mlp_output_.reset(r * shape_.hidden);
}

void LfmModel::Impl::release_prefill_workspace() {
    prefill_tokens_.reset(0);
    prefill_hidden_.reset(0);
    prefill_residual_.reset(0);
    prefill_normed_.reset(0);
    prefill_op_output_.reset(0);
    prefill_q_.reset(0);
    prefill_k_.reset(0);
    prefill_v_.reset(0);
    prefill_conv_projected_.reset(0);
    prefill_gate_up_.reset(0);
    prefill_activated_.reset(0);
    prefill_mlp_output_.reset(0);
}

void LfmModel::Impl::run_mlp_decode(const LayerCommon& common_layer) {
    if (const MoeFfnWeights* moe = as_moe_ffn(common_layer.feed_forward)) {
        (void)moe;
        run_mlp_moe_decode(common_layer);
        return;
    }
    launch_rmsnorm(hidden_.data(), common_layer.ffn_norm, normed_.data(),
                   1, shape_.hidden, shape_.norm_eps,
                   stream_.get());
    if (options_.fused_projections) {
        linear(normed_.data(), *as_dense_ffn(common_layer.feed_forward)->w13, gate_up_.data(),
               1, 2 * shape_.intermediate, shape_.hidden);
    } else {
        const LinearWeight w1 =
            slice_rows(*as_dense_ffn(common_layer.feed_forward)->w13, 0, shape_.intermediate);
        const LinearWeight w3 = slice_rows(
            *as_dense_ffn(common_layer.feed_forward)->w13, shape_.intermediate, shape_.intermediate);
        linear(normed_.data(), w1, gate_up_.data(),
               1, shape_.intermediate, shape_.hidden);
        linear(normed_.data(), w3, gate_up_.data() + shape_.intermediate,
               1, shape_.intermediate, shape_.hidden);
    }
    launch_swiglu_fused(gate_up_.data(), activated_.data(),
                        shape_.intermediate, stream_.get());
    if (options_.fused_residuals) {
        linear(activated_.data(), *as_dense_ffn(common_layer.feed_forward)->w2, hidden_.data(),
               1, shape_.hidden, shape_.intermediate, 1.0f);
    } else {
        linear(activated_.data(), *as_dense_ffn(common_layer.feed_forward)->w2, mlp_output_.data(),
               1, shape_.hidden, shape_.intermediate);
        launch_residual_add(hidden_.data(), mlp_output_.data(),
                            shape_.hidden, stream_.get());
    }
}

void LfmModel::Impl::run_mlp_prefill(const LayerCommon& common_layer, int rows) {
    if (const MoeFfnWeights* moe = as_moe_ffn(common_layer.feed_forward)) {
        (void)moe;
        run_mlp_moe_prefill(common_layer, rows);
        return;
    }
    const size_t matrix_elements =
        static_cast<size_t>(rows) * shape_.intermediate;
    launch_rmsnorm(prefill_hidden_.data(), common_layer.ffn_norm,
                   prefill_normed_.data(), rows, shape_.hidden,
                   shape_.norm_eps, stream_.get());
    if (options_.fused_projections) {
        linear(prefill_normed_.data(), *as_dense_ffn(common_layer.feed_forward)->w13, prefill_gate_up_.data(),
               rows, 2 * shape_.intermediate, shape_.hidden);
    } else {
        const LinearWeight w1 =
            slice_rows(*as_dense_ffn(common_layer.feed_forward)->w13, 0, shape_.intermediate);
        const LinearWeight w3 = slice_rows(
            *as_dense_ffn(common_layer.feed_forward)->w13, shape_.intermediate, shape_.intermediate);
        linear(prefill_normed_.data(), w1, prefill_gate_up_.data(),
               rows, shape_.intermediate, shape_.hidden);
        linear(prefill_normed_.data(), w3,
               prefill_gate_up_.data() + matrix_elements,
               rows, shape_.intermediate, shape_.hidden);
    }
    launch_swiglu_fused(prefill_gate_up_.data(), prefill_activated_.data(),
                        static_cast<int>(matrix_elements), stream_.get());
    if (options_.fused_residuals) {
        linear(prefill_activated_.data(), *as_dense_ffn(common_layer.feed_forward)->w2, prefill_hidden_.data(),
               rows, shape_.hidden, shape_.intermediate, 1.0f);
    } else {
        linear(prefill_activated_.data(), *as_dense_ffn(common_layer.feed_forward)->w2, prefill_mlp_output_.data(),
                rows, shape_.hidden, shape_.intermediate);
        launch_residual_add(prefill_hidden_.data(), prefill_mlp_output_.data(),
                            rows * shape_.hidden, stream_.get());
    }
}

namespace {

// MoE router config / FFN device descriptors are provided inline by
// lfm/detail/model_types.hpp (moe_router_config / moe_ffn_device) so the
// standalone paths and the packed executor share one definition.

} // namespace

void LfmModel::Impl::run_mlp_moe_decode(const LayerCommon& common_layer) {
    const MoeFfnWeights& moe = *as_moe_ffn(common_layer.feed_forward);
    launch_rmsnorm(hidden_.data(), common_layer.ffn_norm, normed_.data(),
                   1, shape_.hidden, shape_.norm_eps, stream_.get());
    // Router: BF16 normed hidden -> float -> top-K experts.
    launch_cast_bf16_to_float(normed_.data(), moe_hidden_float_.data(),
                              shape_.hidden, stream_.get());
    const lfm::MoeRouterConfig cfg = moe_router_config(shape_);
    lfm::MoeRouterDevice rdev;
    rdev.router_weight = moe.router_float;
    rdev.expert_bias = moe.expert_bias;
    rdev.hidden_data = moe_hidden_float_.data();
    rdev.selected_experts = moe_sel_.data();
    rdev.routing_weights = moe_routing_w_.data();
    rdev.rows = 1;
    rdev.hidden_dim = shape_.hidden;
    launch_moe_router(rdev, cfg, moe_router_scratch_.data(), stream_.get());

    // Expert FFN: accumulate the routing-weighted expert outputs into moe_output_.
    moe_output_.zero_async(stream_.get());
    const lfm::MoeFfnDevice fdev = moe_ffn_device(moe, shape_);
    launch_moe_ffn(fdev, moe_sel_.data(), moe_routing_w_.data(),
                   normed_.data(), moe_output_.data(), 1, shape_.experts_per_token,
                   moe_gu_scratch_.data(), moe_act_scratch_.data(), stream_.get());

    // Residual add into the hidden state.
    launch_residual_add(hidden_.data(), moe_output_.data(),
                        shape_.hidden, stream_.get());
}

void LfmModel::Impl::run_mlp_moe_prefill(const LayerCommon& common_layer, int rows) {
    const MoeFfnWeights& moe = *as_moe_ffn(common_layer.feed_forward);
    // Size the prefill scratch to the requested row count.
    moe_pf_hidden_float_.reset(static_cast<size_t>(rows) * shape_.hidden);
    moe_pf_sel_.reset(static_cast<size_t>(rows) * shape_.experts_per_token);
    moe_pf_routing_w_.reset(static_cast<size_t>(rows) * shape_.experts_per_token);
    moe_pf_router_scratch_.reset(static_cast<size_t>(rows) * shape_.num_experts);
    moe_pf_output_.reset(static_cast<size_t>(rows) * shape_.hidden);
    moe_pf_gu_scratch_.reset(
        static_cast<size_t>(rows) * shape_.experts_per_token * 2 * shape_.moe_intermediate);
    moe_pf_act_scratch_.reset(
        static_cast<size_t>(rows) * shape_.experts_per_token * shape_.moe_intermediate);

    launch_rmsnorm(prefill_hidden_.data(), common_layer.ffn_norm,
                   prefill_normed_.data(), rows, shape_.hidden, shape_.norm_eps,
                   stream_.get());
    launch_cast_bf16_to_float(prefill_normed_.data(), moe_pf_hidden_float_.data(),
                              rows * shape_.hidden, stream_.get());
    const lfm::MoeRouterConfig cfg = moe_router_config(shape_);
    lfm::MoeRouterDevice rdev;
    rdev.router_weight = moe.router_float;
    rdev.expert_bias = moe.expert_bias;
    rdev.hidden_data = moe_pf_hidden_float_.data();
    rdev.selected_experts = moe_pf_sel_.data();
    rdev.routing_weights = moe_pf_routing_w_.data();
    rdev.rows = rows;
    rdev.hidden_dim = shape_.hidden;
    launch_moe_router(rdev, cfg, moe_pf_router_scratch_.data(), stream_.get());

    moe_pf_output_.zero_async(stream_.get());
    const lfm::MoeFfnDevice fdev = moe_ffn_device(moe, shape_);
    launch_moe_ffn(fdev, moe_pf_sel_.data(), moe_pf_routing_w_.data(),
                   prefill_normed_.data(), moe_pf_output_.data(), rows,
                   shape_.experts_per_token, moe_pf_gu_scratch_.data(),
                   moe_pf_act_scratch_.data(), stream_.get());

    launch_residual_add(prefill_hidden_.data(), moe_pf_output_.data(),
                        rows * shape_.hidden, stream_.get());
}

void LfmModel::Impl::prefill_batched(const std::vector<int32_t>& tokens) {
    reset();
    const int rows = static_cast<int>(tokens.size());
    allocate_prefill_workspace(rows);

    LFM_CUDA(cudaMemcpyAsync(prefill_tokens_.data(), tokens.data(),
                             tokens.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    launch_mark_seen_batch(prefill_tokens_.data(), rows, seen_tokens_.data(),
                           shape_.vocab_size, stream_.get());
    weight_layout_->embed_batch(
        prefill_tokens_.data(), rows, prefill_hidden_.data(),
        shape_.hidden, stream_.get());

    for (Layer& layer : layers_) {
        LayerCommon& common_layer = common(layer);
        if (!options_.fused_residuals) {
            LFM_CUDA(cudaMemcpyAsync(
                prefill_residual_.data(), prefill_hidden_.data(),
                prefill_hidden_.bytes(), cudaMemcpyDeviceToDevice,
                stream_.get()));
        }
        launch_rmsnorm(prefill_hidden_.data(), common_layer.operator_norm,
                       prefill_normed_.data(), rows, shape_.hidden,
                       shape_.norm_eps, stream_.get());

        if (AttentionLayer* attention = as_attention(layer)) {
            const LinearWeight q_weight =
                slice_rows(*attention->qkv, 0, shape_.q_width);
            const LinearWeight k_weight = slice_rows(
                *attention->qkv, shape_.q_width, shape_.kv_width);
            const LinearWeight v_weight = slice_rows(
                *attention->qkv, shape_.q_width + shape_.kv_width,
                shape_.kv_width);
            linear(prefill_normed_.data(), q_weight, prefill_q_.data(),
                   rows, shape_.q_width, shape_.hidden);
            linear(prefill_normed_.data(), k_weight, prefill_k_.data(),
                   rows, shape_.kv_width, shape_.hidden);
            linear(prefill_normed_.data(), v_weight, prefill_v_.data(),
                   rows, shape_.kv_width, shape_.hidden);

            if (options_.fast_attention) {
                launch_qk_norm_rope_prefill_fast(
                    prefill_q_.data(), prefill_k_.data(),
                    attention->q_norm, attention->k_norm,
                    rope_cos_.data(), rope_sin_.data(), rows,
                    shape_.num_attention_heads, shape_.num_key_value_heads,
                    shape_.head_dim, shape_.norm_eps,
                    stream_.get());
            } else {
                launch_qk_norm_rope_prefill_strict(
                    prefill_q_.data(), prefill_k_.data(),
                    attention->q_norm, attention->k_norm,
                    rope_cos_.data(), rope_sin_.data(), rows,
                    shape_.num_attention_heads, shape_.num_key_value_heads,
                    shape_.head_dim, shape_.norm_eps,
                    stream_.get());
            }
            if (options_.kv_cache_mode == KvCacheMode::Int8) {
                launch_store_kv_int8_prefill(
                    prefill_k_.data(), prefill_v_.data(),
                    attention->key_cache_int8.data(), attention->value_cache_int8.data(),
                    attention->key_cache_scales.data(), attention->value_cache_scales.data(),
                    rows, shape_.num_key_value_heads, shape_.head_dim,
                    stream_.get());
                if (options_.fast_attention) {
                    launch_gqa_prefill_online_int8(
                        prefill_q_.data(), attention->key_cache_int8.data(),
                        attention->value_cache_int8.data(),
                        attention->key_cache_scales.data(),
                        attention->value_cache_scales.data(),
                        prefill_op_output_.data(), rows, shape_.num_attention_heads,
                        shape_.num_key_value_heads, shape_.head_dim, stream_.get());
                } else {
                    launch_gqa_prefill_strict_int8(
                        prefill_q_.data(), attention->key_cache_int8.data(),
                        attention->value_cache_int8.data(),
                        attention->key_cache_scales.data(),
                        attention->value_cache_scales.data(),
                        prefill_op_output_.data(), rows, shape_.num_attention_heads,
                        shape_.num_key_value_heads, shape_.head_dim, stream_.get());
                }
            } else {
                launch_store_kv_prefill(
                    prefill_k_.data(), prefill_v_.data(),
                    attention->key_cache.data(), attention->value_cache.data(),
                    rows, shape_.kv_width, stream_.get());
                if (options_.fast_attention) {
                    launch_gqa_prefill_online(
                        prefill_q_.data(), attention->key_cache.data(),
                        attention->value_cache.data(), prefill_op_output_.data(),
                        rows, shape_.num_attention_heads, shape_.num_key_value_heads,
                        shape_.head_dim, stream_.get());
                } else {
                    launch_gqa_prefill_strict(
                        prefill_q_.data(), attention->key_cache.data(),
                        attention->value_cache.data(), prefill_op_output_.data(),
                        rows, shape_.num_attention_heads, shape_.num_key_value_heads,
                        shape_.head_dim, stream_.get());
                }
            }
            linear(prefill_op_output_.data(), *attention->out,
                   prefill_hidden_.data(), rows, shape_.hidden,
                   shape_.hidden, options_.fused_residuals ? 1.0f : 0.0f);
        } else {
            ConvolutionLayer& convolution = *as_convolution(layer);
            linear(prefill_normed_.data(), *convolution.conv_in,
                   prefill_conv_projected_.data(), rows,
                   3 * shape_.hidden, shape_.hidden);
            launch_conv_prefill(
                prefill_conv_projected_.data(), convolution.conv_weight,
                convolution.conv_state.data(), prefill_op_output_.data(),
                rows, shape_.hidden, shape_.conv_cache,
                stream_.get());
            linear(prefill_op_output_.data(), *convolution.conv_out,
                   prefill_hidden_.data(), rows, shape_.hidden,
                   shape_.hidden, options_.fused_residuals ? 1.0f : 0.0f);
        }

        if (!options_.fused_residuals) {
            launch_residual_add(prefill_hidden_.data(), prefill_residual_.data(),
                                rows * shape_.hidden, stream_.get());
        }
        run_mlp_prefill(common_layer, rows);
    }

    const __nv_bfloat16* last_hidden = prefill_hidden_.data() +
        static_cast<size_t>(rows - 1) * shape_.hidden;
    launch_rmsnorm(last_hidden, final_norm_, normed_.data(),
                   1, shape_.hidden, shape_.norm_eps,
                   stream_.get());
    linear(normed_.data(), *embedding_, logits_.data(),
           1, shape_.vocab_size, shape_.hidden);

    position_ = rows;
    LFM_CUDA(cudaMemcpyAsync(position_device_.data(), &position_,
                             sizeof(position_), cudaMemcpyHostToDevice,
                             stream_.get()));
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    release_prefill_workspace();
    phase_ = SessionPhase::Ready;
}

void LfmModel::Impl::forward_token_host(int32_t token, bool compute_logits) {
    if (position_ >= max_context_) {
        throw std::runtime_error("context limit reached");
    }
    weight_layout_->embed_token(
        token, hidden_.data(), shape_.hidden, stream_.get());

    for (Layer& layer : layers_) {
        LayerCommon& common_layer = common(layer);
        if (!options_.fused_residuals) {
            LFM_CUDA(cudaMemcpyAsync(
                residual_.data(), hidden_.data(), hidden_.bytes(),
                cudaMemcpyDeviceToDevice, stream_.get()));
        }
        launch_rmsnorm(hidden_.data(), common_layer.operator_norm, normed_.data(),
                       1, shape_.hidden, shape_.norm_eps,
                       stream_.get());
        if (AttentionLayer* attention = as_attention(layer)) {
            __nv_bfloat16* q = qkv_output_.data();
            __nv_bfloat16* k = q + shape_.q_width;
            __nv_bfloat16* v = k + shape_.kv_width;
            if (options_.fused_projections) {
                linear(normed_.data(), *attention->qkv, qkv_output_.data(),
                       1, shape_.qkv_width, shape_.hidden);
            } else {
                const LinearWeight q_weight =
                    slice_rows(*attention->qkv, 0, shape_.q_width);
                const LinearWeight k_weight = slice_rows(
                    *attention->qkv, shape_.q_width, shape_.kv_width);
                const LinearWeight v_weight = slice_rows(
                    *attention->qkv, shape_.q_width + shape_.kv_width,
                    shape_.kv_width);
                linear(normed_.data(), q_weight, q,
                       1, shape_.q_width, shape_.hidden);
                linear(normed_.data(), k_weight, k,
                       1, shape_.kv_width, shape_.hidden);
                linear(normed_.data(), v_weight, v,
                       1, shape_.kv_width, shape_.hidden);
            }
            if (options_.fast_attention) {
                launch_qk_norm_rope_fast(
                    q, k, attention->q_norm, attention->k_norm,
                    rope_cos_.data(), rope_sin_.data(),
                    shape_.num_attention_heads, shape_.num_key_value_heads,
                    shape_.head_dim, position_,
                    shape_.norm_eps, stream_.get());
            } else {
                launch_qk_norm_rope_strict(
                    q, k, attention->q_norm, attention->k_norm,
                    rope_cos_.data(), rope_sin_.data(),
                    shape_.num_attention_heads, shape_.num_key_value_heads,
                    shape_.head_dim, position_,
                    shape_.norm_eps, stream_.get());
            }
            if (options_.kv_cache_mode == KvCacheMode::Int8) {
                launch_store_kv_int8(
                    k, v, attention->key_cache_int8.data(),
                    attention->value_cache_int8.data(),
                    attention->key_cache_scales.data(),
                    attention->value_cache_scales.data(), position_,
                    shape_.num_key_value_heads, shape_.head_dim, stream_.get());
                if (options_.fast_attention) {
                    launch_gqa_decode_online_int8(
                        q, attention->key_cache_int8.data(),
                        attention->value_cache_int8.data(),
                        attention->key_cache_scales.data(),
                        attention->value_cache_scales.data(), op_output_.data(),
                        position_ + 1, shape_.num_attention_heads,
                        shape_.num_key_value_heads, shape_.head_dim, stream_.get());
                } else {
                    launch_gqa_decode_strict_int8(
                        q, attention->key_cache_int8.data(),
                        attention->value_cache_int8.data(),
                        attention->key_cache_scales.data(),
                        attention->value_cache_scales.data(), op_output_.data(),
                        position_ + 1, shape_.num_attention_heads,
                        shape_.num_key_value_heads, shape_.head_dim, stream_.get());
                }
            } else {
                launch_store_kv(k, v, attention->key_cache.data(),
                                attention->value_cache.data(), position_,
                                shape_.kv_width, stream_.get());
                if (options_.fast_attention) {
                    launch_gqa_decode_online(
                        q, attention->key_cache.data(), attention->value_cache.data(),
                        op_output_.data(), position_ + 1,
                        shape_.num_attention_heads, shape_.num_key_value_heads,
                        shape_.head_dim, stream_.get());
                } else {
                    launch_gqa_decode_strict(
                        q, attention->key_cache.data(), attention->value_cache.data(),
                        op_output_.data(), position_ + 1,
                        shape_.num_attention_heads, shape_.num_key_value_heads,
                        shape_.head_dim, stream_.get());
                }
            }
            linear(op_output_.data(), *attention->out, hidden_.data(),
                   1, shape_.hidden, shape_.hidden,
                   options_.fused_residuals ? 1.0f : 0.0f);
        } else {
            ConvolutionLayer& convolution = *as_convolution(layer);
            linear(normed_.data(), *convolution.conv_in, conv_projected_.data(),
                   1, 3 * shape_.hidden, shape_.hidden);
            launch_conv_decode(
                conv_projected_.data(), convolution.conv_weight,
                convolution.conv_state.data(), op_output_.data(),
                shape_.hidden, shape_.conv_cache, position_,
                stream_.get());
            linear(op_output_.data(), *convolution.conv_out, hidden_.data(),
                   1, shape_.hidden, shape_.hidden,
                   options_.fused_residuals ? 1.0f : 0.0f);
        }
        if (!options_.fused_residuals) {
            launch_residual_add(hidden_.data(), residual_.data(),
                                shape_.hidden, stream_.get());
        }
        run_mlp_decode(common_layer);
    }
    if (compute_logits) {
        launch_rmsnorm(hidden_.data(), final_norm_, normed_.data(),
                       1, shape_.hidden, shape_.norm_eps,
                       stream_.get());
        linear(normed_.data(), *embedding_, logits_.data(),
               1, shape_.vocab_size, shape_.hidden);
    }
    ++position_;
}


void LfmModel::Impl::forward_token_paged_host(
    int32_t token, bool compute_logits, PhysicalPagedKvCache& paged_kv,
    const uint32_t* device_page_table, int page_table_stride) {
    if (position_ >= max_context_) {
        throw std::runtime_error("context limit reached");
    }
    if (paged_kv.mode() != options_.kv_cache_mode) {
        throw std::invalid_argument("model and physical paged KV modes differ");
    }
    weight_layout_->embed_token(
        token, hidden_.data(), shape_.hidden, stream_.get());

    for (int layer_index = 0; layer_index < shape_.num_hidden_layers; ++layer_index) {
        Layer& layer = layers_[static_cast<size_t>(layer_index)];
        LayerCommon& common_layer = common(layer);
        if (!options_.fused_residuals) {
            LFM_CUDA(cudaMemcpyAsync(residual_.data(), hidden_.data(), hidden_.bytes(),
                                     cudaMemcpyDeviceToDevice, stream_.get()));
        }
        launch_rmsnorm(hidden_.data(), common_layer.operator_norm, normed_.data(),
                       1, shape_.hidden, shape_.norm_eps, stream_.get());
        if (AttentionLayer* attention = as_attention(layer)) {
            __nv_bfloat16* q = qkv_output_.data();
            __nv_bfloat16* k = q + shape_.q_width;
            __nv_bfloat16* v = k + shape_.kv_width;
            if (options_.fused_projections) {
                linear(normed_.data(), *attention->qkv, qkv_output_.data(), 1,
                       shape_.qkv_width, shape_.hidden);
            } else {
                const LinearWeight q_weight =
                    slice_rows(*attention->qkv, 0, shape_.q_width);
                const LinearWeight k_weight = slice_rows(
                    *attention->qkv, shape_.q_width, shape_.kv_width);
                const LinearWeight v_weight = slice_rows(
                    *attention->qkv, shape_.q_width + shape_.kv_width,
                    shape_.kv_width);
                linear(normed_.data(), q_weight, q, 1, shape_.q_width,
                       shape_.hidden);
                linear(normed_.data(), k_weight, k, 1, shape_.kv_width,
                       shape_.hidden);
                linear(normed_.data(), v_weight, v, 1, shape_.kv_width,
                       shape_.hidden);
            }
            if (options_.fast_attention) {
                launch_qk_norm_rope_fast(
                    q, k, attention->q_norm, attention->k_norm, rope_cos_.data(),
                    rope_sin_.data(), shape_.num_attention_heads, shape_.num_key_value_heads,
                    shape_.head_dim, position_, shape_.norm_eps,
                    stream_.get());
            } else {
                launch_qk_norm_rope_strict(
                    q, k, attention->q_norm, attention->k_norm, rope_cos_.data(),
                    rope_sin_.data(), shape_.num_attention_heads, shape_.num_key_value_heads,
                    shape_.head_dim, position_, shape_.norm_eps,
                    stream_.get());
            }
            const int slot = paged_kv.attention_slot(layer_index);
            if (slot < 0) throw std::logic_error("attention layer has no page slot");
            if (options_.kv_cache_mode == KvCacheMode::Int8) {
                launch_store_kv_int8_paged_batch(
                    k, v, paged_kv.key_int8(), paged_kv.value_int8(),
                    paged_kv.key_scales(), paged_kv.value_scales(),
                    device_page_table, page_table_stride, position_device_.data(),
                    1, slot, paged_kv.page_tokens(),
                    paged_kv.attention_layers(), shape_.num_key_value_heads,
                    shape_.head_dim, stream_.get());
                if (use_segmented_attention(position_)) {
                    const int chunks = (position_ + 1 +
                        options_.attention_chunk_tokens - 1) /
                        options_.attention_chunk_tokens;
                    launch_gqa_decode_int8_paged_segmented_batch(
                        q, paged_kv.key_int8(), paged_kv.value_int8(),
                        paged_kv.key_scales(), paged_kv.value_scales(),
                        device_page_table, page_table_stride, op_output_.data(),
                        position_device_.data(), 1, slot,
                        paged_kv.page_tokens(),
                        paged_kv.attention_layers(),
                        shape_.num_attention_heads, shape_.num_key_value_heads,
                        shape_.head_dim, options_.attention_chunk_tokens,
                        chunks, attention_partial_max_.data(),
                        attention_partial_denom_.data(),
                        attention_partial_accum_.data(), stream_.get());
                } else {
                    launch_gqa_decode_int8_paged_batch(
                        q, paged_kv.key_int8(), paged_kv.value_int8(),
                        paged_kv.key_scales(), paged_kv.value_scales(),
                        device_page_table, page_table_stride, op_output_.data(),
                        position_device_.data(), 1, slot,
                        paged_kv.page_tokens(),
                        paged_kv.attention_layers(),
                        shape_.num_attention_heads, shape_.num_key_value_heads,
                        shape_.head_dim, options_.fast_attention,
                        stream_.get());
                }
            } else {
                launch_store_kv_paged_batch(
                    k, v, paged_kv.key_bf16(), paged_kv.value_bf16(),
                    device_page_table, page_table_stride, position_device_.data(),
                    1, slot, paged_kv.page_tokens(),
                    paged_kv.attention_layers(), shape_.num_key_value_heads,
                    shape_.head_dim, stream_.get());
                if (use_segmented_attention(position_)) {
                    const int chunks = (position_ + 1 +
                        options_.attention_chunk_tokens - 1) /
                        options_.attention_chunk_tokens;
                    launch_gqa_decode_paged_segmented_batch(
                        q, paged_kv.key_bf16(), paged_kv.value_bf16(),
                        device_page_table, page_table_stride,
                        op_output_.data(), position_device_.data(), 1, slot,
                        paged_kv.page_tokens(),
                        paged_kv.attention_layers(),
                        shape_.num_attention_heads, shape_.num_key_value_heads,
                        shape_.head_dim, options_.attention_chunk_tokens,
                        chunks, attention_partial_max_.data(),
                        attention_partial_denom_.data(),
                        attention_partial_accum_.data(), stream_.get());
                } else {
                    launch_gqa_decode_paged_batch(
                        q, paged_kv.key_bf16(), paged_kv.value_bf16(),
                        device_page_table, page_table_stride,
                        op_output_.data(), position_device_.data(), 1, slot,
                        paged_kv.page_tokens(),
                        paged_kv.attention_layers(),
                        shape_.num_attention_heads, shape_.num_key_value_heads,
                        shape_.head_dim, options_.fast_attention,
                        stream_.get());
                }
            }
            linear(op_output_.data(), *attention->out, hidden_.data(), 1,
                   shape_.hidden, shape_.hidden,
                   options_.fused_residuals ? 1.0f : 0.0f);
        } else {
            ConvolutionLayer& convolution = *as_convolution(layer);
            linear(normed_.data(), *convolution.conv_in, conv_projected_.data(), 1,
                   3 * shape_.hidden, shape_.hidden);
            launch_conv_decode(conv_projected_.data(), convolution.conv_weight,
                               convolution.conv_state.data(), op_output_.data(),
                               shape_.hidden, shape_.conv_cache,
                               position_, stream_.get());
            linear(op_output_.data(), *convolution.conv_out, hidden_.data(), 1,
                   shape_.hidden, shape_.hidden,
                   options_.fused_residuals ? 1.0f : 0.0f);
        }
        if (!options_.fused_residuals) {
            launch_residual_add(hidden_.data(), residual_.data(),
                                shape_.hidden, stream_.get());
        }
        run_mlp_decode(common_layer);
    }
    if (compute_logits) {
        launch_rmsnorm(hidden_.data(), final_norm_, normed_.data(), 1,
                       shape_.hidden, shape_.norm_eps, stream_.get());
        linear(normed_.data(), *embedding_, logits_.data(), 1,
               shape_.vocab_size, shape_.hidden);
    }
    ++position_;
    LFM_CUDA(cudaMemcpyAsync(position_device_.data(), &position_, sizeof(position_),
                             cudaMemcpyHostToDevice, stream_.get()));
}

void LfmModel::Impl::prefill_chunk_paged(
    const std::vector<int32_t>& tokens, bool begin, bool finalize,
    PhysicalPagedKvCache& paged_kv,
    const std::vector<uint32_t>& page_table) {
    if (tokens.empty()) {
        throw std::invalid_argument("prefill_chunk_paged needs at least one token");
    }
    if (paged_kv.mode() != options_.kv_cache_mode) {
        throw std::invalid_argument("model and physical paged KV modes differ");
    }
    if (begin) {
        release_local_kv_cache();
        reset(false);
        metrics_ = {};
    } else if (position_ == 0) {
        throw std::runtime_error(
            "paged prefill continuation requires an initial chunk or prefix state");
    }
    if (position_ + static_cast<int>(tokens.size()) > max_context_) {
        throw std::invalid_argument("paged prefill chunks exceed max_context");
    }
    const size_t final_position =
        static_cast<size_t>(position_) + tokens.size();
    const size_t pages_needed =
        (final_position + static_cast<size_t>(paged_kv.page_tokens()) - 1) /
        static_cast<size_t>(paged_kv.page_tokens());
    if (page_table.size() < pages_needed ||
        page_table.size() > static_cast<size_t>(paged_kv.max_pages_per_request())) {
        throw std::invalid_argument("paged prefill page table has invalid length");
    }
    if (paged_page_table_.size() < page_table.size()) {
        paged_page_table_.reset(page_table.size());
    }
    LFM_CUDA(cudaMemcpyAsync(paged_page_table_.data(), page_table.data(),
                             page_table.size() * sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    if (paged_prefill_tokens_.size() < tokens.size()) {
        paged_prefill_tokens_.reset(tokens.size());
    }
    LFM_CUDA(cudaMemcpyAsync(paged_prefill_tokens_.data(), tokens.data(),
                             tokens.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    launch_mark_seen_batch(paged_prefill_tokens_.data(),
                           static_cast<int>(tokens.size()), seen_tokens_.data(),
                           shape_.vocab_size, stream_.get());
    phase_ = SessionPhase::Prefilling;
    const auto started = std::chrono::steady_clock::now();
    for (size_t i = 0; i < tokens.size(); ++i) {
        forward_token_paged_host(tokens[i], finalize && i + 1 == tokens.size(),
                                 paged_kv, paged_page_table_.data(),
                                 static_cast<int>(page_table.size()));
    }
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    const auto ended = std::chrono::steady_clock::now();
    metrics_.last_prefill_ms +=
        std::chrono::duration<double, std::milli>(ended - started).count();
    metrics_.prefill_tokens += tokens.size();
    if (finalize) {
        phase_ = SessionPhase::Ready;
        active_segmented_attention_ = use_segmented_attention(position_);
    }
}

void LfmModel::Impl::prefill_legacy(const std::vector<int32_t>& tokens) {
    reset();
    DeviceBuffer<int32_t> input(tokens.size());
    LFM_CUDA(cudaMemcpyAsync(input.data(), tokens.data(),
                             tokens.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    launch_mark_seen_batch(input.data(), static_cast<int>(tokens.size()),
                           seen_tokens_.data(), shape_.vocab_size,
                           stream_.get());
    for (size_t i = 0; i < tokens.size(); ++i) {
        forward_token_host(tokens[i], i + 1 == tokens.size());
    }
    LFM_CUDA(cudaMemcpyAsync(position_device_.data(), &position_,
                             sizeof(position_), cudaMemcpyHostToDevice,
                             stream_.get()));
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    phase_ = SessionPhase::Ready;
}

void LfmModel::Impl::prefill(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) {
        throw std::invalid_argument("prefill needs at least one token");
    }
    if (tokens.size() > static_cast<size_t>(max_context_)) {
        throw std::invalid_argument("prefill exceeds max_context");
    }
    const auto begin = std::chrono::steady_clock::now();
    if (options_.legacy_prefill) {
        prefill_legacy(tokens);
    } else {
        prefill_batched(tokens);
    }
    const auto end = std::chrono::steady_clock::now();
    metrics_.last_prefill_ms =
        std::chrono::duration<double, std::milli>(end - begin).count();
    metrics_.prefill_tokens = tokens.size();
    metrics_.cumulative_decode_ms = 0.0;
    metrics_.decoded_tokens = 0;
}

void LfmModel::Impl::prefill_chunk(const std::vector<int32_t>& tokens,
                              bool begin, bool finalize) {
    if (tokens.empty()) {
        throw std::invalid_argument("prefill_chunk needs at least one token");
    }
    if (begin) {
        reset();
        metrics_ = {};
    } else if (position_ == 0) {
        throw std::runtime_error(
            "prefill_chunk continuation requires an initial chunk");
    }
    if (phase_ == SessionPhase::Ready) {
        throw std::runtime_error("cannot append prefill after finalization");
    }
    if (position_ + static_cast<int>(tokens.size()) > max_context_) {
        throw std::invalid_argument("prefill chunks exceed max_context");
    }
    phase_ = SessionPhase::Prefilling;

    DeviceBuffer<int32_t> input(tokens.size());
    LFM_CUDA(cudaMemcpyAsync(input.data(), tokens.data(),
                             tokens.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    launch_mark_seen_batch(input.data(), static_cast<int>(tokens.size()),
                           seen_tokens_.data(), shape_.vocab_size,
                           stream_.get());
    const auto started = std::chrono::steady_clock::now();
    for (size_t i = 0; i < tokens.size(); ++i) {
        forward_token_host(tokens[i], finalize && i + 1 == tokens.size());
    }
    LFM_CUDA(cudaMemcpyAsync(position_device_.data(), &position_,
                             sizeof(position_), cudaMemcpyHostToDevice,
                             stream_.get()));
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    const auto ended = std::chrono::steady_clock::now();
    metrics_.last_prefill_ms +=
        std::chrono::duration<double, std::milli>(ended - started).count();
    metrics_.prefill_tokens += tokens.size();
    if (finalize) {
        phase_ = SessionPhase::Ready;
        active_segmented_attention_ = use_segmented_attention(position_);
    }
}

void LfmModel::Impl::set_generation_config(GenerationConfig generation) {
    generation.validate();
    if (phase_ == SessionPhase::DecodePending) {
        throw std::runtime_error(
            "cannot change generation configuration during decode");
    }
    generation_ = generation;
    decode_graph_.reset();
    segmented_decode_graph_.reset();
}

void LfmModel::Impl::enqueue_sampling() {
    const int effective_top_k = generation_.greedy() ? 1 : generation_.top_k;
    const float effective_temperature =
        generation_.temperature > 0.0f ? generation_.temperature : 1.0f;
    if (plan_.sampling_kernel() == SamplingKernelKind::Fused) {
        launch_fused_sample_topk(
            logits_.data(), seen_tokens_.data(), sampling_scores_.data(),
            topk_values_.data(), topk_indices_.data(), shape_.vocab_size,
            effective_temperature, generation_.repetition_penalty,
            effective_top_k,
            generation_.greedy() ? 1.0f : generation_.top_p,
            rng_state_.data(), sampled_device_.data(), stream_.get());
    } else {
        launch_prepare_sampling_scores(
            logits_.data(), seen_tokens_.data(), sampling_scores_.data(),
            shape_.vocab_size, effective_temperature,
            generation_.repetition_penalty, stream_.get());
        for (int rank = 0; rank < effective_top_k; ++rank) {
            launch_select_topk(sampling_scores_.data(), topk_values_.data(),
                               topk_indices_.data(), rank, shape_.vocab_size,
                               stream_.get());
        }
        launch_sample_topk(topk_values_.data(), topk_indices_.data(),
                           effective_top_k,
                           generation_.greedy() ? 1.0f : generation_.top_p,
                           rng_state_.data(), sampled_device_.data(),
                           stream_.get());
        launch_mark_seen(sampled_device_.data(), seen_tokens_.data(),
                         shape_.vocab_size, stream_.get());
    }
}

void LfmModel::Impl::enqueue_decode_forward() {
    weight_layout_->embed_token_device(
        sampled_device_.data(), hidden_.data(), shape_.hidden,
        stream_.get());

    for (Layer& layer : layers_) {
        LayerCommon& common_layer = common(layer);
        if (!options_.fused_residuals) {
            LFM_CUDA(cudaMemcpyAsync(
                residual_.data(), hidden_.data(), hidden_.bytes(),
                cudaMemcpyDeviceToDevice, stream_.get()));
        }
        launch_rmsnorm(hidden_.data(), common_layer.operator_norm, normed_.data(),
                       1, shape_.hidden, shape_.norm_eps,
                       stream_.get());
        if (AttentionLayer* attention = as_attention(layer)) {
            __nv_bfloat16* q = qkv_output_.data();
            __nv_bfloat16* k = q + shape_.q_width;
            __nv_bfloat16* v = k + shape_.kv_width;
            if (options_.fused_projections) {
                linear(normed_.data(), *attention->qkv, qkv_output_.data(),
                       1, shape_.qkv_width, shape_.hidden);
            } else {
                const LinearWeight q_weight =
                    slice_rows(*attention->qkv, 0, shape_.q_width);
                const LinearWeight k_weight = slice_rows(
                    *attention->qkv, shape_.q_width, shape_.kv_width);
                const LinearWeight v_weight = slice_rows(
                    *attention->qkv, shape_.q_width + shape_.kv_width,
                    shape_.kv_width);
                linear(normed_.data(), q_weight, q,
                       1, shape_.q_width, shape_.hidden);
                linear(normed_.data(), k_weight, k,
                       1, shape_.kv_width, shape_.hidden);
                linear(normed_.data(), v_weight, v,
                       1, shape_.kv_width, shape_.hidden);
            }
            if (options_.fast_attention) {
                launch_qk_norm_rope_fast_device(
                    q, k, attention->q_norm, attention->k_norm,
                    rope_cos_.data(), rope_sin_.data(),
                    shape_.num_attention_heads, shape_.num_key_value_heads,
                    shape_.head_dim, position_device_.data(),
                    shape_.norm_eps, stream_.get());
            } else {
                launch_qk_norm_rope_strict_device(
                    q, k, attention->q_norm, attention->k_norm,
                    rope_cos_.data(), rope_sin_.data(),
                    shape_.num_attention_heads, shape_.num_key_value_heads,
                    shape_.head_dim, position_device_.data(),
                    shape_.norm_eps, stream_.get());
            }
            if (options_.kv_cache_mode == KvCacheMode::Int8) {
                launch_store_kv_int8_device(
                    k, v, attention->key_cache_int8.data(),
                    attention->value_cache_int8.data(),
                    attention->key_cache_scales.data(),
                    attention->value_cache_scales.data(), position_device_.data(),
                    shape_.num_key_value_heads, shape_.head_dim, stream_.get());
                if (active_segmented_attention_) {
                    launch_gqa_decode_segmented_int8_device(
                        q, attention->key_cache_int8.data(),
                        attention->value_cache_int8.data(),
                        attention->key_cache_scales.data(),
                        attention->value_cache_scales.data(), op_output_.data(),
                        position_device_.data(), shape_.num_attention_heads,
                        shape_.num_key_value_heads, shape_.head_dim,
                        options_.attention_chunk_tokens, attention_chunks_,
                        attention_partial_max_.data(),
                        attention_partial_denom_.data(),
                        attention_partial_accum_.data(), stream_.get());
                } else if (options_.fast_attention) {
                    launch_gqa_decode_online_int8_device(
                        q, attention->key_cache_int8.data(),
                        attention->value_cache_int8.data(),
                        attention->key_cache_scales.data(),
                        attention->value_cache_scales.data(), op_output_.data(),
                        position_device_.data(), shape_.num_attention_heads,
                        shape_.num_key_value_heads, shape_.head_dim, stream_.get());
                } else {
                    launch_gqa_decode_strict_int8_device(
                        q, attention->key_cache_int8.data(),
                        attention->value_cache_int8.data(),
                        attention->key_cache_scales.data(),
                        attention->value_cache_scales.data(), op_output_.data(),
                        position_device_.data(), shape_.num_attention_heads,
                        shape_.num_key_value_heads, shape_.head_dim, stream_.get());
                }
            } else {
                launch_store_kv_device(
                    k, v, attention->key_cache.data(), attention->value_cache.data(),
                    position_device_.data(), shape_.kv_width, stream_.get());
                if (active_segmented_attention_) {
                    launch_gqa_decode_segmented_device(
                        q, attention->key_cache.data(), attention->value_cache.data(),
                        op_output_.data(), position_device_.data(),
                        shape_.num_attention_heads, shape_.num_key_value_heads,
                        shape_.head_dim, options_.attention_chunk_tokens,
                        attention_chunks_, attention_partial_max_.data(),
                        attention_partial_denom_.data(),
                        attention_partial_accum_.data(), stream_.get());
                } else if (options_.fast_attention) {
                    launch_gqa_decode_online_device(
                        q, attention->key_cache.data(), attention->value_cache.data(),
                        op_output_.data(), position_device_.data(),
                        shape_.num_attention_heads, shape_.num_key_value_heads,
                        shape_.head_dim, stream_.get());
                } else {
                    launch_gqa_decode_strict_device(
                        q, attention->key_cache.data(), attention->value_cache.data(),
                        op_output_.data(), position_device_.data(),
                        shape_.num_attention_heads, shape_.num_key_value_heads,
                        shape_.head_dim, stream_.get());
                }
            }
            linear(op_output_.data(), *attention->out, hidden_.data(),
                   1, shape_.hidden, shape_.hidden,
                   options_.fused_residuals ? 1.0f : 0.0f);
        } else {
            ConvolutionLayer& convolution = *as_convolution(layer);
            linear(normed_.data(), *convolution.conv_in, conv_projected_.data(),
                   1, 3 * shape_.hidden, shape_.hidden);
            launch_conv_decode_device(
                conv_projected_.data(), convolution.conv_weight,
                convolution.conv_state.data(), op_output_.data(),
                shape_.hidden, shape_.conv_cache,
                position_device_.data(), stream_.get());
            linear(op_output_.data(), *convolution.conv_out, hidden_.data(),
                   1, shape_.hidden, shape_.hidden,
                   options_.fused_residuals ? 1.0f : 0.0f);
        }
        if (!options_.fused_residuals) {
            launch_residual_add(hidden_.data(), residual_.data(),
                                shape_.hidden, stream_.get());
        }
        run_mlp_decode(common_layer);
    }
    launch_rmsnorm(hidden_.data(), final_norm_, normed_.data(),
                   1, shape_.hidden, shape_.norm_eps,
                   stream_.get());
    linear(normed_.data(), *embedding_, logits_.data(),
           1, shape_.vocab_size, shape_.hidden);
}

void LfmModel::Impl::enqueue_decode_step() {
    enqueue_sampling();
    enqueue_decode_forward();
    launch_increment_position(position_device_.data(), stream_.get());
}

bool LfmModel::Impl::use_segmented_attention(int host_position) const {
    return plan_.segmented_attention(host_position);
}

CudaGraphExec& LfmModel::Impl::graph_for_attention(bool segmented) {
    return segmented ? segmented_decode_graph_ : decode_graph_;
}

void LfmModel::Impl::capture_decode_graph(bool segmented) {
    if (!options_.cuda_graph) return;
    CudaGraphExec& graph = graph_for_attention(segmented);
    if (graph.ready()) return;
    active_segmented_attention_ = segmented;
    graph.capture_begin(stream_.get());
    try {
        enqueue_decode_step();
        graph.capture_end(stream_.get());
    } catch (...) {
        graph.abort_capture(stream_.get());
        throw;
    }
}

int32_t LfmModel::Impl::decode() {
    decode_async_begin();
    return decode_async_finish();
}

void LfmModel::Impl::decode_async_begin() {
    if (!local_kv_cache_available_) {
        throw std::runtime_error(
            "lane decode is unavailable after transferring KV to the shared paged cache");
    }
    if (phase_ != SessionPhase::Ready) {
        throw std::runtime_error("decode requires a successful prefill");
    }
    if (phase_ == SessionPhase::DecodePending) {
        throw std::runtime_error("decode_async_begin called twice");
    }
    if (position_ >= max_context_) {
        throw std::runtime_error("context limit reached");
    }
    decode_async_begin_time_ = std::chrono::steady_clock::now();
    const bool segmented = use_segmented_attention(position_);
    active_segmented_attention_ = segmented;
    if (options_.cuda_graph) {
        CudaGraphExec& graph = graph_for_attention(segmented);
        if (!graph.ready()) capture_decode_graph(segmented);
        graph.launch(stream_.get());
    } else {
        enqueue_decode_step();
    }
    LFM_CUDA(cudaMemcpyAsync(sampled_host_.data(), sampled_device_.data(),
                             sizeof(int32_t), cudaMemcpyDeviceToHost,
                             stream_.get()));
    phase_ = SessionPhase::DecodePending;
}

int32_t LfmModel::Impl::decode_async_finish() {
    if (phase_ != SessionPhase::DecodePending) {
        throw std::runtime_error("decode_async_finish without begin");
    }
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    phase_ = SessionPhase::Ready;
    ++position_;
    const auto ended = std::chrono::steady_clock::now();
    metrics_.cumulative_decode_ms +=
        std::chrono::duration<double, std::milli>(
            ended - decode_async_begin_time_).count();
    ++metrics_.decoded_tokens;
    return sampled_host_.data()[0];
}

DecodeBenchmark LfmModel::Impl::benchmark_decode(int warmup_steps,
                                                int measured_steps) {
    if (phase_ != SessionPhase::Ready) {
        throw std::runtime_error("benchmark_decode requires a successful prefill");
    }
    if (warmup_steps < 0 || measured_steps <= 0) {
        throw std::invalid_argument(
            "benchmark steps require warmup >= 0 and measured > 0");
    }
    const int total_steps = warmup_steps + measured_steps;
    if (position_ + total_steps > max_context_) {
        throw std::runtime_error("decode benchmark exceeds context limit");
    }
    if (options_.cuda_graph) {
        const int final_position = position_ + total_steps - 1;
        const bool starts_segmented = use_segmented_attention(position_);
        const bool ends_segmented = use_segmented_attention(final_position);
        if (!starts_segmented || !ends_segmented) capture_decode_graph(false);
        if (starts_segmented || ends_segmented) capture_decode_graph(true);
    }

    int simulated_position = position_;
    auto launch_step = [&]() {
        const bool segmented = use_segmented_attention(simulated_position);
        active_segmented_attention_ = segmented;
        if (options_.cuda_graph) {
            CudaGraphExec& graph = graph_for_attention(segmented);
            if (!graph.ready()) capture_decode_graph(segmented);
            graph.launch(stream_.get());
        } else {
            enqueue_decode_step();
        }
        ++simulated_position;
    };
    for (int i = 0; i < warmup_steps; ++i) launch_step();
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));

    CudaEvent begin;
    CudaEvent end;
    begin.record(stream_.get());
    for (int i = 0; i < measured_steps; ++i) launch_step();
    end.record(stream_.get());
    end.synchronize();

    position_ += total_steps;
    DecodeBenchmark result;
    result.warmup_steps = warmup_steps;
    result.measured_steps = measured_steps;
    result.elapsed_ms = CudaEvent::elapsed_ms(begin, end);
    return result;
}

ModelMemoryStats LfmModel::Impl::memory_stats() const {
    ModelMemoryStats stats;
    stats.weights = weights_ ? weights_->memory_bytes() : 0;
    for (const Layer& layer : layers_) {
        if (const AttentionLayer* attention = as_attention(layer)) {
            stats.kv_cache += attention->key_cache.bytes() + attention->value_cache.bytes() +
                attention->key_cache_int8.bytes() + attention->value_cache_int8.bytes() +
                attention->key_cache_scales.bytes() + attention->value_cache_scales.bytes();
        } else {
            stats.conv_state += as_convolution(layer)->conv_state.bytes();
        }
    }
    stats.rope_tables = rope_cos_.bytes() + rope_sin_.bytes();
    stats.activations =
        hidden_.bytes() + residual_.bytes() + normed_.bytes() +
        op_output_.bytes() + qkv_output_.bytes() + conv_projected_.bytes() +
        gate_up_.bytes() + activated_.bytes() + mlp_output_.bytes() +
        logits_.bytes() + paged_page_table_.bytes() +
        paged_prefill_tokens_.bytes() + prefill_tokens_.bytes() +
        prefill_hidden_.bytes() +
        prefill_residual_.bytes() + prefill_normed_.bytes() +
        prefill_op_output_.bytes() + prefill_q_.bytes() + prefill_k_.bytes() +
        prefill_v_.bytes() + prefill_conv_projected_.bytes() +
        prefill_gate_up_.bytes() + prefill_activated_.bytes() +
        prefill_mlp_output_.bytes();
    stats.sampling =
        position_device_.bytes() + sampled_device_.bytes() +
        seen_tokens_.bytes() + sampling_scores_.bytes() +
        topk_values_.bytes() + topk_indices_.bytes() + rng_state_.bytes();
    stats.matmul_workspace = gemm_ ? gemm_->workspace_bytes() : 0;
    stats.attention_workspace = attention_partial_max_.bytes() +
        attention_partial_denom_.bytes() + attention_partial_accum_.bytes();
    return stats;
}

// Build the SessionState snapshot that SessionStore consumes. Centralizes
// the per-call wiring (shape, variant, kv mode, per-layer buffer pointers)
// so save/load/export/restore stay one-liners in the host.
SessionStore::SessionState LfmModel::Impl::make_session_state() {
    SessionStore::SessionState state{
        .shape = shape_,
        .max_context = max_context_,
        .position = position_,
        .kv_cache_mode = options_.kv_cache_mode,
        .variant = variant_,
        .stream = stream_.get(),
        .seen_tokens = &seen_tokens_,
        .logits = &logits_,
        .rng_state = &rng_state_,
    };
    state.layer_buffers.reserve(layers_.size());
    for (Layer& layer : layers_) {
        SessionStore::SessionState::LayerBuffers buffers{};
        if (AttentionLayer* attention = as_attention(layer)) {
            buffers.is_attention = true;
            buffers.key_cache_bf16 = attention->key_cache.data();
            buffers.value_cache_bf16 = attention->value_cache.data();
            buffers.key_cache_int8 = attention->key_cache_int8.data();
            buffers.value_cache_int8 = attention->value_cache_int8.data();
            buffers.key_cache_scales = attention->key_cache_scales.data();
            buffers.value_cache_scales = attention->value_cache_scales.data();
        } else if (ConvolutionLayer* convolution = as_convolution(layer)) {
            buffers.is_attention = false;
            buffers.conv_state = convolution->conv_state.data();
            buffers.conv_state_elements = convolution->conv_state.size();
        }
        state.layer_buffers.push_back(buffers);
    }
    return state;
}

void LfmModel::Impl::save_session(const std::string& path) {
    if (!local_kv_cache_available_) {
        throw std::runtime_error(
            "save_session requires a model with a local contiguous KV cache");
    }
    if (phase_ != SessionPhase::Ready) {
        throw std::runtime_error("cannot save a session before prefill or load_session");
    }
    SessionStore::SessionState state = make_session_state();
    SessionStore::save(path, state);
}

void LfmModel::Impl::load_session(const std::string& path) {
    reset();
    SessionStore::SessionState state = make_session_state();
    SessionStore::load(path, state);
    position_ = state.position;
    LFM_CUDA(cudaMemcpy(position_device_.data(), &position_, sizeof(position_),
                        cudaMemcpyHostToDevice));
    phase_ = SessionPhase::Ready;
    active_segmented_attention_ = use_segmented_attention(position_);
    metrics_ = {};
}


PrefixState LfmModel::Impl::export_prefix_state() const {
    if (phase_ != SessionPhase::Ready) {
        throw std::runtime_error("cannot export prefix state before prefill");
    }
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    SessionStore::SessionState state = const_cast<Impl*>(this)->make_session_state();
    auto snapshot = SessionStore::export_prefix(state);
    PrefixState out;
    out.position = snapshot.position;
    out.seen_tokens = std::move(snapshot.seen_tokens);
    out.logits_bf16 = std::move(snapshot.logits_bf16);
    out.conv_state_bf16 = std::move(snapshot.conv_state_bf16);
    return out;
}

void LfmModel::Impl::restore_prefix_state(const PrefixState& state) {
    SessionStore::PrefixSnapshot snapshot;
    snapshot.position = state.position;
    snapshot.seen_tokens = state.seen_tokens;
    snapshot.logits_bf16 = state.logits_bf16;
    snapshot.conv_state_bf16 = state.conv_state_bf16;
    SessionStore::SessionState session = make_session_state();
    const uint64_t request_seed = generation_.seed;
    SessionStore::restore_prefix(snapshot, session, request_seed);
    phase_ = SessionPhase::Prefilling;
    position_ = session.position;
    LFM_CUDA(cudaMemcpyAsync(position_device_.data(), &position_,
                             sizeof(position_), cudaMemcpyHostToDevice,
                             stream_.get()));
    phase_ = SessionPhase::Ready;
    active_segmented_attention_ = use_segmented_attention(position_);
    metrics_ = {};
}

void LfmModel::Impl::release_local_kv_cache() {
    if (!local_kv_cache_available_) return;
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    for (Layer& layer : layers_) {
        AttentionLayer* attention = as_attention(layer);
        if (!attention) continue;
        attention->key_cache.reset(0);
        attention->value_cache.reset(0);
        attention->key_cache_int8.reset(0);
        attention->value_cache_int8.reset(0);
        attention->key_cache_scales.reset(0);
        attention->value_cache_scales.reset(0);
    }
    local_kv_cache_available_ = false;
    decode_graph_.reset();
    segmented_decode_graph_.reset();
}

std::vector<float> LfmModel::Impl::copy_logits() {
    if (phase_ != SessionPhase::Ready) {
        throw std::runtime_error("logits are unavailable before prefill");
    }
    std::vector<__nv_bfloat16> bf16_logits(shape_.vocab_size);
    LFM_CUDA(cudaMemcpyAsync(
        bf16_logits.data(), logits_.data(), logits_.bytes(),
        cudaMemcpyDeviceToHost, stream_.get()));
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));

    std::vector<float> result(shape_.vocab_size);
    for (int i = 0; i < shape_.vocab_size; ++i) {
        result[static_cast<size_t>(i)] =
            __bfloat162float(bf16_logits[static_cast<size_t>(i)]);
    }
    return result;
}


LfmModel::LfmModel(const std::string& safetensors_path,
                   int max_context,
                   ModelOptions options,
                   GenerationConfig generation)
    : impl_(std::make_unique<Impl>(safetensors_path, max_context,
                                   std::move(options), std::move(generation))),
      session_view_(*this),
      diagnostics_view_(*this),
      persistence_view_(*this) {}

LfmModel::~LfmModel() = default;

void LfmModel::reset(bool allocate_local_kv) { impl_->reset(allocate_local_kv); }
void LfmModel::prefill(const std::vector<int32_t>& tokens) { impl_->prefill(tokens); }
void LfmModel::prefill_chunk(const std::vector<int32_t>& tokens,
                             bool begin, bool finalize) {
    impl_->prefill_chunk(tokens, begin, finalize);
}
void LfmModel::prefill_chunk_paged(const std::vector<int32_t>& tokens,
                                   bool begin, bool finalize,
                                   PhysicalPagedKvCache& paged_kv,
                                   const std::vector<uint32_t>& page_table) {
    impl_->prefill_chunk_paged(tokens, begin, finalize, paged_kv, page_table);
}
int32_t LfmModel::decode() { return impl_->decode(); }
void LfmModel::decode_async_begin() { impl_->decode_async_begin(); }
int32_t LfmModel::decode_async_finish() { return impl_->decode_async_finish(); }
void LfmModel::set_generation_config(GenerationConfig generation) {
    impl_->set_generation_config(std::move(generation));
}
std::vector<float> LfmModel::copy_logits() { return impl_->copy_logits(); }
DecodeBenchmark LfmModel::benchmark_decode(int warmup_steps, int measured_steps) {
    return impl_->benchmark_decode(warmup_steps, measured_steps);
}
ModelMemoryStats LfmModel::memory_stats() const { return impl_->memory_stats(); }
RuntimeMetrics LfmModel::runtime_metrics() const { return impl_->runtime_metrics(); }
void LfmModel::clear_runtime_metrics() { impl_->clear_runtime_metrics(); }
void LfmModel::save_session(const std::string& path) { impl_->save_session(path); }
void LfmModel::load_session(const std::string& path) { impl_->load_session(path); }
PrefixState LfmModel::export_prefix_state() const { return impl_->export_prefix_state(); }
void LfmModel::restore_prefix_state(const PrefixState& state) {
    impl_->restore_prefix_state(state);
}
void LfmModel::release_local_kv_cache() { impl_->release_local_kv_cache(); }
bool LfmModel::local_kv_cache_available() const {
    return impl_->local_kv_cache_available();
}
SessionPhase LfmModel::phase() const { return impl_->phase(); }
bool LfmModel::ready_for_decode() const { return impl_->ready_for_decode(); }
bool LfmModel::decode_pending() const { return impl_->decode_pending(); }
int LfmModel::position() const { return impl_->position(); }
int LfmModel::vocab_size() const { return impl_->shape_.vocab_size; }

IPackedSession& LfmModel::packed_session() { return *impl_; }
const IPackedSession& LfmModel::packed_session() const { return *impl_; }
bool LfmModel::cuda_graph_ready() const { return impl_->cuda_graph_ready(); }

void LfmInferenceSession::reset(bool allocate_local_kv) {
    owner_->impl_->reset(allocate_local_kv);
}
void LfmInferenceSession::prefill(const std::vector<int32_t>& tokens) {
    owner_->impl_->prefill(tokens);
}
void LfmInferenceSession::prefill_chunk(const std::vector<int32_t>& tokens,
                                        bool begin, bool finalize) {
    owner_->impl_->prefill_chunk(tokens, begin, finalize);
}
void LfmInferenceSession::prefill_chunk_paged(
    const std::vector<int32_t>& tokens, bool begin, bool finalize,
    PhysicalPagedKvCache& paged_kv,
    const std::vector<uint32_t>& page_table) {
    owner_->impl_->prefill_chunk_paged(tokens, begin, finalize,
                                      paged_kv, page_table);
}
int32_t LfmInferenceSession::decode() { return owner_->impl_->decode(); }
void LfmInferenceSession::decode_async_begin() {
    owner_->impl_->decode_async_begin();
}
int32_t LfmInferenceSession::decode_async_finish() {
    return owner_->impl_->decode_async_finish();
}
void LfmInferenceSession::set_generation_config(GenerationConfig generation) {
    owner_->impl_->set_generation_config(std::move(generation));
}
void LfmInferenceSession::release_local_kv_cache() {
    owner_->impl_->release_local_kv_cache();
}
bool LfmInferenceSession::local_kv_cache_available() const {
    return owner_->impl_->local_kv_cache_available();
}
SessionPhase LfmInferenceSession::phase() const { return owner_->impl_->phase(); }
bool LfmInferenceSession::ready_for_decode() const {
    return owner_->impl_->ready_for_decode();
}
bool LfmInferenceSession::decode_pending() const {
    return owner_->impl_->decode_pending();
}
int LfmInferenceSession::position() const { return owner_->impl_->position(); }

std::vector<float> LfmDiagnostics::copy_logits() const {
    return owner_->impl_->copy_logits();
}
DecodeBenchmark LfmDiagnostics::benchmark_decode(int warmup_steps,
                                                  int measured_steps) {
    return owner_->impl_->benchmark_decode(warmup_steps, measured_steps);
}
ModelMemoryStats LfmDiagnostics::memory_stats() const {
    return owner_->impl_->memory_stats();
}
RuntimeMetrics LfmDiagnostics::runtime_metrics() const {
    return owner_->impl_->runtime_metrics();
}
void LfmDiagnostics::clear_runtime_metrics() {
    owner_->impl_->clear_runtime_metrics();
}
bool LfmDiagnostics::cuda_graph_ready() const {
    return owner_->impl_->cuda_graph_ready();
}

void LfmPersistence::save_session(const std::string& path) const {
    owner_->impl_->save_session(path);
}
void LfmPersistence::load_session(const std::string& path) {
    owner_->impl_->load_session(path);
}
PrefixState LfmPersistence::export_prefix_state() const {
    return owner_->impl_->export_prefix_state();
}
void LfmPersistence::restore_prefix_state(const PrefixState& state) {
    owner_->impl_->restore_prefix_state(state);
}

} // namespace lfm
