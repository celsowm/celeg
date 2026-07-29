#include "lfm/detail/model/impl.hpp"
#include "lfm/detail/checkpoint/bootstrap.hpp"
#include "lfm/backend/cuda/kernels/kernels.cuh"
#include "lfm/model/weights/quantization.hpp"
#include "lfm/backend/cuda/paged_kv.hpp"
#include "lfm/model/config/variant.hpp"
#include "lfm/model/weights/layout.hpp"
#include "lfm/runtime/weights_topology.hpp"
#include "lfm/model/weights/loader.hpp"
#include "lfm/checkpoint/formats/gguf.hpp"
#include "lfm/checkpoint/repositories/gguf.hpp"
#include "lfm/backend/cuda/kernels/gguf.cuh"
#include "lfm/runtime/moe.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <utility>

namespace lfm {
namespace {

std::string layer_name(int index, const std::string& suffix) {
    return "model.layers." + std::to_string(index) + "." + suffix;
}

} // namespace

struct LtPlan;  // defined in include/lfm/detail/model/types.hpp

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
        case LinearStorageKind::Q4_K:
        case LinearStorageKind::Q6_K:
            if (bf16 || int8 || int4 || scales || gguf_segments.empty()) {
                throw std::runtime_error("invalid GGUF-quantized linear weight storage");
            }
            for (const auto& segment : gguf_segments) {
                if (!segment.blocks || segment.rows <= 0 || segment.cols != cols ||
                    (segment.type != GgmlType::Q4_K && segment.type != GgmlType::Q6_K) ||
                    segment.row_bytes == 0) {
                    throw std::runtime_error("invalid GGUF linear segment");
                }
            }
            break;
    }
}

LfmModel::Impl::Impl(const std::string& model_path,
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
    // Load the model topology either from a GGUF checkpoint's embedded metadata
    // or from a config.json next to the safetensors file. GGUF checkpoints carry
    // their tokenizer + hyper-parameters inside the container, so there is no
    // sidecar config.json to read.
    const detail::ModelBootstrap bootstrap =
        detail::load_model_bootstrap(std::filesystem::path(model_path));
    const bool is_gguf = bootstrap.is_gguf;
    if (is_gguf && options_.weight_mode != WeightMode::Bf16) {
        throw std::invalid_argument(
            "GGUF CUDA weights are natively Q4_K/Q6_K; INT4/INT8 requantization is unsupported");
    }
    std::shared_ptr<GgufFile> gguf_file = bootstrap.gguf_file;
    const ModelConfig& config = bootstrap.config;
    shape_ = bootstrap.shape;
    variant_ = bootstrap.variant;
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
        moe_output_accum_.reset(static_cast<size_t>(shape_.hidden));
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
    weights_ = WeightLoader::acquire(
        model_path, options_.weight_mode,
        options_.expert_offload.fingerprint());
    weight_loader_ = std::make_unique<WeightLoader>(weights_, options_.weight_mode);

    // Only one constructor populates a shared checkpoint at a time. Other
    // sessions wait, then reuse the immutable device buffers.
    std::unique_lock<std::mutex> shared_weights_lock(weights_->mutex);
    std::shared_ptr<IWeightRepository> repo_owner;
    if (is_gguf) {
        repo_owner = std::make_shared<GgufRepository>(gguf_file);
    } else {
        repo_owner = std::make_shared<SafeTensorRepository>(model_path);
    }
    weights_->repo = repo_owner;
    const IWeightRepository& repo = *repo_owner;
    const auto embedding_spec = weights::token_embedding(shape_);
    embedding_ = weight_loader_->load_linear_weight(
        repo, std::string(embedding_spec.canonical_name), embedding_spec.shape);
    const auto norm_spec = weights::final_norm(shape_);
    final_norm_ = weight_loader_->load_weight(
        repo, std::string(norm_spec.canonical_name), norm_spec.shape);
    {
        if (embedding_->gguf_quantized()) {
            if (embedding_->gguf_segments.size() != 1) {
                throw std::runtime_error("GGUF embedding must use one native segment");
            }
            weight_layout_ = make_gguf_weight_layout(embedding_->gguf_segments.front());
        } else if (options_.weight_mode == WeightMode::Int8) {
            weight_layout_ = make_weight_layout(
                options_.weight_mode, embedding_->int8, embedding_->scales);
        } else if (options_.weight_mode == WeightMode::Int4) {
            weight_layout_ = make_weight_layout(
                options_.weight_mode, embedding_->int4, embedding_->scales);
        } else {
            weight_layout_ = make_weight_layout(
                options_.weight_mode, embedding_->bf16, embedding_->scales);
        }
    }
    // Untied LM head: load the separate lm_head weight for the final logits
    // projection. Some checkpoints omit the lm_head tensor yet leave
    // tie_word_embeddings unset; in that case the head is effectively tied to
    // the embedding table, so we fall back to it instead of erroring.
    const auto lm_head_spec = weights::language_model_head(
        shape_, config.tie_word_embeddings);
    if (!lm_head_spec.optional &&
        repo.contains(std::string(lm_head_spec.canonical_name))) {
        lm_head_ = weight_loader_->load_linear_weight(
            repo, std::string(lm_head_spec.canonical_name), lm_head_spec.shape);
    }

    // Resolve the MoE expert-offload plan before loading experts. Snapshot the
    // free VRAM now (embeddings/final norm already uploaded; attention/conv and
    // experts are loaded below) and estimate the always-resident non-expert
    // weight footprint analytically from the topology so the planner can decide
    // how many experts fit in the GPU cache per layer.
    if (options_.expert_offload.enabled() &&
        shape_.architecture == ArchitectureKind::MoeLfm2) {
        size_t free_bytes = 0, total_bytes = 0;
        LFM_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
        const int moe_layers = moe_layer_count(shape_);
        const size_t bpe = bytes_per_expert_bf16(shape_);
        // All experts are BF16; non-expert weights = everything else already or
        // about to be resident on the GPU.
        const size_t all_expert_bytes =
            static_cast<size_t>(shape_.num_experts) *
            static_cast<size_t>(moe_layers) * bpe;
        // Conservative non-expert estimate: total checkpoint minus experts.
        // embed + lm-head + attention + conv + dense FFN + router + norms.
        const size_t embed_bytes =
            static_cast<size_t>(shape_.vocab_size) * shape_.hidden *
            sizeof(__nv_bfloat16);
        const size_t per_attn = static_cast<size_t>(shape_.qkv_width) * shape_.hidden +
            static_cast<size_t>(shape_.hidden) * shape_.q_width;
        const size_t attn_bytes = per_attn *
            static_cast<size_t>(shape_.attention_layer_count) * sizeof(__nv_bfloat16);
        const size_t dense_ffn_bytes =
            static_cast<size_t>(shape_.num_dense_layers) *
            (3ull * shape_.dense_intermediate * shape_.hidden) * sizeof(__nv_bfloat16);
        const size_t router_bytes =
            static_cast<size_t>(moe_layers) * shape_.num_experts * shape_.hidden *
            (sizeof(__nv_bfloat16) + sizeof(float));
        const size_t non_expert_bytes =
            embed_bytes + attn_bytes + dense_ffn_bytes + router_bytes +
            (64ull << 20);  // norms/conv/bias slack
        (void)all_expert_bytes;

        ExpertOffloadPlanInputs pin;
        pin.shape = shape_;
        pin.options = options_.expert_offload;
        pin.gpu_free_bytes = free_bytes;
        pin.non_expert_weight_bytes = non_expert_bytes;
        pin.workspace_bytes = options_.lt_workspace_bytes + (256ull << 20);
        pin.context_tokens = max_context_;
        expert_offload_plan_ = plan_expert_offload(pin);
        expert_transfer_stream_ = std::make_unique<CudaStream>();

        weights_->expert_offload_plan = expert_offload_plan_;
        if (options_.expert_offload.backing == ExpertBackingMode::DiskCached && !weights_->pinned_expert_cache) {
            size_t bpe = bytes_per_expert_bf16(shape_);
            size_t gu_bytes = 2 * shape_.moe_intermediate * shape_.hidden * sizeof(__nv_bfloat16);
            size_t dn_bytes = shape_.hidden * shape_.moe_intermediate * sizeof(__nv_bfloat16);
            weights_->pinned_expert_cache = std::make_unique<PinnedExpertCache>(
                options_.expert_offload.host_expert_cache_bytes,
                bpe, gu_bytes, dn_bytes);
        }
        if (options_.expert_offload.backing == ExpertBackingMode::DiskCached && !weights_->expert_io_manager) {
            weights_->expert_io_manager = std::make_unique<ExpertIoManager>(
                options_.expert_offload.io_workers,
                options_.expert_offload.io_queue_depth);
        }
        if (!options_.expert_offload.expert_sidecar_path.empty() && !weights_->expert_sidecar) {
            auto sidecar = std::make_unique<ExpertSidecar>();
            int moe_layers = moe_layer_count(shape_);
            if (sidecar->load(options_.expert_offload.expert_sidecar_path,
                             moe_layers, shape_.num_experts,
                             shape_.moe_intermediate, shape_.hidden)) {
                weights_->expert_sidecar = std::move(sidecar);
                std::fprintf(stderr, "Loaded compatible expert sidecar from %s\n",
                             options_.expert_offload.expert_sidecar_path.c_str());
            } else {
                std::fprintf(stderr, "WARNING: Sidecar %s is incompatible or could not be loaded; falling back to safetensors.\n",
                             options_.expert_offload.expert_sidecar_path.c_str());
            }
        }
        if (!options_.expert_offload.usage_profile_path.empty()) {
            weights_->usage_profile_path = options_.expert_offload.usage_profile_path;
            if (weights_->usage_stats.layers.empty()) {
                int moe_layers = moe_layer_count(shape_);
                if (weights_->usage_stats.load(weights_->usage_profile_path, moe_layers, shape_.num_experts)) {
                    std::fprintf(stderr, "Loaded persistent expert usage statistics from %s\n",
                                 weights_->usage_profile_path.c_str());
                } else {
                    weights_->usage_stats.layers.assign(
                        static_cast<size_t>(moe_layers),
                        std::vector<ExpertUsageEntry>(static_cast<size_t>(shape_.num_experts)));
                }
            }
        }
        std::fprintf(stderr, "%s", expert_offload_plan_.report().c_str());
    }
    expert_caches_.resize(static_cast<size_t>(shape_.num_hidden_layers));
    if (weights_->expert_controllers.empty()) {
        weights_->expert_controllers.resize(static_cast<size_t>(shape_.num_hidden_layers));
    }
    expert_catalog_.resize(static_cast<size_t>(shape_.num_hidden_layers));
    if (weights_->expert_catalog.empty()) {
        weights_->expert_catalog.resize(static_cast<size_t>(shape_.num_hidden_layers));
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
                // Checkpoint naming varies: LFM2.5 ships
                // "feed_forward.expert_bias.weight"; the earlier LFM2 MoE
                // release ships "feed_forward.expert_bias" (no .weight suffix).
                const std::string bias_name = repo.contains(
                    layer_name(i, "feed_forward.expert_bias.weight"))
                    ? layer_name(i, "feed_forward.expert_bias.weight")
                    : layer_name(i, "feed_forward.expert_bias");
                expert_bias = weight_loader_->load_f32_weight(
                    repo, bias_name, {static_cast<int64_t>(E)});
            }
            const LinearWeight* router = weight_loader_->load_router(
                repo, i, E, shape_.hidden);

            // Cache a device float copy of the router weight for the CUDA
            // router kernel (it consumes float, the loaded weight is BF16).
            DeviceBuffer<float>& router_float = moe_router_float_[static_cast<size_t>(i)];
            router_float.reset(static_cast<size_t>(E) * shape_.hidden);
            launch_cast_bf16_to_float(
                router->bf16, router_float.data(),
                static_cast<int>(E) * shape_.hidden, stream_.get());

            MoeFfnWeights moe_weights{};
            moe_weights.router = router;
            moe_weights.expert_bias = expert_bias;
            moe_weights.router_float = router_float.data();

            if (expert_offload_plan_.enabled) {
                if (options_.expert_offload.backing == ExpertBackingMode::DiskCached) {
                    std::vector<ExpertLocation> catalog =
                        weight_loader_->build_expert_catalog(repo, i, E, inter, shape_.hidden);
                    expert_catalog_[static_cast<size_t>(i)] = catalog;
                    if (weights_->expert_catalog[static_cast<size_t>(i)].empty()) {
                        weights_->expert_catalog[static_cast<size_t>(i)] = catalog;
                    }

                    size_t gate_up_bytes = 2 * inter * shape_.hidden * sizeof(__nv_bfloat16);
                    size_t down_bytes = shape_.hidden * inter * sizeof(__nv_bfloat16);
                    auto cache = std::make_unique<ExpertLayerCache>(
                        E, expert_offload_plan_.experts_per_layer,
                        gate_up_bytes, down_bytes);
                    cache->set_policy(options_.expert_offload.policy);
                    std::vector<const __nv_bfloat16*> empty_host_dev(static_cast<size_t>(E), nullptr);
                    cache->set_host_sources(empty_host_dev, empty_host_dev);

                    auto controller = std::make_unique<ResidencyController>();
                    controller->cache = std::move(cache);
                    controller->transfer_stream = std::make_unique<CudaStream>();

                    if (expert_offload_plan_.experts_per_layer > 0) {
                        for (int s = 0; s < expert_offload_plan_.experts_per_layer; ++s) {
                            const ExpertLocation& loc = catalog[static_cast<size_t>(s)];
                            ExpertHostLease lease = weights_->pinned_expert_cache->acquire(i, s, [&](std::span<std::byte> gu_dest, std::span<std::byte> dn_dest) {
                                if (weights_->expert_sidecar) {
                                    weights_->expert_sidecar->read_expert(i, s, gu_dest, dn_dest);
                                } else {
                                    repo.read(loc.w1, gu_dest.subspan(0, loc.w1.bytes));
                                    repo.read(loc.w3, gu_dest.subspan(loc.w1.bytes, loc.w3.bytes));
                                    repo.read(loc.w2, dn_dest);
                                }
                            });
                            controller->cache->promote(s, s, reinterpret_cast<const __nv_bfloat16*>(lease.gate_up()),
                                           reinterpret_cast<const __nv_bfloat16*>(lease.down()),
                                           controller->transfer_stream->get());
                            auto ev = std::make_unique<CudaEvent>();
                            ev->record(controller->transfer_stream->get());
                            controller->inflight_transfers.push_back({std::move(lease), std::move(ev)});
                        }
                    }
                    LFM_CUDA(cudaStreamSynchronize(controller->transfer_stream->get()));
                    moe_weights.gate_up_ptrs = controller->cache->gate_up_ptrs();
                    moe_weights.down_ptrs = controller->cache->down_ptrs();
                    weights_->expert_controllers[static_cast<size_t>(i)] = std::move(controller);
                    expert_caches_[static_cast<size_t>(i)] = weights_->expert_controllers[static_cast<size_t>(i)]->cache.get();
                } else {
                    // Host-backed experts + per-layer GPU cache. Load expert bytes
                    // into the host store (no eager device upload), build the cache,
                    // and seed it with the first `experts_per_layer` experts.
                    WeightLoader::HostExpertLayer host_layer =
                        weight_loader_->load_moe_experts_host(
                            repo, i, E, inter, shape_.hidden, host_expert_store_,
                            options_.expert_offload.host_mode);
                    auto cache = std::make_unique<ExpertLayerCache>(
                        E, expert_offload_plan_.experts_per_layer,
                        host_layer.gate_up_bytes, host_layer.down_bytes);
                    cache->set_policy(options_.expert_offload.policy);
                    cache->set_host_sources(host_layer.gate_up_host_dev,
                                            host_layer.down_host_dev);

                    auto controller = std::make_unique<ResidencyController>();
                    controller->cache = std::move(cache);
                    controller->transfer_stream = std::make_unique<CudaStream>();

                    std::vector<int> seed(static_cast<size_t>(
                        expert_offload_plan_.experts_per_layer));
                    for (int s = 0; s < expert_offload_plan_.experts_per_layer; ++s) {
                        seed[static_cast<size_t>(s)] = s;
                    }
                    controller->cache->seed(seed, controller->transfer_stream->get());
                    LFM_CUDA(cudaStreamSynchronize(controller->transfer_stream->get()));
                    moe_weights.gate_up_ptrs = controller->cache->gate_up_ptrs();
                    moe_weights.down_ptrs = controller->cache->down_ptrs();
                    weights_->expert_controllers[static_cast<size_t>(i)] = std::move(controller);
                    expert_caches_[static_cast<size_t>(i)] = weights_->expert_controllers[static_cast<size_t>(i)]->cache.get();
                }
            } else {
                moe_weights.gate_up =
                    weight_loader_->load_moe_gate_up(repo, i, E, inter, shape_.hidden);
                moe_weights.down =
                    weight_loader_->load_moe_down(repo, i, E, inter, shape_.hidden);
            }

            common_layer.feed_forward = moe_weights;
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

LfmModel::Impl::~Impl() {
    if (weights_ && !weights_->usage_profile_path.empty()) {
        weights_->usage_stats.save(weights_->usage_profile_path);
    }
    if (moe_sel_host_ != nullptr) {
        cudaFreeHost(moe_sel_host_);
        moe_sel_host_ = nullptr;
    }
    if (moe_route_scores_host_ != nullptr) {
        cudaFreeHost(moe_route_scores_host_);
        moe_route_scores_host_ = nullptr;
    }
}


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
    linear(normed_.data(), *logits_weight(), logits_.data(),
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
    // Zero convolution state (running buffer that must start empty).  KV
    // caches are intentionally NOT zeroed here: attention only reads
    // positions 0..position_ and every used slot is overwritten before
    // becoming visible after the position reset above.
    for (Layer& layer : layers_) {
        if (AttentionLayer* attention = as_attention(layer)) {
            (void)attention;
            continue;
        }
        as_convolution(layer)->conv_state.zero_async(stream_.get());
    }
    // Prime the FFN-done, router-done and prefetch-done events so the offload
    // transfer stream can start promoting experts on the first layer of the
    // next forward pass.
    LFM_CUDA(cudaEventRecord(ffn_done_event_.get(), stream_.get()));
    LFM_CUDA(cudaEventRecord(router_done_event_.get(), stream_.get()));
    LFM_CUDA(cudaEventRecord(prefetch_done_event_.get(), stream_.get()));
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
}


void LfmModel::Impl::allocate_prefill_workspace(int rows) {
    const size_t r = static_cast<size_t>(rows);
    prefill_tokens_.reserve(r);
    prefill_hidden_.reserve(r * shape_.hidden);
    prefill_residual_.reserve(r * shape_.hidden);
    prefill_normed_.reserve(r * shape_.hidden);
    prefill_op_output_.reserve(r * shape_.hidden);
    prefill_q_.reserve(r * shape_.q_width);
    prefill_k_.reserve(r * shape_.kv_width);
    prefill_v_.reserve(r * shape_.kv_width);
    prefill_conv_projected_.reserve(r * 3 * shape_.hidden);
    prefill_gate_up_.reserve(r * 2 * shape_.intermediate);
    prefill_activated_.reserve(r * shape_.intermediate);
    prefill_mlp_output_.reserve(r * shape_.hidden);
}

void LfmModel::Impl::release_prefill_workspace() {
}

} // namespace lfm


