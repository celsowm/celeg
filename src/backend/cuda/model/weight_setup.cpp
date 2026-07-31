#include "lfm/detail/model/impl.hpp"
#include "lfm/detail/checkpoint/bootstrap.hpp"
#include "lfm/backend/cuda/kernels/kernels.cuh"
#include "lfm/backend/cuda/paged_kv.hpp"
#include "lfm/model/weights/policy.hpp"
#include "lfm/model/weights/quantization.hpp"
#include "lfm/model/weights/layout.hpp"
#include "lfm/runtime/weights_topology.hpp"
#include "lfm/model/weights/loader.hpp"
#include "lfm/checkpoint/formats/gguf.hpp"
#include "lfm/checkpoint/repositories/gguf.hpp"
#include "lfm/runtime/moe.hpp"

#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
namespace lfm {

namespace {
std::string layer_name(int index, const std::string& suffix) {
    return "model.layers." + std::to_string(index) + "." + suffix;
}

std::string tensor_name(const ITensorNamingPolicy& policy, TensorRole role,
                        int layer = -1) {
    const auto names = policy.candidates({role, layer, -1, {}});
    if (names.empty()) throw std::logic_error("tensor naming policy returned no candidates");
    return names.front();
}
} // namespace

void Model::Impl::load_checkpoint_weights(
    const std::string& model_path,
    const detail::ModelBootstrap& bootstrap) {
    const bool is_gguf = bootstrap.is_gguf;
    const std::shared_ptr<GgufFile> gguf_file = bootstrap.gguf_file;
    const ModelConfig& config = bootstrap.config;
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
    embedding_ = weight_loader_->load_linear_weight(
        repo, tensor_name(*tensor_naming_, TensorRole::TokenEmbedding),
        {shape_.vocab_size, shape_.hidden});
    final_norm_ = weight_loader_->load_weight(
        repo, tensor_name(*tensor_naming_, TensorRole::FinalNorm),
        {shape_.hidden});
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
    const std::string lm_head_name =
        tensor_name(*tensor_naming_, TensorRole::LanguageModelHead);
    if (!config.tie_word_embeddings && repo.contains(lm_head_name)) {
        lm_head_ = weight_loader_->load_linear_weight(
            repo, lm_head_name, {shape_.vocab_size, shape_.hidden});
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
            repo, tensor_name(*tensor_naming_, TensorRole::AttentionInputNorm, i),
            {shape_.hidden});
        common_layer.ffn_norm = weight_loader_->load_weight(
            repo, tensor_name(*tensor_naming_, TensorRole::FfnInputNorm, i),
            {shape_.hidden});
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
            const LinearWeight* router = weight_loader_->load_router_weight(
                repo, i, E, shape_.hidden);

            // Cache a device float copy of the router weight for the CUDA
            // router kernel (it consumes float, the loaded weight is BF16 —
            // always non-null after Phase 1.1's load_router_weight contract).
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
                const HostTensorView expert_probe = repo.tensor(
                    layer_name(i, "feed_forward.experts.0.w1.weight"));
                if (expert_probe.dtype == TensorDType::Quantized) {
                    throw std::invalid_argument(
                        "native GGUF MoE experts do not support BF16 offload; "
                        "disable expert offload to keep packed Q4/Q6 weights resident");
                }
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
                                    const auto& reader =
                                        require_random_access_tensor_reader(repo);
                                    reader.read(loc.w1, gu_dest.subspan(0, loc.w1.bytes));
                                    reader.read(loc.w3, gu_dest.subspan(loc.w1.bytes, loc.w3.bytes));
                                    reader.read(loc.w2, dn_dest);
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
                    {tensor_name(*tensor_naming_, TensorRole::FfnGate, i),
                     {shape_.intermediate, shape_.hidden}},
                    {tensor_name(*tensor_naming_, TensorRole::FfnUp, i),
                     {shape_.intermediate, shape_.hidden}},
                });
            const LinearWeight* w2 = weight_loader_->load_linear_weight(
                repo, tensor_name(*tensor_naming_, TensorRole::FfnDown, i),
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
                    {tensor_name(*tensor_naming_, TensorRole::AttentionQuery, i),
                     {shape_.q_width, shape_.hidden}},
                    {tensor_name(*tensor_naming_, TensorRole::AttentionKey, i),
                     {shape_.kv_width, shape_.hidden}},
                    {tensor_name(*tensor_naming_, TensorRole::AttentionValue, i),
                     {shape_.kv_width, shape_.hidden}},
                });
            attention_layer.out = weight_loader_->load_linear_weight(
                repo, tensor_name(*tensor_naming_, TensorRole::AttentionOutput, i),
                {shape_.hidden, shape_.hidden});
            if (shape_.query_key_norm) {
                attention_layer.q_norm = weight_loader_->load_weight(
                    repo, layer_name(i, "self_attn.q_layernorm.weight"),
                    {shape_.head_dim});
                attention_layer.k_norm = weight_loader_->load_weight(
                    repo, layer_name(i, "self_attn.k_layernorm.weight"),
                    {shape_.head_dim});
            }

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

}

} // namespace lfm
