#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/detail/checkpoint/bootstrap.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/paged_kv.hpp"
#include "celeg/backend/cuda/weight_policy.hpp"
#include "celeg/model/weights/quantization.hpp"
#include "celeg/backend/cuda/weight_layout.hpp"
#include "celeg/runtime/weights_topology.hpp"
#include "celeg/backend/cuda/weights_loader.hpp"
#include "celeg/backend/cuda/moe.hpp"

#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
namespace celeg {

namespace {
void* allocate_pinned_host(std::size_t bytes) {
    void* pointer = nullptr;
    return cudaMallocHost(&pointer, bytes) == cudaSuccess ? pointer : nullptr;
}

void deallocate_pinned_host(void* pointer) {
    if (pointer) cudaFreeHost(pointer);
}

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

void CudaCompiledModel::load_checkpoint_weights(
    const std::string& model_path,
    const detail::ModelBootstrap& bootstrap) {
    // Immutable device weights are shared across all request lanes that use
    // the same checkpoint, device and quantization mode. The WeightLoader
    // owns the process-wide cache and the SafeTensor I/O + quantization;
    // the compiled model only retains the resulting shared_ptr + the layer views.
    resources_.weights_ = WeightLoader::acquire(
        model_path, resources_.options_.weight_mode,
        resources_.options_.expert_offload.fingerprint());
    resources_.weight_loader_ = std::make_unique<WeightLoader>(
        resources_.weights_, resources_.options_.weight_mode);

    // Only one constructor populates a shared checkpoint at a time. Other
    // sessions wait, then reuse the immutable device buffers.
    std::unique_lock<std::mutex> shared_weights_lock(resources_.weights_->mutex);
    resources_.weights_->repo = bootstrap.checkpoint.repository;
    const IWeightRepository& repo = *resources_.weights_->repo;
    resources_.embedding_ = resources_.weight_loader_->load_linear_weight(
        repo, tensor_name(*resources_.tensor_naming_, TensorRole::TokenEmbedding),
        {resources_.shape_.vocab_size, resources_.shape_.hidden});
    resources_.final_norm_ = resources_.weight_loader_->load_weight(
        repo, tensor_name(*resources_.tensor_naming_, TensorRole::FinalNorm),
        {resources_.shape_.hidden});
    if (resources_.shape_.has_per_layer_input) {
        const int ple = resources_.shape_.per_layer_input_size;
        resources_.per_layer_embedding_ = resources_.weight_loader_->load_linear_weight(
            repo, tensor_name(*resources_.tensor_naming_, TensorRole::PerLayerEmbedding),
            {resources_.shape_.vocab_size,
             resources_.shape_.num_hidden_layers * ple});
        resources_.per_layer_context_projection_ = resources_.weight_loader_->load_linear_weight(
            repo, tensor_name(*resources_.tensor_naming_, TensorRole::PerLayerContextProjection),
            {resources_.shape_.num_hidden_layers * ple, resources_.shape_.hidden});
        resources_.per_layer_projection_norm_ = resources_.weight_loader_->load_weight(
            repo, tensor_name(*resources_.tensor_naming_, TensorRole::PerLayerProjectionNorm),
            {ple});
    }
    {
        if (resources_.embedding_->gguf_quantized()) {
            if (resources_.embedding_->gguf_segments.size() != 1) {
                throw std::runtime_error("GGUF embedding must use one native segment");
            }
            resources_.weight_layout_ = make_gguf_weight_layout(resources_.embedding_->gguf_segments.front());
        } else if (resources_.options_.weight_mode == WeightMode::Int8) {
            resources_.weight_layout_ = make_weight_layout(
                resources_.options_.weight_mode, resources_.embedding_->int8, resources_.embedding_->scales);
        } else if (resources_.options_.weight_mode == WeightMode::Int4) {
            resources_.weight_layout_ = make_weight_layout(
                resources_.options_.weight_mode, resources_.embedding_->int4, resources_.embedding_->scales);
        } else {
            resources_.weight_layout_ = make_weight_layout(
                resources_.options_.weight_mode, resources_.embedding_->bf16, resources_.embedding_->scales);
        }
    }
    // Untied LM head: load the separate lm_head weight for the final logits
    // projection. Some checkpoints omit the lm_head tensor yet leave
    // tie_word_embeddings unset; in that case the head is effectively tied to
    // the embedding table, so we fall back to it instead of erroring.
    const std::string lm_head_name =
        tensor_name(*resources_.tensor_naming_, TensorRole::LanguageModelHead);
    if (!resources_.model_.capabilities.tied_embeddings && repo.contains(lm_head_name)) {
        resources_.lm_head_ = resources_.weight_loader_->load_linear_weight(
            repo, lm_head_name, {resources_.shape_.vocab_size, resources_.shape_.hidden});
    }

    // Resolve the MoE expert-offload plan before loading experts. Snapshot the
    // free VRAM now (embeddings/final norm already uploaded; attention/conv and
    // experts are loaded below) and estimate the always-resident non-expert
    // weight footprint analytically from the topology so the planner can decide
    // how many experts fit in the GPU cache per layer.
    if (resources_.options_.expert_offload.enabled() &&
        resources_.shape_.num_experts > 0) {
        size_t free_bytes = 0, total_bytes = 0;
        CELEG_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
        const int moe_layers = moe_layer_count(resources_.shape_);
        const size_t bpe = bytes_per_expert_bf16(resources_.shape_);
        // All experts are BF16; non-expert weights = everything else already or
        // about to be resident on the GPU.
        const size_t all_expert_bytes =
            static_cast<size_t>(resources_.shape_.num_experts) *
            static_cast<size_t>(moe_layers) * bpe;
        // Conservative non-expert estimate: total checkpoint minus experts.
        // embed + lm-head + attention + conv + dense FFN + router + norms.
        const size_t embed_bytes =
            static_cast<size_t>(resources_.shape_.vocab_size) * resources_.shape_.hidden *
            sizeof(__nv_bfloat16);
        const size_t per_attn = static_cast<size_t>(resources_.shape_.maximum_attention_projection_width()) *
            resources_.shape_.hidden;
        const size_t attn_bytes = per_attn *
            static_cast<size_t>(resources_.shape_.attention_layer_count) * sizeof(__nv_bfloat16);
        const size_t dense_ffn_bytes =
            static_cast<size_t>(resources_.shape_.num_dense_layers) *
            (3ull * resources_.shape_.dense_intermediate * resources_.shape_.hidden) * sizeof(__nv_bfloat16);
        const size_t router_bytes =
            static_cast<size_t>(moe_layers) * resources_.shape_.num_experts * resources_.shape_.hidden *
            (sizeof(__nv_bfloat16) + sizeof(float));
        const size_t non_expert_bytes =
            embed_bytes + attn_bytes + dense_ffn_bytes + router_bytes +
            (64ull << 20);  // norms/conv/bias slack
        (void)all_expert_bytes;

        ExpertOffloadPlanInputs pin;
        pin.shape = resources_.shape_;
        pin.options = resources_.options_.expert_offload;
        pin.gpu_free_bytes = free_bytes;
        pin.non_expert_weight_bytes = non_expert_bytes;
        pin.workspace_bytes = resources_.options_.lt_workspace_bytes + (256ull << 20);
        pin.context_tokens = max_context_;
        workspace_.expert_offload_plan_ = plan_expert_offload(pin);
        workspace_.expert_transfer_stream_ = std::make_unique<CudaStream>();

        resources_.weights_->expert_offload_plan = workspace_.expert_offload_plan_;
        if (resources_.options_.expert_offload.backing == ExpertBackingMode::DiskCached && !resources_.weights_->pinned_expert_cache) {
            size_t bpe = bytes_per_expert_bf16(resources_.shape_);
            size_t gu_bytes = 2 * resources_.shape_.moe_intermediate * resources_.shape_.hidden * sizeof(__nv_bfloat16);
            size_t dn_bytes = resources_.shape_.hidden * resources_.shape_.moe_intermediate * sizeof(__nv_bfloat16);
            resources_.weights_->pinned_expert_cache = std::make_unique<PinnedExpertCache>(
                resources_.options_.expert_offload.host_expert_cache_bytes,
                bpe, gu_bytes, dn_bytes,
                allocate_pinned_host, deallocate_pinned_host);
        }
        if (resources_.options_.expert_offload.backing == ExpertBackingMode::DiskCached && !resources_.weights_->expert_io_manager) {
            resources_.weights_->expert_io_manager = std::make_unique<ExpertIoManager>(
                resources_.options_.expert_offload.io_workers,
                resources_.options_.expert_offload.io_queue_depth);
        }
        if (!resources_.options_.expert_offload.expert_sidecar_path.empty() && !resources_.weights_->expert_sidecar) {
            auto sidecar = std::make_unique<ExpertSidecar>();
            int moe_layers = moe_layer_count(resources_.shape_);
            if (sidecar->load(resources_.options_.expert_offload.expert_sidecar_path,
                             moe_layers, resources_.shape_.num_experts,
                             resources_.shape_.moe_intermediate, resources_.shape_.hidden)) {
                resources_.weights_->expert_sidecar = std::move(sidecar);
                std::fprintf(stderr, "Loaded compatible expert sidecar from %s\n",
                             resources_.options_.expert_offload.expert_sidecar_path.c_str());
            } else {
                std::fprintf(stderr, "WARNING: Sidecar %s is incompatible or could not be loaded; falling back to safetensors.\n",
                             resources_.options_.expert_offload.expert_sidecar_path.c_str());
            }
        }
        if (!resources_.options_.expert_offload.usage_profile_path.empty()) {
            resources_.weights_->usage_profile_path = resources_.options_.expert_offload.usage_profile_path;
            if (resources_.weights_->usage_stats.layers.empty()) {
                int moe_layers = moe_layer_count(resources_.shape_);
                if (resources_.weights_->usage_stats.load(resources_.weights_->usage_profile_path, moe_layers, resources_.shape_.num_experts)) {
                    std::fprintf(stderr, "Loaded persistent expert usage statistics from %s\n",
                                 resources_.weights_->usage_profile_path.c_str());
                } else {
                    resources_.weights_->usage_stats.layers.assign(
                        static_cast<size_t>(moe_layers),
                        std::vector<ExpertUsageEntry>(static_cast<size_t>(resources_.shape_.num_experts)));
                }
            }
        }
        std::fprintf(stderr, "%s", workspace_.expert_offload_plan_.report().c_str());
    }
    workspace_.expert_caches_.resize(static_cast<size_t>(resources_.shape_.num_hidden_layers));
    if (resources_.weights_->expert_controllers.empty()) {
        resources_.weights_->expert_controllers.resize(static_cast<size_t>(resources_.shape_.num_hidden_layers));
    }
    workspace_.expert_catalog_.resize(static_cast<size_t>(resources_.shape_.num_hidden_layers));
    if (resources_.weights_->expert_catalog.empty()) {
        resources_.weights_->expert_catalog.resize(static_cast<size_t>(resources_.shape_.num_hidden_layers));
    }

    resources_.layers_.reserve(static_cast<size_t>(resources_.shape_.num_hidden_layers));
    std::vector<int> shared_owner(2, -1);
    for (int i = 0; i < resources_.shape_.num_hidden_layers; ++i) {
        LayerCommon common_layer;
        common_layer.operator_norm = resources_.weight_loader_->load_weight(
            repo, tensor_name(*resources_.tensor_naming_, TensorRole::AttentionInputNorm, i),
            {resources_.shape_.hidden});
        common_layer.ffn_norm = resources_.weight_loader_->load_weight(
            repo, tensor_name(*resources_.tensor_naming_, TensorRole::FfnInputNorm, i),
            {resources_.shape_.hidden});
        if (resources_.shape_.has_split_attention_norms) {
            common_layer.post_attention_norm = resources_.weight_loader_->load_weight(
                repo, tensor_name(*resources_.tensor_naming_, TensorRole::AttentionPostNorm, i),
                {resources_.shape_.hidden});
            common_layer.post_feed_forward_norm = resources_.weight_loader_->load_weight(
                repo, tensor_name(*resources_.tensor_naming_, TensorRole::FfnOutputNorm, i),
                {resources_.shape_.hidden});
        }
        if (resources_.shape_.has_per_layer_input) {
            common_layer.per_layer_input_gate = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(*resources_.tensor_naming_, TensorRole::PerLayerInputGate, i),
                {resources_.shape_.per_layer_input_size, resources_.shape_.hidden});
            common_layer.per_layer_projection = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(*resources_.tensor_naming_, TensorRole::PerLayerProjection, i),
                {resources_.shape_.hidden, resources_.shape_.per_layer_input_size});
            common_layer.per_layer_input_norm = resources_.weight_loader_->load_weight(
                repo, tensor_name(*resources_.tensor_naming_, TensorRole::PerLayerInputNorm, i),
                {resources_.shape_.hidden});
            common_layer.layer_scalar = resources_.weight_loader_->load_weight(
                repo, tensor_name(*resources_.tensor_naming_, TensorRole::LayerScalar, i), {1});
        }
        if (resources_.shape_.layer_uses_moe(i)) {
            // Mixture-of-experts feed-forward for this layer.
            const int E = resources_.shape_.num_experts;
            const int inter = resources_.shape_.moe_intermediate;
            const float* expert_bias = nullptr;
            if (resources_.shape_.use_expert_bias) {
                // Checkpoint naming varies: LFM2.5 ships
                // "feed_forward.expert_bias.weight"; the earlier LFM2 MoE
                // release ships "feed_forward.expert_bias" (no .weight suffix).
                const std::string bias_name = repo.contains(
                    layer_name(i, "feed_forward.expert_bias.weight"))
                    ? layer_name(i, "feed_forward.expert_bias.weight")
                    : layer_name(i, "feed_forward.expert_bias");
                expert_bias = resources_.weight_loader_->load_f32_weight(
                    repo, bias_name, {static_cast<int64_t>(E)});
            }
            const LinearWeight* router = resources_.weight_loader_->load_router_weight(
                repo, i, E, resources_.shape_.hidden);

            // Cache a device float copy of the router weight for the CUDA
            // router kernel (it consumes float, the loaded weight is BF16 —
            // always non-null after Phase 1.1's load_router_weight contract).
            DeviceBuffer<float>& router_float = workspace_.moe_router_float_[static_cast<size_t>(i)];
            router_float.reset(static_cast<size_t>(E) * resources_.shape_.hidden);
            launch_cast_bf16_to_float(
                router->bf16, router_float.data(),
                static_cast<int>(E) * resources_.shape_.hidden, stream_.get());

            MoeFfnWeights moe_weights{};
            moe_weights.router = router;
            moe_weights.expert_bias = expert_bias;
            moe_weights.router_float = router_float.data();

            if (workspace_.expert_offload_plan_.enabled) {
                const HostTensorView expert_probe = repo.tensor(
                    layer_name(i, "feed_forward.experts.0.w1.weight"));
                if (expert_probe.dtype == TensorDType::Quantized) {
                    throw std::invalid_argument(
                        "native GGUF MoE experts do not support BF16 offload; "
                        "disable expert offload to keep packed Q4/Q6 weights resident");
                }
                if (resources_.options_.expert_offload.backing == ExpertBackingMode::DiskCached) {
                    std::vector<ExpertLocation> catalog =
                        resources_.weight_loader_->build_expert_catalog(repo, i, E, inter, resources_.shape_.hidden);
                    workspace_.expert_catalog_[static_cast<size_t>(i)] = catalog;
                    if (resources_.weights_->expert_catalog[static_cast<size_t>(i)].empty()) {
                        resources_.weights_->expert_catalog[static_cast<size_t>(i)] = catalog;
                    }

                    size_t gate_up_bytes = 2 * inter * resources_.shape_.hidden * sizeof(__nv_bfloat16);
                    size_t down_bytes = resources_.shape_.hidden * inter * sizeof(__nv_bfloat16);
                    auto cache = std::make_unique<ExpertLayerCache>(
                        E, workspace_.expert_offload_plan_.experts_per_layer,
                        gate_up_bytes, down_bytes);
                    cache->set_policy(resources_.options_.expert_offload.policy);
                    std::vector<const __nv_bfloat16*> empty_host_dev(static_cast<size_t>(E), nullptr);
                    cache->set_host_sources(empty_host_dev, empty_host_dev);

                    auto controller = std::make_unique<ResidencyController>();
                    controller->cache = std::move(cache);
                    controller->transfer_stream = std::make_unique<CudaStream>();

                    if (workspace_.expert_offload_plan_.experts_per_layer > 0) {
                        for (int s = 0; s < workspace_.expert_offload_plan_.experts_per_layer; ++s) {
                            const ExpertLocation& loc = catalog[static_cast<size_t>(s)];
                            ExpertHostLease lease = resources_.weights_->pinned_expert_cache->acquire(i, s, [&](std::span<std::byte> gu_dest, std::span<std::byte> dn_dest) {
                                if (resources_.weights_->expert_sidecar) {
                                    resources_.weights_->expert_sidecar->read_expert(i, s, gu_dest, dn_dest);
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
                    CELEG_CUDA(cudaStreamSynchronize(controller->transfer_stream->get()));
                    moe_weights.gate_up_ptrs = controller->cache->gate_up_ptrs();
                    moe_weights.down_ptrs = controller->cache->down_ptrs();
                    resources_.weights_->expert_controllers[static_cast<size_t>(i)] = std::move(controller);
                    workspace_.expert_caches_[static_cast<size_t>(i)] = resources_.weights_->expert_controllers[static_cast<size_t>(i)]->cache.get();
                } else {
                    // Host-backed experts + per-layer GPU cache. Load expert bytes
                    // into the host store (no eager device upload), build the cache,
                    // and seed it with the first `experts_per_layer` experts.
                    WeightLoader::HostExpertLayer host_layer =
                        resources_.weight_loader_->load_moe_experts_host(
                            repo, i, E, inter, resources_.shape_.hidden, workspace_.host_expert_store_,
                            resources_.options_.expert_offload.host_mode);
                    auto cache = std::make_unique<ExpertLayerCache>(
                        E, workspace_.expert_offload_plan_.experts_per_layer,
                        host_layer.gate_up_bytes, host_layer.down_bytes);
                    cache->set_policy(resources_.options_.expert_offload.policy);
                    cache->set_host_sources(host_layer.gate_up_host_dev,
                                            host_layer.down_host_dev);

                    auto controller = std::make_unique<ResidencyController>();
                    controller->cache = std::move(cache);
                    controller->transfer_stream = std::make_unique<CudaStream>();

                    std::vector<int> seed(static_cast<size_t>(
                        workspace_.expert_offload_plan_.experts_per_layer));
                    for (int s = 0; s < workspace_.expert_offload_plan_.experts_per_layer; ++s) {
                        seed[static_cast<size_t>(s)] = s;
                    }
                    controller->cache->seed(seed, controller->transfer_stream->get());
                    CELEG_CUDA(cudaStreamSynchronize(controller->transfer_stream->get()));
                    moe_weights.gate_up_ptrs = controller->cache->gate_up_ptrs();
                    moe_weights.down_ptrs = controller->cache->down_ptrs();
                    resources_.weights_->expert_controllers[static_cast<size_t>(i)] = std::move(controller);
                    workspace_.expert_caches_[static_cast<size_t>(i)] = resources_.weights_->expert_controllers[static_cast<size_t>(i)]->cache.get();
                }
            } else {
                moe_weights.gate_up =
                    resources_.weight_loader_->load_moe_gate_up(repo, i, E, inter, resources_.shape_.hidden);
                moe_weights.down =
                    resources_.weight_loader_->load_moe_down(repo, i, E, inter, resources_.shape_.hidden);
            }

            common_layer.feed_forward = moe_weights;
        } else {
            const int intermediate = resources_.shape_.feed_forward_intermediates.empty()
                ? resources_.shape_.intermediate
                : resources_.shape_.feed_forward_intermediates.at(static_cast<size_t>(i));
            const LinearWeight* w13 = resources_.weight_loader_->load_concat_linear_weight(
                repo, layer_name(i, "feed_forward.w13.weight"),
                {
                    {tensor_name(*resources_.tensor_naming_, TensorRole::FfnGate, i),
                     {intermediate, resources_.shape_.hidden}},
                    {tensor_name(*resources_.tensor_naming_, TensorRole::FfnUp, i),
                     {intermediate, resources_.shape_.hidden}},
                });
            const LinearWeight* w2 = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(*resources_.tensor_naming_, TensorRole::FfnDown, i),
                {resources_.shape_.hidden, intermediate});
            common_layer.feed_forward = DenseFfnWeights{w13, w2};
        }

        const MixerKind layer_type =
            resources_.shape_.mixer_kinds[static_cast<size_t>(i)];
        if (layer_type == MixerKind::Attention) {
            AttentionLayer attention_layer;
            attention_layer.common = common_layer;
            attention_layer.layout = resources_.shape_.attention_layout(i);
            const AttentionSpec& layout = attention_layer.layout;
            attention_layer.query = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(*resources_.tensor_naming_, TensorRole::AttentionQuery, i),
                {layout.query_width(), resources_.shape_.hidden});
            if (!layout.kv_sharing.shared() || layout.kv_sharing.publishes) {
                attention_layer.key = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(*resources_.tensor_naming_, TensorRole::AttentionKey, i),
                    {layout.key_value_width(), resources_.shape_.hidden});
                attention_layer.value = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(*resources_.tensor_naming_, TensorRole::AttentionValue, i),
                    {layout.key_value_width(), resources_.shape_.hidden});
            } else {
                if (layout.kv_sharing.group < 0 ||
                    layout.kv_sharing.group >= static_cast<int>(shared_owner.size()) ||
                    shared_owner[static_cast<size_t>(layout.kv_sharing.group)] < 0) {
                    throw std::runtime_error("CUDA shared KV consumer has no owner");
                }
                attention_layer.kv_owner_layer =
                    shared_owner[static_cast<size_t>(layout.kv_sharing.group)];
            }
            attention_layer.out = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(*resources_.tensor_naming_, TensorRole::AttentionOutput, i),
                {resources_.shape_.hidden, layout.query_width()});
            if (layout.query_key_norm) {
                attention_layer.q_norm = resources_.weight_loader_->load_weight(
                    repo, tensor_name(*resources_.tensor_naming_, TensorRole::AttentionQueryNorm, i),
                    {layout.head_dim});
                if (attention_layer.key) {
                    attention_layer.k_norm = resources_.weight_loader_->load_weight(
                        repo, tensor_name(*resources_.tensor_naming_, TensorRole::AttentionKeyNorm, i),
                        {layout.head_dim});
                }
            }

            if (resources_.options_.allocate_local_kv_cache && attention_layer.key) {
                const size_t cache_elements = static_cast<size_t>(max_context_) *
                    static_cast<size_t>(layout.key_value_width());
                if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
                    attention_layer.key_cache_int8.reset(cache_elements);
                    attention_layer.value_cache_int8.reset(cache_elements);
                    const size_t scale_elements =
                        static_cast<size_t>(max_context_) *
                        static_cast<size_t>(layout.key_value_heads);
                    attention_layer.key_cache_scales.reset(scale_elements);
                    attention_layer.value_cache_scales.reset(scale_elements);
                } else {
                    attention_layer.key_cache.reset(cache_elements);
                    attention_layer.value_cache.reset(cache_elements);
                }
            }
            if (layout.kv_sharing.publishes) {
                shared_owner[static_cast<size_t>(layout.kv_sharing.group)] = i;
                attention_layer.kv_owner_layer = i;
            }
            if (!layout.kv_sharing.shared()) attention_layer.kv_owner_layer = i;
            resources_.layers_.emplace_back(std::move(attention_layer));
        } else {
            ConvolutionLayer convolution_layer;
            convolution_layer.common = common_layer;
            convolution_layer.conv_in = resources_.weight_loader_->load_linear_weight(
                repo, layer_name(i, "conv.in_proj.weight"),
                {3 * resources_.shape_.hidden, resources_.shape_.hidden});
            convolution_layer.conv_weight = resources_.weight_loader_->load_weight(
                repo, layer_name(i, "conv.conv.weight"),
                {resources_.shape_.hidden, 1, resources_.shape_.conv_cache});
            convolution_layer.conv_out = resources_.weight_loader_->load_linear_weight(
                repo, layer_name(i, "conv.out_proj.weight"),
                {resources_.shape_.hidden, resources_.shape_.hidden});
            convolution_layer.conv_state.reset(
                static_cast<size_t>(resources_.shape_.conv_cache) * resources_.shape_.hidden);
            resources_.layers_.emplace_back(std::move(convolution_layer));
        }
    }

}

} // namespace celeg
