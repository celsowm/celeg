#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/detail/checkpoint/bootstrap.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/paged_kv.hpp"
#include "celeg/backend/cuda/weight_policy.hpp"
#include "celeg/model/weights/quantization.hpp"
#include "celeg/backend/cuda/weight_layout.hpp"
#include "celeg/runtime/weights_topology.hpp"
#include "celeg/backend/cuda/weights_loader.hpp"
#include "celeg/backend/cuda/weight_setup_support.hpp"
#include "celeg/backend/cuda/weight_setup.hpp"
#include "celeg/backend/cuda/moe.hpp"
#include "celeg/backend/cuda/moe/expert_source.hpp"

#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
namespace celeg {

namespace {
std::string layer_name(int index, const std::string& suffix) {
    return cuda_layer_name(index, suffix);
}

std::string tensor_name(std::span<const TensorRequest> requests, TensorRole role,
                        int layer = -1) {
    return cuda_tensor_name(requests, role, layer);
}

} // namespace

void CudaCompiledModel::load_checkpoint_weights(
    const std::string& model_path,
    const detail::ModelBootstrap& bootstrap) {
    CudaWeightSetup::load(*this, model_path, bootstrap,
        [this](const IWeightRepository& repo) {
    // Resolve the MoE expert-offload plan before loading experts. Snapshot the
    // free VRAM now (embeddings/final norm already uploaded; attention/conv and
    // experts are loaded below) and estimate the always-resident non-expert
    // weight footprint analytically from the topology so the planner can decide
    // how many experts fit in the GPU cache per layer.
    configure_cuda_expert_resources(*this);
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
        const bool nemotron = resources_.shape_.mamba2_layer_count > 0;
        common_layer.operator_norm = resources_.weight_loader_->load_weight(
            repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::AttentionInputNorm, i),
            {resources_.shape_.hidden});
        if (!nemotron) {
            common_layer.ffn_norm = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::FfnInputNorm, i),
                {resources_.shape_.hidden});
        }
        if (!nemotron && resources_.shape_.has_split_attention_norms) {
            common_layer.post_attention_norm = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::AttentionPostNorm, i),
                {resources_.shape_.hidden});
            common_layer.post_feed_forward_norm = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::FfnOutputNorm, i),
                {resources_.shape_.hidden});
        }
        if (resources_.program_.per_layer_input.enabled) {
            common_layer.per_layer_input_gate = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::PerLayerInputGate, i),
                {resources_.shape_.per_layer_input_size, resources_.shape_.hidden});
            common_layer.per_layer_projection = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::PerLayerProjection, i),
                {resources_.shape_.hidden, resources_.shape_.per_layer_input_size});
            common_layer.per_layer_input_norm = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::PerLayerInputNorm, i),
                {resources_.shape_.hidden});
            common_layer.layer_scalar = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::LayerScalar, i), {1});
        }
        if (nemotron) {
            // Nemotron-H owns its block-specific projections in the layer
            // variant below; there is no generic post-mixer FFN descriptor.
            common_layer.feed_forward = DenseFfnWeights{};
        } else if (resources_.shape_.layer_uses_moe(i)) {
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
                            ExpertHostLease lease = resources_.weights_->host_expert_cache->acquire(i, s, [&](std::span<std::byte> payload) {
                                if (!resources_.weights_->expert_source) {
                                    throw std::runtime_error("CUDA expert source is not initialized");
                                }
                                resources_.weights_->expert_source->read(i, s, payload);
                            });
                            controller->cache->promote(
                                           s, s,
                                           reinterpret_cast<const __nv_bfloat16*>(lease.payload()),
                                           reinterpret_cast<const __nv_bfloat16*>(lease.payload() + loc.w1.bytes + loc.w3.bytes),
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
                    {tensor_name(resources_.model_.weight_plan.requests, TensorRole::FfnGate, i),
                     {intermediate, resources_.shape_.hidden}},
                    {tensor_name(resources_.model_.weight_plan.requests, TensorRole::FfnUp, i),
                     {intermediate, resources_.shape_.hidden}},
                });
            const LinearWeight* w2 = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::FfnDown, i),
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
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::AttentionQuery, i),
                {layout.query_width(), resources_.shape_.hidden});
            if (!layout.kv_sharing.shared() || layout.kv_sharing.publishes) {
                attention_layer.key = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::AttentionKey, i),
                    {layout.key_value_width(), resources_.shape_.hidden});
                attention_layer.value = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::AttentionValue, i),
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
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::AttentionOutput, i),
                {resources_.shape_.hidden, layout.query_width()});
            if (layout.query_key_norm) {
                attention_layer.q_norm = resources_.weight_loader_->load_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::AttentionQueryNorm, i),
                    {layout.head_dim});
                if (attention_layer.key) {
                    attention_layer.k_norm = resources_.weight_loader_->load_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::AttentionKeyNorm, i),
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
        } else if (layer_type == MixerKind::Mamba2) {
            Mamba2Layer mamba_layer;
            mamba_layer.common = common_layer;
            mamba_layer.spec = resources_.shape_.mamba2_layouts.at(static_cast<size_t>(i));
            const Mamba2Spec& spec = mamba_layer.spec;
            const int conv_dim = spec.intermediate_size +
                2 * spec.group_count * spec.state_size;
            mamba_layer.in = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2Input, i),
                {2 * spec.intermediate_size + 2 * spec.group_count * spec.state_size +
                 spec.num_heads, resources_.shape_.hidden});
            mamba_layer.conv_weight = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2Conv, i),
                {conv_dim, 1, spec.conv_kernel});
            mamba_layer.conv_bias = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2ConvBias, i),
                {conv_dim});
            mamba_layer.dt_bias = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2DtBias, i),
                {spec.num_heads});
            mamba_layer.a_log = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2ALog, i),
                {spec.num_heads});
            mamba_layer.d = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2D, i),
                {spec.num_heads});
            mamba_layer.norm = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2Norm, i),
                {spec.intermediate_size});
            mamba_layer.out = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::Mamba2Output, i),
                {resources_.shape_.hidden, spec.intermediate_size});
            mamba_layer.conv_state.reset(static_cast<size_t>(conv_dim) * spec.conv_kernel);
            mamba_layer.ssm_state.reset(static_cast<size_t>(spec.intermediate_size) * spec.state_size);
            resources_.layers_.emplace_back(std::move(mamba_layer));
        } else if (layer_type == MixerKind::MlpOnly) {
            MlpOnlyLayer mlp_layer;
            mlp_layer.common = common_layer;
            mlp_layer.spec = resources_.shape_.mlp_only_layouts.at(static_cast<size_t>(i));
            mlp_layer.up = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::FfnUp, i),
                {mlp_layer.spec.intermediate_size, resources_.shape_.hidden});
            mlp_layer.down = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::FfnDown, i),
                {resources_.shape_.hidden, mlp_layer.spec.intermediate_size});
            resources_.layers_.emplace_back(std::move(mlp_layer));
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
        });
}

} // namespace celeg
