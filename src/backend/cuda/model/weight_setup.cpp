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

int last_full_attention_layer(const RuntimeTopology& shape) {
    for (int layer = shape.num_hidden_layers - 1; layer >= 0; --layer) {
        if (shape.mixer_kinds.at(static_cast<size_t>(layer)) == MixerKind::Attention) {
            return layer;
        }
    }
    return -1;
}

void load_mtp_resources(CudaCompiledModel& model, const IWeightRepository& repo) {
    CudaModelResources& resources = model.resources_;
    if (!resources.options_.enable_mtp) return;
    if (resources.shape_.mtp_num_hidden_layers <= 0) {
        throw std::invalid_argument("MTP was requested but the checkpoint has no MTP layers");
    }
    if (!repo.contains("mtp.fc.weight")) {
        throw std::runtime_error("MTP was requested but mtp.fc.weight is missing");
    }
    if (!resources.options_.allocate_local_kv_cache) {
        throw std::invalid_argument("MTP requires a local KV cache; disable packed/paged execution");
    }
    const int full_layer = last_full_attention_layer(resources.shape_);
    if (full_layer < 0) {
        throw std::runtime_error("MTP requires a full-attention target layer");
    }
    const AttentionSpec mtp_layout = [&]() {
        AttentionSpec layout = resources.shape_.attention_layout(full_layer);
        layout.kv_sharing = {};
        return layout;
    }();

    CudaMtpResources& mtp = resources.mtp_;
    mtp.enabled = true;
    mtp.layer_count = resources.shape_.mtp_num_hidden_layers;
    mtp.fc = resources.weight_loader_->load_linear_weight(
        repo, "mtp.fc.weight",
        {resources.shape_.hidden, 2 * resources.shape_.hidden});
    mtp.pre_fc_norm_embedding = resources.weight_loader_->load_rms_norm_weight(
        repo, "mtp.pre_fc_norm_embedding.weight", {resources.shape_.hidden},
        resources.shape_.numerical_policy.rms_norm_add_one);
    mtp.pre_fc_norm_hidden = resources.weight_loader_->load_rms_norm_weight(
        repo, "mtp.pre_fc_norm_hidden.weight", {resources.shape_.hidden},
        resources.shape_.numerical_policy.rms_norm_add_one);
    mtp.norm = resources.weight_loader_->load_rms_norm_weight(
        repo, "mtp.norm.weight", {resources.shape_.hidden},
        resources.shape_.numerical_policy.rms_norm_add_one);
    mtp.logits = resources.lm_head_ ? resources.lm_head_ : resources.embedding_;
    mtp.layers.reserve(static_cast<size_t>(mtp.layer_count));

    for (int index = 0; index < mtp.layer_count; ++index) {
        const std::string prefix = "mtp.layers." + std::to_string(index);
        LayerCommon common_layer;
        common_layer.operator_norm = resources.weight_loader_->load_rms_norm_weight(
            repo, prefix + ".input_layernorm.weight", {resources.shape_.hidden},
            resources.shape_.numerical_policy.rms_norm_add_one);
        common_layer.ffn_norm = resources.weight_loader_->load_rms_norm_weight(
            repo, prefix + ".post_attention_layernorm.weight", {resources.shape_.hidden},
            resources.shape_.numerical_policy.rms_norm_add_one);

        if (resources.shape_.num_experts > 0) {
            const int E = resources.shape_.num_experts;
            const int inter = resources.shape_.moe_intermediate;
            MoeFfnWeights moe{};
            moe.router = resources.weight_loader_->load_router_weight_named(
                repo, prefix + ".mlp.gate.weight", E, resources.shape_.hidden);
            const int resource_layer = resources.shape_.num_hidden_layers + index;
            DeviceBuffer<float>& router_float =
                model.workspace_.moe_router_float_[static_cast<size_t>(resource_layer)];
            router_float.reset(static_cast<size_t>(E) * resources.shape_.hidden);
            launch_cast_bf16_to_float(moe.router->bf16, router_float.data(),
                                      E * resources.shape_.hidden, model.stream_.get());
            moe.router_float = router_float.data();

            if (resources.shape_.shared_expert_intermediate > 0) {
                const int shared = resources.shape_.shared_expert_intermediate;
                moe.shared_w13 = resources.weight_loader_->load_concat_linear_weight(
                    repo, prefix + ".mlp.shared_expert.w13.weight",
                    {{prefix + ".mlp.shared_expert.gate_proj.weight",
                      {shared, resources.shape_.hidden}},
                     {prefix + ".mlp.shared_expert.up_proj.weight",
                      {shared, resources.shape_.hidden}}});
                moe.shared_w2 = resources.weight_loader_->load_linear_weight(
                    repo, prefix + ".mlp.shared_expert.down_proj.weight",
                    {resources.shape_.hidden, shared});
                moe.shared_gate = resources.weight_loader_->load_linear_weight(
                    repo, prefix + ".mlp.shared_expert_gate.weight",
                    {1, resources.shape_.hidden});
            }

            const std::string experts_prefix = prefix + ".mlp.experts";
            if (model.workspace_.expert_offload_plan_.enabled) {
                std::vector<ExpertLocation> catalog =
                    resources.weight_loader_->build_expert_catalog_named(
                        repo, experts_prefix, "gate_proj", "up_proj", "down_proj",
                        E, inter, resources.shape_.hidden);
                model.workspace_.expert_catalog_[static_cast<size_t>(resource_layer)] = catalog;
                if (resources.weights_->expert_catalog[static_cast<size_t>(resource_layer)].empty()) {
                    resources.weights_->expert_catalog[static_cast<size_t>(resource_layer)] = catalog;
                }
                const size_t gate_up_bytes = 2ull * static_cast<size_t>(inter) *
                    resources.shape_.hidden * sizeof(__nv_bfloat16);
                const size_t down_bytes = static_cast<size_t>(resources.shape_.hidden) *
                    inter * sizeof(__nv_bfloat16);
                auto cache = std::make_unique<ExpertLayerCache>(
                    E, model.workspace_.expert_offload_plan_.experts_per_layer,
                    gate_up_bytes, down_bytes);
                cache->set_policy(resources.options_.expert_offload.policy);
                auto controller = std::make_unique<ResidencyController>();
                controller->cache = std::move(cache);
                controller->transfer_stream = std::make_unique<CudaStream>();
                if (resources.options_.expert_offload.backing == ExpertBackingMode::DiskCached) {
                    std::vector<const __nv_bfloat16*> empty(static_cast<size_t>(E), nullptr);
                    controller->cache->set_host_sources(empty, empty);
                    std::vector<ExpertHostLease> seed_leases(static_cast<size_t>(
                        model.workspace_.expert_offload_plan_.experts_per_layer));
                    for (int seed = 0; seed < model.workspace_.expert_offload_plan_.experts_per_layer; ++seed) {
                        const ExpertLocation& location = catalog[static_cast<size_t>(seed)];
                        seed_leases[static_cast<size_t>(seed)] =
                            resources.weights_->host_expert_cache->acquire(
                            resource_layer, seed, [&](std::span<std::byte> payload) {
                                if (!resources.weights_->expert_source) {
                                    throw std::runtime_error("CUDA expert source is not initialized");
                                }
                                resources.weights_->expert_source->read(resource_layer, seed, payload);
                            });
                        ExpertHostLease& lease = seed_leases[static_cast<size_t>(seed)];
                        controller->cache->promote(
                            seed, seed,
                            reinterpret_cast<const __nv_bfloat16*>(lease.payload()),
                            reinterpret_cast<const __nv_bfloat16*>(
                                lease.payload() + location.w1.bytes + location.w3.bytes),
                            controller->transfer_stream->get());
                    }
                    CELEG_CUDA(cudaStreamSynchronize(controller->transfer_stream->get()));
                } else {
                    WeightLoader::HostExpertLayer host_layer =
                        resources.weight_loader_->load_moe_experts_host_named(
                            repo, experts_prefix, "gate_proj", "up_proj", "down_proj",
                            E, inter, resources.shape_.hidden,
                            model.workspace_.host_expert_store_,
                            resources.options_.expert_offload.host_mode);
                    controller->cache->set_host_sources(host_layer.gate_up_host_dev,
                                                        host_layer.down_host_dev);
                    std::vector<int> seed(static_cast<size_t>(
                        model.workspace_.expert_offload_plan_.experts_per_layer));
                    for (int slot = 0; slot < static_cast<int>(seed.size()); ++slot) {
                        seed[static_cast<size_t>(slot)] = slot;
                    }
                    controller->cache->seed(seed, controller->transfer_stream->get());
                    CELEG_CUDA(cudaStreamSynchronize(controller->transfer_stream->get()));
                }
                moe.gate_up_ptrs = controller->cache->gate_up_ptrs();
                moe.down_ptrs = controller->cache->down_ptrs();
                resources.weights_->expert_controllers[static_cast<size_t>(resource_layer)] =
                    std::move(controller);
                model.workspace_.expert_caches_[static_cast<size_t>(resource_layer)] =
                    resources.weights_->expert_controllers[static_cast<size_t>(resource_layer)]->cache.get();
            } else {
                moe.gate_up = resources.weight_loader_->load_moe_gate_up_named(
                    repo, experts_prefix, "gate_proj", "up_proj", E, inter,
                    resources.shape_.hidden);
                moe.down = resources.weight_loader_->load_moe_down_named(
                    repo, experts_prefix, "down_proj", E, inter, resources.shape_.hidden);
            }
            common_layer.feed_forward = moe;
        } else {
            const int inter = resources.shape_.intermediate;
            const LinearWeight* w13 = resources.weight_loader_->load_concat_linear_weight(
                repo, prefix + ".mlp.w13.weight",
                {{prefix + ".mlp.gate_proj.weight", {inter, resources.shape_.hidden}},
                 {prefix + ".mlp.up_proj.weight", {inter, resources.shape_.hidden}}});
            const LinearWeight* w2 = resources.weight_loader_->load_linear_weight(
                repo, prefix + ".mlp.down_proj.weight",
                {resources.shape_.hidden, inter});
            common_layer.feed_forward = DenseFfnWeights{w13, w2};
        }

        AttentionLayer attention;
        attention.common = common_layer;
        attention.layout = mtp_layout;
        attention.query = resources.weight_loader_->load_linear_weight(
            repo, prefix + ".self_attn.q_proj.weight",
            {mtp_layout.query_projection_width(), resources.shape_.hidden});
        attention.key = resources.weight_loader_->load_linear_weight(
            repo, prefix + ".self_attn.k_proj.weight",
            {mtp_layout.key_value_width(), resources.shape_.hidden});
        attention.value = resources.weight_loader_->load_linear_weight(
            repo, prefix + ".self_attn.v_proj.weight",
            {mtp_layout.key_value_width(), resources.shape_.hidden});
        attention.out = resources.weight_loader_->load_linear_weight(
            repo, prefix + ".self_attn.o_proj.weight",
            {resources.shape_.hidden, mtp_layout.query_width()});
        attention.q_norm = resources.weight_loader_->load_rms_norm_weight(
            repo, prefix + ".self_attn.q_norm.weight", {mtp_layout.head_dim},
            resources.shape_.numerical_policy.rms_norm_add_one);
        attention.k_norm = resources.weight_loader_->load_rms_norm_weight(
            repo, prefix + ".self_attn.k_norm.weight", {mtp_layout.head_dim},
            resources.shape_.numerical_policy.rms_norm_add_one);
        const size_t cache_elements = static_cast<size_t>(model.max_context_) *
            static_cast<size_t>(mtp_layout.key_value_width());
        if (resources.options_.kv_cache_mode == KvCacheMode::Int8) {
            attention.key_cache_int8.reset(cache_elements);
            attention.value_cache_int8.reset(cache_elements);
            const size_t scales = static_cast<size_t>(model.max_context_) *
                static_cast<size_t>(mtp_layout.key_value_heads);
            attention.key_cache_scales.reset(scales);
            attention.value_cache_scales.reset(scales);
        } else {
            attention.key_cache.reset(cache_elements);
            attention.value_cache.reset(cache_elements);
        }
        mtp.layers.emplace_back(std::move(attention));
    }
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
    const int mtp_layer_count = resources_.options_.enable_mtp
        ? resources_.shape_.mtp_num_hidden_layers : 0;
    const int resource_layer_count = resources_.shape_.num_hidden_layers +
        mtp_layer_count;
    workspace_.expert_caches_.resize(static_cast<size_t>(resource_layer_count));
    if (resources_.weights_->expert_controllers.size() <
        static_cast<size_t>(resource_layer_count)) {
        resources_.weights_->expert_controllers.resize(
            static_cast<size_t>(resource_layer_count));
    }
    workspace_.expert_catalog_.resize(static_cast<size_t>(resource_layer_count));
    if (resources_.weights_->expert_catalog.size() <
        static_cast<size_t>(resource_layer_count)) {
        resources_.weights_->expert_catalog.resize(
            static_cast<size_t>(resource_layer_count));
    }

    resources_.layers_.reserve(static_cast<size_t>(resources_.shape_.num_hidden_layers));
    std::vector<int> shared_owner(2, -1);
    for (int i = 0; i < resources_.shape_.num_hidden_layers; ++i) {
        LayerCommon common_layer;
        const bool sequential_mixer_model = resources_.shape_.mamba2_layer_count > 0;
        common_layer.operator_norm = resources_.weight_loader_->load_rms_norm_weight(
            repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::AttentionInputNorm, i),
            {resources_.shape_.hidden}, resources_.shape_.numerical_policy.rms_norm_add_one);
        if (!sequential_mixer_model) {
            common_layer.ffn_norm = resources_.weight_loader_->load_rms_norm_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::FfnInputNorm, i),
                {resources_.shape_.hidden}, resources_.shape_.numerical_policy.rms_norm_add_one);
        }
        if (!sequential_mixer_model && resources_.shape_.has_split_attention_norms) {
            common_layer.post_attention_norm = resources_.weight_loader_->load_rms_norm_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::AttentionPostNorm, i),
                {resources_.shape_.hidden}, resources_.shape_.numerical_policy.rms_norm_add_one);
            common_layer.post_feed_forward_norm = resources_.weight_loader_->load_rms_norm_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::FfnOutputNorm, i),
                {resources_.shape_.hidden}, resources_.shape_.numerical_policy.rms_norm_add_one);
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
        if (sequential_mixer_model) {
            // Sequential mixer models own their block-specific projections in the layer
            // variant below; there is no generic post-mixer FFN descriptor.
            common_layer.feed_forward = DenseFfnWeights{};
        } else if (resources_.shape_.layer_uses_moe(i)) {
            // Mixture-of-experts feed-forward for this layer.
            const int E = resources_.shape_.num_experts;
            const int inter = resources_.shape_.moe_intermediate;
            const float* expert_bias = nullptr;
            if (resources_.shape_.use_expert_bias) {
                // Checkpoint naming varies: some formats use
                // "feed_forward.expert_bias.weight"; others use
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

            if (resources_.shape_.shared_expert_intermediate > 0) {
                const int shared_inter = resources_.shape_.shared_expert_intermediate;
                moe_weights.shared_w13 = resources_.weight_loader_->load_concat_linear_weight(
                    repo, layer_name(i, "shared_expert.w13.weight"),
                    {
                        {tensor_name(resources_.model_.weight_plan.requests,
                                     TensorRole::MoeSharedGate, i),
                         {shared_inter, resources_.shape_.hidden}},
                        {tensor_name(resources_.model_.weight_plan.requests,
                                     TensorRole::MoeSharedUp, i),
                         {shared_inter, resources_.shape_.hidden}},
                    });
                moe_weights.shared_w2 = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::MoeSharedDown, i),
                    {resources_.shape_.hidden, shared_inter});
                moe_weights.shared_gate = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::MoeSharedGateWeight, i),
                    {1, resources_.shape_.hidden});
            }

            if (workspace_.expert_offload_plan_.enabled) {
                const bool packed_expert_model = resources_.shape_.shared_expert_intermediate > 0;
                const std::string probe_name = packed_expert_model
                    ? "model.language_model.layers." + std::to_string(i) +
                      ".mlp.experts.gate_up_proj"
                    : layer_name(i, "feed_forward.experts.0.w1.weight");
                const HostTensorView expert_probe = repo.tensor(probe_name);
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
                    // Initial seeding is fully synchronized above, so the
                    // host payloads are no longer needed.  Do not keep one
                    // lease per seeded expert alive across all MoE layers:
                    // that would exhaust a bounded disk-backed HostExpertCache
                    // before inference starts.
                    controller->inflight_transfers.clear();
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
                if (resources_.shape_.shared_expert_intermediate > 0) {
                    moe_weights.gate_up = resources_.weight_loader_->load_expert_linear_weight(
                        repo, tensor_name(resources_.model_.weight_plan.requests,
                                          TensorRole::MoePackedGateUp, i),
                        E, 2 * inter, resources_.shape_.hidden);
                    moe_weights.down = resources_.weight_loader_->load_expert_linear_weight(
                        repo, tensor_name(resources_.model_.weight_plan.requests,
                                          TensorRole::MoePackedDown, i),
                        E, resources_.shape_.hidden, inter);
                } else {
                    moe_weights.gate_up = resources_.weight_loader_->load_moe_gate_up(
                        repo, i, E, inter, resources_.shape_.hidden);
                    moe_weights.down = resources_.weight_loader_->load_moe_down(
                        repo, i, E, inter, resources_.shape_.hidden);
                }
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
            const std::string query_name = tensor_name(
                resources_.model_.weight_plan.requests, TensorRole::AttentionQuery, i);
            const AttentionSpec& layout = attention_layer.layout;
            if (!layout.uses_latent_state()) {
                const HostTensorView query_source = repo.tensor(query_name);
                if (query_source.shape.size() == 2 &&
                    query_source.shape[0] == 2LL * attention_layer.layout.query_width()) {
                    attention_layer.layout.query_gate = true;
                }
            }
            if (const auto* alibi = std::get_if<AlibiBiasSpec>(&layout.bias)) {
                if (static_cast<int>(alibi->slopes.size()) != layout.query_heads) {
                    throw std::invalid_argument("CUDA ALiBi slope count does not match query heads");
                }
                attention_layer.alibi_slopes.reset(alibi->slopes.size());
                CELEG_CUDA(cudaMemcpy(
                    attention_layer.alibi_slopes.data(), alibi->slopes.data(),
                    attention_layer.alibi_slopes.bytes(), cudaMemcpyHostToDevice));
            }
            if (layout.uses_latent_state()) {
                if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
                    throw std::invalid_argument(
                        "CUDA latent attention currently requires BF16 state storage");
                }
                const auto& latent = *layout.latent_state();
                attention_layer.latent_query = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::AttentionLatentQuery, i),
                    {layout.latent_query_content_width(), resources_.shape_.hidden});
                if (layout.latent_query_rope_width() != 0) {
                    attention_layer.latent_query_rope = resources_.weight_loader_->load_linear_weight(
                        repo, tensor_name(resources_.model_.weight_plan.requests,
                                          TensorRole::AttentionLatentQueryRope, i),
                        {layout.latent_query_rope_width(), resources_.shape_.hidden});
                }
                const bool owns_latent_state =
                    !layout.kv_sharing.shared() || layout.kv_sharing.publishes;
                if (owns_latent_state) {
                    attention_layer.latent_key = resources_.weight_loader_->load_linear_weight(
                        repo, tensor_name(resources_.model_.weight_plan.requests,
                                          TensorRole::AttentionLatentKey, i),
                        {latent.latent_rank, resources_.shape_.hidden});
                    attention_layer.latent_value = resources_.weight_loader_->load_linear_weight(
                        repo, tensor_name(resources_.model_.weight_plan.requests,
                                          TensorRole::AttentionLatentValue, i),
                        {latent.latent_rank, resources_.shape_.hidden});
                    if (latent.decoupled_rope && latent.rope_head_dim != 0) {
                        attention_layer.latent_key_rope = resources_.weight_loader_->load_linear_weight(
                            repo, tensor_name(resources_.model_.weight_plan.requests,
                                              TensorRole::AttentionLatentKeyRope, i),
                            {latent.rope_head_dim, resources_.shape_.hidden});
                    }
                }
                attention_layer.out = resources_.weight_loader_->load_linear_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests,
                                      TensorRole::AttentionLatentOutput, i),
                    {resources_.shape_.hidden, layout.latent_query_content_width()});
                if (resources_.options_.allocate_local_kv_cache && owns_latent_state) {
                    attention_layer.latent_key_cache.reset(
                        static_cast<size_t>(max_context_) * latent.latent_rank);
                    attention_layer.latent_value_cache.reset(
                        static_cast<size_t>(max_context_) * latent.latent_rank);
                    if (latent.decoupled_rope && latent.rope_head_dim != 0) {
                        attention_layer.latent_key_rope_cache.reset(
                            static_cast<size_t>(max_context_) * latent.rope_head_dim);
                    }
                }
                if (layout.kv_sharing.publishes) {
                    shared_owner[static_cast<size_t>(layout.kv_sharing.group)] = i;
                    attention_layer.kv_owner_layer = i;
                }
                if (!layout.kv_sharing.shared()) attention_layer.kv_owner_layer = i;
                resources_.layers_.emplace_back(std::move(attention_layer));
                continue;
            }
            attention_layer.query = resources_.weight_loader_->load_linear_weight(
                repo, query_name,
                {layout.query_projection_width(), resources_.shape_.hidden});
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
                attention_layer.q_norm = resources_.weight_loader_->load_rms_norm_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::AttentionQueryNorm, i),
                    {layout.head_dim}, resources_.shape_.numerical_policy.rms_norm_add_one);
                if (attention_layer.key) {
                    attention_layer.k_norm = resources_.weight_loader_->load_rms_norm_weight(
                    repo, tensor_name(resources_.model_.weight_plan.requests, TensorRole::AttentionKeyNorm, i),
                        {layout.head_dim}, resources_.shape_.numerical_policy.rms_norm_add_one);
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
        } else if (layer_type == MixerKind::GatedDeltaNet) {
            GatedDeltaNetLayer gated_delta_layer;
            gated_delta_layer.common = common_layer;
            gated_delta_layer.spec = resources_.shape_.gated_delta_net_layouts.at(
                static_cast<size_t>(i));
            const GatedDeltaNetSpec& spec = gated_delta_layer.spec;
            const int key_width = spec.key_heads * spec.key_head_dim;
            const int value_width = spec.value_heads * spec.value_head_dim;
            const int qkv_width = 2 * key_width + value_width;
            gated_delta_layer.qkv = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests,
                                  TensorRole::GatedDeltaNetQkv, i),
                {qkv_width, resources_.shape_.hidden});
            gated_delta_layer.z = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests,
                                  TensorRole::GatedDeltaNetZ, i),
                {value_width, resources_.shape_.hidden});
            gated_delta_layer.b = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests,
                                  TensorRole::GatedDeltaNetBeta, i),
                {spec.value_heads, resources_.shape_.hidden});
            gated_delta_layer.a = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests,
                                  TensorRole::GatedDeltaNetAlpha, i),
                {spec.value_heads, resources_.shape_.hidden});
            gated_delta_layer.conv_weight = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests,
                                  TensorRole::GatedDeltaNetConv, i),
                {qkv_width, 1, spec.conv_kernel});
            gated_delta_layer.dt_bias = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests,
                                  TensorRole::GatedDeltaNetDtBias, i),
                {spec.value_heads});
            gated_delta_layer.a_log = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests,
                                  TensorRole::GatedDeltaNetALog, i),
                {spec.value_heads});
            gated_delta_layer.norm = resources_.weight_loader_->load_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests,
                                  TensorRole::GatedDeltaNetNorm, i),
                {spec.value_head_dim});
            gated_delta_layer.out = resources_.weight_loader_->load_linear_weight(
                repo, tensor_name(resources_.model_.weight_plan.requests,
                                  TensorRole::GatedDeltaNetOutput, i),
                {resources_.shape_.hidden, value_width});
            const int conv_dim = qkv_width;
            gated_delta_layer.conv_state.reset(static_cast<size_t>(conv_dim) * spec.conv_kernel);
            gated_delta_layer.recurrent_state.reset(static_cast<size_t>(spec.value_heads) *
                spec.key_head_dim * spec.value_head_dim);
            resources_.layers_.emplace_back(std::move(gated_delta_layer));
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
    load_mtp_resources(*this, repo);
        });
}

} // namespace celeg
