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

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace celeg {
namespace {

int last_full_attention_layer(const RuntimeTopology& shape) {
    for (int layer = shape.num_hidden_layers - 1; layer >= 0; --layer) {
        if (shape.mixer_kinds.at(static_cast<size_t>(layer)) == MixerKind::Attention) {
            return layer;
        }
    }
    return -1;
}

} // namespace

void load_mtp_weights(CudaCompiledModel& model, const IWeightRepository& repo) {
    CudaModelResources& resources = model.resources_;
    if (!resources.options_.enable_mtp) return;
    if (resources.shape_.checkpoint.mtp_num_hidden_layers <= 0) {
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
    mtp.layer_count = resources.shape_.checkpoint.mtp_num_hidden_layers;
    mtp.fc = resources.weight_loader_->load_linear_weight(
        repo, "mtp.fc.weight",
        {resources.shape_.hidden, 2 * resources.shape_.hidden});
    mtp.pre_fc_norm_embedding = resources.weight_loader_->load_rms_norm_weight(
        repo, "mtp.pre_fc_norm_embedding.weight", {resources.shape_.hidden},
        NormWeightKind::Scale);
    mtp.pre_fc_norm_hidden = resources.weight_loader_->load_rms_norm_weight(
        repo, "mtp.pre_fc_norm_hidden.weight", {resources.shape_.hidden},
        NormWeightKind::Scale);
    mtp.norm = resources.weight_loader_->load_rms_norm_weight(
        repo, "mtp.norm.weight", {resources.shape_.hidden},
        NormWeightKind::Scale);
    mtp.logits = resources.lm_head_ ? resources.lm_head_ : resources.embedding_;
    mtp.layers.reserve(static_cast<size_t>(mtp.layer_count));

    for (int index = 0; index < mtp.layer_count; ++index) {
        const std::string prefix = "mtp.layers." + std::to_string(index);
        LayerCommon common_layer;
        common_layer.operator_norm = resources.weight_loader_->load_rms_norm_weight(
            repo, prefix + ".input_layernorm.weight", {resources.shape_.hidden},
            NormWeightKind::Scale);
        common_layer.ffn_norm = resources.weight_loader_->load_rms_norm_weight(
            repo, prefix + ".post_attention_layernorm.weight", {resources.shape_.hidden},
            NormWeightKind::Scale);

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
            NormWeightKind::Scale);
        attention.k_norm = resources.weight_loader_->load_rms_norm_weight(
            repo, prefix + ".self_attn.k_norm.weight", {mtp_layout.head_dim},
            NormWeightKind::Scale);
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

} // namespace celeg
