#include "detail/compiled_model.hpp"
#include "checkpoint/detail/bootstrap.hpp"
#include "kernels/kernels.cuh"
#include "backend/cuda/paged_kv.hpp"
#include "backend/cuda/weight_policy.hpp"
#include "celeg/model/weights/quantization.hpp"
#include "backend/cuda/weight_layout.hpp"
#include "backend/cuda/weights_loader.hpp"
#include "backend/cuda/weight_setup_support.hpp"
#include "backend/cuda/weight_setup.hpp"
#include "backend/cuda/moe.hpp"
#include "backend/cuda/moe/expert_source.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace celeg {
namespace {

/// MTP layers are not part of the canonical weight plan yet (extensibility
/// plan, Sprint C residual gap), so their routed-expert spellings are
/// constructed here, at the one place that knows the MTP checkpoint
/// convention, and then flow through the same resolved-name loader API the
/// planned layers use.
MoeExpertTensorNames mtp_expert_names(const std::string& experts_prefix,
                                      int num_experts) {
    MoeExpertTensorNames names;
    names.gate.reserve(static_cast<size_t>(num_experts));
    names.up.reserve(static_cast<size_t>(num_experts));
    names.down.reserve(static_cast<size_t>(num_experts));
    for (int expert = 0; expert < num_experts; ++expert) {
        const std::string prefix =
            experts_prefix + "." + std::to_string(expert);
        names.gate.push_back(prefix + ".gate_proj.weight");
        names.up.push_back(prefix + ".up_proj.weight");
        names.down.push_back(prefix + ".down_proj.weight");
    }
    return names;
}

}

void load_mtp_weights(CudaCompiledModel& model, const IWeightRepository& repo) {
    CudaModelResources& resources = model.resources_;
    if (!resources.options().enable_mtp) return;
    if (resources.dims().mtp_num_hidden_layers <= 0) {
        throw std::invalid_argument("MTP was requested but the checkpoint has no MTP layers");
    }
    if (!repo.contains("mtp.fc.weight")) {
        throw std::runtime_error("MTP was requested but mtp.fc.weight is missing");
    }
    if (!resources.options().allocate_local_kv_cache) {
        throw std::invalid_argument("MTP requires a local KV cache; disable packed/paged execution");
    }
    const CompiledAttentionProgram* full_attention = resources.program_.last_attention();
    if (!full_attention) {
        throw std::runtime_error("MTP requires a full-attention target layer");
    }
    AttentionSpec mtp_layout = full_attention->semantics;
    mtp_layout.kv_sharing = {};

    CudaMtpResources& mtp = resources.mtp_;
    mtp.enabled = true;
    mtp.layer_count = resources.dims().mtp_num_hidden_layers;
    mtp.fc = resources.weight_loader_->load_linear_weight(
        repo, "mtp.fc.weight",
        {resources.program_.hidden, 2 * resources.program_.hidden});
    mtp.pre_fc_norm_embedding = resources.weight_loader_->load_rms_norm_weight(
        repo, "mtp.pre_fc_norm_embedding.weight", {resources.program_.hidden},
        NormWeightKind::Scale);
    mtp.pre_fc_norm_hidden = resources.weight_loader_->load_rms_norm_weight(
        repo, "mtp.pre_fc_norm_hidden.weight", {resources.program_.hidden},
        NormWeightKind::Scale);
    mtp.norm = resources.weight_loader_->load_rms_norm_weight(
        repo, "mtp.norm.weight", {resources.program_.hidden},
        NormWeightKind::Scale);
    mtp.logits = resources.lm_head_ ? resources.lm_head_ : resources.embedding_;
    mtp.layers.reserve(static_cast<size_t>(mtp.layer_count));
    const MoeLayerProgram* moe_program = resources.program_.first_moe();
    const CompiledDenseFeedForwardProgram* dense_program =
        resources.program_.first_dense_feed_forward();

    for (int index = 0; index < mtp.layer_count; ++index) {
        const std::string prefix = "mtp.layers." + std::to_string(index);
        LayerCommon common_layer;
        common_layer.mixer_norm_before = resources.weight_loader_->load_rms_norm_weight(
            repo, prefix + ".input_layernorm.weight", {resources.program_.hidden},
            NormWeightKind::Scale);
        common_layer.feed_forward_norm_before = resources.weight_loader_->load_rms_norm_weight(
            repo, prefix + ".post_attention_layernorm.weight", {resources.program_.hidden},
            NormWeightKind::Scale);

        if (moe_program) {
            const MoeLayerProgram& moe_semantics = *moe_program;
            const int E = moe_semantics.router.expert_count;
            const int inter = moe_semantics.routed.mlp.intermediate_size;
            MoeFfnWeights moe{};
            moe.router = resources.weight_loader_->load_router_weight_named(
                repo, prefix + ".mlp.gate.weight", E, resources.program_.hidden);
            const auto* router_bf16 = std::get_if<Bf16LinearStorage>(&moe.router->storage);
            if (!router_bf16 || !router_bf16->data) {
                throw std::logic_error("CUDA MTP router requires BF16 storage");
            }
            const int resource_layer = resources.shape().num_hidden_layers + index;
            DeviceBuffer<float>& router_float =
                model.workspace_.moe_router_float_[static_cast<size_t>(resource_layer)];
            router_float.reset(static_cast<size_t>(E) * resources.program_.hidden);
            launch_cast_bf16_to_float(
                router_bf16->data, router_float.data(),
                E * resources.program_.hidden, model.stream_.get());
            moe.router_float = router_float.data();

            if (moe_semantics.shared) {
                const int shared = moe_semantics.shared->mlp.intermediate_size;
                moe.shared_w13 = resources.weight_loader_->load_concat_linear_weight(
                    repo, prefix + ".mlp.shared_expert.w13.weight",
                    {{prefix + ".mlp.shared_expert.gate_proj.weight",
                      {shared, resources.program_.hidden}},
                     {prefix + ".mlp.shared_expert.up_proj.weight",
                      {shared, resources.program_.hidden}}});
                moe.shared_w2 = resources.weight_loader_->load_linear_weight(
                    repo, prefix + ".mlp.shared_expert.down_proj.weight",
                    {resources.program_.hidden, shared});
                moe.shared_gate = resources.weight_loader_->load_linear_weight(
                    repo, prefix + ".mlp.shared_expert_gate.weight",
                    {1, resources.program_.hidden});
            }

            const std::string experts_prefix = prefix + ".mlp.experts";
            const MoeExpertTensorNames expert_names =
                mtp_expert_names(experts_prefix, E);
            if (model.workspace_.expert_offload_plan_.enabled) {
                std::vector<ExpertLocation> catalog =
                    resources.weight_loader_->build_expert_catalog(
                        repo, expert_names, E, inter,
                        resources.program_.hidden);
                model.workspace_.expert_catalog_[static_cast<size_t>(resource_layer)] = catalog;
                if (resources.weights_->expert_catalog[static_cast<size_t>(resource_layer)].empty()) {
                    resources.weights_->expert_catalog[static_cast<size_t>(resource_layer)] = catalog;
                }
                const size_t gate_up_bytes = 2ull * static_cast<size_t>(inter) *
                    resources.program_.hidden * sizeof(__nv_bfloat16);
                const size_t down_bytes = static_cast<size_t>(resources.program_.hidden) *
                    inter * sizeof(__nv_bfloat16);
                auto controller = std::make_unique<ResidencyController>(
                    E, model.workspace_.expert_offload_plan_.experts_per_layer,
                    gate_up_bytes, down_bytes,
                    resources.options().expert_offload.policy);
                if (resources.options().expert_offload.backing == ExpertBackingMode::DiskCached) {
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
                        resources.weight_loader_->load_moe_experts_host(
                            repo, expert_names, E, inter,
                            resources.program_.hidden,
                            model.workspace_.host_expert_store_,
                            resources.options().expert_offload.host_mode);
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
                moe.storage = OffloadedExpertWeights{
                    controller->cache->gate_up_ptrs(), controller->cache->down_ptrs()};
                resources.weights_->expert_controllers[static_cast<size_t>(resource_layer)] =
                    std::move(controller);
                model.workspace_.expert_caches_[static_cast<size_t>(resource_layer)] =
                    resources.weights_->expert_controllers[static_cast<size_t>(resource_layer)]->cache.get();
            } else {
                const ExpertLinearWeight* gate_up =
                    resources.weight_loader_->load_moe_gate_up(
                        repo, expert_names, E, inter,
                        resources.program_.hidden);
                const ExpertLinearWeight* down =
                    resources.weight_loader_->load_moe_down(
                        repo, expert_names, E, inter, resources.program_.hidden);
                moe.storage = ResidentExpertWeights{gate_up, down};
            }
            common_layer.feed_forward = moe;
        } else {
            if (!dense_program) {
                throw std::runtime_error("MTP requires a compiled dense FFN width");
            }
            const int inter = dense_program->intermediate_size;
            if (inter <= 0) {
                throw std::runtime_error("MTP requires a compiled dense FFN width");
            }
            const LinearWeight* w13 = resources.weight_loader_->load_concat_linear_weight(
                repo, prefix + ".mlp.w13.weight",
                {{prefix + ".mlp.gate_proj.weight", {inter, resources.program_.hidden}},
                 {prefix + ".mlp.up_proj.weight", {inter, resources.program_.hidden}}});
            const LinearWeight* w2 = resources.weight_loader_->load_linear_weight(
                repo, prefix + ".mlp.down_proj.weight",
                {resources.program_.hidden, inter});
            common_layer.feed_forward = DenseFfnWeights{w13, w2};
        }

        AttentionLayer attention;
        attention.common = common_layer;
        attention.layout = mtp_layout;
        attention.query = resources.weight_loader_->load_linear_weight(
            repo, prefix + ".self_attn.q_proj.weight",
            {mtp_layout.query_projection_width(), resources.program_.hidden});
        attention.key = resources.weight_loader_->load_linear_weight(
            repo, prefix + ".self_attn.k_proj.weight",
            {mtp_layout.key_value_width(), resources.program_.hidden});
        attention.value = resources.weight_loader_->load_linear_weight(
            repo, prefix + ".self_attn.v_proj.weight",
            {mtp_layout.key_value_width(), resources.program_.hidden});
        attention.out = resources.weight_loader_->load_linear_weight(
            repo, prefix + ".self_attn.o_proj.weight",
            {resources.program_.hidden, mtp_layout.query_width()});
        attention.q_norm = resources.weight_loader_->load_rms_norm_weight(
            repo, prefix + ".self_attn.q_norm.weight", {mtp_layout.head_dim},
            NormWeightKind::Scale);
        attention.k_norm = resources.weight_loader_->load_rms_norm_weight(
            repo, prefix + ".self_attn.k_norm.weight", {mtp_layout.head_dim},
            NormWeightKind::Scale);
        const size_t cache_elements = static_cast<size_t>(model.max_context_) *
            static_cast<size_t>(mtp_layout.key_value_width());
        if (resources.options().kv_cache_mode == KvCacheMode::Int8) {
            attention.state = OrdinaryInt8KvState{};
            auto& ordinary_state = std::get<OrdinaryInt8KvState>(attention.state);
            ordinary_state.key_cache.reset(cache_elements);
            ordinary_state.value_cache.reset(cache_elements);
            const size_t scales = static_cast<size_t>(model.max_context_) *
                static_cast<size_t>(mtp_layout.key_value_heads);
            ordinary_state.key_scales.reset(scales);
            ordinary_state.value_scales.reset(scales);
        } else {
            attention.state = OrdinaryBf16KvState{};
            auto& ordinary_state = std::get<OrdinaryBf16KvState>(attention.state);
            ordinary_state.key_cache.reset(cache_elements);
            ordinary_state.value_cache.reset(cache_elements);
        }
        mtp.layers.emplace_back(std::move(attention));
    }
}

}
