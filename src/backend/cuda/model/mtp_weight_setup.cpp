#include "detail/compiled_model.hpp"
#include "checkpoint/detail/bootstrap.hpp"
#include "backend/cuda/paged_kv.hpp"
#include "backend/cuda/weight_policy.hpp"
#include "celeg/model/weights/quantization.hpp"
#include "backend/cuda/weight_layout.hpp"
#include "backend/cuda/weights_loader.hpp"
#include "backend/cuda/weight_setup_support.hpp"
#include "backend/cuda/weight_setup.hpp"
#include "backend/cuda/moe.hpp"
#include "backend/cuda/moe/expert_source.hpp"
#include "attention_state_setup.hpp"
#include "expert_residency_setup.hpp"
#include "layer_weight_setup.hpp"
#include "moe_weight_setup.hpp"

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
            const int resource_layer = resources.shape().num_hidden_layers + index;
            MoeFfnWeights moe = bind_cuda_moe_router_weight(
                model, repo, prefix + ".mlp.gate.weight",
                resource_layer, E);

            if (moe_semantics.shared) {
                bind_cuda_shared_expert(
                    model, repo,
                    CudaSharedExpertNames{
                        prefix + ".mlp.shared_expert.w13.weight",
                        prefix + ".mlp.shared_expert.gate_proj.weight",
                        prefix + ".mlp.shared_expert.up_proj.weight",
                        prefix + ".mlp.shared_expert.down_proj.weight",
                        prefix + ".mlp.shared_expert_gate.weight"},
                    moe_semantics.shared->mlp.intermediate_size,
                    moe);
            }

            const std::string experts_prefix = prefix + ".mlp.experts";
            const MoeExpertTensorNames expert_names =
                mtp_expert_names(experts_prefix, E);
            if (model.workspace_.expert_offload_plan_.enabled) {
                if (resources.options().expert_offload.backing == ExpertBackingMode::DiskCached) {
                    PreparedDiskExpertResidency prepared =
                        prepare_cuda_disk_expert_residency(
                            model, repo, expert_names, resource_layer, E, inter);
                    ResidencyController& controller = *prepared.controller;
                    std::vector<ExpertHostLease> seed_leases(static_cast<size_t>(
                        model.workspace_.expert_offload_plan_.experts_per_layer));
                    for (int seed = 0;
                         seed < model.workspace_.expert_offload_plan_.experts_per_layer;
                         ++seed) {
                        const ExpertLocation& location =
                            prepared.catalog[static_cast<size_t>(seed)];
                        seed_leases[static_cast<size_t>(seed)] =
                            resources.weights_->host_expert_cache->acquire(
                                resource_layer, seed, [&](std::span<std::byte> payload) {
                                    if (!resources.weights_->expert_source) {
                                        throw std::runtime_error(
                                            "CUDA expert source is not initialized");
                                    }
                                    resources.weights_->expert_source->read(
                                        resource_layer, seed, payload);
                                });
                        ExpertHostLease& lease = seed_leases[static_cast<size_t>(seed)];
                        promote_cuda_disk_expert_payload(
                            controller, seed, location, lease.payload());
                    }
                    CELEG_CUDA(cudaStreamSynchronize(controller.transfer_stream->get()));
                    moe.storage = install_cuda_expert_controller(
                        model, resource_layer, std::move(prepared.controller));
                } else {
                    moe.storage = bind_cuda_host_expert_residency(
                        model, repo, expert_names, resource_layer, E, inter);
                }
            } else {
                moe.storage = bind_cuda_resident_experts(
                    model, repo, expert_names, E, inter);
            }
            common_layer.feed_forward = moe;
        } else {
            if (!dense_program || dense_program->intermediate_size <= 0) {
                throw std::runtime_error("MTP requires a compiled dense FFN width");
            }
            common_layer.feed_forward = bind_cuda_dense_ffn(
                model, repo,
                CudaDenseFfnNames{
                    prefix + ".mlp.w13.weight",
                    prefix + ".mlp.gate_proj.weight",
                    prefix + ".mlp.up_proj.weight",
                    prefix + ".mlp.down_proj.weight"},
                dense_program->intermediate_size);
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
        initialize_cuda_ordinary_attention_state(
            attention, mtp_layout, resources.options().kv_cache_mode,
            model.max_context_, true);
        mtp.layers.emplace_back(std::move(attention));
    }
}

}
