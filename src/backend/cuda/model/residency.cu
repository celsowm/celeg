#include "detail/compiled_model.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/moe.hpp"
#include "celeg/backend/cuda/moe/expert_source.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace celeg {
namespace {


}

void CudaExpertResidencyCoordinator::ensure(ExpertResidencyRequest request) {
    request.validate();
    SharedModelWeights& weights = *weights_;
    ExpertResidencyWorkspace& workspace = *request.workspace;
    auto& expert_offload_plan = weights.expert_offload_plan;
    auto& expert_controllers = weights.expert_controllers;
    auto& host_expert_cache = weights.host_expert_cache;
    auto& expert_io_manager = weights.expert_io_manager;
    auto& expert_catalog = weights.expert_catalog;
    auto& usage_stats = weights.usage_stats;
    auto& usage_profile_path = weights.usage_profile_path;
    const int layer = request.layer;
    const int* sel_dev = request.selected_device;
    const int rows = request.rows;
    const int K = request.experts_per_token;
    const int num_experts = request.expert_count;
    cudaStream_t compute_stream = request.compute_stream;
    const float* route_scores_dev = request.route_scores_device;
    CudaEvent& router_done_event = *workspace.router_done;
    CudaEvent& ffn_done_event = *workspace.ffn_done;
    CudaEvent& promote_done_event = *workspace.promote_done;
    CudaEvent& prefetch_done_event = *workspace.prefetch_done;
    std::vector<int>& cold_expert_host = *workspace.cold_experts_host;
    std::vector<float>& cold_scores_host = *workspace.cold_scores_host;
    std::vector<int>& active_expert_host = *workspace.active_experts_host;
    std::vector<ExpertHostLease>& loaded_leases = *workspace.loaded_leases;
    std::vector<std::future<void>>& futures = *workspace.io_futures;
    std::vector<int>& prefetch_idx = *workspace.prefetch_indices;
    std::vector<int>& prefetch_ranked = *workspace.prefetch_ranked;
    std::vector<float>& prefetch_scores = *workspace.prefetch_scores;
    if (!expert_offload_plan.enabled) return;
    if (layer < 0 || static_cast<size_t>(layer) >= expert_controllers.size()) return;
    ResidencyController* controller = expert_controllers[static_cast<size_t>(layer)].get();
    if (!controller || !controller->cache) return;

    std::lock_guard<std::mutex> lock(controller->mutex);
    ExpertLayerCache* cache = controller->cache.get();
    cudaStream_t transfer = controller->transfer_stream->get();

    CELEG_CUDA(cudaStreamWaitEvent(transfer, router_done_event.get(), 0));
    CELEG_CUDA(cudaStreamWaitEvent(transfer, ffn_done_event.get(), 0));
    CELEG_CUDA(cudaStreamWaitEvent(transfer, prefetch_done_event.get(), 0));

    const int cold_count = cache->resolve_on_device(
        sel_dev, route_scores_dev, rows, K, transfer,
        cold_expert_host, cold_scores_host, active_expert_host);

    for (const int expert : active_expert_host) {
        cache->pin_active(expert);
    }

    for (auto it = controller->inflight_transfers.begin(); it != controller->inflight_transfers.end(); ) {
        cudaError_t status = cudaEventQuery(it->event->get());
        if (status == cudaSuccess) {
            it = controller->inflight_transfers.erase(it);
        } else if (status == cudaErrorNotReady) {
            ++it;
        } else {
            CELEG_CUDA(status);
        }
    }

    if (cold_count != 0) {
        const float default_score = ExpertLayerCache::kUnseen;
        loaded_leases.clear();
        loaded_leases.resize(static_cast<size_t>(cold_count));
        futures.clear();
        for (int i = 0; i < cold_count; ++i) {
            const int e = cold_expert_host[static_cast<size_t>(i)];
            cache->record_miss();
            if (!usage_profile_path.empty() && layer >= 0 &&
                static_cast<size_t>(layer) < usage_stats.layers.size()) {
                auto& entry = usage_stats.layers[static_cast<size_t>(layer)][static_cast<size_t>(e)];
                entry.selection_count++;
                entry.ssd_misses++;
            }
            if (host_expert_cache && expert_io_manager) {
                futures.push_back(expert_io_manager->submit(
                    [this, layer, e, &lease_dest = loaded_leases[static_cast<size_t>(i)]]() {
                        SharedModelWeights& inner_weights = *weights_;
                        ExpertHostLease lease = inner_weights.host_expert_cache->acquire(
                            layer, e, [this, layer, e](std::span<std::byte> payload) {
                                SharedModelWeights& callback_weights = *weights_;
                                if (!callback_weights.expert_source) {
                                    throw std::runtime_error("CUDA expert source is not initialized");
                                }
                                callback_weights.expert_source->read(layer, e, payload);
                            });
                        lease_dest = std::move(lease);
                    }));
            }
        }

        for (auto& future : futures) future.get();

        for (int i = 0; i < cold_count; ++i) {
            const int e = cold_expert_host[static_cast<size_t>(i)];
            float score = default_score;
            if (route_scores_dev != nullptr) score = cold_scores_host[static_cast<size_t>(e)];

            if (host_expert_cache && expert_io_manager) {
                ExpertHostLease& lease = loaded_leases[static_cast<size_t>(i)];
                if (lease.valid()) {
                    const ExpertLocation& loc = expert_catalog[static_cast<size_t>(layer)][static_cast<size_t>(e)];
                    const size_t gate_up_bytes = loc.w1.bytes + loc.w3.bytes;
                    cache->ensure_resident(
                        e, reinterpret_cast<const __nv_bfloat16*>(lease.payload()),
                        reinterpret_cast<const __nv_bfloat16*>(lease.payload() + gate_up_bytes),
                        transfer, score);
                    cache->pin_active(e, score);
                    auto event = std::make_unique<CudaEvent>();
                    event->record(transfer);
                    controller->inflight_transfers.push_back({std::move(lease), std::move(event)});
                }
            } else if (host_expert_cache) {
                ExpertHostLease lease = host_expert_cache->acquire(
                    layer, e, [this, layer, e](std::span<std::byte> payload) {
                        SharedModelWeights& callback_weights = *weights_;
                        if (!callback_weights.expert_source) {
                            throw std::runtime_error("CUDA expert source is not initialized");
                        }
                        callback_weights.expert_source->read(layer, e, payload);
                    });
                const ExpertLocation& loc = expert_catalog[static_cast<size_t>(layer)][static_cast<size_t>(e)];
                const size_t gate_up_bytes = loc.w1.bytes + loc.w3.bytes;
                cache->ensure_resident(
                    e, reinterpret_cast<const __nv_bfloat16*>(lease.payload()),
                    reinterpret_cast<const __nv_bfloat16*>(lease.payload() + gate_up_bytes),
                    transfer, score);
                cache->pin_active(e, score);
                auto event = std::make_unique<CudaEvent>();
                event->record(transfer);
                controller->inflight_transfers.push_back({std::move(lease), std::move(event)});
            } else {
                cache->ensure_resident(e, transfer, score);
                cache->pin_active(e, score);
            }
        }
        cache->sync_residency_tables(transfer);
        if (host_expert_cache) {
            CELEG_CUDA(cudaStreamSynchronize(transfer));
            controller->inflight_transfers.clear();
        }
    }

    for (const int expert : active_expert_host) cache->pin_active(expert);

    CELEG_CUDA(cudaEventRecord(promote_done_event.get(), transfer));
    CELEG_CUDA(cudaStreamWaitEvent(compute_stream, promote_done_event.get(), 0));

    if (expert_offload_plan.prefetch_experts > 0 && route_scores_dev != nullptr) {
        const int take = std::min<int>(K + expert_offload_plan.prefetch_experts, num_experts);
        prefetch_idx.resize(static_cast<size_t>(num_experts));
        for (int e = 0; e < num_experts; ++e) prefetch_idx[static_cast<size_t>(e)] = e;
        std::partial_sort(
            prefetch_idx.begin(), prefetch_idx.begin() + take, prefetch_idx.end(),
            [&cold_scores_host](int a, int b) {
                return cold_scores_host[static_cast<size_t>(a)] >
                       cold_scores_host[static_cast<size_t>(b)];
            });
        prefetch_ranked.resize(static_cast<size_t>(take));
        prefetch_scores.resize(static_cast<size_t>(take));
        for (int i = 0; i < take; ++i) {
            const int e = prefetch_idx[static_cast<size_t>(i)];
            prefetch_ranked[static_cast<size_t>(i)] = e;
            prefetch_scores[static_cast<size_t>(i)] = cold_scores_host[static_cast<size_t>(e)];
        }
        cache->prefetch_list(prefetch_ranked, prefetch_scores,
                             expert_offload_plan.prefetch_experts, transfer);
    } else if (expert_offload_plan.prefetch_experts > 0) {
        cache->prefetch(expert_offload_plan.prefetch_experts, transfer);
    }
    CELEG_CUDA(cudaEventRecord(prefetch_done_event.get(), transfer));
}

void CudaCompiledModel::run_mlp_moe_decode(const LayerCommon& common_layer,
                                           int layer) {
    const MoeFfnWeights& moe = *as_moe_ffn(common_layer.feed_forward);
    const CompiledLayerProgram& layer_semantics = resources_.program_.layers.at(
        static_cast<size_t>(layer));
    const MoeLayerProgram& semantics = std::get<MoeLayerProgram>(layer_semantics.feed_forward);
    if (layer_semantics.feed_forward_norm.before) {
        launch_rmsnorm(workspace_.hidden_.data(), common_layer.feed_forward_norm_before,
                       workspace_.normed_.data(), 1, resources_.program_.hidden,
                       layer_semantics.feed_forward_norm.before->epsilon, stream_.get());
    } else {
        CELEG_CUDA(cudaMemcpyAsync(workspace_.normed_.data(), workspace_.hidden_.data(),
                                  workspace_.hidden_.bytes(), cudaMemcpyDeviceToDevice,
                                  stream_.get()));
    }
    launch_cast_bf16_to_float(workspace_.normed_.data(), workspace_.moe_hidden_float_.data(),
                              resources_.program_.hidden, stream_.get());
    const MoeRouterConfig cfg = moe_router_config(semantics);
    MoeRouterDevice router;
    router.router_weight = moe.router_float;
    router.expert_bias = moe.expert_bias;
    router.hidden_data = workspace_.moe_hidden_float_.data();
    router.selected_experts = workspace_.moe_sel_.data();
    router.routing_weights = workspace_.moe_routing_w_.data();
    router.rows = 1;
    router.hidden_dim = resources_.program_.hidden;
    launch_moe_router(router, cfg, workspace_.moe_router_scratch_.data(), stream_.get());
    CELEG_CUDA(cudaEventRecord(workspace_.router_done_event_.get(), stream_.get()));

    resources_.weights_->residency_coordinator->ensure(ExpertResidencyRequest{
        layer, workspace_.moe_sel_.data(), 1, semantics.router.experts_per_token,
        semantics.router.expert_count, stream_.get(),
        workspace_.moe_router_scratch_.data(), &workspace_.residency_workspace_});

    workspace_.moe_output_accum_.zero_async(stream_.get());
    const MoeFfnDevice ffn = moe_ffn_device(moe, semantics);
    launch_moe_ffn(ffn, workspace_.moe_sel_.data(), workspace_.moe_routing_w_.data(),
                   workspace_.normed_.data(), workspace_.moe_output_accum_.data(), 1,
                   semantics.router.experts_per_token,
                   workspace_.moe_gu_scratch_.data(), workspace_.moe_act_scratch_.data(),
                   stream_.get());
    launch_finalize_moe_output(workspace_.moe_output_accum_.data(),
                               workspace_.moe_output_.data(),
                               resources_.program_.hidden, stream_.get());
    if (moe.shared_w13 != nullptr) {
        const int shared_intermediate = semantics.shared->mlp.intermediate_size;
        linear(workspace_.normed_.data(), *moe.shared_w13, workspace_.gate_up_.data(),
               1, 2 * shared_intermediate, resources_.program_.hidden);
        launch_swiglu_fused(workspace_.gate_up_.data(), workspace_.activated_.data(),
                            shared_intermediate, stream_.get());
        linear(workspace_.activated_.data(), *moe.shared_w2, workspace_.mlp_output_.data(),
               1, resources_.program_.hidden, shared_intermediate);
        if (moe.shared_gate != nullptr) {
            linear(workspace_.normed_.data(), *moe.shared_gate,
                   workspace_.moe_shared_gate_.data(), 1, 1,
                   resources_.program_.hidden);
            launch_sigmoid_scale_by_scalar(workspace_.mlp_output_.data(),
                                           workspace_.moe_shared_gate_.data(),
                                           resources_.program_.hidden, stream_.get());
        }
        launch_residual_add(workspace_.moe_output_.data(), workspace_.mlp_output_.data(),
                            resources_.program_.hidden, stream_.get());
    }
    CELEG_CUDA(cudaEventRecord(workspace_.ffn_done_event_.get(), stream_.get()));
    launch_scale(workspace_.moe_output_.data(), resources_.program_.hidden,
                 layer_semantics.residual.multiplier, stream_.get());
    if (layer_semantics.feed_forward_norm.after) {
        launch_rmsnorm(workspace_.moe_output_.data(), common_layer.feed_forward_norm_after,
                       workspace_.moe_output_.data(), 1, resources_.program_.hidden,
                       layer_semantics.feed_forward_norm.after->epsilon, stream_.get());
    }
    launch_residual_add(workspace_.hidden_.data(), workspace_.moe_output_.data(),
                        resources_.program_.hidden, stream_.get());
}

void CudaCompiledModel::run_mlp_moe_prefill(const LayerCommon& common_layer, int rows,
                                            int layer) {
    const MoeFfnWeights& moe = *as_moe_ffn(common_layer.feed_forward);
    const CompiledLayerProgram& layer_semantics = resources_.program_.layers.at(
        static_cast<size_t>(layer));
    const MoeLayerProgram& semantics = std::get<MoeLayerProgram>(layer_semantics.feed_forward);
    workspace_.moe_pf_hidden_float_.reserve(static_cast<size_t>(rows) * resources_.program_.hidden);
    workspace_.moe_pf_sel_.reserve(static_cast<size_t>(rows) * semantics.router.experts_per_token);
    workspace_.moe_pf_routing_w_.reserve(static_cast<size_t>(rows) * semantics.router.experts_per_token);
    workspace_.moe_pf_router_scratch_.reserve(static_cast<size_t>(rows) * semantics.router.expert_count);
    workspace_.moe_pf_output_accum_.reserve(static_cast<size_t>(rows) * resources_.program_.hidden);
    workspace_.moe_pf_output_.reserve(static_cast<size_t>(rows) * resources_.program_.hidden);
    workspace_.moe_pf_gu_scratch_.reserve(
        static_cast<size_t>(rows) * semantics.router.experts_per_token * 2 *
        semantics.routed.mlp.intermediate_size);
    workspace_.moe_pf_act_scratch_.reserve(
        static_cast<size_t>(rows) * semantics.router.experts_per_token *
        semantics.routed.mlp.intermediate_size);

    if (layer_semantics.feed_forward_norm.before) {
        launch_rmsnorm(workspace_.prefill_hidden_.data(), common_layer.feed_forward_norm_before,
                       workspace_.prefill_normed_.data(), rows, resources_.program_.hidden,
                       layer_semantics.feed_forward_norm.before->epsilon, stream_.get());
    } else {
        CELEG_CUDA(cudaMemcpyAsync(
            workspace_.prefill_normed_.data(), workspace_.prefill_hidden_.data(),
            static_cast<size_t>(rows) * resources_.program_.hidden * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice, stream_.get()));
    }
    launch_cast_bf16_to_float(workspace_.prefill_normed_.data(),
                              workspace_.moe_pf_hidden_float_.data(),
                              rows * resources_.program_.hidden, stream_.get());
    const MoeRouterConfig cfg = moe_router_config(semantics);
    MoeRouterDevice router;
    router.router_weight = moe.router_float;
    router.expert_bias = moe.expert_bias;
    router.hidden_data = workspace_.moe_pf_hidden_float_.data();
    router.selected_experts = workspace_.moe_pf_sel_.data();
    router.routing_weights = workspace_.moe_pf_routing_w_.data();
    router.rows = rows;
    router.hidden_dim = resources_.program_.hidden;
    launch_moe_router(router, cfg, workspace_.moe_pf_router_scratch_.data(), stream_.get());
    CELEG_CUDA(cudaEventRecord(workspace_.router_done_event_.get(), stream_.get()));

    resources_.weights_->residency_coordinator->ensure(ExpertResidencyRequest{
        layer, workspace_.moe_pf_sel_.data(), rows, semantics.router.experts_per_token,
        semantics.router.expert_count, stream_.get(), nullptr,
        &workspace_.residency_workspace_});

    workspace_.moe_pf_output_accum_.zero_async(stream_.get());
    const MoeFfnDevice ffn = moe_ffn_device(moe, semantics);
    launch_moe_ffn(ffn, workspace_.moe_pf_sel_.data(), workspace_.moe_pf_routing_w_.data(),
                   workspace_.prefill_normed_.data(), workspace_.moe_pf_output_accum_.data(), rows,
                   semantics.router.experts_per_token, workspace_.moe_pf_gu_scratch_.data(),
                   workspace_.moe_pf_act_scratch_.data(), stream_.get());
    launch_finalize_moe_output(workspace_.moe_pf_output_accum_.data(),
                               workspace_.moe_pf_output_.data(),
                               rows * resources_.program_.hidden, stream_.get());
    if (moe.shared_w13 != nullptr) {
        const int shared_intermediate = semantics.shared->mlp.intermediate_size;
        linear(workspace_.prefill_normed_.data(), *moe.shared_w13,
               workspace_.prefill_gate_up_.data(), rows, 2 * shared_intermediate,
               resources_.program_.hidden);
        launch_swiglu_interleaved(workspace_.prefill_gate_up_.data(),
                                  workspace_.prefill_activated_.data(), rows,
                                  shared_intermediate, stream_.get());
        linear(workspace_.prefill_activated_.data(), *moe.shared_w2,
               workspace_.prefill_mlp_output_.data(), rows,
               resources_.program_.hidden, shared_intermediate);
        if (moe.shared_gate != nullptr) {
            linear(workspace_.prefill_normed_.data(), *moe.shared_gate,
                   workspace_.moe_pf_shared_gate_.data(), rows, 1,
                   resources_.program_.hidden);
            launch_sigmoid_multiply_strided(workspace_.prefill_mlp_output_.data(),
                                            workspace_.moe_pf_shared_gate_.data(),
                                            rows, resources_.program_.hidden, stream_.get());
        }
        launch_residual_add(workspace_.moe_pf_output_.data(),
                            workspace_.prefill_mlp_output_.data(),
                            rows * resources_.program_.hidden, stream_.get());
    }
    CELEG_CUDA(cudaEventRecord(workspace_.ffn_done_event_.get(), stream_.get()));
    launch_scale(workspace_.moe_pf_output_.data(), rows * resources_.program_.hidden,
                 layer_semantics.residual.multiplier, stream_.get());
    if (layer_semantics.feed_forward_norm.after) {
        launch_rmsnorm(workspace_.moe_pf_output_.data(), common_layer.feed_forward_norm_after,
                       workspace_.moe_pf_output_.data(), rows, resources_.program_.hidden,
                       layer_semantics.feed_forward_norm.after->epsilon, stream_.get());
    }
    launch_residual_add(workspace_.prefill_hidden_.data(), workspace_.moe_pf_output_.data(),
                        rows * resources_.program_.hidden, stream_.get());
}

}
