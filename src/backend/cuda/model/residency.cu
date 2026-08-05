#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/moe.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace celeg {
namespace {

__global__ void mask_expert_selection_kernel(const int* src_sel,
                                             int* dest_sel,
                                             const std::uint8_t* expert_active,
                                             int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int e = src_sel[idx];
    if (e >= 0 && !expert_active[e]) {
        dest_sel[idx] = -1;
    } else {
        dest_sel[idx] = e;
    }
}

// MoE router config / FFN device descriptors are provided inline by
// celeg/detail/model/types.hpp (moe_router_config / moe_ffn_device) so the
// standalone paths and the packed executor share one definition.

} // namespace

void SharedModelWeights::ensure_moe_experts_resident(ExpertResidencyRequest request) {
    request.validate();
    const int layer = request.layer;
    const int* sel_dev = request.selected_device;
    const int rows = request.rows;
    const int K = request.experts_per_token;
    const int num_experts = request.expert_count;
    cudaStream_t compute_stream = request.compute_stream;
    const float* route_scores_dev = request.route_scores_device;
    CudaEvent& router_done_event = *request.router_done;
    CudaEvent& ffn_done_event = *request.ffn_done;
    CudaEvent& promote_done_event = *request.promote_done;
    CudaEvent& prefetch_done_event = *request.prefetch_done;
    std::vector<int>& cold_expert_host = *request.cold_experts_host;
    std::vector<float>& cold_scores_host = *request.cold_scores_host;
    std::vector<int>& prefetch_idx = *request.prefetch_indices;
    std::vector<int>& prefetch_ranked = *request.prefetch_ranked;
    std::vector<float>& prefetch_scores = *request.prefetch_scores;
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

    // Device-side residency check: reads sel_dev + expert_slot_dev_ on GPU,
    // outputs a compact list of cold experts. This avoids D2H-copying the
    // full selection and router scores; only the (typically tiny) cold list
    // is transferred back.
    const int cold_count = cache->resolve_on_device(
        sel_dev, route_scores_dev, rows, K, transfer,
        cold_expert_host, cold_scores_host);

    // Release any leases for completed transfers
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

    // Touch resident experts and promote cold ones.
    if (cold_count == 0) {
        // The device resolver proved that every selected pointer is resident.
        // Do not copy the full selection back to the host on this hot path.
    } else {
        // Promote cold experts. The cold list has unique expert indices.
        const float default_score = ExpertLayerCache::kUnseen;
        std::vector<ExpertHostLease> loaded_leases(static_cast<size_t>(cold_count));
        std::vector<std::future<void>> futures;
        for (int i = 0; i < cold_count; ++i) {
            const int e = cold_expert_host[static_cast<size_t>(i)];
            cache->record_miss();

            // Record usage stats (miss)
            if (!usage_profile_path.empty() && layer >= 0 && static_cast<size_t>(layer) < usage_stats.layers.size()) {
                auto& entry = usage_stats.layers[static_cast<size_t>(layer)][static_cast<size_t>(e)];
                entry.selection_count++;
                entry.ssd_misses++;
            }

            if (pinned_expert_cache && expert_io_manager) {
                // Async parallel load using ExpertIoManager!
                futures.push_back(expert_io_manager->submit([this, layer, e, &lease_dest = loaded_leases[static_cast<size_t>(i)]]() {
                    const ExpertLocation& loc = expert_catalog[static_cast<size_t>(layer)][static_cast<size_t>(e)];
                    ExpertHostLease lease = pinned_expert_cache->acquire(layer, e, [&](std::span<std::byte> gu_dest, std::span<std::byte> dn_dest) {
                        if (expert_sidecar) {
                            expert_sidecar->read_expert(layer, e, gu_dest, dn_dest);
                        } else {
                            const auto& reader =
                                require_random_access_tensor_reader(*repo);
                            reader.read(loc.w1, gu_dest.subspan(0, loc.w1.bytes));
                            reader.read(loc.w3, gu_dest.subspan(loc.w1.bytes, loc.w3.bytes));
                            reader.read(loc.w2, dn_dest);
                        }
                    });
                    lease_dest = std::move(lease);
                }));
            }
        }

        // Wait for all async loads to complete!
        for (auto& f : futures) {
            f.get();
        }

        // Promote loaded experts sequentially
        for (int i = 0; i < cold_count; ++i) {
            const int e = cold_expert_host[static_cast<size_t>(i)];
            float score = default_score;
            if (route_scores_dev != nullptr) {
                score = cold_scores_host[static_cast<size_t>(e)];
            }

            if (pinned_expert_cache && expert_io_manager) {
                ExpertHostLease& lease = loaded_leases[static_cast<size_t>(i)];
                if (lease.valid()) {
                    cache->ensure_resident(e, reinterpret_cast<const __nv_bfloat16*>(lease.gate_up()),
                                           reinterpret_cast<const __nv_bfloat16*>(lease.down()),
                                           transfer, score);

                    auto ev = std::make_unique<CudaEvent>();
                    ev->record(transfer);
                    controller->inflight_transfers.push_back({std::move(lease), std::move(ev)});
                }
            } else if (pinned_expert_cache) {
                // Sync fallback (if no io_manager is present)
                const ExpertLocation& loc = expert_catalog[static_cast<size_t>(layer)][static_cast<size_t>(e)];
                ExpertHostLease lease = pinned_expert_cache->acquire(layer, e, [&](std::span<std::byte> gu_dest, std::span<std::byte> dn_dest) {
                    if (expert_sidecar) {
                        expert_sidecar->read_expert(layer, e, gu_dest, dn_dest);
                    } else {
                        const auto& reader =
                            require_random_access_tensor_reader(*repo);
                        reader.read(loc.w1, gu_dest.subspan(0, loc.w1.bytes));
                        reader.read(loc.w3, gu_dest.subspan(loc.w1.bytes, loc.w3.bytes));
                        reader.read(loc.w2, dn_dest);
                    }
                });

                cache->ensure_resident(e, reinterpret_cast<const __nv_bfloat16*>(lease.gate_up()),
                                       reinterpret_cast<const __nv_bfloat16*>(lease.down()),
                                       transfer, score);

                auto ev = std::make_unique<CudaEvent>();
                ev->record(transfer);
                controller->inflight_transfers.push_back({std::move(lease), std::move(ev)});
            } else {
                // Host-resident mode
                cache->ensure_resident(e, transfer, score);
            }
        }
        // Publish all pointer and slot changes once after the promotion batch.
        cache->sync_residency_tables(transfer);

    }

    // Compute stream must wait for the on-demand promotions to land before the
    // FFN reads the pointer table.
    CELEG_CUDA(cudaEventRecord(promote_done_event.get(), transfer));
    CELEG_CUDA(cudaStreamWaitEvent(compute_stream, promote_done_event.get(), 0));

    // Speculatively prefetch experts so they are GPU-resident before the *next*
    // token's FFN. Reuses host buffers to avoid per-call allocation.
    if (expert_offload_plan.prefetch_experts > 0 && route_scores_dev != nullptr) {
        const int E = num_experts;
        const int take = std::min<int>(
            K + expert_offload_plan.prefetch_experts, E);
        prefetch_idx.resize(static_cast<size_t>(E));
        for (int e = 0; e < E; ++e) prefetch_idx[static_cast<size_t>(e)] = e;
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

void CudaCompiledModel::ensure_moe_experts_resident(int layer, const int* sel_dev,
                                                   int rows,
                                                   cudaStream_t compute_stream,
                                                   const float* route_scores_dev) {
    if (!resources_.weights_) return;
    resources_.weights_->ensure_moe_experts_resident(ExpertResidencyRequest{
        layer, sel_dev, rows, resources_.shape_.experts_per_token,
        resources_.shape_.num_experts, compute_stream, route_scores_dev,
        &workspace_.router_done_event_, &workspace_.ffn_done_event_,
        &workspace_.promote_done_event_, &workspace_.prefetch_done_event_,
        &workspace_.cold_expert_host_, &workspace_.cold_scores_host_,
        &workspace_.prefetch_idx_, &workspace_.prefetch_ranked_,
        &workspace_.prefetch_scores_});
}

void CudaCompiledModel::ensure_moe_experts_resident_packed(
    int layer, const int* sel_dev, int rows, cudaStream_t stream,
    const float* route_scores_dev) {
    ensure_moe_experts_resident(layer, sel_dev, rows, stream, route_scores_dev);
}

void CudaCompiledModel::run_mlp_moe_decode(const LayerCommon& common_layer,
                                         int layer) {
    const MoeFfnWeights& moe = *as_moe_ffn(common_layer.feed_forward);
    launch_rmsnorm(workspace_.hidden_.data(), common_layer.ffn_norm, workspace_.normed_.data(),
                    1, resources_.shape_.hidden, resources_.shape_.numerical_policy.norm_eps, stream_.get());
    // Router: BF16 normed hidden -> float -> top-K experts.
    launch_cast_bf16_to_float(workspace_.normed_.data(), workspace_.moe_hidden_float_.data(),
                               resources_.shape_.hidden, stream_.get());
    const celeg::MoeRouterConfig cfg = moe_router_config(resources_.shape_);
    celeg::MoeRouterDevice rdev;
    rdev.router_weight = moe.router_float;
    rdev.expert_bias = moe.expert_bias;
    rdev.hidden_data = workspace_.moe_hidden_float_.data();
    rdev.selected_experts = workspace_.moe_sel_.data();
    rdev.routing_weights = workspace_.moe_routing_w_.data();
    rdev.rows = 1;
    rdev.hidden_dim = resources_.shape_.hidden;
    launch_moe_router(rdev, cfg, workspace_.moe_router_scratch_.data(), stream_.get());
    CELEG_CUDA(cudaEventRecord(workspace_.router_done_event_.get(), stream_.get()));

    // Promote any cold experts selected by the router before the FFN reads them.
    ensure_moe_experts_resident(layer, workspace_.moe_sel_.data(), 1, stream_.get(),
                                workspace_.moe_router_scratch_.data());

    // Expert FFN: accumulate the routing-weighted expert outputs into the
    // FP32 output accumulator and then cast into the BF16 workspace_.moe_output_.
    workspace_.moe_output_accum_.zero_async(stream_.get());
    const celeg::MoeFfnDevice fdev = moe_ffn_device(moe, resources_.shape_);
    launch_moe_ffn(fdev, workspace_.moe_sel_.data(), workspace_.moe_routing_w_.data(),
                    workspace_.normed_.data(), workspace_.moe_output_accum_.data(), 1, resources_.shape_.experts_per_token,
                    workspace_.moe_gu_scratch_.data(), workspace_.moe_act_scratch_.data(), stream_.get());
    launch_finalize_moe_output(workspace_.moe_output_accum_.data(), workspace_.moe_output_.data(),
                                resources_.shape_.hidden, stream_.get());
    CELEG_CUDA(cudaEventRecord(workspace_.ffn_done_event_.get(), stream_.get()));

    // Residual add into the hidden state.
    launch_residual_add(workspace_.hidden_.data(), workspace_.moe_output_.data(),
                         resources_.shape_.hidden, stream_.get());
}

void CudaCompiledModel::run_mlp_moe_prefill(const LayerCommon& common_layer, int rows,
                                         int layer) {
    const MoeFfnWeights& moe = *as_moe_ffn(common_layer.feed_forward);
    // Size the prefill scratch to the requested row count.
    workspace_.moe_pf_hidden_float_.reserve(static_cast<size_t>(rows) * resources_.shape_.hidden);
    workspace_.moe_pf_sel_.reserve(static_cast<size_t>(rows) * resources_.shape_.experts_per_token);
    workspace_.moe_pf_sel_masked_.reserve(static_cast<size_t>(rows) * resources_.shape_.experts_per_token);
    workspace_.expert_active_dev_.reserve(static_cast<size_t>(resources_.shape_.num_experts));
    workspace_.moe_pf_routing_w_.reserve(static_cast<size_t>(rows) * resources_.shape_.experts_per_token);
    workspace_.moe_pf_router_scratch_.reserve(static_cast<size_t>(rows) * resources_.shape_.num_experts);
    workspace_.moe_pf_output_accum_.reserve(static_cast<size_t>(rows) * resources_.shape_.hidden);
    workspace_.moe_pf_output_.reserve(static_cast<size_t>(rows) * resources_.shape_.hidden);
    workspace_.moe_pf_gu_scratch_.reserve(
        static_cast<size_t>(rows) * resources_.shape_.experts_per_token * 2 * resources_.shape_.moe_intermediate);
    workspace_.moe_pf_act_scratch_.reserve(
        static_cast<size_t>(rows) * resources_.shape_.experts_per_token * resources_.shape_.moe_intermediate);

    launch_rmsnorm(workspace_.prefill_hidden_.data(), common_layer.ffn_norm,
                   workspace_.prefill_normed_.data(), rows, resources_.shape_.hidden, resources_.shape_.numerical_policy.norm_eps,
                   stream_.get());
    launch_cast_bf16_to_float(workspace_.prefill_normed_.data(), workspace_.moe_pf_hidden_float_.data(),
                              rows * resources_.shape_.hidden, stream_.get());
    const celeg::MoeRouterConfig cfg = moe_router_config(resources_.shape_);
    celeg::MoeRouterDevice rdev;
    rdev.router_weight = moe.router_float;
    rdev.expert_bias = moe.expert_bias;
    rdev.hidden_data = workspace_.moe_pf_hidden_float_.data();
    rdev.selected_experts = workspace_.moe_pf_sel_.data();
    rdev.routing_weights = workspace_.moe_pf_routing_w_.data();
    rdev.rows = rows;
    rdev.hidden_dim = resources_.shape_.hidden;
    launch_moe_router(rdev, cfg, workspace_.moe_pf_router_scratch_.data(), stream_.get());
    CELEG_CUDA(cudaEventRecord(workspace_.router_done_event_.get(), stream_.get()));

    const bool is_disk_backed = workspace_.expert_offload_plan_.enabled && (resources_.options_.expert_offload.backing == ExpertBackingMode::DiskCached);

    if (is_disk_backed) {
        // Disk-backed expert-major streamed prefill path!
        ExpertLayerCache* cache = workspace_.expert_caches_[static_cast<size_t>(layer)];
        const int capacity = cache->capacity();

        // 1. Resolve residency on device to get the cold list
        std::vector<int> cold_experts;
        std::vector<float> cold_scores;
        cache->resolve_on_device(workspace_.moe_pf_sel_.data(), nullptr, rows, resources_.shape_.experts_per_token,
                                 stream_.get(), cold_experts, cold_scores);

        workspace_.moe_pf_output_accum_.zero_async(stream_.get());

        // 2. Read back the selection table to know all used experts in this prefill chunk
        const size_t total_selections = static_cast<size_t>(rows) * resources_.shape_.experts_per_token;
        std::vector<int> sel_host(total_selections);
        CELEG_CUDA(cudaMemcpyAsync(sel_host.data(), workspace_.moe_pf_sel_.data(),
                                 total_selections * sizeof(int),
                                 cudaMemcpyDeviceToHost, stream_.get()));
        CELEG_CUDA(cudaStreamSynchronize(stream_.get()));

        std::vector<int> used_experts;
        std::vector<bool> seen(static_cast<size_t>(resources_.shape_.num_experts), false);
        for (int e : sel_host) {
            if (e >= 0 && e < resources_.shape_.num_experts && !seen[static_cast<size_t>(e)]) {
                seen[static_cast<size_t>(e)] = true;
                used_experts.push_back(e);
            }
        }

        // 3. Process all used experts in batches of size up to capacity
        for (size_t offset = 0; offset < used_experts.size(); offset += capacity) {
            size_t batch_size = std::min<size_t>(capacity, used_experts.size() - offset);
            std::vector<int> batch(used_experts.begin() + offset, used_experts.begin() + offset + batch_size);

            // Promote all experts in this batch to GPU
            for (int e : batch) {
                if (!cache->resident(e)) {
                    const ExpertLocation& loc = workspace_.expert_catalog_[static_cast<size_t>(layer)][static_cast<size_t>(e)];
                    ExpertHostLease lease = resources_.weights_->pinned_expert_cache->acquire(layer, e, [&](std::span<std::byte> gu_dest, std::span<std::byte> dn_dest) {
                        if (resources_.weights_->expert_sidecar) {
                            resources_.weights_->expert_sidecar->read_expert(layer, e, gu_dest, dn_dest);
                        } else {
                            const auto& reader =
                                require_random_access_tensor_reader(*resources_.weights_->repo);
                            reader.read(loc.w1, gu_dest.subspan(0, loc.w1.bytes));
                            reader.read(loc.w3, gu_dest.subspan(loc.w1.bytes, loc.w3.bytes));
                            reader.read(loc.w2, dn_dest);
                        }
                    });
                    if (cache->ensure_resident(e,
                                               reinterpret_cast<const __nv_bfloat16*>(lease.gate_up()),
                                               reinterpret_cast<const __nv_bfloat16*>(lease.down()),
                                               stream_.get())) {
                        auto ev = std::make_unique<CudaEvent>();
                        ev->record(stream_.get());
                        std::lock_guard<std::mutex> ctrl_lock(resources_.weights_->expert_controllers[static_cast<size_t>(layer)]->mutex);
                        resources_.weights_->expert_controllers[static_cast<size_t>(layer)]->inflight_transfers.push_back({std::move(lease), std::move(ev)});
                    }
                } else {
                    cache->touch(e);
                }
            }

            // Sync stream to ensure promotions are ordered
            CELEG_CUDA(cudaStreamSynchronize(stream_.get()));

            // Construct a temporary device pointer table where ONLY the experts in the current batch are non-null
            std::vector<const __nv_bfloat16*> temp_gu(static_cast<size_t>(resources_.shape_.num_experts), nullptr);
            std::vector<const __nv_bfloat16*> temp_dn(static_cast<size_t>(resources_.shape_.num_experts), nullptr);
            std::vector<std::uint8_t> active_flags(static_cast<size_t>(resources_.shape_.num_experts), 0);

            for (int e : batch) {
                temp_gu[static_cast<size_t>(e)] = cache->expert_gate_up_dev(e);
                temp_dn[static_cast<size_t>(e)] = cache->expert_down_dev(e);
                active_flags[static_cast<size_t>(e)] = 1;
            }

            CELEG_CUDA(cudaMemcpyAsync(const_cast<const __nv_bfloat16**>(cache->gate_up_ptrs()), temp_gu.data(),
                                     static_cast<size_t>(resources_.shape_.num_experts) * sizeof(const __nv_bfloat16*),
                                     cudaMemcpyHostToDevice, stream_.get()));
            CELEG_CUDA(cudaMemcpyAsync(const_cast<const __nv_bfloat16**>(cache->down_ptrs()), temp_dn.data(),
                                     static_cast<size_t>(resources_.shape_.num_experts) * sizeof(const __nv_bfloat16*),
                                     cudaMemcpyHostToDevice, stream_.get()));

            // Populate active flags on device
            CELEG_CUDA(cudaMemcpyAsync(workspace_.expert_active_dev_.data(), active_flags.data(),
                                     static_cast<size_t>(resources_.shape_.num_experts) * sizeof(std::uint8_t),
                                     cudaMemcpyHostToDevice, stream_.get()));

            // Launch mask kernel to safely replace deactivated experts with -1
            const int block = 256;
            const int total_elems = rows * resources_.shape_.experts_per_token;
            const int grid = (total_elems + block - 1) / block;
            mask_expert_selection_kernel<<<grid, block, 0, stream_.get()>>>(
                workspace_.moe_pf_sel_.data(), workspace_.moe_pf_sel_masked_.data(), workspace_.expert_active_dev_.data(), total_elems);

            // Launch FFN for the chunk using the safely masked selection matrix
            celeg::MoeFfnDevice fdev = moe_ffn_device(moe, resources_.shape_);
            fdev.gate_up_ptrs = cache->gate_up_ptrs();
            fdev.down_ptrs = cache->down_ptrs();

            launch_moe_ffn(fdev, workspace_.moe_pf_sel_masked_.data(), workspace_.moe_pf_routing_w_.data(),
                            workspace_.prefill_normed_.data(), workspace_.moe_pf_output_accum_.data(), rows,
                            resources_.shape_.experts_per_token, workspace_.moe_pf_gu_scratch_.data(),
                            workspace_.moe_pf_act_scratch_.data(), stream_.get());
        }

        // Restore the full pointer table of the cache (pointing to all currently resident experts)
        std::vector<const __nv_bfloat16*> full_gu(static_cast<size_t>(resources_.shape_.num_experts), nullptr);
        std::vector<const __nv_bfloat16*> full_dn(static_cast<size_t>(resources_.shape_.num_experts), nullptr);
        for (int e = 0; e < resources_.shape_.num_experts; ++e) {
            full_gu[static_cast<size_t>(e)] = cache->expert_gate_up_dev(e);
            full_dn[static_cast<size_t>(e)] = cache->expert_down_dev(e);
        }
        CELEG_CUDA(cudaMemcpyAsync(const_cast<const __nv_bfloat16**>(cache->gate_up_ptrs()), full_gu.data(),
                                 static_cast<size_t>(resources_.shape_.num_experts) * sizeof(const __nv_bfloat16*),
                                 cudaMemcpyHostToDevice, stream_.get()));
        CELEG_CUDA(cudaMemcpyAsync(const_cast<const __nv_bfloat16**>(cache->down_ptrs()), full_dn.data(),
                                 static_cast<size_t>(resources_.shape_.num_experts) * sizeof(const __nv_bfloat16*),
                                 cudaMemcpyHostToDevice, stream_.get()));

        // Finalize outputs
        launch_finalize_moe_output(workspace_.moe_pf_output_accum_.data(), workspace_.moe_pf_output_.data(),
                                    rows * resources_.shape_.hidden, stream_.get());
        CELEG_CUDA(cudaEventRecord(workspace_.ffn_done_event_.get(), stream_.get()));

        launch_residual_add(workspace_.prefill_hidden_.data(), workspace_.moe_pf_output_.data(),
                            rows * resources_.shape_.hidden, stream_.get());
    } else {
        // Promote any cold experts selected by the router before the FFN reads them.
        ensure_moe_experts_resident(layer, workspace_.moe_pf_sel_.data(), rows, stream_.get());

        workspace_.moe_pf_output_accum_.zero_async(stream_.get());
        const celeg::MoeFfnDevice fdev = moe_ffn_device(moe, resources_.shape_);
        launch_moe_ffn(fdev, workspace_.moe_pf_sel_.data(), workspace_.moe_pf_routing_w_.data(),
                        workspace_.prefill_normed_.data(), workspace_.moe_pf_output_accum_.data(), rows,
                        resources_.shape_.experts_per_token, workspace_.moe_pf_gu_scratch_.data(),
                        workspace_.moe_pf_act_scratch_.data(), stream_.get());
        launch_finalize_moe_output(workspace_.moe_pf_output_accum_.data(), workspace_.moe_pf_output_.data(),
                                    rows * resources_.shape_.hidden, stream_.get());
        CELEG_CUDA(cudaEventRecord(workspace_.ffn_done_event_.get(), stream_.get()));

        launch_residual_add(workspace_.prefill_hidden_.data(), workspace_.moe_pf_output_.data(),
                            rows * resources_.shape_.hidden, stream_.get());
    }
}

} // namespace celeg
