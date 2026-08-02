#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/moe.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace celeg {
void CudaCompiledModel::initialize_per_layer_input_device(const int32_t* token) {
    if (!resources_.shape_.has_per_layer_input) return;
    if (!resources_.per_layer_embedding_ || !resources_.per_layer_embedding_->bf16 ||
        !resources_.per_layer_context_projection_ || !resources_.per_layer_projection_norm_) {
        throw std::logic_error("Gemma per-layer input weights are not BF16-resident");
    }
    const int layers = resources_.shape_.num_hidden_layers;
    const int ple = resources_.shape_.per_layer_input_size;
    const int width = layers * ple;
    launch_embedding_slice_device(token, resources_.per_layer_embedding_->bf16,
                                  width, 0, workspace_.per_layer_token_.data(), width,
                                  stream_.get());
    launch_scale(workspace_.per_layer_token_.data(), width,
                 std::sqrt(static_cast<float>(ple)), stream_.get());
    linear(workspace_.hidden_.data(), *resources_.per_layer_context_projection_,
           workspace_.per_layer_context_.data(), 1, width, resources_.shape_.hidden);
    launch_scale(workspace_.per_layer_context_.data(), width,
                 1.0f / std::sqrt(static_cast<float>(resources_.shape_.hidden)), stream_.get());
    launch_rmsnorm(workspace_.per_layer_context_.data(), resources_.per_layer_projection_norm_,
                   workspace_.per_layer_context_.data(), layers, ple,
                   resources_.shape_.norm_eps, stream_.get());
    launch_residual_add(workspace_.per_layer_context_.data(), workspace_.per_layer_token_.data(),
                        width, stream_.get());
    launch_scale(workspace_.per_layer_context_.data(), width,
                 0.7071067811865475f, stream_.get());
}

void CudaCompiledModel::initialize_per_layer_input_host(int32_t token) {
    if (!resources_.shape_.has_per_layer_input) return;
    if (!resources_.per_layer_embedding_ || !resources_.per_layer_embedding_->bf16 ||
        !resources_.per_layer_context_projection_ || !resources_.per_layer_projection_norm_) {
        throw std::logic_error("Gemma per-layer input weights are not BF16-resident");
    }
    const int layers = resources_.shape_.num_hidden_layers;
    const int ple = resources_.shape_.per_layer_input_size;
    const int width = layers * ple;
    launch_embedding_slice(token, resources_.per_layer_embedding_->bf16, width, 0,
                           workspace_.per_layer_token_.data(), width, stream_.get());
    launch_scale(workspace_.per_layer_token_.data(), width,
                 std::sqrt(static_cast<float>(ple)), stream_.get());
    linear(workspace_.hidden_.data(), *resources_.per_layer_context_projection_,
           workspace_.per_layer_context_.data(), 1, width, resources_.shape_.hidden);
    launch_scale(workspace_.per_layer_context_.data(), width,
                 1.0f / std::sqrt(static_cast<float>(resources_.shape_.hidden)), stream_.get());
    launch_rmsnorm(workspace_.per_layer_context_.data(), resources_.per_layer_projection_norm_,
                   workspace_.per_layer_context_.data(), layers, ple,
                   resources_.shape_.norm_eps, stream_.get());
    launch_residual_add(workspace_.per_layer_context_.data(), workspace_.per_layer_token_.data(),
                        width, stream_.get());
    launch_scale(workspace_.per_layer_context_.data(), width,
                 0.7071067811865475f, stream_.get());
}

void CudaCompiledModel::initialize_per_layer_input_batch(const int32_t* tokens, int rows) {
    if (!resources_.shape_.has_per_layer_input) return;
    if (!resources_.per_layer_embedding_ || !resources_.per_layer_embedding_->bf16 ||
        !resources_.per_layer_context_projection_ || !resources_.per_layer_projection_norm_) {
        throw std::logic_error("Gemma per-layer input weights are not BF16-resident");
    }
    const int layers = resources_.shape_.num_hidden_layers;
    const int ple = resources_.shape_.per_layer_input_size;
    const int width = layers * ple;
    launch_embedding_slice_batch(tokens, rows, resources_.per_layer_embedding_->bf16,
                                 width, 0, workspace_.prefill_per_layer_token_.data(),
                                 width, stream_.get());
    launch_scale(workspace_.prefill_per_layer_token_.data(), rows * width,
                 std::sqrt(static_cast<float>(ple)), stream_.get());
    linear(workspace_.prefill_hidden_.data(), *resources_.per_layer_context_projection_,
           workspace_.prefill_per_layer_context_.data(), rows, width,
           resources_.shape_.hidden);
    launch_scale(workspace_.prefill_per_layer_context_.data(), rows * width,
                 1.0f / std::sqrt(static_cast<float>(resources_.shape_.hidden)), stream_.get());
    launch_rmsnorm(workspace_.prefill_per_layer_context_.data(), resources_.per_layer_projection_norm_,
                   workspace_.prefill_per_layer_context_.data(), rows * layers, ple,
                   resources_.shape_.norm_eps, stream_.get());
    launch_residual_add(workspace_.prefill_per_layer_context_.data(),
                        workspace_.prefill_per_layer_token_.data(), rows * width,
                        stream_.get());
    launch_scale(workspace_.prefill_per_layer_context_.data(), rows * width,
                 0.7071067811865475f, stream_.get());
}

void CudaCompiledModel::run_mlp_decode(const LayerCommon& common_layer, int layer) {
    if (const MoeFfnWeights* moe = as_moe_ffn(common_layer.feed_forward)) {
        (void)moe;
        run_mlp_moe_decode(common_layer, layer);
    } else {
        launch_rmsnorm(workspace_.hidden_.data(), common_layer.ffn_norm, workspace_.normed_.data(),
                       1, resources_.shape_.hidden, resources_.shape_.norm_eps,
                       stream_.get());
        const int intermediate = resources_.shape_.feed_forward_intermediates.empty()
            ? resources_.shape_.intermediate
            : resources_.shape_.feed_forward_intermediates.at(static_cast<size_t>(layer));
        if (resources_.options_.fused_projections) {
            linear(workspace_.normed_.data(), *as_dense_ffn(common_layer.feed_forward)->w13, workspace_.gate_up_.data(),
                   1, 2 * intermediate, resources_.shape_.hidden);
        } else {
            const LinearWeight w1 =
                slice_rows(*as_dense_ffn(common_layer.feed_forward)->w13, 0, intermediate);
            const LinearWeight w3 = slice_rows(
                *as_dense_ffn(common_layer.feed_forward)->w13, intermediate, intermediate);
            linear(workspace_.normed_.data(), w1, workspace_.gate_up_.data(),
                   1, intermediate, resources_.shape_.hidden);
            linear(workspace_.normed_.data(), w3, workspace_.gate_up_.data() + intermediate,
                   1, intermediate, resources_.shape_.hidden);
        }
        const bool gelu_tanh = !resources_.shape_.feed_forward_activations.empty() &&
            resources_.shape_.feed_forward_activations.at(static_cast<size_t>(layer)) == ActivationKind::GeluTanh;
        if (gelu_tanh) {
            launch_gated_gelu_tanh(workspace_.gate_up_.data(), workspace_.activated_.data(),
                                   intermediate, stream_.get());
        } else {
            launch_swiglu_fused(workspace_.gate_up_.data(), workspace_.activated_.data(),
                                intermediate, stream_.get());
        }
        const bool split_output = common_layer.post_feed_forward_norm != nullptr;
        if (resources_.options_.fused_residuals && !split_output) {
            linear(workspace_.activated_.data(), *as_dense_ffn(common_layer.feed_forward)->w2, workspace_.hidden_.data(),
                   1, resources_.shape_.hidden, intermediate, 1.0f);
        } else {
            linear(workspace_.activated_.data(), *as_dense_ffn(common_layer.feed_forward)->w2, workspace_.mlp_output_.data(),
                   1, resources_.shape_.hidden, intermediate);
            if (split_output) {
                launch_rmsnorm(workspace_.mlp_output_.data(), common_layer.post_feed_forward_norm,
                               workspace_.mlp_output_.data(), 1, resources_.shape_.hidden,
                               resources_.shape_.norm_eps, stream_.get());
            }
            launch_scale(workspace_.mlp_output_.data(), resources_.shape_.hidden, resources_.shape_.residual_multiplier,
                         stream_.get());
            launch_residual_add(workspace_.hidden_.data(), workspace_.mlp_output_.data(),
                                resources_.shape_.hidden, stream_.get());
        }
    }
    run_per_layer_input_decode(common_layer, layer);
}

void CudaCompiledModel::run_mlp_prefill(const LayerCommon& common_layer, int rows,
                                     int layer) {
    if (const MoeFfnWeights* moe = as_moe_ffn(common_layer.feed_forward)) {
        (void)moe;
        run_mlp_moe_prefill(common_layer, rows, layer);
    } else {
        const int intermediate = resources_.shape_.feed_forward_intermediates.empty()
            ? resources_.shape_.intermediate
            : resources_.shape_.feed_forward_intermediates.at(static_cast<size_t>(layer));
        const size_t matrix_elements = static_cast<size_t>(rows) * intermediate;
        launch_rmsnorm(workspace_.prefill_hidden_.data(), common_layer.ffn_norm,
                       workspace_.prefill_normed_.data(), rows, resources_.shape_.hidden,
                       resources_.shape_.norm_eps, stream_.get());
        if (resources_.options_.fused_projections) {
        linear(workspace_.prefill_normed_.data(), *as_dense_ffn(common_layer.feed_forward)->w13, workspace_.prefill_gate_up_.data(),
               rows, 2 * intermediate, resources_.shape_.hidden);
        // The fused GEMM writes one contiguous [gate|up] pair per row
        // (row stride 2*intermediate), unlike the split-call path below
        // which writes all gates then all ups as two separate planes.
        if (!resources_.shape_.feed_forward_activations.empty() &&
            resources_.shape_.feed_forward_activations.at(static_cast<size_t>(layer)) == ActivationKind::GeluTanh) {
            launch_gated_gelu_tanh(workspace_.prefill_gate_up_.data(),
                                   workspace_.prefill_activated_.data(),
                                   static_cast<int>(matrix_elements), stream_.get());
        } else {
            launch_swiglu_interleaved(workspace_.prefill_gate_up_.data(),
                                      workspace_.prefill_activated_.data(), rows,
                                      intermediate, stream_.get());
        }
        } else {
        const LinearWeight w1 =
            slice_rows(*as_dense_ffn(common_layer.feed_forward)->w13, 0, intermediate);
        const LinearWeight w3 = slice_rows(
            *as_dense_ffn(common_layer.feed_forward)->w13, intermediate, intermediate);
        linear(workspace_.prefill_normed_.data(), w1, workspace_.prefill_gate_up_.data(),
               rows, intermediate, resources_.shape_.hidden);
        linear(workspace_.prefill_normed_.data(), w3,
               workspace_.prefill_gate_up_.data() + matrix_elements,
               rows, intermediate, resources_.shape_.hidden);
        if (!resources_.shape_.feed_forward_activations.empty() &&
            resources_.shape_.feed_forward_activations.at(static_cast<size_t>(layer)) == ActivationKind::GeluTanh) {
            launch_gated_gelu_tanh(workspace_.prefill_gate_up_.data(), workspace_.prefill_activated_.data(),
                                   static_cast<int>(matrix_elements), stream_.get());
        } else {
            launch_swiglu_fused(workspace_.prefill_gate_up_.data(), workspace_.prefill_activated_.data(),
                                static_cast<int>(matrix_elements), stream_.get());
        }
        }
        const bool split_output = common_layer.post_feed_forward_norm != nullptr;
        if (resources_.options_.fused_residuals && !split_output) {
        linear(workspace_.prefill_activated_.data(), *as_dense_ffn(common_layer.feed_forward)->w2, workspace_.prefill_hidden_.data(),
               rows, resources_.shape_.hidden, intermediate, 1.0f);
        } else {
        linear(workspace_.prefill_activated_.data(), *as_dense_ffn(common_layer.feed_forward)->w2, workspace_.prefill_mlp_output_.data(),
                rows, resources_.shape_.hidden, intermediate);
        if (split_output) {
            launch_rmsnorm(workspace_.prefill_mlp_output_.data(), common_layer.post_feed_forward_norm,
                           workspace_.prefill_mlp_output_.data(), rows, resources_.shape_.hidden,
                           resources_.shape_.norm_eps, stream_.get());
        }
        launch_scale(workspace_.prefill_mlp_output_.data(), rows * resources_.shape_.hidden,
                     resources_.shape_.residual_multiplier, stream_.get());
        launch_residual_add(workspace_.prefill_hidden_.data(), workspace_.prefill_mlp_output_.data(),
                            rows * resources_.shape_.hidden, stream_.get());
        }
    }
    run_per_layer_input_prefill(common_layer, rows, layer);
}

void CudaCompiledModel::run_per_layer_input_decode(const LayerCommon& common_layer,
                                                   int layer) {
    if (!resources_.shape_.has_per_layer_input) return;
    const int ple = resources_.shape_.per_layer_input_size;
    launch_gelu_tanh(workspace_.per_layer_gate_.data(), workspace_.per_layer_gate_.data(),
                     ple, stream_.get());
    const __nv_bfloat16* context = workspace_.per_layer_context_.data() +
        static_cast<size_t>(layer) * ple;
    launch_multiply(workspace_.per_layer_gate_.data(), context, ple, stream_.get());
    CELEG_CUDA(cudaMemcpyAsync(workspace_.residual_.data(), workspace_.hidden_.data(),
                               workspace_.hidden_.bytes(), cudaMemcpyDeviceToDevice,
                               stream_.get()));
    linear(workspace_.per_layer_gate_.data(), *common_layer.per_layer_projection,
           workspace_.hidden_.data(), 1, resources_.shape_.hidden, ple);
    launch_rmsnorm(workspace_.hidden_.data(), common_layer.per_layer_input_norm,
                   workspace_.hidden_.data(), 1, resources_.shape_.hidden,
                   resources_.shape_.norm_eps, stream_.get());
    launch_scale_by_scalar(workspace_.hidden_.data(), common_layer.layer_scalar,
                           resources_.shape_.hidden, stream_.get());
    launch_residual_add(workspace_.hidden_.data(), workspace_.residual_.data(),
                        resources_.shape_.hidden, stream_.get());
}

void CudaCompiledModel::run_per_layer_input_prefill(const LayerCommon& common_layer,
                                                    int rows, int layer) {
    if (!resources_.shape_.has_per_layer_input) return;
    const int layers = resources_.shape_.num_hidden_layers;
    const int ple = resources_.shape_.per_layer_input_size;
    linear(workspace_.prefill_hidden_.data(), *common_layer.per_layer_input_gate,
           workspace_.prefill_per_layer_gate_.data(), rows, ple,
           resources_.shape_.hidden);
    launch_gelu_tanh(workspace_.prefill_per_layer_gate_.data(),
                     workspace_.prefill_per_layer_gate_.data(), rows * ple,
                     stream_.get());
    launch_multiply_strided(workspace_.prefill_per_layer_gate_.data(),
                            workspace_.prefill_per_layer_context_.data(), rows, ple,
                            layers * ple, layer * ple, stream_.get());
    CELEG_CUDA(cudaMemcpyAsync(workspace_.prefill_residual_.data(),
                               workspace_.prefill_hidden_.data(),
                               workspace_.prefill_hidden_.bytes(),
                               cudaMemcpyDeviceToDevice, stream_.get()));
    linear(workspace_.prefill_per_layer_gate_.data(), *common_layer.per_layer_projection,
           workspace_.prefill_hidden_.data(), rows, resources_.shape_.hidden, ple);
    launch_rmsnorm(workspace_.prefill_hidden_.data(), common_layer.per_layer_input_norm,
                   workspace_.prefill_hidden_.data(), rows, resources_.shape_.hidden,
                   resources_.shape_.norm_eps, stream_.get());
    launch_scale_by_scalar(workspace_.prefill_hidden_.data(), common_layer.layer_scalar,
                           rows * resources_.shape_.hidden, stream_.get());
    launch_residual_add(workspace_.prefill_hidden_.data(), workspace_.prefill_residual_.data(),
                        rows * resources_.shape_.hidden, stream_.get());
}

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

void SharedModelWeights::ensure_moe_experts_resident(
    int layer, const int* sel_dev, int rows, int K, int num_experts,
    cudaStream_t compute_stream, const float* route_scores_dev,
    CudaEvent& router_done_event, CudaEvent& ffn_done_event,
    CudaEvent& promote_done_event, CudaEvent& prefetch_done_event,
    std::vector<int>& cold_expert_host, std::vector<float>& cold_scores_host,
    std::vector<int>& prefetch_idx, std::vector<int>& prefetch_ranked,
    std::vector<float>& prefetch_scores) {
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
        // All selected experts are GPU-resident; touch them for LRU scoring.
        const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(K);
        for (size_t i = 0; i < total; ++i) {
            const int e = cold_expert_host[i];
            const float score = route_scores_dev != nullptr
                ? cold_scores_host[static_cast<size_t>(e)]
                : ExpertLayerCache::kUnseen;
            cache->record_hit();
            cache->touch(e, score);

            // Record usage stats
            if (!usage_profile_path.empty() && layer >= 0 && static_cast<size_t>(layer) < usage_stats.layers.size()) {
                auto& entry = usage_stats.layers[static_cast<size_t>(layer)][static_cast<size_t>(e)];
                entry.selection_count++;
                entry.gpu_cache_hits++;
            }
        }
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

        // Also touch resident experts from the full selection that weren't cold.
        if (rows >= 1) {
            const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(K);
            std::vector<int> sel_host(total);
            CELEG_CUDA(cudaMemcpyAsync(sel_host.data(), sel_dev,
                                     total * sizeof(int),
                                     cudaMemcpyDeviceToHost, transfer));
            CELEG_CUDA(cudaStreamSynchronize(transfer));
            for (size_t i = 0; i < total; ++i) {
                const int e = sel_host[i];
                if (cache->resident(e)) {
                    const float score = route_scores_dev != nullptr
                        ? cold_scores_host[static_cast<size_t>(e)]
                        : ExpertLayerCache::kUnseen;
                    cache->record_hit();
                    cache->touch(e, score);

                    // Record usage stats
                    if (!usage_profile_path.empty() && layer >= 0 && static_cast<size_t>(layer) < usage_stats.layers.size()) {
                        auto& entry = usage_stats.layers[static_cast<size_t>(layer)][static_cast<size_t>(e)];
                        entry.selection_count++;
                        entry.gpu_cache_hits++;
                    }
                }
            }
        }
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
    resources_.weights_->ensure_moe_experts_resident(
        layer, sel_dev, rows, resources_.shape_.experts_per_token, resources_.shape_.num_experts,
        compute_stream, route_scores_dev,
        workspace_.router_done_event_, workspace_.ffn_done_event_,
        workspace_.promote_done_event_, workspace_.prefetch_done_event_,
        workspace_.cold_expert_host_, workspace_.cold_scores_host_,
        workspace_.prefetch_idx_, workspace_.prefetch_ranked_,
        workspace_.prefetch_scores_);
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
                    1, resources_.shape_.hidden, resources_.shape_.norm_eps, stream_.get());
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
                   workspace_.prefill_normed_.data(), rows, resources_.shape_.hidden, resources_.shape_.norm_eps,
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
