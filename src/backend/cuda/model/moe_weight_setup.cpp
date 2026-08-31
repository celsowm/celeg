#include "moe_weight_setup.hpp"

#include "detail/compiled_model.hpp"
#include "kernels/kernels.cuh"
#include "backend/cuda/weight_setup_support.hpp"
#include "celeg/checkpoint/packed/int4.hpp"
#include "backend/cuda/moe.hpp"
#include "backend/cuda/moe/expert_source.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

namespace celeg {

void bind_cuda_moe_feed_forward(CudaCompiledModel& model,
                                const IWeightRepository& repo,
                                const MoeLayerProgram& semantics,
                                int layer_index,
                                LayerCommon& common_layer) {
    CudaModelResources& resources = model.resources_;
    CudaWorkspace& workspace = model.workspace_;
    const int expert_count = semantics.router.expert_count;
    const int intermediate = semantics.routed.mlp.intermediate_size;
    const MoeExpertTensorNames expert_names = moe_expert_tensor_names(
        resources.model_.weight_plan.requests, layer_index, expert_count);

    const float* expert_bias = nullptr;
    if (semantics.router.has_expert_bias) {
        expert_bias = resources.weight_loader_->load_f32_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::MoeRouterBias, layer_index),
            {static_cast<int64_t>(expert_count)});
    }

    const LinearWeight* router = resources.weight_loader_->load_router_weight_named(
        repo,
        cuda_tensor_name(resources.model_.weight_plan.requests,
                         TensorRole::MoeRouter, layer_index),
        expert_count, resources.program_.hidden);
    const auto* router_bf16 = std::get_if<Bf16LinearStorage>(&router->storage);
    if (!router_bf16 || !router_bf16->data) {
        throw std::logic_error("CUDA MoE router requires BF16 storage");
    }

    DeviceBuffer<float>& router_float =
        workspace.moe_router_float_[static_cast<size_t>(layer_index)];
    router_float.reset(static_cast<size_t>(expert_count) * resources.program_.hidden);
    launch_cast_bf16_to_float(
        router_bf16->data, router_float.data(),
        expert_count * resources.program_.hidden, model.stream_.get());

    MoeFfnWeights moe_weights{};
    moe_weights.router = router;
    moe_weights.expert_bias = expert_bias;
    moe_weights.router_float = router_float.data();

    if (semantics.shared) {
        const int shared_intermediate = semantics.shared->mlp.intermediate_size;
        moe_weights.shared_w13 = resources.weight_loader_->load_concat_linear_weight(
            repo, cuda_layer_name(layer_index, "shared_expert.w13.weight"),
            {
                {cuda_tensor_name(resources.model_.weight_plan.requests,
                                  TensorRole::MoeSharedGate, layer_index),
                 {shared_intermediate, resources.program_.hidden}},
                {cuda_tensor_name(resources.model_.weight_plan.requests,
                                  TensorRole::MoeSharedUp, layer_index),
                 {shared_intermediate, resources.program_.hidden}},
            });
        moe_weights.shared_w2 = resources.weight_loader_->load_linear_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::MoeSharedDown, layer_index),
            {resources.program_.hidden, shared_intermediate});
        const auto gate_request = std::find_if(
            resources.model_.weight_plan.requests.begin(),
            resources.model_.weight_plan.requests.end(),
            [layer_index](const TensorRequest& request) {
                return request.role == TensorRole::MoeSharedGateWeight &&
                       request.layer == layer_index && request.expert == -1;
            });
        if (gate_request != resources.model_.weight_plan.requests.end()) {
            moe_weights.shared_gate = resources.weight_loader_->load_linear_weight(
                repo,
                cuda_tensor_name(resources.model_.weight_plan.requests,
                                 TensorRole::MoeSharedGateWeight, layer_index),
                {1, resources.program_.hidden});
        }
    }

    if (workspace.expert_offload_plan_.enabled) {
        const std::string& probe_name = expert_names.packed()
            ? expert_names.packed_gate_up : expert_names.gate.front();
        if (!has_packed_int4_matrix(repo, probe_name)) {
            const HostTensorView expert_probe = repo.tensor(probe_name);
            if (expert_probe.dtype == TensorDType::Quantized) {
                throw std::invalid_argument(
                    "native GGUF MoE experts do not support BF16 offload; "
                    "disable expert offload to keep packed Q4/Q6 weights resident");
            }
        }

        if (resources.options().expert_offload.backing == ExpertBackingMode::DiskCached) {
            std::vector<ExpertLocation> catalog =
                resources.weight_loader_->build_expert_catalog(
                    repo, expert_names, expert_count, intermediate,
                    resources.program_.hidden);
            workspace.expert_catalog_[static_cast<size_t>(layer_index)] = catalog;
            if (resources.weights_->expert_catalog[static_cast<size_t>(layer_index)].empty()) {
                resources.weights_->expert_catalog[static_cast<size_t>(layer_index)] = catalog;
            }

            const size_t gate_up_bytes =
                2ull * static_cast<size_t>(intermediate) * resources.program_.hidden *
                sizeof(__nv_bfloat16);
            const size_t down_bytes =
                static_cast<size_t>(resources.program_.hidden) * intermediate *
                sizeof(__nv_bfloat16);
            auto controller = std::make_unique<ResidencyController>(
                expert_count, workspace.expert_offload_plan_.experts_per_layer,
                gate_up_bytes, down_bytes,
                resources.options().expert_offload.policy);
            std::vector<const __nv_bfloat16*> empty_host_dev(
                static_cast<size_t>(expert_count), nullptr);
            controller->cache->set_host_sources(empty_host_dev, empty_host_dev);

            if (workspace.expert_offload_plan_.experts_per_layer > 0) {
                for (int seed = 0;
                     seed < workspace.expert_offload_plan_.experts_per_layer;
                     ++seed) {
                    const ExpertLocation& location = catalog[static_cast<size_t>(seed)];
                    ExpertHostLease lease = resources.weights_->host_expert_cache->acquire(
                        layer_index, seed, [&](std::span<std::byte> payload) {
                            if (!resources.weights_->expert_source) {
                                throw std::runtime_error(
                                    "CUDA expert source is not initialized");
                            }
                            resources.weights_->expert_source->read(
                                layer_index, seed, payload);
                        });
                    controller->cache->promote(
                        seed, seed,
                        reinterpret_cast<const __nv_bfloat16*>(lease.payload()),
                        reinterpret_cast<const __nv_bfloat16*>(
                            lease.payload() + location.w1.bytes + location.w3.bytes),
                        controller->transfer_stream->get());
                    auto event = std::make_unique<CudaEvent>();
                    event->record(controller->transfer_stream->get());
                    controller->inflight_transfers.push_back(
                        {std::move(lease), std::move(event)});
                }
            }
            CELEG_CUDA(cudaStreamSynchronize(controller->transfer_stream->get()));
            controller->inflight_transfers.clear();
            moe_weights.storage = OffloadedExpertWeights{
                controller->cache->gate_up_ptrs(), controller->cache->down_ptrs()};
            resources.weights_->expert_controllers[static_cast<size_t>(layer_index)] =
                std::move(controller);
            workspace.expert_caches_[static_cast<size_t>(layer_index)] =
                resources.weights_->expert_controllers[static_cast<size_t>(layer_index)]
                    ->cache.get();
        } else {
            WeightLoader::HostExpertLayer host_layer =
                resources.weight_loader_->load_moe_experts_host(
                    repo, expert_names, expert_count, intermediate,
                    resources.program_.hidden, workspace.host_expert_store_,
                    resources.options().expert_offload.host_mode);
            auto controller = std::make_unique<ResidencyController>(
                expert_count, workspace.expert_offload_plan_.experts_per_layer,
                host_layer.gate_up_bytes, host_layer.down_bytes,
                resources.options().expert_offload.policy);
            controller->cache->set_host_sources(host_layer.gate_up_host_dev,
                                                host_layer.down_host_dev);

            std::vector<int> seed(static_cast<size_t>(
                workspace.expert_offload_plan_.experts_per_layer));
            for (int slot = 0; slot < static_cast<int>(seed.size()); ++slot) {
                seed[static_cast<size_t>(slot)] = slot;
            }
            controller->cache->seed(seed, controller->transfer_stream->get());
            CELEG_CUDA(cudaStreamSynchronize(controller->transfer_stream->get()));
            moe_weights.storage = OffloadedExpertWeights{
                controller->cache->gate_up_ptrs(), controller->cache->down_ptrs()};
            resources.weights_->expert_controllers[static_cast<size_t>(layer_index)] =
                std::move(controller);
            workspace.expert_caches_[static_cast<size_t>(layer_index)] =
                resources.weights_->expert_controllers[static_cast<size_t>(layer_index)]
                    ->cache.get();
        }
    } else {
        if (expert_names.packed()) {
            const ExpertLinearWeight* gate_up =
                resources.weight_loader_->load_expert_linear_weight(
                    repo, expert_names.packed_gate_up,
                    expert_count, 2 * intermediate, resources.program_.hidden);
            const ExpertLinearWeight* down =
                resources.weight_loader_->load_expert_linear_weight(
                    repo, expert_names.packed_down,
                    expert_count, resources.program_.hidden, intermediate);
            moe_weights.storage = ResidentExpertWeights{gate_up, down};
        } else {
            const ExpertLinearWeight* gate_up =
                resources.weight_loader_->load_moe_gate_up(
                    repo, expert_names, expert_count, intermediate,
                    resources.program_.hidden);
            const ExpertLinearWeight* down =
                resources.weight_loader_->load_moe_down(
                    repo, expert_names, expert_count, intermediate,
                    resources.program_.hidden);
            moe_weights.storage = ResidentExpertWeights{gate_up, down};
        }
    }

    common_layer.feed_forward = moe_weights;
}

}
