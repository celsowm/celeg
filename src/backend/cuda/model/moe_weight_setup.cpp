#include "moe_weight_setup.hpp"

#include "detail/compiled_model.hpp"
#include "kernels/kernels.cuh"
#include "backend/cuda/weight_setup_support.hpp"
#include "celeg/checkpoint/packed/int4.hpp"
#include "backend/cuda/moe.hpp"
#include "backend/cuda/moe/expert_source.hpp"
#include "expert_residency_setup.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

namespace celeg {

MoeFfnWeights bind_cuda_moe_router_weight(
    CudaCompiledModel& model,
    const IWeightRepository& repo,
    const std::string& router_name,
    int resource_layer,
    int expert_count,
    const float* expert_bias) {
    CudaModelResources& resources = model.resources_;
    const LinearWeight* router = resources.weight_loader_->load_router_weight_named(
        repo, router_name, expert_count, resources.program_.hidden);
    const auto* router_bf16 = std::get_if<Bf16LinearStorage>(&router->storage);
    if (!router_bf16 || !router_bf16->data) {
        throw std::logic_error("CUDA MoE router requires BF16 storage");
    }

    DeviceBuffer<float>& router_float =
        model.workspace_.moe_router_float_[static_cast<size_t>(resource_layer)];
    router_float.reset(static_cast<size_t>(expert_count) * resources.program_.hidden);
    launch_cast_bf16_to_float(
        router_bf16->data, router_float.data(),
        expert_count * resources.program_.hidden, model.stream_.get());

    MoeFfnWeights weights{};
    weights.router = router;
    weights.expert_bias = expert_bias;
    weights.router_float = router_float.data();
    return weights;
}

void bind_cuda_shared_expert(
    CudaCompiledModel& model,
    const IWeightRepository& repo,
    const CudaSharedExpertNames& names,
    int intermediate,
    MoeFfnWeights& weights) {
    CudaModelResources& resources = model.resources_;
    weights.shared_w13 = resources.weight_loader_->load_concat_linear_weight(
        repo, names.synthetic_w13,
        {
            {names.gate, {intermediate, resources.program_.hidden}},
            {names.up, {intermediate, resources.program_.hidden}},
        });
    weights.shared_w2 = resources.weight_loader_->load_linear_weight(
        repo, names.down, {resources.program_.hidden, intermediate});
    if (names.gate_weight) {
        weights.shared_gate = resources.weight_loader_->load_linear_weight(
            repo, *names.gate_weight, {1, resources.program_.hidden});
    }
}

ResidentExpertWeights bind_cuda_resident_experts(
    CudaCompiledModel& model,
    const IWeightRepository& repo,
    const MoeExpertTensorNames& expert_names,
    int expert_count,
    int intermediate) {
    CudaModelResources& resources = model.resources_;
    if (expert_names.packed()) {
        const ExpertLinearWeight* gate_up =
            resources.weight_loader_->load_expert_linear_weight(
                repo, expert_names.packed_gate_up,
                expert_count, 2 * intermediate, resources.program_.hidden);
        const ExpertLinearWeight* down =
            resources.weight_loader_->load_expert_linear_weight(
                repo, expert_names.packed_down,
                expert_count, resources.program_.hidden, intermediate);
        return ResidentExpertWeights{gate_up, down};
    }

    const ExpertLinearWeight* gate_up =
        resources.weight_loader_->load_moe_gate_up(
            repo, expert_names, expert_count, intermediate,
            resources.program_.hidden);
    const ExpertLinearWeight* down =
        resources.weight_loader_->load_moe_down(
            repo, expert_names, expert_count, intermediate,
            resources.program_.hidden);
    return ResidentExpertWeights{gate_up, down};
}

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

    MoeFfnWeights moe_weights = bind_cuda_moe_router_weight(
        model, repo,
        cuda_tensor_name(resources.model_.weight_plan.requests,
                         TensorRole::MoeRouter, layer_index),
        layer_index, expert_count, expert_bias);

    if (semantics.shared) {
        CudaSharedExpertNames names{
            cuda_layer_name(layer_index, "shared_expert.w13.weight"),
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::MoeSharedGate, layer_index),
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::MoeSharedUp, layer_index),
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::MoeSharedDown, layer_index),
            std::nullopt};
        const auto gate_request = std::find_if(
            resources.model_.weight_plan.requests.begin(),
            resources.model_.weight_plan.requests.end(),
            [layer_index](const TensorRequest& request) {
                return request.role == TensorRole::MoeSharedGateWeight &&
                       request.layer == layer_index && request.expert == -1;
            });
        if (gate_request != resources.model_.weight_plan.requests.end()) {
            names.gate_weight = cuda_tensor_name(
                resources.model_.weight_plan.requests,
                TensorRole::MoeSharedGateWeight, layer_index);
        }
        bind_cuda_shared_expert(
            model, repo, names, semantics.shared->mlp.intermediate_size,
            moe_weights);
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
            PreparedDiskExpertResidency prepared =
                prepare_cuda_disk_expert_residency(
                    model, repo, expert_names, layer_index,
                    expert_count, intermediate);
            ResidencyController& controller = *prepared.controller;
            for (int seed = 0;
                 seed < workspace.expert_offload_plan_.experts_per_layer;
                 ++seed) {
                const ExpertLocation& location =
                    prepared.catalog[static_cast<size_t>(seed)];
                ExpertHostLease lease = resources.weights_->host_expert_cache->acquire(
                    layer_index, seed, [&](std::span<std::byte> payload) {
                        if (!resources.weights_->expert_source) {
                            throw std::runtime_error(
                                "CUDA expert source is not initialized");
                        }
                        resources.weights_->expert_source->read(
                            layer_index, seed, payload);
                    });
                promote_cuda_disk_expert_payload(
                    controller, seed, location, lease.payload());
                auto event = std::make_unique<CudaEvent>();
                event->record(controller.transfer_stream->get());
                controller.inflight_transfers.push_back(
                    {std::move(lease), std::move(event)});
            }
            CELEG_CUDA(cudaStreamSynchronize(controller.transfer_stream->get()));
            controller.inflight_transfers.clear();
            moe_weights.storage = install_cuda_expert_controller(
                model, layer_index, std::move(prepared.controller));
        } else {
            moe_weights.storage = bind_cuda_host_expert_residency(
                model, repo, expert_names, layer_index,
                expert_count, intermediate);
        }
    } else {
        moe_weights.storage = bind_cuda_resident_experts(
            model, repo, expert_names, expert_count, intermediate);
    }

    common_layer.feed_forward = moe_weights;
}

}
