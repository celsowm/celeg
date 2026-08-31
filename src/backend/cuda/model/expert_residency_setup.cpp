#include "expert_residency_setup.hpp"

#include "detail/compiled_model.hpp"

#include <stdexcept>
#include <vector>

namespace celeg {

OffloadedExpertWeights install_cuda_expert_controller(
    CudaCompiledModel& model,
    int resource_layer,
    std::unique_ptr<ResidencyController> controller) {
    if (!controller || !controller->cache) {
        throw std::invalid_argument("CUDA expert controller is not initialized");
    }

    OffloadedExpertWeights storage{
        controller->cache->gate_up_ptrs(),
        controller->cache->down_ptrs()};
    model.resources_.weights_->expert_controllers[
        static_cast<size_t>(resource_layer)] = std::move(controller);
    model.workspace_.expert_caches_[static_cast<size_t>(resource_layer)] =
        model.resources_.weights_->expert_controllers[
            static_cast<size_t>(resource_layer)]->cache.get();
    return storage;
}

OffloadedExpertWeights bind_cuda_host_expert_residency(
    CudaCompiledModel& model,
    const IWeightRepository& repo,
    const MoeExpertTensorNames& expert_names,
    int resource_layer,
    int expert_count,
    int intermediate) {
    CudaModelResources& resources = model.resources_;
    CudaWorkspace& workspace = model.workspace_;
    if (resources.options().expert_offload.backing == ExpertBackingMode::DiskCached) {
        throw std::invalid_argument(
            "host expert residency binding does not accept DiskCached backing");
    }

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
    return install_cuda_expert_controller(
        model, resource_layer, std::move(controller));
}

PreparedDiskExpertResidency prepare_cuda_disk_expert_residency(
    CudaCompiledModel& model,
    const IWeightRepository& repo,
    const MoeExpertTensorNames& expert_names,
    int resource_layer,
    int expert_count,
    int intermediate) {
    CudaModelResources& resources = model.resources_;
    CudaWorkspace& workspace = model.workspace_;
    if (resources.options().expert_offload.backing != ExpertBackingMode::DiskCached) {
        throw std::invalid_argument(
            "disk expert residency preparation requires DiskCached backing");
    }

    PreparedDiskExpertResidency prepared;
    prepared.catalog = resources.weight_loader_->build_expert_catalog(
        repo, expert_names, expert_count, intermediate,
        resources.program_.hidden);
    workspace.expert_catalog_[static_cast<size_t>(resource_layer)] =
        prepared.catalog;
    if (resources.weights_->expert_catalog[
            static_cast<size_t>(resource_layer)].empty()) {
        resources.weights_->expert_catalog[
            static_cast<size_t>(resource_layer)] = prepared.catalog;
    }

    const size_t gate_up_bytes =
        2ull * static_cast<size_t>(intermediate) * resources.program_.hidden *
        sizeof(__nv_bfloat16);
    const size_t down_bytes =
        static_cast<size_t>(resources.program_.hidden) * intermediate *
        sizeof(__nv_bfloat16);
    prepared.controller = std::make_unique<ResidencyController>(
        expert_count, workspace.expert_offload_plan_.experts_per_layer,
        gate_up_bytes, down_bytes,
        resources.options().expert_offload.policy);
    std::vector<const __nv_bfloat16*> empty(
        static_cast<size_t>(expert_count), nullptr);
    prepared.controller->cache->set_host_sources(empty, empty);
    return prepared;
}

void promote_cuda_disk_expert_payload(
    ResidencyController& controller,
    int slot,
    const ExpertLocation& location,
    const std::byte* payload) {
    if (!payload) {
        throw std::invalid_argument("CUDA disk expert payload is null");
    }
    controller.cache->promote(
        slot, slot,
        reinterpret_cast<const __nv_bfloat16*>(payload),
        reinterpret_cast<const __nv_bfloat16*>(
            payload + location.w1.bytes + location.w3.bytes),
        controller.transfer_stream->get());
}

}
