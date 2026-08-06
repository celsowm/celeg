#include "celeg/backend/cuda/weight_setup_support.hpp"

#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/moe/offload.hpp"

#include <cstdio>
#include <memory>

namespace celeg {
namespace {

void* allocate_pinned_host(std::size_t bytes) {
    void* pointer = nullptr;
    return cudaMallocHost(&pointer, bytes) == cudaSuccess ? pointer : nullptr;
}

void deallocate_pinned_host(void* pointer) {
    if (pointer) cudaFreeHost(pointer);
}

} // namespace

void configure_cuda_expert_resources(CudaCompiledModel& model) {
    CudaModelResources& resources = model.resources_;
    CudaWorkspace& workspace = model.workspace_;
    if (!resources.options_.expert_offload.enabled() ||
        resources.shape_.num_experts <= 0) {
        return;
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CELEG_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    const int moe_layers = moe_layer_count(resources.shape_);
    const size_t expert_bytes = bytes_per_expert_bf16(resources.shape_);
    const size_t embed_bytes = static_cast<size_t>(resources.shape_.vocab_size) *
        resources.shape_.hidden * sizeof(__nv_bfloat16);
    const size_t attention_bytes = static_cast<size_t>(
        resources.shape_.maximum_attention_projection_width()) *
        resources.shape_.hidden *
        static_cast<size_t>(resources.shape_.attention_layer_count) *
        sizeof(__nv_bfloat16);
    const size_t dense_ffn_bytes = static_cast<size_t>(resources.shape_.num_dense_layers) *
        (3ull * resources.shape_.dense_intermediate * resources.shape_.hidden) *
        sizeof(__nv_bfloat16);
    const size_t router_bytes = static_cast<size_t>(moe_layers) *
        resources.shape_.num_experts * resources.shape_.hidden *
        (sizeof(__nv_bfloat16) + sizeof(float));

    ExpertOffloadPlanInputs inputs;
    inputs.shape = resources.shape_;
    inputs.options = resources.options_.expert_offload;
    inputs.gpu_free_bytes = free_bytes;
    inputs.non_expert_weight_bytes = embed_bytes + attention_bytes +
        dense_ffn_bytes + router_bytes + (64ull << 20);
    inputs.workspace_bytes = resources.options_.lt_workspace_bytes + (256ull << 20);
    inputs.context_tokens = model.max_context_;
    workspace.expert_offload_plan_ = plan_expert_offload(inputs);
    workspace.expert_transfer_stream_ = std::make_unique<CudaStream>();
    resources.weights_->expert_offload_plan = workspace.expert_offload_plan_;

    if (resources.options_.expert_offload.backing == ExpertBackingMode::DiskCached &&
        !resources.weights_->host_expert_cache) {
        resources.weights_->host_expert_cache = std::make_unique<HostExpertCache>(
            resources.options_.expert_offload.host_expert_cache_bytes,
            expert_bytes, allocate_pinned_host, deallocate_pinned_host);
    }
    if (resources.options_.expert_offload.backing == ExpertBackingMode::DiskCached &&
        !resources.weights_->expert_io_manager) {
        resources.weights_->expert_io_manager = std::make_unique<ExpertIoManager>(
            resources.options_.expert_offload.io_workers,
            resources.options_.expert_offload.io_queue_depth);
    }
    if (!resources.options_.expert_offload.expert_sidecar_path.empty() &&
        !resources.weights_->expert_sidecar) {
        auto sidecar = std::make_unique<ExpertSidecar>();
        if (sidecar->load(resources.options_.expert_offload.expert_sidecar_path,
                          moe_layers, resources.shape_.num_experts,
                          resources.shape_.moe_intermediate, resources.shape_.hidden)) {
            resources.weights_->expert_sidecar = std::move(sidecar);
            std::fprintf(stderr, "Loaded compatible expert sidecar from %s\n",
                         resources.options_.expert_offload.expert_sidecar_path.c_str());
        } else {
            std::fprintf(stderr, "WARNING: Sidecar %s is incompatible or could not be loaded; falling back to safetensors.\n",
                         resources.options_.expert_offload.expert_sidecar_path.c_str());
        }
    }
    if (!resources.options_.expert_offload.usage_profile_path.empty()) {
        resources.weights_->usage_profile_path =
            resources.options_.expert_offload.usage_profile_path;
        if (resources.weights_->usage_stats.layers.empty()) {
            if (!resources.weights_->usage_stats.load(
                    resources.weights_->usage_profile_path, moe_layers,
                    resources.shape_.num_experts)) {
                resources.weights_->usage_stats.layers.assign(
                    static_cast<size_t>(moe_layers),
                    std::vector<ExpertUsageEntry>(
                        static_cast<size_t>(resources.shape_.num_experts)));
            } else {
                std::fprintf(stderr,
                             "Loaded persistent expert usage statistics from %s\n",
                             resources.weights_->usage_profile_path.c_str());
            }
        }
    }
    std::fprintf(stderr, "%s", workspace.expert_offload_plan_.report().c_str());
}

} // namespace celeg
