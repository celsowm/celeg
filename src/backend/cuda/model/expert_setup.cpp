#include "celeg/backend/cuda/weight_setup_support.hpp"

#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/moe/offload.hpp"

#include <cstdio>
#include <memory>

namespace celeg {
namespace {

std::size_t bf16_bytes(std::size_t elements) {
    return elements * sizeof(__nv_bfloat16);
}

std::size_t estimate_non_expert_weights(const RuntimeTopology& shape,
                                        bool tied_embeddings) {
    std::size_t bytes = bf16_bytes(static_cast<std::size_t>(shape.vocab_size) *
                                   static_cast<std::size_t>(shape.hidden));
    if (!tied_embeddings) bytes += bytes;
    bytes += bf16_bytes(static_cast<std::size_t>(shape.hidden));

    for (int layer = 0; layer < shape.num_hidden_layers; ++layer) {
        // input_layernorm and the FFN input norm are present in all normal
        // decoder blocks. Split-norm families add two more vectors below.
        bytes += bf16_bytes(2ull * static_cast<std::size_t>(shape.hidden));
        if (shape.has_split_attention_norms) {
            bytes += bf16_bytes(2ull * static_cast<std::size_t>(shape.hidden));
        }

        const MixerKind mixer = shape.mixer_kinds.at(static_cast<size_t>(layer));
        if (mixer == MixerKind::Attention) {
            const AttentionSpec& attention = shape.attention_layout(layer);
            bytes += bf16_bytes(static_cast<std::size_t>(attention.query_projection_width()) * shape.hidden);
            if (!attention.kv_sharing.shared() || attention.kv_sharing.publishes) {
                bytes += bf16_bytes(2ull * static_cast<std::size_t>(attention.key_value_width()) * shape.hidden);
            }
            bytes += bf16_bytes(static_cast<std::size_t>(shape.hidden) * attention.query_width());
            if (attention.query_key_norm) {
                bytes += bf16_bytes(2ull * static_cast<std::size_t>(attention.head_dim));
            }
        } else if (mixer == MixerKind::GatedDeltaNet) {
            const GatedDeltaNetSpec& spec = shape.gated_delta_net_layouts.at(
                static_cast<size_t>(layer));
            const std::size_t key_width = static_cast<std::size_t>(spec.key_heads) * spec.key_head_dim;
            const std::size_t value_width = static_cast<std::size_t>(spec.value_heads) * spec.value_head_dim;
            const std::size_t qkv_width = 2ull * key_width + value_width;
            bytes += bf16_bytes(qkv_width * shape.hidden);
            bytes += bf16_bytes(value_width * shape.hidden);
            bytes += bf16_bytes(2ull * static_cast<std::size_t>(spec.value_heads) * shape.hidden);
            bytes += bf16_bytes(static_cast<std::size_t>(shape.hidden) * value_width);
            bytes += bf16_bytes(qkv_width * spec.conv_kernel +
                                2ull * static_cast<std::size_t>(spec.value_heads) +
                                static_cast<std::size_t>(spec.value_head_dim));
        }

        if (shape.layer_uses_moe(layer)) {
            const std::size_t router_elements = static_cast<std::size_t>(shape.num_experts) * shape.hidden;
            bytes += bf16_bytes(router_elements) + router_elements * sizeof(float);
            if (shape.shared_expert_intermediate > 0) {
                const std::size_t shared = static_cast<std::size_t>(shape.shared_expert_intermediate);
                bytes += bf16_bytes(3ull * shared * shape.hidden + shape.hidden);
            }
        } else {
            const int intermediate = shape.feed_forward_intermediates.empty()
                ? shape.intermediate
                : shape.feed_forward_intermediates.at(static_cast<size_t>(layer));
            bytes += bf16_bytes(3ull * static_cast<std::size_t>(intermediate) * shape.hidden);
        }
    }
    return bytes;
}

std::size_t estimate_mtp_non_expert_weights(const RuntimeTopology& shape) {
    if (shape.mtp_num_hidden_layers <= 0) return 0;
    const auto bf16_bytes = [](std::size_t elements) {
        return elements * sizeof(__nv_bfloat16);
    };
    const int full_attention_layer = [&]() {
        for (int layer = shape.num_hidden_layers - 1; layer >= 0; --layer) {
            if (shape.mixer_kinds.at(static_cast<size_t>(layer)) == MixerKind::Attention) {
                return layer;
            }
        }
        return -1;
    }();
    if (full_attention_layer < 0) {
        throw std::runtime_error("MTP requires a full-attention target layer");
    }
    const AttentionSpec& attention = shape.attention_layout(full_attention_layer);
    std::size_t bytes = bf16_bytes(2ull * shape.hidden * shape.hidden);
    bytes += bf16_bytes(3ull * shape.hidden); // two pre-fc norms and final norm
    for (int layer = 0; layer < shape.mtp_num_hidden_layers; ++layer) {
        bytes += bf16_bytes(2ull * shape.hidden); // input and post-attention norms
        bytes += bf16_bytes(static_cast<size_t>(attention.query_projection_width()) * shape.hidden);
        bytes += bf16_bytes(2ull * static_cast<size_t>(attention.key_value_width()) * shape.hidden);
        bytes += bf16_bytes(static_cast<size_t>(shape.hidden) * attention.query_width());
        if (attention.query_key_norm) bytes += bf16_bytes(2ull * attention.head_dim);
        if (shape.num_experts > 0) {
            bytes += bf16_bytes(static_cast<size_t>(shape.num_experts) * shape.hidden);
            bytes += static_cast<size_t>(shape.num_experts) * shape.hidden * sizeof(float);
            if (shape.shared_expert_intermediate > 0) {
                const size_t shared = static_cast<size_t>(shape.shared_expert_intermediate);
                bytes += bf16_bytes(3ull * shared * shape.hidden + shape.hidden);
            }
        } else {
            bytes += bf16_bytes(3ull * static_cast<size_t>(shape.intermediate) * shape.hidden);
        }
    }
    return bytes;
}

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
    const int moe_layers = moe_layer_count(resources.shape_) +
        (resources.options_.enable_mtp && resources.shape_.num_experts > 0
            ? resources.shape_.mtp_num_hidden_layers : 0);
    const size_t expert_bytes = bytes_per_expert_bf16(resources.shape_);
    ExpertOffloadPlanInputs inputs;
    inputs.shape = resources.shape_;
    inputs.options = resources.options_.expert_offload;
    inputs.gpu_free_bytes = free_bytes;
    inputs.non_expert_weight_bytes = estimate_non_expert_weights(
        resources.shape_, resources.model_.capabilities.tied_embeddings) +
        (resources.options_.enable_mtp
            ? estimate_mtp_non_expert_weights(resources.shape_) : 0) +
        (64ull << 20);
    inputs.workspace_bytes = resources.options_.lt_workspace_bytes + (256ull << 20);
    inputs.context_tokens = model.max_context_;
    if (resources.options_.enable_mtp) {
        inputs.extra_moe_layers = resources.shape_.num_experts > 0
            ? resources.shape_.mtp_num_hidden_layers : 0;
        const int full_attention_layer = [&]() {
            for (int layer = resources.shape_.num_hidden_layers - 1; layer >= 0; --layer) {
                if (resources.shape_.mixer_kinds.at(static_cast<size_t>(layer)) == MixerKind::Attention) {
                    return layer;
                }
            }
            return -1;
        }();
        if (full_attention_layer >= 0) {
            const AttentionSpec& attention = resources.shape_.attention_layout(full_attention_layer);
            inputs.extra_kv_reservation_bytes = static_cast<size_t>(
                resources.shape_.mtp_num_hidden_layers) * 2ull *
                static_cast<size_t>(attention.key_value_width()) *
                sizeof(__nv_bfloat16) * static_cast<size_t>(model.max_context_);
        }
    }
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
