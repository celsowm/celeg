#include "celeg/backend/cuda/weight_setup_support.hpp"

#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/moe/offload.hpp"

#include <cstdio>
#include <memory>
#include <stdexcept>

namespace celeg {
namespace {

std::size_t bf16_bytes(std::size_t elements) {
    return elements * sizeof(__nv_bfloat16);
}

std::size_t estimate_non_expert_weights(const ExecutionTopology& shape,
                                        const CheckpointDimensions& dims,
                                        const CompiledModelProgram& program,
                                        bool tied_embeddings) {
    std::size_t bytes = bf16_bytes(static_cast<std::size_t>(dims.vocab_size) *
                                   static_cast<std::size_t>(shape.hidden));
    if (!tied_embeddings) bytes += bytes;
    bytes += bf16_bytes(static_cast<std::size_t>(shape.hidden));

    for (int layer = 0; layer < static_cast<int>(program.layers.size()); ++layer) {
        // input_layernorm and the FFN input norm are present in all normal
        // decoder blocks. Split-norm families add two more vectors below.
        bytes += bf16_bytes(2ull * static_cast<std::size_t>(shape.hidden));
        const CompiledLayerProgram& layer_program = program.layers.at(
            static_cast<size_t>(layer));
        if (layer_program.post_attention_norm.enabled()) {
            bytes += bf16_bytes(static_cast<std::size_t>(shape.hidden));
        }
        if (layer_program.post_feed_forward_norm.enabled()) {
            bytes += bf16_bytes(static_cast<std::size_t>(shape.hidden));
        }

        if (layer_program.mixer == CompiledMixer::Attention) {
            const AttentionSpec& attention = layer_program.attention.value();
            bytes += bf16_bytes(static_cast<std::size_t>(attention.query_projection_width()) * shape.hidden);
            if (!attention.kv_sharing.shared() || attention.kv_sharing.publishes) {
                bytes += bf16_bytes(2ull * static_cast<std::size_t>(attention.key_value_width()) * shape.hidden);
            }
            bytes += bf16_bytes(static_cast<std::size_t>(shape.hidden) * attention.query_width());
            if (attention.has_query_key_norm()) {
                bytes += bf16_bytes(2ull * static_cast<std::size_t>(attention.head_dim));
            }
        } else if (layer_program.mixer == CompiledMixer::GatedDeltaNet) {
            const GatedDeltaNetSpec& spec = layer_program.gated_delta_net.value();
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

        if (layer_program.moe) {
            const std::size_t router_elements = static_cast<std::size_t>(
                layer_program.moe->router.expert_count) * shape.hidden;
            bytes += bf16_bytes(router_elements) + router_elements * sizeof(float);
            if (layer_program.moe->shared) {
                const std::size_t shared = static_cast<std::size_t>(
                    layer_program.moe->shared->mlp.intermediate_size);
                bytes += bf16_bytes(3ull * shared * shape.hidden + shape.hidden);
            }
        } else {
            const int intermediate = layer_program.feed_forward_intermediate;
            if (intermediate <= 0) {
                throw std::runtime_error("compiled layer has no FFN width for weight estimate");
            }
            bytes += bf16_bytes(3ull * static_cast<std::size_t>(intermediate) * shape.hidden);
        }
    }
    return bytes;
}

std::size_t estimate_mtp_non_expert_weights(const ExecutionTopology& shape,
                                            const CheckpointDimensions& dims,
                                            const CompiledModelProgram& program) {
    if (dims.mtp_num_hidden_layers <= 0) return 0;
    const auto bf16_bytes = [](std::size_t elements) {
        return elements * sizeof(__nv_bfloat16);
    };
    const int full_attention_layer = [&]() {
        for (int layer = static_cast<int>(program.layers.size()) - 1; layer >= 0; --layer) {
            if (program.layers.at(static_cast<size_t>(layer)).mixer == CompiledMixer::Attention) {
                return layer;
            }
        }
        return -1;
    }();
    if (full_attention_layer < 0) {
        throw std::runtime_error("MTP requires a full-attention target layer");
    }
    const AttentionSpec& attention = program.layers.at(
        static_cast<size_t>(full_attention_layer)).attention.value();
    std::size_t bytes = bf16_bytes(2ull * shape.hidden * shape.hidden);
    bytes += bf16_bytes(3ull * shape.hidden); // two pre-fc norms and final norm
    for (int layer = 0; layer < dims.mtp_num_hidden_layers; ++layer) {
        bytes += bf16_bytes(2ull * shape.hidden); // input and post-attention norms
        bytes += bf16_bytes(static_cast<size_t>(attention.query_projection_width()) * shape.hidden);
        bytes += bf16_bytes(2ull * static_cast<size_t>(attention.key_value_width()) * shape.hidden);
        bytes += bf16_bytes(static_cast<size_t>(shape.hidden) * attention.query_width());
        if (attention.has_query_key_norm()) bytes += bf16_bytes(2ull * attention.head_dim);
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
        !resources.program_.has_moe()) {
        return;
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CELEG_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    const int moe_layers = moe_layer_count(resources.program_) +
        (resources.options_.enable_mtp && resources.program_.has_moe()
            ? resources.dims_.mtp_num_hidden_layers : 0);
    const size_t expert_bytes = bytes_per_expert_bf16(resources.program_);
    ExpertOffloadPlanInputs inputs(resources.program_);
    inputs.options = resources.options_.expert_offload;
    inputs.gpu_free_bytes = free_bytes;
    inputs.non_expert_weight_bytes = estimate_non_expert_weights(
        resources.shape_, resources.dims_, resources.program_, resources.model_.capabilities.tied_embeddings) +
        (resources.options_.enable_mtp
            ? estimate_mtp_non_expert_weights(resources.shape_, resources.dims_,
                                              resources.program_) : 0) +
        (64ull << 20);
    inputs.workspace_bytes = resources.options_.lt_workspace_bytes + (256ull << 20);
    inputs.context_tokens = model.max_context_;
    if (resources.options_.enable_mtp) {
        inputs.extra_moe_layers = resources.program_.has_moe()
            ? resources.dims_.mtp_num_hidden_layers : 0;
        const int full_attention_layer = [&]() {
            for (int layer = static_cast<int>(resources.program_.layers.size()) - 1;
                 layer >= 0; --layer) {
                if (resources.program_.layers.at(static_cast<size_t>(layer)).mixer ==
                    CompiledMixer::Attention) {
                    return layer;
                }
            }
            return -1;
        }();
        if (full_attention_layer >= 0) {
            const AttentionSpec& attention = resources.program_.layers.at(
                static_cast<size_t>(full_attention_layer)).attention.value();
            inputs.extra_kv_reservation_bytes = static_cast<size_t>(
                resources.dims_.mtp_num_hidden_layers) * 2ull *
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
