#include "celeg/backend/cuda/weight_setup_support.hpp"

#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/moe/offload.hpp"

#include <cstdio>
#include <algorithm>
#include <memory>
#include <stdexcept>

namespace celeg {
namespace {

std::size_t bf16_bytes(std::size_t elements) {
    return elements * sizeof(__nv_bfloat16);
}

std::size_t estimate_non_expert_weights(const CheckpointDimensions& dims,
                                        const CompiledModelProgram& program,
                                        bool tied_embeddings) {
    std::size_t bytes = bf16_bytes(static_cast<std::size_t>(dims.vocab_size) *
                                   static_cast<std::size_t>(program.hidden));
    if (!tied_embeddings) bytes += bytes;
    bytes += bf16_bytes(static_cast<std::size_t>(program.hidden));

    for (const CompiledLayerProgram& layer : program.layers) {
        bytes += bf16_bytes(static_cast<std::size_t>(program.hidden));
        if (!std::holds_alternative<std::monostate>(layer.feed_forward)) {
            bytes += bf16_bytes(static_cast<std::size_t>(program.hidden));
        }
        if (layer.post_attention_norm.enabled()) {
            bytes += bf16_bytes(static_cast<std::size_t>(program.hidden));
        }
        if (layer.post_feed_forward_norm.enabled()) {
            bytes += bf16_bytes(static_cast<std::size_t>(program.hidden));
        }

        if (const auto* compiled = std::get_if<CompiledAttentionProgram>(&layer.mixer)) {
            const AttentionSpec& attention = compiled->semantics;
            bytes += bf16_bytes(static_cast<std::size_t>(
                attention.query_projection_width()) * program.hidden);
            if (!attention.kv_sharing.shared() || attention.kv_sharing.publishes) {
                bytes += bf16_bytes(2ull * static_cast<std::size_t>(
                    attention.key_value_width()) * program.hidden);
            }
            bytes += bf16_bytes(static_cast<std::size_t>(program.hidden) *
                                attention.query_width());
            if (attention.has_query_key_norm()) {
                bytes += bf16_bytes(2ull * static_cast<std::size_t>(attention.head_dim));
            }
        } else if (const auto* spec = std::get_if<GatedDeltaNetSpec>(&layer.mixer)) {
            const std::size_t key_width = static_cast<std::size_t>(spec->key_heads) *
                                          spec->key_head_dim;
            const std::size_t value_width = static_cast<std::size_t>(spec->value_heads) *
                                            spec->value_head_dim;
            const std::size_t qkv_width = 2ull * key_width + value_width;
            bytes += bf16_bytes(qkv_width * program.hidden);
            bytes += bf16_bytes(value_width * program.hidden);
            bytes += bf16_bytes(2ull * static_cast<std::size_t>(spec->value_heads) *
                                program.hidden);
            bytes += bf16_bytes(static_cast<std::size_t>(program.hidden) * value_width);
            bytes += bf16_bytes(qkv_width * spec->conv_kernel +
                                2ull * static_cast<std::size_t>(spec->value_heads) +
                                static_cast<std::size_t>(spec->value_head_dim));
        }

        if (const auto* moe = std::get_if<MoeLayerProgram>(&layer.feed_forward)) {
            const std::size_t router_elements = static_cast<std::size_t>(
                moe->router.expert_count) * program.hidden;
            bytes += bf16_bytes(router_elements) + router_elements * sizeof(float);
            if (moe->shared) {
                const std::size_t shared = static_cast<std::size_t>(
                    moe->shared->mlp.intermediate_size);
                bytes += bf16_bytes(3ull * shared * program.hidden + program.hidden);
            }
        } else if (const auto* dense =
                       std::get_if<CompiledDenseFeedForwardProgram>(&layer.feed_forward)) {
            if (dense->intermediate_size <= 0) {
                throw std::runtime_error(
                    "compiled dense layer has no FFN width for weight estimate");
            }
            bytes += bf16_bytes(3ull * static_cast<std::size_t>(
                dense->intermediate_size) * program.hidden);
        }
    }
    return bytes;
}

std::size_t estimate_mtp_non_expert_weights(const CheckpointDimensions& dims,
                                            const CompiledModelProgram& program) {
    if (dims.mtp_num_hidden_layers <= 0) return 0;
    const int full_attention_layer = [&]() {
        for (int layer = static_cast<int>(program.layers.size()) - 1; layer >= 0; --layer) {
            if (std::holds_alternative<CompiledAttentionProgram>(
                    program.layers.at(static_cast<size_t>(layer)).mixer)) {
                return layer;
            }
        }
        return -1;
    }();
    if (full_attention_layer < 0) {
        throw std::runtime_error("MTP requires a full-attention target layer");
    }
    const AttentionSpec& attention = std::get<CompiledAttentionProgram>(
        program.layers.at(static_cast<size_t>(full_attention_layer)).mixer).semantics;
    const auto moe_layer = std::find_if(
        program.layers.begin(), program.layers.end(),
        [](const CompiledLayerProgram& layer) {
            return std::holds_alternative<MoeLayerProgram>(layer.feed_forward);
        });
    const auto dense_layer = std::find_if(
        program.layers.begin(), program.layers.end(),
        [](const CompiledLayerProgram& layer) {
            return std::holds_alternative<CompiledDenseFeedForwardProgram>(
                layer.feed_forward);
        });
    std::size_t bytes = bf16_bytes(2ull * program.hidden * program.hidden);
    bytes += bf16_bytes(3ull * program.hidden);
    for (int layer = 0; layer < dims.mtp_num_hidden_layers; ++layer) {
        bytes += bf16_bytes(2ull * program.hidden);
        bytes += bf16_bytes(static_cast<size_t>(attention.query_projection_width()) *
                            program.hidden);
        bytes += bf16_bytes(2ull * static_cast<size_t>(attention.key_value_width()) *
                            program.hidden);
        bytes += bf16_bytes(static_cast<size_t>(program.hidden) * attention.query_width());
        if (attention.has_query_key_norm()) {
            bytes += bf16_bytes(2ull * static_cast<size_t>(attention.head_dim));
        }
        if (moe_layer != program.layers.end()) {
            const auto& moe = std::get<MoeLayerProgram>(moe_layer->feed_forward);
            const int experts = moe.router.expert_count;
            bytes += bf16_bytes(static_cast<size_t>(experts) * program.hidden);
            bytes += static_cast<size_t>(experts) * program.hidden * sizeof(float);
            if (moe.shared) {
                const size_t shared = static_cast<size_t>(moe.shared->mlp.intermediate_size);
                bytes += bf16_bytes(3ull * shared * program.hidden + program.hidden);
            }
        } else {
            if (dense_layer == program.layers.end()) {
                throw std::runtime_error("MTP requires a compiled dense FFN width");
            }
            const int intermediate = std::get<CompiledDenseFeedForwardProgram>(
                dense_layer->feed_forward).intermediate_size;
            if (intermediate <= 0) {
                throw std::runtime_error("MTP requires a compiled dense FFN width");
            }
            bytes += bf16_bytes(3ull * static_cast<std::size_t>(intermediate) *
                                program.hidden);
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

}

void configure_cuda_expert_resources(CudaCompiledModel& model) {
    CudaModelResources& resources = model.resources_;
    CudaWorkspace& workspace = model.workspace_;
    if (!resources.options_.expert_offload.enabled() ||
        !resources.program_.has_moe()) {
        return;
    }
    int maximum_experts = 0;
    int maximum_moe_intermediate = 0;
    for (const CompiledLayerProgram& layer : resources.program_.layers) {
        const auto* moe = std::get_if<MoeLayerProgram>(&layer.feed_forward);
        if (!moe) continue;
        maximum_experts = std::max(maximum_experts, moe->router.expert_count);
        maximum_moe_intermediate = std::max(
            maximum_moe_intermediate, moe->routed.mlp.intermediate_size);
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
        resources.dims_, resources.program_, resources.model_.capabilities.tied_embeddings) +
        (resources.options_.enable_mtp
            ? estimate_mtp_non_expert_weights(resources.dims_, resources.program_) : 0) +
        (64ull << 20);
    inputs.workspace_bytes = resources.options_.lt_workspace_bytes + (256ull << 20);
    inputs.context_tokens = model.max_context_;
    if (resources.options_.enable_mtp) {
        inputs.extra_moe_layers = resources.program_.has_moe()
            ? resources.dims_.mtp_num_hidden_layers : 0;
        const int full_attention_layer = [&]() {
            for (int layer = static_cast<int>(resources.program_.layers.size()) - 1;
                 layer >= 0; --layer) {
                if (std::holds_alternative<CompiledAttentionProgram>(
                        resources.program_.layers.at(static_cast<size_t>(layer)).mixer)) {
                    return layer;
                }
            }
            return -1;
        }();
        if (full_attention_layer >= 0) {
            const AttentionSpec& attention = std::get<CompiledAttentionProgram>(
                resources.program_.layers.at(
                    static_cast<size_t>(full_attention_layer)).mixer).semantics;
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
                          moe_layers, maximum_experts,
                          maximum_moe_intermediate, resources.program_.hidden)) {
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
                    maximum_experts)) {
                resources.weights_->usage_stats.layers.assign(
                    static_cast<size_t>(moe_layers),
                    std::vector<ExpertUsageEntry>(
                        static_cast<size_t>(maximum_experts)));
            } else {
                std::fprintf(stderr,
                             "Loaded persistent expert usage statistics from %s\n",
                             resources.weights_->usage_profile_path.c_str());
            }
        }
    }
    std::fprintf(stderr, "%s", workspace.expert_offload_plan_.report().c_str());
}

}
