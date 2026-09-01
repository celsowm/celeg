#include "attention_weight_setup.hpp"

#include "attention_state_setup.hpp"
#include "detail/compiled_model.hpp"
#include "backend/cuda/weight_setup_support.hpp"

#include <stdexcept>
#include <string>

namespace celeg {
namespace {

int attention_norm_width(const NormSpec& norm, int heads, int head_dim) {
    return norm.granularity == NormGranularity::PerHead
        ? head_dim
        : heads * head_dim;
}

}

bool bind_cuda_attention_layer(CudaCompiledModel& model,
                               const IWeightRepository& repo,
                               const CompiledLayerProgram& semantics,
                               int layer_index,
                               const LayerCommon& common_layer,
                               std::vector<int>& shared_owner) {
    const auto* compiled_attention =
        std::get_if<CompiledAttentionProgram>(&semantics.mixer);
    if (!compiled_attention) return false;

    CudaModelResources& resources = model.resources_;
    AttentionLayer attention_layer;
    attention_layer.common = common_layer;
    attention_layer.layout = compiled_attention->semantics;
    const AttentionSpec& layout = attention_layer.layout;

    if (const auto* alibi = std::get_if<AlibiBiasSpec>(&layout.bias)) {
        if (static_cast<int>(alibi->slopes.size()) != layout.query_heads) {
            throw std::invalid_argument(
                "CUDA ALiBi slope count does not match query heads");
        }
        attention_layer.alibi_slopes.reset(alibi->slopes.size());
        CELEG_CUDA(cudaMemcpy(
            attention_layer.alibi_slopes.data(), alibi->slopes.data(),
            attention_layer.alibi_slopes.bytes(), cudaMemcpyHostToDevice));
    }

    if (layout.uses_latent_state()) {
        if (resources.options().kv_cache_mode == KvCacheMode::Int8) {
            throw std::invalid_argument(
                "CUDA latent attention currently requires BF16 state storage");
        }
        const auto& latent = *layout.latent_state();
        const auto* factorized = latent.factorized_projection();
        const bool owns_latent_state =
            !std::holds_alternative<SharedKvConsumer>(layout.kv_sharing);
        if (factorized) {
            attention_layer.latent_query_projection =
                resources.weight_loader_->load_linear_weight(
                    repo,
                    cuda_tensor_name(resources.model_.weight_plan.requests,
                                     TensorRole::AttentionLatentQueryProjection,
                                     layer_index),
                    {factorized->query_rank, resources.program_.hidden});
            attention_layer.latent_query_expansion =
                resources.weight_loader_->load_linear_weight(
                    repo,
                    cuda_tensor_name(resources.model_.weight_plan.requests,
                                     TensorRole::AttentionLatentQueryExpansion,
                                     layer_index),
                    {layout.query_heads *
                         (latent.nope_head_dim + latent.rope_head_dim),
                     factorized->query_rank});
            attention_layer.latent_query_norm =
                resources.weight_loader_->load_rms_norm_weight(
                    repo,
                    cuda_tensor_name(resources.model_.weight_plan.requests,
                                     TensorRole::AttentionLatentQueryNorm,
                                     layer_index),
                    {factorized->query_rank},
                    factorized->query_latent_norm.weight_kind);
            attention_layer.latent_key_projection =
                resources.weight_loader_->load_linear_weight(
                    repo,
                    cuda_tensor_name(resources.model_.weight_plan.requests,
                                     TensorRole::AttentionLatentKeyProjection,
                                     layer_index),
                    {latent.latent_rank + latent.rope_head_dim,
                     resources.program_.hidden});
            attention_layer.latent_key_norm =
                resources.weight_loader_->load_rms_norm_weight(
                    repo,
                    cuda_tensor_name(resources.model_.weight_plan.requests,
                                     TensorRole::AttentionLatentKeyNorm,
                                     layer_index),
                    {latent.latent_rank},
                    factorized->key_latent_norm.weight_kind);
            attention_layer.latent_expansion =
                resources.weight_loader_->load_linear_weight(
                    repo,
                    cuda_tensor_name(resources.model_.weight_plan.requests,
                                     TensorRole::AttentionLatentExpansion,
                                     layer_index),
                    {layout.query_heads *
                         (latent.nope_head_dim + factorized->value_head_dim),
                     latent.latent_rank});
            attention_layer.gate = resources.weight_loader_->load_linear_weight(
                repo,
                cuda_tensor_name(resources.model_.weight_plan.requests,
                                 TensorRole::AttentionGate, layer_index),
                {layout.output_gate_width(), resources.program_.hidden});
            attention_layer.out = resources.weight_loader_->load_linear_weight(
                repo,
                cuda_tensor_name(resources.model_.weight_plan.requests,
                                 TensorRole::AttentionLatentOutput, layer_index),
                {resources.program_.hidden, layout.latent_output_width()});
            if (!std::holds_alternative<Bf16LinearStorage>(
                    attention_layer.latent_query_expansion->storage) ||
                !std::holds_alternative<Bf16LinearStorage>(
                    attention_layer.latent_expansion->storage)) {
                throw std::invalid_argument(
                    "CUDA factorized latent attention currently requires BF16 expansion weights");
            }
        } else {
            attention_layer.latent_query =
                resources.weight_loader_->load_linear_weight(
                    repo,
                    cuda_tensor_name(resources.model_.weight_plan.requests,
                                     TensorRole::AttentionLatentQuery,
                                     layer_index),
                    {layout.latent_query_content_width(), resources.program_.hidden});
            if (layout.latent_query_rope_width() != 0) {
                attention_layer.latent_query_rope =
                    resources.weight_loader_->load_linear_weight(
                        repo,
                        cuda_tensor_name(resources.model_.weight_plan.requests,
                                         TensorRole::AttentionLatentQueryRope,
                                         layer_index),
                        {layout.latent_query_rope_width(), resources.program_.hidden});
            }
            if (owns_latent_state) {
                attention_layer.latent_key =
                    resources.weight_loader_->load_linear_weight(
                        repo,
                        cuda_tensor_name(resources.model_.weight_plan.requests,
                                         TensorRole::AttentionLatentKey,
                                         layer_index),
                        {latent.latent_rank, resources.program_.hidden});
                attention_layer.latent_value =
                    resources.weight_loader_->load_linear_weight(
                        repo,
                        cuda_tensor_name(resources.model_.weight_plan.requests,
                                         TensorRole::AttentionLatentValue,
                                         layer_index),
                        {latent.latent_rank, resources.program_.hidden});
                if (latent.decoupled_rope && latent.rope_head_dim != 0) {
                    attention_layer.latent_key_rope =
                        resources.weight_loader_->load_linear_weight(
                            repo,
                            cuda_tensor_name(resources.model_.weight_plan.requests,
                                             TensorRole::AttentionLatentKeyRope,
                                             layer_index),
                            {latent.rope_head_dim, resources.program_.hidden});
                }
            }
            attention_layer.out = resources.weight_loader_->load_linear_weight(
                repo,
                cuda_tensor_name(resources.model_.weight_plan.requests,
                                 TensorRole::AttentionLatentOutput, layer_index),
                {resources.program_.hidden, layout.latent_query_content_width()});
        }

        attention_layer.state = LatentAttentionRuntimeState{};
        if (resources.options().allocate_local_kv_cache && owns_latent_state) {
            auto& latent_state =
                std::get<LatentAttentionRuntimeState>(attention_layer.state);
            latent_state.latent_key_cache.reset(
                static_cast<size_t>(model.max_context_) * latent.latent_rank);
            latent_state.latent_value_cache.reset(
                static_cast<size_t>(model.max_context_) * latent.latent_rank);
            if (latent.decoupled_rope && latent.rope_head_dim != 0) {
                latent_state.latent_key_rope_cache.reset(
                    static_cast<size_t>(model.max_context_) * latent.rope_head_dim);
            }
        }
        if (kv_sharing_publishes(layout.kv_sharing)) {
            shared_owner[static_cast<size_t>(kv_sharing_group(layout.kv_sharing))] =
                layer_index;
            attention_layer.kv_owner_layer = layer_index;
        }
        if (!kv_sharing_shared(layout.kv_sharing)) {
            attention_layer.kv_owner_layer = layer_index;
        }
        resources.layers_.emplace_back(std::move(attention_layer));
        return true;
    }

    attention_layer.query = resources.weight_loader_->load_linear_weight(
        repo,
        cuda_tensor_name(resources.model_.weight_plan.requests,
                         TensorRole::AttentionQuery, layer_index),
        {layout.query_projection_width(), resources.program_.hidden});
    if (layout.output_gate.has_value() && !layout.output_gate->packed_with_query) {
        attention_layer.gate = resources.weight_loader_->load_linear_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::AttentionGate, layer_index),
            {layout.query_width(), resources.program_.hidden});
    }
    if (!std::holds_alternative<SharedKvConsumer>(layout.kv_sharing)) {
        attention_layer.key = resources.weight_loader_->load_linear_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::AttentionKey, layer_index),
            {layout.key_value_width(), resources.program_.hidden});
        attention_layer.value = resources.weight_loader_->load_linear_weight(
            repo,
            cuda_tensor_name(resources.model_.weight_plan.requests,
                             TensorRole::AttentionValue, layer_index),
            {layout.key_value_width(), resources.program_.hidden});
    } else {
        const int group = kv_sharing_group(layout.kv_sharing);
        if (group < 0 || group >= static_cast<int>(shared_owner.size()) ||
            shared_owner[static_cast<size_t>(group)] < 0) {
            throw std::runtime_error("CUDA shared KV consumer has no owner");
        }
        attention_layer.kv_owner_layer = shared_owner[static_cast<size_t>(group)];
    }
    attention_layer.out = resources.weight_loader_->load_linear_weight(
        repo,
        cuda_tensor_name(resources.model_.weight_plan.requests,
                         TensorRole::AttentionOutput, layer_index),
        {resources.program_.hidden, layout.query_width()});
    if (layout.query_norm.has_value()) {
        const int query_norm_width = attention_norm_width(
            *layout.query_norm, layout.query_heads, layout.head_dim);
        attention_layer.q_norm = resources.weight_loader_->load_rms_norm_weight(
            repo,
            layout.query_norm->weightless()
                ? std::string{}
                : cuda_tensor_name(resources.model_.weight_plan.requests,
                                   TensorRole::AttentionQueryNorm, layer_index),
            {query_norm_width}, layout.query_norm->weight_kind);
    }
    if (layout.key_norm.has_value() && attention_layer.key) {
        const int key_norm_width = attention_norm_width(
            *layout.key_norm, layout.key_value_heads, layout.head_dim);
        attention_layer.k_norm = resources.weight_loader_->load_rms_norm_weight(
            repo,
            layout.key_norm->weightless()
                ? std::string{}
                : cuda_tensor_name(resources.model_.weight_plan.requests,
                                   TensorRole::AttentionKeyNorm, layer_index),
            {key_norm_width}, layout.key_norm->weight_kind);
    }

    initialize_cuda_ordinary_attention_state(
        attention_layer, layout, resources.options().kv_cache_mode,
        model.max_context_,
        resources.options().allocate_local_kv_cache && attention_layer.key);
    if (kv_sharing_publishes(layout.kv_sharing)) {
        shared_owner[static_cast<size_t>(kv_sharing_group(layout.kv_sharing))] =
            layer_index;
        attention_layer.kv_owner_layer = layer_index;
    }
    if (!kv_sharing_shared(layout.kv_sharing)) {
        attention_layer.kv_owner_layer = layer_index;
    }
    resources.layers_.emplace_back(std::move(attention_layer));
    return true;
}

}
