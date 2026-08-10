#include "weight_requirements.hpp"

#include <optional>
#include <utility>

namespace celeg::descriptor_detail {

void append_attention_weight_requirements(ResolvedModel& model,
                                          const RuntimeTopology& topology,
                                          int layer, int physical_layer) {
    const AttentionSpec& attention = topology.attention_layout(layer);
    const int query_width = attention.query_width();
    const int key_value_width = attention.key_value_width();
    const auto append = [&](TensorRole role, std::vector<int64_t> shape) {
        model.weight_plan.requests.push_back(
            {role, layer, -1, std::move(shape), std::nullopt, physical_layer});
    };
    if (attention.uses_external_memory()) {
        append(TensorRole::AttentionQuery,
               {attention.query_projection_width(), topology.hidden});
        append(TensorRole::AttentionOutput,
               {topology.hidden, attention.query_width()});
    } else if (attention.uses_latent_state()) {
        append(TensorRole::AttentionLatentQuery,
               {attention.latent_query_content_width(), topology.hidden});
        if (attention.latent_query_rope_width() != 0) {
            append(TensorRole::AttentionLatentQueryRope,
                   {attention.latent_query_rope_width(), topology.hidden});
        }
        const auto& latent = *attention.latent_state();
        append(TensorRole::AttentionLatentKey, {latent.latent_rank, topology.hidden});
        append(TensorRole::AttentionLatentValue, {latent.latent_rank, topology.hidden});
        if (latent.decoupled_rope && latent.rope_head_dim != 0) {
            append(TensorRole::AttentionLatentKeyRope,
                   {latent.rope_head_dim, topology.hidden});
        }
        append(TensorRole::AttentionLatentOutput,
               {topology.hidden, attention.latent_query_content_width()});
    } else {
        append(TensorRole::AttentionQuery, {query_width, topology.hidden});
        if (attention.query_norm.enabled()) {
            append(TensorRole::AttentionQueryNorm, {attention.head_dim});
        }
        if (!attention.kv_sharing.shared() || attention.kv_sharing.publishes) {
            append(TensorRole::AttentionKey, {key_value_width, topology.hidden});
            append(TensorRole::AttentionValue, {key_value_width, topology.hidden});
            if (attention.key_norm.enabled()) {
                append(TensorRole::AttentionKeyNorm, {attention.head_dim});
            }
        }
        append(TensorRole::AttentionOutput, {topology.hidden, query_width});
    }
    if (attention.output_gate.enabled() && !attention.output_gate.packed_with_query) {
        append(TensorRole::AttentionGate,
               {attention.query_width(), topology.hidden});
    }
    if (const auto* relative =
            std::get_if<RelativePositionBiasSpec>(&attention.bias)) {
        append(TensorRole::AttentionRelativePositionBias,
               {attention.query_heads * relative->bucket_count});
    }
    const LayerSpec& layer_spec = model.graph.layers.at(static_cast<size_t>(layer));
    if (topology.has_split_attention_norms) {
        if (layer_spec.post_attention_norm.enabled() &&
            !layer_spec.post_attention_norm.weightless()) {
            model.weight_plan.requests.push_back(
                {TensorRole::AttentionPostNorm, layer, -1, {topology.hidden},
                 std::nullopt, physical_layer, layer_spec.post_attention_norm.weight_kind});
        }
        if (layer_spec.post_feed_forward_norm.enabled() &&
            !layer_spec.post_feed_forward_norm.weightless()) {
            model.weight_plan.requests.push_back(
                {TensorRole::FfnOutputNorm, layer, -1, {topology.hidden},
                 std::nullopt, physical_layer, layer_spec.post_feed_forward_norm.weight_kind});
        }
    }
}

void append_short_conv_weight_requirements(ResolvedModel& model,
                                           const RuntimeTopology& topology,
                                           int layer, int physical_layer) {
    const auto append = [&](TensorRole role, std::vector<int64_t> shape) {
        model.weight_plan.requests.push_back(
            {role, layer, -1, std::move(shape), std::nullopt, physical_layer});
    };
    append(TensorRole::ShortConvInput, {3 * topology.hidden, topology.hidden});
    append(TensorRole::ShortConvKernel,
           {topology.hidden, 1, topology.conv_cache});
    append(TensorRole::ShortConvOutput, {topology.hidden, topology.hidden});
}

void append_gated_delta_net_weight_requirements(ResolvedModel& model,
                                                const RuntimeTopology& topology,
                                                int layer, int physical_layer) {
    const auto& recurrent = topology.gated_delta_net_layouts.at(static_cast<size_t>(layer));
    const int qkv = 2 * recurrent.key_heads * recurrent.key_head_dim +
        recurrent.value_heads * recurrent.value_head_dim;
    const auto append = [&](TensorRole role, std::vector<int64_t> shape) {
        model.weight_plan.requests.push_back(
            {role, layer, -1, std::move(shape), std::nullopt, physical_layer});
    };
    append(TensorRole::GatedDeltaNetQkv, {qkv, topology.hidden});
    append(TensorRole::GatedDeltaNetZ,
           {recurrent.value_heads * recurrent.value_head_dim, topology.hidden});
    append(TensorRole::GatedDeltaNetAlpha, {recurrent.value_heads, topology.hidden});
    append(TensorRole::GatedDeltaNetBeta, {recurrent.value_heads, topology.hidden});
    append(TensorRole::GatedDeltaNetDtBias, {recurrent.value_heads});
    append(TensorRole::GatedDeltaNetALog, {recurrent.value_heads});
    append(TensorRole::GatedDeltaNetConv, {qkv, 1, recurrent.conv_kernel});
    append(TensorRole::GatedDeltaNetNorm, {recurrent.value_head_dim});
    append(TensorRole::GatedDeltaNetOutput,
           {topology.hidden, recurrent.value_heads * recurrent.value_head_dim});
}

void append_mamba2_weight_requirements(ResolvedModel& model,
                                       const RuntimeTopology& topology,
                                       int layer, int physical_layer) {
    const auto& recurrent = topology.mamba2_layouts.at(static_cast<size_t>(layer));
    const int conv_dim = recurrent.intermediate_size +
        2 * recurrent.group_count * recurrent.state_size;
    const auto append = [&](TensorRole role, std::vector<int64_t> shape) {
        model.weight_plan.requests.push_back(
            {role, layer, -1, std::move(shape), std::nullopt, physical_layer});
    };
    append(TensorRole::Mamba2Input,
           {2 * recurrent.intermediate_size +
            2 * recurrent.group_count * recurrent.state_size +
            recurrent.num_heads, topology.hidden});
    append(TensorRole::Mamba2Conv, {conv_dim, 1, recurrent.conv_kernel});
    append(TensorRole::Mamba2ConvBias, {conv_dim});
    append(TensorRole::Mamba2DtBias, {recurrent.num_heads});
    append(TensorRole::Mamba2ALog, {recurrent.num_heads});
    append(TensorRole::Mamba2D, {recurrent.num_heads});
    append(TensorRole::Mamba2Norm, {recurrent.intermediate_size});
    append(TensorRole::Mamba2Output,
           {topology.hidden, recurrent.intermediate_size});
}

void append_dense_ffn_weight_requirements(ResolvedModel& model,
                                          const RuntimeTopology& topology,
                                          int layer, int physical_layer,
                                          int intermediate) {
    const auto append = [&](TensorRole role, std::vector<int64_t> shape) {
        model.weight_plan.requests.push_back(
            {role, layer, -1, std::move(shape), std::nullopt, physical_layer});
    };
    append(TensorRole::FfnGate, {intermediate, topology.hidden});
    append(TensorRole::FfnUp, {intermediate, topology.hidden});
    append(TensorRole::FfnDown, {topology.hidden, intermediate});
    if (topology.has_split_attention_norms) {
        append(TensorRole::FfnOutputNorm, {topology.hidden});
    }
}

void append_mlp_only_weight_requirements(ResolvedModel& model,
                                          const RuntimeTopology& topology,
                                          int layer, int physical_layer) {
    const auto& mlp = topology.mlp_only_layouts.at(static_cast<size_t>(layer));
    const auto append = [&](TensorRole role, std::vector<int64_t> shape) {
        model.weight_plan.requests.push_back(
            {role, layer, -1, std::move(shape), std::nullopt, physical_layer});
    };
    append(TensorRole::FfnUp, {mlp.intermediate_size, topology.hidden});
    append(TensorRole::FfnDown, {topology.hidden, mlp.intermediate_size});
}

void append_moe_weight_requirements(ResolvedModel& model,
                                    const RuntimeTopology& topology,
                                    int layer, int physical_layer) {
    const auto append = [&](TensorRole role, int expert, std::vector<int64_t> shape) {
        model.weight_plan.requests.push_back(
            {role, layer, expert, std::move(shape), std::nullopt, physical_layer});
    };
    append(TensorRole::MoeRouter, -1, {topology.num_experts, topology.hidden});
    if (topology.shared_expert_intermediate > 0) {
        append(TensorRole::MoePackedGateUp, -1,
               {topology.num_experts, 2 * topology.moe_intermediate, topology.hidden});
        append(TensorRole::MoePackedDown, -1,
               {topology.num_experts, topology.hidden, topology.moe_intermediate});
        append(TensorRole::MoeSharedGate, -1,
               {topology.shared_expert_intermediate, topology.hidden});
        append(TensorRole::MoeSharedUp, -1,
               {topology.shared_expert_intermediate, topology.hidden});
        append(TensorRole::MoeSharedDown, -1,
               {topology.hidden, topology.shared_expert_intermediate});
        append(TensorRole::MoeSharedGateWeight, -1, {1, topology.hidden});
    } else {
        for (int expert = 0; expert < topology.num_experts; ++expert) {
            append(TensorRole::MoeExpertGate, expert,
                   {topology.moe_intermediate, topology.hidden});
            append(TensorRole::MoeExpertUp, expert,
                   {topology.moe_intermediate, topology.hidden});
            append(TensorRole::MoeExpertDown, expert,
                   {topology.hidden, topology.moe_intermediate});
        }
    }
}

} // namespace celeg::descriptor_detail
