#include "descriptor_detail.hpp"

#include <string>
#include <utility>

namespace celeg::descriptor_detail {
namespace {

class DescriptorNamingPolicy final : public ITensorNamingPolicy {
public:
    explicit DescriptorNamingPolicy(const Descriptor& descriptor) {
        for (const BindingPattern& pattern : descriptor.bindings) {
            bindings_[pattern.role] = pattern.candidates;
        }
    }

    std::vector<std::string> candidates(const TensorRequest& request) const override {
        const auto it = bindings_.find(request.role);
        if (it == bindings_.end()) return {};
        std::vector<std::string> result;
        result.reserve(it->second.size());
        for (const std::string& pattern : it->second) {
            std::string value = pattern;
            replace(value, "{layer}", request.layer >= 0 ? std::to_string(request.layer) : "");
            replace(value, "{physical_layer}", request.physical_layer >= 0
                ? std::to_string(request.physical_layer) : "");
            replace(value, "{expert}", request.expert >= 0 ? std::to_string(request.expert) : "");
            result.push_back(std::move(value));
        }
        return result;
    }

private:
    static void replace(std::string& value, std::string_view needle,
                        std::string_view replacement) {
        size_t position = 0;
        while ((position = value.find(needle, position)) != std::string::npos) {
            value.replace(position, needle.size(), replacement);
            position += replacement.size();
        }
    }

    std::unordered_map<TensorRole, std::vector<std::string>> bindings_;
};

void add_descriptor_request(ResolvedModel& model, TensorRole role, int layer, int expert,
                            std::vector<int64_t> shape, int physical_layer = -1) {
    model.weight_plan.requests.push_back({role, layer, expert, std::move(shape),
                                          std::nullopt, physical_layer});
}

} // namespace

std::unique_ptr<ITensorNamingPolicy> create_naming_policy(const Descriptor& descriptor) {
    return std::make_unique<DescriptorNamingPolicy>(descriptor);
}

void build_weight_plan(ResolvedModel& model, const Descriptor&,
                       const ITensorNamingPolicy& naming_policy) {
    const RuntimeTopology& topology = model.topology;
    model.weight_plan.requests.clear();
    add_descriptor_request(model, TensorRole::TokenEmbedding, -1, -1,
                           {topology.vocab_size, topology.hidden});
    add_descriptor_request(model, TensorRole::LanguageModelHead, -1, -1,
                           {topology.vocab_size, topology.hidden});
    add_descriptor_request(model, TensorRole::FinalNorm, -1, -1, {topology.hidden});
    if (topology.has_per_layer_input) {
        const int width = topology.num_hidden_layers * topology.per_layer_input_size;
        add_descriptor_request(model, TensorRole::PerLayerEmbedding, -1, -1,
                               {topology.vocab_size, width});
        add_descriptor_request(model, TensorRole::PerLayerContextProjection, -1, -1,
                               {width, topology.hidden});
        add_descriptor_request(model, TensorRole::PerLayerProjectionNorm, -1, -1,
                               {topology.per_layer_input_size});
    }
    for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
        const int physical_layer = topology.checkpoint_layer_for_layer.empty()
            ? layer : topology.checkpoint_layer_for_layer.at(static_cast<size_t>(layer));
        const AttentionSpec& attention = topology.attention_layout(layer);
        const int query_width = attention.query_width();
        const int key_value_width = attention.key_value_width();
        const int intermediate = topology.feed_forward_intermediates.empty()
            ? topology.intermediate
            : topology.feed_forward_intermediates.at(static_cast<size_t>(layer));
        add_descriptor_request(model, TensorRole::AttentionInputNorm, layer, -1,
                               {topology.hidden}, physical_layer);
        if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::Attention) {
            if (attention.uses_external_memory()) {
                add_descriptor_request(model, TensorRole::AttentionQuery, layer, -1,
                                       {attention.query_projection_width(), topology.hidden},
                                       physical_layer);
                add_descriptor_request(model, TensorRole::AttentionOutput, layer, -1,
                                       {topology.hidden, attention.query_width()},
                                       physical_layer);
            } else if (attention.uses_latent_state()) {
                add_descriptor_request(model, TensorRole::AttentionLatentQuery, layer, -1,
                                       {attention.latent_query_content_width(), topology.hidden},
                                       physical_layer);
                if (attention.latent_query_rope_width() != 0) {
                    add_descriptor_request(model, TensorRole::AttentionLatentQueryRope,
                                           layer, -1,
                                           {attention.latent_query_rope_width(), topology.hidden},
                                           physical_layer);
                }
                const auto& latent = *attention.latent_state();
                add_descriptor_request(model, TensorRole::AttentionLatentKey, layer, -1,
                                       {latent.latent_rank, topology.hidden}, physical_layer);
                add_descriptor_request(model, TensorRole::AttentionLatentValue, layer, -1,
                                       {latent.latent_rank, topology.hidden}, physical_layer);
                if (latent.decoupled_rope && latent.rope_head_dim != 0) {
                    add_descriptor_request(model, TensorRole::AttentionLatentKeyRope, layer, -1,
                                           {latent.rope_head_dim, topology.hidden},
                                           physical_layer);
                }
                add_descriptor_request(model, TensorRole::AttentionLatentOutput, layer, -1,
                                       {topology.hidden, attention.latent_query_content_width()},
                                       physical_layer);
            } else {
            add_descriptor_request(model, TensorRole::AttentionQuery, layer, -1,
                                   {query_width, topology.hidden}, physical_layer);
            if (attention.query_key_norm) {
                add_descriptor_request(model, TensorRole::AttentionQueryNorm, layer, -1,
                                       {attention.head_dim}, physical_layer);
            }
            if (!attention.kv_sharing.shared() || attention.kv_sharing.publishes) {
                add_descriptor_request(model, TensorRole::AttentionKey, layer, -1,
                                       {key_value_width, topology.hidden}, physical_layer);
                add_descriptor_request(model, TensorRole::AttentionValue, layer, -1,
                                       {key_value_width, topology.hidden}, physical_layer);
                if (attention.query_key_norm) {
                    add_descriptor_request(model, TensorRole::AttentionKeyNorm, layer, -1,
                                           {attention.head_dim}, physical_layer);
                }
            }
            add_descriptor_request(model, TensorRole::AttentionOutput, layer, -1,
                                   {topology.hidden, query_width}, physical_layer);
            }
            if (const auto* relative =
                    std::get_if<RelativePositionBiasSpec>(&attention.bias)) {
                add_descriptor_request(
                    model, TensorRole::AttentionRelativePositionBias, layer, -1,
                    {attention.query_heads * relative->bucket_count}, physical_layer);
            }
            if (topology.has_split_attention_norms) {
                add_descriptor_request(model, TensorRole::AttentionPostNorm, layer, -1,
                                       {topology.hidden}, physical_layer);
            }
        } else if (topology.mixer_kinds[static_cast<size_t>(layer)] ==
                   MixerKind::ShortConvolution) {
            add_descriptor_request(model, TensorRole::ShortConvInput, layer, -1,
                                   {3 * topology.hidden, topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::ShortConvKernel, layer, -1,
                                   {topology.hidden, 1, topology.conv_cache}, physical_layer);
            add_descriptor_request(model, TensorRole::ShortConvOutput, layer, -1,
                                   {topology.hidden, topology.hidden}, physical_layer);
        } else if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::GatedDeltaNet) {
            const auto& recurrent = topology.gated_delta_net_layouts.at(static_cast<size_t>(layer));
            const int qkv = 2 * recurrent.key_heads * recurrent.key_head_dim +
                recurrent.value_heads * recurrent.value_head_dim;
            add_descriptor_request(model, TensorRole::GatedDeltaNetQkv, layer, -1,
                                   {qkv, topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::GatedDeltaNetZ, layer, -1,
                                   {recurrent.value_heads * recurrent.value_head_dim, topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::GatedDeltaNetAlpha, layer, -1,
                                   {recurrent.value_heads, topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::GatedDeltaNetBeta, layer, -1,
                                   {recurrent.value_heads, topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::GatedDeltaNetDtBias, layer, -1,
                                   {recurrent.value_heads}, physical_layer);
            add_descriptor_request(model, TensorRole::GatedDeltaNetALog, layer, -1,
                                   {recurrent.value_heads}, physical_layer);
            add_descriptor_request(model, TensorRole::GatedDeltaNetConv, layer, -1,
                                   {qkv, 1, recurrent.conv_kernel}, physical_layer);
            add_descriptor_request(model, TensorRole::GatedDeltaNetNorm, layer, -1,
                                   {recurrent.value_head_dim}, physical_layer);
            add_descriptor_request(model, TensorRole::GatedDeltaNetOutput, layer, -1,
                                   {topology.hidden, recurrent.value_heads * recurrent.value_head_dim}, physical_layer);
        } else if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::Mamba2) {
            const auto& recurrent = topology.mamba2_layouts.at(static_cast<size_t>(layer));
            const int conv_dim = recurrent.intermediate_size +
                2 * recurrent.group_count * recurrent.state_size;
            add_descriptor_request(model, TensorRole::Mamba2Input, layer, -1,
                                   {2 * recurrent.intermediate_size +
                                    2 * recurrent.group_count * recurrent.state_size +
                                    recurrent.num_heads, topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::Mamba2Conv, layer, -1,
                                   {conv_dim, 1, recurrent.conv_kernel}, physical_layer);
            add_descriptor_request(model, TensorRole::Mamba2ConvBias, layer, -1,
                                   {conv_dim}, physical_layer);
            add_descriptor_request(model, TensorRole::Mamba2DtBias, layer, -1,
                                   {recurrent.num_heads}, physical_layer);
            add_descriptor_request(model, TensorRole::Mamba2ALog, layer, -1,
                                   {recurrent.num_heads}, physical_layer);
            add_descriptor_request(model, TensorRole::Mamba2D, layer, -1,
                                   {recurrent.num_heads}, physical_layer);
            add_descriptor_request(model, TensorRole::Mamba2Norm, layer, -1,
                                   {recurrent.intermediate_size}, physical_layer);
            add_descriptor_request(model, TensorRole::Mamba2Output, layer, -1,
                                   {topology.hidden, recurrent.intermediate_size}, physical_layer);
        }
        const bool mlp_only = topology.mixer_kinds[static_cast<size_t>(layer)] ==
            MixerKind::MlpOnly;
        if (mlp_only) {
            const auto& mlp = topology.mlp_only_layouts.at(static_cast<size_t>(layer));
            add_descriptor_request(model, TensorRole::FfnUp, layer, -1,
                                   {mlp.intermediate_size, topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::FfnDown, layer, -1,
                                   {topology.hidden, mlp.intermediate_size}, physical_layer);
        } else {
            add_descriptor_request(model, TensorRole::FfnInputNorm, layer, -1,
                                   {topology.hidden}, physical_layer);
        }
        if (!mlp_only && topology.layer_uses_moe(layer)) {
            add_descriptor_request(model, TensorRole::MoeRouter, layer, -1,
                                   {topology.num_experts, topology.hidden}, physical_layer);
            if (topology.shared_expert_intermediate > 0) {
                add_descriptor_request(model, TensorRole::MoePackedGateUp, layer, -1,
                                       {topology.num_experts, 2 * topology.moe_intermediate,
                                        topology.hidden}, physical_layer);
                add_descriptor_request(model, TensorRole::MoePackedDown, layer, -1,
                                       {topology.num_experts, topology.hidden,
                                        topology.moe_intermediate}, physical_layer);
                add_descriptor_request(model, TensorRole::MoeSharedGate, layer, -1,
                                       {topology.shared_expert_intermediate, topology.hidden}, physical_layer);
                add_descriptor_request(model, TensorRole::MoeSharedUp, layer, -1,
                                       {topology.shared_expert_intermediate, topology.hidden}, physical_layer);
                add_descriptor_request(model, TensorRole::MoeSharedDown, layer, -1,
                                       {topology.hidden, topology.shared_expert_intermediate}, physical_layer);
                add_descriptor_request(model, TensorRole::MoeSharedGateWeight, layer, -1,
                                       {1, topology.hidden}, physical_layer);
            } else {
                for (int expert = 0; expert < topology.num_experts; ++expert) {
                    add_descriptor_request(model, TensorRole::MoeExpertGate, layer, expert,
                                           {topology.moe_intermediate, topology.hidden}, physical_layer);
                    add_descriptor_request(model, TensorRole::MoeExpertUp, layer, expert,
                                           {topology.moe_intermediate, topology.hidden}, physical_layer);
                    add_descriptor_request(model, TensorRole::MoeExpertDown, layer, expert,
                                           {topology.hidden, topology.moe_intermediate}, physical_layer);
                }
            }
        } else if (!mlp_only) {
            add_descriptor_request(model, TensorRole::FfnGate, layer, -1,
                                   {intermediate, topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::FfnUp, layer, -1,
                                   {intermediate, topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::FfnDown, layer, -1,
                                   {topology.hidden, intermediate}, physical_layer);
            if (topology.has_split_attention_norms) {
                add_descriptor_request(model, TensorRole::FfnOutputNorm, layer, -1,
                                       {topology.hidden}, physical_layer);
            }
        }
        if (topology.has_per_layer_input) {
            add_descriptor_request(model, TensorRole::PerLayerInputGate, layer, -1,
                                   {topology.per_layer_input_size, topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::PerLayerProjection, layer, -1,
                                   {topology.hidden, topology.per_layer_input_size}, physical_layer);
            add_descriptor_request(model, TensorRole::PerLayerInputNorm, layer, -1,
                                   {topology.hidden}, physical_layer);
            add_descriptor_request(model, TensorRole::LayerScalar, layer, -1, {1}, physical_layer);
        }
    }
    resolve_weight_plan(model, naming_policy);
}

} // namespace celeg::descriptor_detail
