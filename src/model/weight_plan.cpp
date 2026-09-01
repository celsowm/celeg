#include "celeg/model/weight_plan.hpp"

#include "celeg/checkpoint/packed/fp8.hpp"
#include "celeg/checkpoint/packed/int4.hpp"
#include "celeg/checkpoint/packed/nvfp4.hpp"
#include "celeg/model/weights/roles.hpp"

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace celeg {
namespace {

void add_request(ResolvedModel& model, TensorRole role, int layer, int expert,
                 std::vector<int64_t> shape, int physical_layer = -1,
                 NormWeightKind norm_weight_kind = NormWeightKind::Scale) {
    model.weight_plan.requests.push_back({role, layer, expert, std::move(shape),
                                          std::nullopt, physical_layer,
                                          norm_weight_kind});
}

void add_norm_request(ResolvedModel& model, TensorRole role, int layer,
                      const std::optional<NormSpec>& spec, std::vector<int64_t> shape,
                      int physical_layer) {
    if (!spec || spec->weightless()) return;
    add_request(model, role, layer, -1, std::move(shape), physical_layer,
                spec->weight_kind);
}

int attention_norm_width(const NormSpec& norm, int per_head_width, int whole_width) {
    return norm.granularity == NormGranularity::PerHead
        ? per_head_width
        : whole_width;
}

void append_attention(ResolvedModel& model, const AttentionSpec& attention,
                      int layer, int physical_layer) {
    const int hidden = model.graph.hidden;
    const auto append = [&](TensorRole role, std::vector<int64_t> shape) {
        add_request(model, role, layer, -1, std::move(shape), physical_layer);
    };
    if (attention.uses_external_memory()) {
        const int query_width = attention.query_width();
        append(TensorRole::AttentionQuery,
               {attention.query_projection_width(), hidden});
        if (attention.query_norm.has_value()) {
            add_norm_request(
                model, TensorRole::AttentionQueryNorm, layer,
                attention.query_norm,
                {attention_norm_width(*attention.query_norm,
                                      attention.head_dim,
                                      query_width)},
                physical_layer);
        }
        append(TensorRole::AttentionOutput, {hidden, query_width});
    } else if (attention.uses_latent_state()) {
        const auto& latent = *attention.latent_state();
        if (const auto* factorized = latent.factorized_projection()) {
            append(TensorRole::AttentionLatentQueryProjection,
                   {factorized->query_rank, hidden});
            append(TensorRole::AttentionLatentQueryExpansion,
                   {attention.query_heads * (latent.nope_head_dim + latent.rope_head_dim),
                    factorized->query_rank});
            append(TensorRole::AttentionLatentQueryNorm, {factorized->query_rank});
            append(TensorRole::AttentionLatentKeyProjection,
                   {latent.latent_rank + latent.rope_head_dim, hidden});
            append(TensorRole::AttentionLatentKeyNorm, {latent.latent_rank});
            append(TensorRole::AttentionLatentExpansion,
                   {attention.query_heads * (latent.nope_head_dim + factorized->value_head_dim),
                    latent.latent_rank});
            append(TensorRole::AttentionLatentOutput,
                   {hidden, attention.latent_output_width()});
        } else {
            append(TensorRole::AttentionLatentQuery,
                   {attention.latent_query_content_width(), hidden});
            if (attention.latent_query_rope_width() != 0) {
                append(TensorRole::AttentionLatentQueryRope,
                       {attention.latent_query_rope_width(), hidden});
            }
            append(TensorRole::AttentionLatentKey, {latent.latent_rank, hidden});
            append(TensorRole::AttentionLatentValue, {latent.latent_rank, hidden});
            if (latent.decoupled_rope && latent.rope_head_dim != 0) {
                append(TensorRole::AttentionLatentKeyRope,
                       {latent.rope_head_dim, hidden});
            }
            append(TensorRole::AttentionLatentOutput,
                   {hidden, attention.latent_query_content_width()});
        }
    } else {
        const int query_width = attention.query_width();
        const int key_value_width = attention.key_value_width();
        append(TensorRole::AttentionQuery,
               {attention.query_projection_width(), hidden});
        if (attention.query_norm.has_value()) {
            add_norm_request(
                model, TensorRole::AttentionQueryNorm, layer,
                attention.query_norm,
                {attention_norm_width(*attention.query_norm,
                                      attention.head_dim,
                                      query_width)},
                physical_layer);
        }
        if (!std::holds_alternative<SharedKvConsumer>(attention.kv_sharing)) {
            append(TensorRole::AttentionKey, {key_value_width, hidden});
            append(TensorRole::AttentionValue, {key_value_width, hidden});
            if (attention.key_norm.has_value()) {
                add_norm_request(
                    model, TensorRole::AttentionKeyNorm, layer,
                    attention.key_norm,
                    {attention_norm_width(*attention.key_norm,
                                          attention.head_dim,
                                          key_value_width)},
                    physical_layer);
            }
        }
        append(TensorRole::AttentionOutput, {hidden, query_width});
    }
    if (attention.output_gate.has_value() && !attention.output_gate->packed_with_query) {
        append(TensorRole::AttentionGate,
               {attention.output_gate_width(), hidden});
    }
    if (const auto* relative =
            std::get_if<RelativePositionBiasSpec>(&attention.bias)) {
        append(TensorRole::AttentionRelativePositionBias,
               {attention.query_heads * relative->bucket_count});
    }
}

void append_mixer(ResolvedModel& model, const LayerSpec& layer,
                  int layer_index, int physical_layer) {
    const int hidden = model.graph.hidden;
    const auto append = [&](TensorRole role, std::vector<int64_t> shape) {
        add_request(model, role, layer_index, -1, std::move(shape), physical_layer);
    };
    std::visit([&](const auto& mixer) {
        using Mixer = std::decay_t<decltype(mixer)>;
        if constexpr (std::is_same_v<Mixer, AttentionSpec>) {
            append_attention(model, mixer, layer_index, physical_layer);
        } else if constexpr (std::is_same_v<Mixer, ShortConvolutionSpec>) {
            append(TensorRole::ShortConvInput, {3 * hidden, hidden});
            append(TensorRole::ShortConvKernel, {hidden, 1, mixer.cache_length});
            append(TensorRole::ShortConvOutput, {hidden, hidden});
        } else if constexpr (std::is_same_v<Mixer, GatedDeltaNetSpec>) {
            const int qkv = mixer.qkv_width();
            if (mixer.factorized_projections) {
                append(TensorRole::GatedDeltaNetQuery,
                       {mixer.key_heads * mixer.key_head_dim, hidden});
                append(TensorRole::GatedDeltaNetKey,
                       {mixer.key_heads * mixer.key_head_dim, hidden});
                append(TensorRole::GatedDeltaNetValue,
                       {mixer.value_width(), hidden});
                append(TensorRole::GatedDeltaNetDecay,
                       {mixer.decay_width(), hidden});
                append(TensorRole::GatedDeltaNetOutputGate,
                       {mixer.value_width(), hidden});
                append(TensorRole::GatedDeltaNetQueryConv,
                       {mixer.key_heads * mixer.key_head_dim, 1, mixer.conv_kernel});
                append(TensorRole::GatedDeltaNetKeyConv,
                       {mixer.key_heads * mixer.key_head_dim, 1, mixer.conv_kernel});
                append(TensorRole::GatedDeltaNetValueConv,
                       {mixer.value_heads * mixer.value_head_dim, 1, mixer.conv_kernel});
            } else {
                append(TensorRole::GatedDeltaNetQkv, {qkv, hidden});
                append(TensorRole::GatedDeltaNetZ, {mixer.value_width(), hidden});
                append(TensorRole::GatedDeltaNetAlpha, {mixer.value_heads, hidden});
            }
            append(TensorRole::GatedDeltaNetBeta, {mixer.value_heads, hidden});
            append(TensorRole::GatedDeltaNetDtBias, {mixer.decay_width()});
            append(TensorRole::GatedDeltaNetALog, {mixer.value_heads});
            append(TensorRole::GatedDeltaNetConv, {qkv, 1, mixer.conv_kernel});
            append(TensorRole::GatedDeltaNetNorm, {mixer.value_head_dim});
            append(TensorRole::GatedDeltaNetOutput,
                   {hidden, mixer.value_width()});
        } else if constexpr (std::is_same_v<Mixer, Mamba2Spec>) {
            const int conv_dim = mixer.intermediate_size +
                2 * mixer.group_count * mixer.state_size;
            append(TensorRole::Mamba2Input,
                   {2 * mixer.intermediate_size + 2 * mixer.group_count *
                    mixer.state_size + mixer.num_heads, hidden});
            append(TensorRole::Mamba2Conv, {conv_dim, 1, mixer.conv_kernel});
            append(TensorRole::Mamba2ConvBias, {conv_dim});
            append(TensorRole::Mamba2DtBias, {mixer.num_heads});
            append(TensorRole::Mamba2ALog, {mixer.num_heads});
            append(TensorRole::Mamba2D, {mixer.num_heads});
            append(TensorRole::Mamba2Norm, {mixer.intermediate_size});
            append(TensorRole::Mamba2Output, {hidden, mixer.intermediate_size});
        } else if constexpr (std::is_same_v<Mixer, MlpBlockSpec>) {
            append(TensorRole::FfnUp, {mixer.intermediate_size, hidden});
            append(TensorRole::FfnDown, {hidden, mixer.intermediate_size});
        } else {
            static_assert(always_false_v<Mixer>, "unhandled mixer weight requirements");
        }
    }, layer.mixer);
}

/// A naming candidate is loadable when the repository stores it directly or
/// when it is a checkpoint-packed int4/FP8/NVFP4 tensor addressed by its
/// virtual base name (physical entries carry the _packed/_scale/_shape or
/// _scale/_global_scale suffixes). This is the same repository-level
/// convention the weight codecs consume, so plan resolution and loaders
/// agree on what a resolved name means.
bool repository_has_tensor(const IWeightRepository& repository,
                           const std::string& name) {
    return repository.contains(name) || has_packed_int4_matrix(repository, name) ||
        has_packed_fp8_matrix(repository, name) || has_packed_nvfp4_matrix(repository, name);
}

// Decides, once, whether this MoE layer's routed-expert weights are packed
// into a single [num_experts, ...] tensor pair or stored as individually
// named per-expert tensors. This is checkpoint-family information that
// backends must consume from the resolved plan rather than re-deriving by
// probing `repository->contains()` on literal tensor-name spellings during
// setup (see docs/EXTENSIBILITY_REFACTORING_PLAN.md, Phase 3).
bool moe_routed_experts_are_packed(int layer, int physical_layer,
                                   const ITensorNamingPolicy& naming_policy,
                                   const IWeightRepository* repository) {
    if (repository == nullptr) return false;
    const TensorRequest probe{TensorRole::MoePackedGateUp, layer, -1, {},
                              std::nullopt, physical_layer};
    for (const std::string& candidate : naming_policy.candidates(probe)) {
        if (repository_has_tensor(*repository, candidate)) return true;
    }
    return false;
}

void append_moe(ResolvedModel& model, const MixtureOfExpertsSpec& moe,
                int layer, int physical_layer,
                const ITensorNamingPolicy& naming_policy,
                const IWeightRepository* repository) {
    const int hidden = model.graph.hidden;
    const auto append = [&](TensorRole role, int expert, std::vector<int64_t> shape) {
        add_request(model, role, layer, expert, std::move(shape), physical_layer);
    };
    append(TensorRole::MoeRouter, -1, {moe.num_experts, hidden});
    if (moe.use_expert_bias) {
        append(TensorRole::MoeRouterBias, -1, {moe.num_experts});
    }
    if (moe_routed_experts_are_packed(layer, physical_layer, naming_policy, repository)) {
        append(TensorRole::MoePackedGateUp, -1,
               {moe.num_experts, 2 * moe.intermediate_size, hidden});
        append(TensorRole::MoePackedDown, -1,
               {moe.num_experts, hidden, moe.intermediate_size});
    } else {
        for (int expert = 0; expert < moe.num_experts; ++expert) {
            append(TensorRole::MoeExpertGate, expert,
                   {moe.intermediate_size, hidden});
            append(TensorRole::MoeExpertUp, expert,
                   {moe.intermediate_size, hidden});
            append(TensorRole::MoeExpertDown, expert,
                   {hidden, moe.intermediate_size});
        }
    }
    if (moe.shared) {
        append(TensorRole::MoeSharedGate, -1,
               {moe.shared->intermediate_size, hidden});
        append(TensorRole::MoeSharedUp, -1,
               {moe.shared->intermediate_size, hidden});
        append(TensorRole::MoeSharedDown, -1,
               {hidden, moe.shared->intermediate_size});
    }
}

}

void build_weight_plan_from_graph(ResolvedModel& model,
                                  const ITensorNamingPolicy& naming_policy,
                                  const IWeightRepository* repository) {
    const ModelGraph& graph = model.graph;
    const CheckpointDimensions& dimensions = model.topology.dims;
    if (graph.layers.empty() || graph.hidden <= 0) {
        throw std::invalid_argument("cannot build weight plan from an incomplete graph");
    }
    model.weight_plan.requests.clear();
    add_request(model, TensorRole::TokenEmbedding, -1, -1,
                {model.topology.dims.vocab_size, graph.hidden});
    add_norm_request(model, TensorRole::FinalNorm, -1, graph.final_norm,
                     {graph.hidden}, -1);
    add_request(model, TensorRole::LanguageModelHead, -1, -1,
                {model.topology.dims.vocab_size, graph.hidden});
    if (graph.per_layer_input) {
        const int layer_count = static_cast<int>(graph.layers.size());
        const int input_size = graph.per_layer_input->input_size;
        add_request(model, TensorRole::PerLayerEmbedding, -1, -1,
                    {model.topology.dims.vocab_size, layer_count * input_size});
        add_request(model, TensorRole::PerLayerContextProjection, -1, -1,
                    {layer_count * input_size, graph.hidden});
        add_request(model, TensorRole::PerLayerProjectionNorm, -1, -1,
                    {input_size});
    }
    for (int layer_index = 0;
         layer_index < static_cast<int>(graph.layers.size()); ++layer_index) {
        const LayerSpec& layer = graph.layers[static_cast<size_t>(layer_index)];
        const int physical_layer = dimensions.checkpoint_layer_for_layer.empty()
            ? layer_index
            : dimensions.checkpoint_layer_for_layer.at(static_cast<size_t>(layer_index));
        add_norm_request(model, TensorRole::AttentionInputNorm, layer_index,
                         layer.mixer_norm.before, {graph.hidden}, physical_layer);
        append_mixer(model, layer, layer_index, physical_layer);
        const bool has_feed_forward =
            !std::holds_alternative<std::monostate>(layer.feed_forward);
        if (has_feed_forward) {
            add_norm_request(model, TensorRole::FfnInputNorm, layer_index,
                             layer.feed_forward_norm.before, {graph.hidden}, physical_layer);
        }
        std::visit([&](const auto& feed_forward) {
            using FeedForward = std::decay_t<decltype(feed_forward)>;
            if constexpr (std::is_same_v<FeedForward, std::monostate>) {
                return;
            } else if constexpr (std::is_same_v<FeedForward, DenseFeedForwardSpec>) {
                add_request(model, TensorRole::FfnGate, layer_index, -1,
                            {feed_forward.intermediate_size, graph.hidden}, physical_layer);
                add_request(model, TensorRole::FfnUp, layer_index, -1,
                            {feed_forward.intermediate_size, graph.hidden}, physical_layer);
                add_request(model, TensorRole::FfnDown, layer_index, -1,
                            {graph.hidden, feed_forward.intermediate_size}, physical_layer);
            } else if constexpr (std::is_same_v<FeedForward, MixtureOfExpertsSpec>) {
                append_moe(model, feed_forward, layer_index, physical_layer,
                          naming_policy, repository);
            } else {
                static_assert(always_false_v<FeedForward>,
                              "unhandled feed-forward weight requirements");
            }
        }, layer.feed_forward);
        add_norm_request(model, TensorRole::AttentionPostNorm, layer_index,
                         layer.mixer_norm.after, {graph.hidden}, physical_layer);
        add_norm_request(model, TensorRole::FfnOutputNorm, layer_index,
                         layer.feed_forward_norm.after, {graph.hidden}, physical_layer);
        if (graph.per_layer_input) {
            add_request(model, TensorRole::PerLayerInputGate, layer_index, -1,
                        {graph.per_layer_input->input_size, graph.hidden}, physical_layer);
            add_request(model, TensorRole::PerLayerProjection, layer_index, -1,
                        {graph.hidden, graph.per_layer_input->input_size}, physical_layer);
            add_request(model, TensorRole::PerLayerInputNorm, layer_index, -1,
                        {graph.hidden}, physical_layer);
            add_request(model, TensorRole::LayerScalar, layer_index, -1,
                        {1}, physical_layer);
        }
    }
    resolve_weight_plan(model, naming_policy, repository);
}

void resolve_weight_plan(ResolvedModel& model,
                         const ITensorNamingPolicy& naming_policy,
                         const IWeightRepository* repository) {
    for (TensorRequest& request : model.weight_plan.requests) {
        const auto names = naming_policy.candidates(request);
        if (repository != nullptr) {
            for (const std::string& name : names) {
                if (repository_has_tensor(*repository, name)) {
                    request.source_name = name;
                    break;
                }
            }
        } else if (!names.empty()) {
            request.source_name = names.front();
        }
    }
}

}