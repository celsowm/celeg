#include "detail/model_internal.hpp"

#include "celeg/backend/cpu/weight_codec.hpp"
#include "celeg/checkpoint/tensor_names.hpp"
#include "celeg/model/weights/quantization.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace celeg {
namespace {

std::string tensor_name(std::span<const TensorRequest> requests, TensorRole role,
                        int layer = -1, int expert = -1) {
    return resolved_tensor_name(requests, role, layer, expert);
}

} // namespace

CpuCompiledModel::CommonWeights CpuCompiledModel::Shared::load_common(
    IWeightRepository* source, CpuPackReader* reader, CpuPackWriter* writer,
    int layer) {
    CommonWeights common;
    const CompiledLayerProgram& layer_program = program.layers.at(
        static_cast<size_t>(layer));
    const int hidden = program.hidden;
    const auto load_norm = [&](TensorRole role, const NormSpec& spec) {
        if (!spec.enabled() || spec.weightless()) {
            return std::vector<float>(static_cast<size_t>(hidden), 1.0f);
        }
        std::vector<float> values = load_vector(source, reader, writer,
            tensor_name(weight_requests, role, layer), {hidden});
        if (spec.weight_kind == NormWeightKind::OnePlusScale) {
            for (float& value : values) value += 1.0f;
        }
        return values;
    };
    common.operator_norm = load_norm(TensorRole::AttentionInputNorm,
                                     layer_program.operator_norm);
    if (!layer_program.execute_feed_forward) {
        common.ffn_norm = common.operator_norm;
        if (layer_program.mixer == CompiledMixer::MlpOnly) {
            const int intermediate = layer_program.feed_forward_intermediate;
            common.mlp_up = load_matrix(source, reader, writer,
                tensor_name(weight_requests, TensorRole::FfnUp, layer),
                {intermediate, program.hidden});
            common.w2 = load_matrix(source, reader, writer,
                tensor_name(weight_requests, TensorRole::FfnDown, layer),
                {program.hidden, intermediate});
        }
        return common;
    }
    if (layer_program.post_attention_norm.enabled()) {
        common.post_attention_norm = load_norm(TensorRole::AttentionPostNorm,
                                               layer_program.post_attention_norm);
    }
    common.ffn_norm = load_norm(TensorRole::FfnInputNorm,
                                layer_program.feed_forward_norm);
    if (layer_program.post_feed_forward_norm.enabled()) {
        common.post_feed_forward_norm = load_norm(TensorRole::FfnOutputNorm,
                                                  layer_program.post_feed_forward_norm);
    }
    const int intermediate = layer_program.feed_forward_intermediate;
    if (intermediate <= 0) {
        throw std::runtime_error("compiled dense layer has no FFN width");
    }
    common.w13 = load_concat(source, reader, writer,
        layer_name(layer, "feed_forward.w13.weight"), {
            {tensor_name(weight_requests, TensorRole::FfnGate, layer),
             {intermediate, shape.hidden}},
            {tensor_name(weight_requests, TensorRole::FfnUp, layer),
             {intermediate, shape.hidden}},
        });
    common.w2 = load_matrix(source, reader, writer,
        tensor_name(weight_requests, TensorRole::FfnDown, layer),
        {shape.hidden, intermediate});
    if (program.per_layer_input.enabled) {
        common.per_layer_input_gate = load_matrix(source, reader, writer,
            tensor_name(weight_requests, TensorRole::PerLayerInputGate, layer),
            {shape.per_layer_input_size, shape.hidden});
        common.per_layer_projection = load_matrix(source, reader, writer,
            tensor_name(weight_requests, TensorRole::PerLayerProjection, layer),
            {shape.hidden, shape.per_layer_input_size});
        common.per_layer_input_norm = load_vector(source, reader, writer,
            tensor_name(weight_requests, TensorRole::PerLayerInputNorm, layer),
            {shape.hidden});
        const std::vector<float> scalar = load_vector(source, reader, writer,
            tensor_name(weight_requests, TensorRole::LayerScalar, layer), {1});
        common.layer_scalar = scalar.front();
    }
    return common;
}

void CpuCompiledModel::Shared::load_weights() {
    std::unique_ptr<CpuPackReader> reader;
    if (!pack_file.empty() && std::filesystem::exists(pack_file)) {
        try {
            reader = std::make_unique<CpuPackReader>(pack_file);
            if (reader->metadata().source_id != source_id ||
                reader->metadata().group_size != group_size) {
                reader.reset();
            } else {
                loaded_pack = true;
            }
        } catch (const std::exception& error) {
            std::clog << "CPU pack cache rejected, rebuilding: "
                      << error.what() << '\n';
            reader.reset();
        } catch (...) {
            std::clog << "CPU pack cache rejected, rebuilding: unknown exception\n";
            reader.reset();
        }
    }

    std::unique_ptr<CpuPackWriter> writer;
    if (!reader) {
        if (!pack_file.empty()) {
            CpuPackMetadata metadata;
            metadata.source_id = source_id;
            metadata.isa = cpu_isa_name(options.isa);
            metadata.group_size = static_cast<uint32_t>(group_size);
            writer = std::make_unique<CpuPackWriter>(pack_file, std::move(metadata));
        }
    }

    IWeightRepository* source = reader ? nullptr : repository.get();
    const bool disk_cached_experts =
        options.expert_backing == CpuExpertBacking::DiskCached &&
        !native_checkpoint;
    weight_store.embedding = load_matrix(source, reader.get(), writer.get(),
        tensor_name(weight_requests, TensorRole::TokenEmbedding),
        {dims.vocab_size, shape.hidden});
    if (!tie_word_embeddings) {
        weight_store.lm_head = load_matrix(source, reader.get(), writer.get(),
            tensor_name(weight_requests, TensorRole::LanguageModelHead),
            {dims.vocab_size, shape.hidden});
    }
    if (program.final_norm.weightless()) {
        weight_store.final_norm.assign(static_cast<size_t>(shape.hidden), 1.0f);
    } else {
        weight_store.final_norm = load_vector(source, reader.get(), writer.get(),
            tensor_name(weight_requests, TensorRole::FinalNorm),
            {shape.hidden});
        if (program.final_norm.weight_kind == NormWeightKind::OnePlusScale) {
            for (float& value : weight_store.final_norm) value += 1.0f;
        }
    }
    if (program.embedding_transform.post_norm) {
        const NormSpec& spec = *program.embedding_transform.post_norm;
        if (spec.weightless()) {
            weight_store.embedding_norm.assign(static_cast<size_t>(shape.hidden), 1.0f);
        } else {
            weight_store.embedding_norm = load_vector(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::FinalNorm), {shape.hidden});
            if (spec.weight_kind == NormWeightKind::OnePlusScale) {
                for (float& value : weight_store.embedding_norm) value += 1.0f;
            }
        }
    }
    if (program.per_layer_input.enabled) {
        weight_store.per_layer_embedding = load_matrix(source, reader.get(), writer.get(),
            tensor_name(weight_requests, TensorRole::PerLayerEmbedding),
                            {dims.vocab_size, shape.num_hidden_layers * shape.per_layer_input_size});
        weight_store.per_layer_context_projection = load_matrix(source, reader.get(), writer.get(),
            tensor_name(weight_requests, TensorRole::PerLayerContextProjection),
            {shape.num_hidden_layers * shape.per_layer_input_size, shape.hidden});
        weight_store.per_layer_projection_norm = load_vector(source, reader.get(), writer.get(),
            tensor_name(weight_requests, TensorRole::PerLayerProjectionNorm),
            {shape.per_layer_input_size});
    }

    const auto load_operator = [&](int index,
                                   const CompiledLayerProgram& layer_program)
        -> std::variant<AttentionWeights, ConvolutionWeights, GatedDeltaNetWeights,
                        Mamba2Weights, MlpOnlyWeights> {
        if (layer_program.mixer == CompiledMixer::Attention) {
            AttentionWeights layer;
            const AttentionSpec& attention = layer_program.attention.value();
            if (attention.uses_latent_state()) {
                const auto& latent = *attention.latent_state();
                if (latent.factorized) {
                    layer.latent_q_projection = load_matrix(
                        source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionLatentQueryProjection, index),
                        {latent.query_rank, shape.hidden});
                    layer.latent_q_expansion = load_matrix(
                        source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionLatentQueryExpansion, index),
                        {attention.query_heads * (latent.nope_head_dim + latent.rope_head_dim),
                         latent.query_rank});
                    layer.latent_q_norm = load_vector(
                        source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionLatentQueryNorm, index),
                        {latent.query_rank});
                    layer.latent_k_projection = load_matrix(
                        source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionLatentKeyProjection, index),
                        {latent.latent_rank + latent.rope_head_dim, shape.hidden});
                    layer.latent_k_norm = load_vector(
                        source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionLatentKeyNorm, index),
                        {latent.latent_rank});
                    layer.latent_expansion = load_matrix(
                        source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionLatentExpansion, index),
                        {attention.query_heads * (latent.nope_head_dim + latent.value_head_dim),
                         latent.latent_rank});
                    layer.out = load_matrix(source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionLatentOutput, index),
                        {shape.hidden, attention.latent_output_width()});
                    layer.gate = load_matrix(source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionGate, index),
                        {attention.output_gate_width(), shape.hidden});
                    return layer;
                }
                layer.q = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::AttentionLatentQuery, index),
                    {attention.latent_query_content_width(), shape.hidden});
                if (attention.latent_query_rope_width() != 0) {
                    layer.latent_q_rope = load_matrix(
                        source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionLatentQueryRope, index),
                        {attention.latent_query_rope_width(), shape.hidden});
                }
                layer.k = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::AttentionLatentKey, index),
                    {latent.latent_rank, shape.hidden});
                layer.v = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::AttentionLatentValue, index),
                    {latent.latent_rank, shape.hidden});
                if (latent.decoupled_rope && latent.rope_head_dim != 0) {
                    layer.latent_k_rope = load_matrix(
                        source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionLatentKeyRope, index),
                        {latent.rope_head_dim, shape.hidden});
                }
                layer.out = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::AttentionLatentOutput, index),
                    {shape.hidden, attention.latent_query_content_width()});
                if (const auto* relative =
                        std::get_if<RelativePositionBiasSpec>(&attention.bias)) {
                    layer.relative_bias = load_vector(
                        source, reader.get(), writer.get(),
                        tensor_name(weight_requests,
                                    TensorRole::AttentionRelativePositionBias, index),
                        {attention.query_heads * relative->bucket_count});
                }
                return layer;
            }
            layer.q = load_matrix(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::AttentionQuery, index),
                {attention.query_projection_width(), shape.hidden});
            if (!attention.uses_external_memory() &&
                (!attention.kv_sharing.shared() || attention.kv_sharing.publishes)) {
                layer.k = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::AttentionKey, index),
                    {attention.key_value_width(), shape.hidden});
                layer.v = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::AttentionValue, index),
                    {attention.key_value_width(), shape.hidden});
            }
            layer.out = load_matrix(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::AttentionOutput, index),
                {shape.hidden, attention.query_width()});
            layer.q_norm = attention.query_norm.enabled()
                ? (attention.query_norm.weightless()
                    ? std::vector<float>(static_cast<size_t>(attention.head_dim), 1.0f)
                    : load_vector(source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionQueryNorm, index),
                        {attention.head_dim}))
                : std::vector<float>(static_cast<size_t>(attention.head_dim), 1.0f);
            layer.k_norm = attention.key_norm.enabled()
                ? (attention.key_norm.weightless()
                    ? std::vector<float>(static_cast<size_t>(attention.head_dim), 1.0f)
                    : load_vector(source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionKeyNorm, index),
                        {attention.head_dim}))
                : std::vector<float>(static_cast<size_t>(attention.head_dim), 1.0f);
            if (attention.query_norm.weight_kind == NormWeightKind::OnePlusScale) {
                for (float& value : layer.q_norm) value += 1.0f;
            }
            if (attention.key_norm.weight_kind == NormWeightKind::OnePlusScale) {
                for (float& value : layer.k_norm) value += 1.0f;
            }
            if (attention.output_gate.enabled() && !attention.output_gate.packed_with_query) {
                layer.gate = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::AttentionGate, index),
                    {attention.query_width(), shape.hidden});
            }
            if (const auto* relative =
                    std::get_if<RelativePositionBiasSpec>(&attention.bias)) {
                layer.relative_bias = load_vector(
                    source, reader.get(), writer.get(),
                    tensor_name(weight_requests,
                                TensorRole::AttentionRelativePositionBias, index),
                    {attention.query_heads * relative->bucket_count});
            }
            return layer;
        }

        if (layer_program.mixer == CompiledMixer::GatedDeltaNet) {
            GatedDeltaNetWeights layer;
            layer.spec = layer_program.gated_delta_net.value();
            const auto& spec = layer.spec;
            const int key_width = spec.key_heads * spec.key_head_dim;
            const int value_width = spec.value_heads * spec.value_head_dim;
            const int qkv_width = 2 * key_width + value_width;
            if (spec.factorized_projections) {
                layer.q = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::GatedDeltaNetQuery, index),
                    {key_width, shape.hidden});
                layer.k = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::GatedDeltaNetKey, index),
                    {key_width, shape.hidden});
                layer.v = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::GatedDeltaNetValue, index),
                    {value_width, shape.hidden});
                layer.z = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::GatedDeltaNetOutputGate, index),
                    {value_width, shape.hidden});
            } else {
                layer.qkv = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::GatedDeltaNetQkv, index),
                    {qkv_width, shape.hidden});
                layer.z = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::GatedDeltaNetZ, index),
                    {value_width, shape.hidden});
            }
            layer.b = load_matrix(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::GatedDeltaNetBeta, index),
                {spec.value_heads, shape.hidden});
            layer.a = load_matrix(source, reader.get(), writer.get(),
                tensor_name(weight_requests, spec.factorized_projections
                    ? TensorRole::GatedDeltaNetDecay : TensorRole::GatedDeltaNetAlpha, index),
                {spec.decay_width(), shape.hidden});
            const int conv_dim = qkv_width;
            if (spec.factorized_projections) {
                layer.q_conv_weight = load_vector(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::GatedDeltaNetQueryConv, index),
                    {key_width, 1, spec.conv_kernel});
                layer.k_conv_weight = load_vector(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::GatedDeltaNetKeyConv, index),
                    {key_width, 1, spec.conv_kernel});
                layer.v_conv_weight = load_vector(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::GatedDeltaNetValueConv, index),
                    {value_width, 1, spec.conv_kernel});
                layer.conv_weight.reserve(static_cast<size_t>(conv_dim) * spec.conv_kernel);
                layer.conv_weight.insert(layer.conv_weight.end(), layer.q_conv_weight.begin(),
                                         layer.q_conv_weight.end());
                layer.conv_weight.insert(layer.conv_weight.end(), layer.k_conv_weight.begin(),
                                         layer.k_conv_weight.end());
                layer.conv_weight.insert(layer.conv_weight.end(), layer.v_conv_weight.begin(),
                                         layer.v_conv_weight.end());
            } else {
                layer.conv_weight = load_vector(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::GatedDeltaNetConv, index),
                    {conv_dim, 1, spec.conv_kernel});
            }
            layer.dt_bias = load_vector(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::GatedDeltaNetDtBias, index),
                {spec.decay_width()});
            layer.a_log = load_vector(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::GatedDeltaNetALog, index),
                {spec.value_heads});
            layer.norm = load_vector(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::GatedDeltaNetNorm, index),
                {spec.value_head_dim});
            layer.out = load_matrix(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::GatedDeltaNetOutput, index),
                {shape.hidden, value_width});
            return layer;
        }

        if (layer_program.mixer == CompiledMixer::Mamba2) {
            Mamba2Weights layer;
            const auto& spec = layer_program.mamba2.value();
            const int conv_dim = spec.intermediate_size + 2 * spec.group_count * spec.state_size;
            layer.in = load_matrix(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::Mamba2Input, index),
                {2 * spec.intermediate_size + 2 * spec.group_count * spec.state_size +
                 spec.num_heads, shape.hidden});
            layer.conv_weight = load_vector(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::Mamba2Conv, index),
                {conv_dim, 1, spec.conv_kernel});
            layer.conv_bias = load_vector(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::Mamba2ConvBias, index), {conv_dim});
            layer.dt_bias = load_vector(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::Mamba2DtBias, index), {spec.num_heads});
            layer.a_log = load_vector(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::Mamba2ALog, index), {spec.num_heads});
            layer.d = load_vector(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::Mamba2D, index), {spec.num_heads});
            layer.norm = load_vector(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::Mamba2Norm, index), {spec.intermediate_size});
            layer.out = load_matrix(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::Mamba2Output, index),
                {shape.hidden, spec.intermediate_size});
            return layer;
        }
        if (layer_program.mixer == CompiledMixer::MlpOnly) return MlpOnlyWeights{};

        ConvolutionWeights layer;
        const ShortConvolutionSpec& convolution =
            layer_program.short_convolution.value();
        layer.in = load_matrix(source, reader.get(), writer.get(),
            layer_name(index, "conv.in_proj.weight"),
            {3 * shape.hidden, shape.hidden});
        const std::vector<float> channel_major =
            load_vector(source, reader.get(), writer.get(),
            layer_name(index, "conv.conv.weight"),
            {shape.hidden, 1, convolution.cache_length});
        layer.weight_tap_major.resize(channel_major.size());
        for (int tap = 0; tap < convolution.cache_length; ++tap) {
            for (int channel = 0; channel < shape.hidden; ++channel) {
                layer.weight_tap_major[static_cast<size_t>(tap) * shape.hidden + channel] =
                    channel_major[static_cast<size_t>(channel) * shape.conv_cache + tap];
            }
        }
        layer.out = load_matrix(source, reader.get(), writer.get(),
            layer_name(index, "conv.out_proj.weight"),
            {shape.hidden, shape.hidden});
        return layer;
    };

    if (shape.num_experts > 0) {
        weight_store.layers.reserve(static_cast<size_t>(shape.num_hidden_layers));
        for (int index = 0; index < shape.num_hidden_layers; ++index) {
            const CompiledLayerProgram& layer_program = program.layers.at(
                static_cast<size_t>(index));
            if (layer_program.feed_forward != CompiledFeedForward::MixtureOfExperts) {
                CommonWeights common =
                    load_common(source, reader.get(), writer.get(), index);
                auto layer = load_operator(index, layer_program);
                std::visit([&](auto& value) {
                    value.common = std::move(common);
                    weight_store.layers.emplace_back(std::move(value));
                }, layer);
                continue;
            }

            MoeWeights layer;
            const auto has_request = [&](TensorRole role, int expert = -1) {
                return std::any_of(weight_requests.begin(), weight_requests.end(),
                    [&](const TensorRequest& request) {
                        return request.role == role && request.layer == index &&
                               request.expert == expert;
                    });
            };
            const bool individual_expert_model =
                has_request(TensorRole::MoeExpertGate, 0);
            const MoeLayerProgram& moe_semantics =
                program.layers.at(static_cast<size_t>(index)).moe.value();
            const bool packed_expert_model = !individual_expert_model &&
                moe_semantics.shared.has_value();
            layer.common.operator_norm = load_vector(source, reader.get(), writer.get(),
                has_request(TensorRole::AttentionInputNorm)
                    ? tensor_name(weight_requests, TensorRole::AttentionInputNorm, index)
                    : layer_name(index, "operator_norm.weight"), {shape.hidden});
            layer.common.ffn_norm = load_vector(source, reader.get(), writer.get(),
                has_request(TensorRole::FfnInputNorm)
                    ? tensor_name(weight_requests, TensorRole::FfnInputNorm, index)
                    : layer_name(index, "ffn_norm.weight"), {shape.hidden});
            if (packed_expert_model) {
                for (float& value : layer.common.operator_norm) value += 1.0f;
                for (float& value : layer.common.ffn_norm) value += 1.0f;
            }
            layer.operator_layer = load_operator(index, layer_program);
            layer.num_experts = shape.num_experts;
            layer.experts_per_token = shape.experts_per_token;
            layer.normalize_topk = shape.normalize_topk;
            layer.use_expert_bias = shape.use_expert_bias;
            layer.routed_scaling_factor = shape.routed_scaling_factor;
            layer.router = load_vector(source, reader.get(), writer.get(),
                has_request(TensorRole::MoeRouter)
                    ? tensor_name(weight_requests, TensorRole::MoeRouter, index)
                    : layer_name(index, "feed_forward.gate.weight"),
                {shape.num_experts, shape.hidden});

            if (packed_expert_model) {
                const int shared_intermediate = moe_semantics.shared->mlp.intermediate_size;
                layer.shared_w13 = load_concat(source, reader.get(), writer.get(),
                    layer_name(index, "shared_expert.w13.weight"), {
                        {tensor_name(weight_requests, TensorRole::MoeSharedGate, index),
                         {shared_intermediate, shape.hidden}},
                        {tensor_name(weight_requests, TensorRole::MoeSharedUp, index),
                         {shared_intermediate, shape.hidden}},
                    });
                layer.shared_w2 = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::MoeSharedDown, index),
                    {shape.hidden, shared_intermediate});
                layer.shared_gate = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::MoeSharedGateWeight, index),
                    {1, shape.hidden});
                layer.has_shared_gate = true;
                layer.expert_w13 = CpuWeightCodec(source, reader.get(), writer.get(), group_size)
                    .packed_matrices(
                        tensor_name(weight_requests, TensorRole::MoePackedGateUp, index),
                        {moe_semantics.router.expert_count,
                         2 * moe_semantics.routed.mlp.intermediate_size, program.hidden});
                layer.expert_w2 = CpuWeightCodec(source, reader.get(), writer.get(), group_size)
                    .packed_matrices(
                        tensor_name(weight_requests, TensorRole::MoePackedDown, index),
                        {moe_semantics.router.expert_count, program.hidden,
                         moe_semantics.routed.mlp.intermediate_size});
            } else if (individual_expert_model && moe_semantics.shared.has_value()) {
                const int shared_intermediate = moe_semantics.shared->mlp.intermediate_size;
                layer.shared_w13 = load_concat(source, reader.get(), writer.get(),
                    layer_name(index, "mlp.shared_experts.w13.weight"), {
                        {tensor_name(weight_requests, TensorRole::MoeSharedGate, index),
                         {shared_intermediate, shape.hidden}},
                        {tensor_name(weight_requests, TensorRole::MoeSharedUp, index),
                         {shared_intermediate, shape.hidden}},
                    });
                layer.shared_w2 = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::MoeSharedDown, index),
                    {shape.hidden, shared_intermediate});
            }

            const std::string bias_name = has_request(TensorRole::MoeRouterBias)
                ? tensor_name(weight_requests, TensorRole::MoeRouterBias, index)
                : layer_name(index, "feed_forward.expert_bias.weight");
            if ((source && source->contains(bias_name)) ||
                (reader && reader->contains(bias_name))) {
                layer.router_bias = load_vector(source, reader.get(), writer.get(),
                    bias_name, {shape.num_experts});
            }

            if (!disk_cached_experts) {
                layer.expert_w13.resize(static_cast<size_t>(shape.num_experts));
                layer.expert_w2.resize(static_cast<size_t>(shape.num_experts));
            }
            for (int expert = 0; expert < shape.num_experts; ++expert) {
                const int moe_inter = program.layers.at(static_cast<size_t>(index))
                    .moe->routed.mlp.intermediate_size;
                if (moe_inter <= 0) {
                    throw std::runtime_error("compiled MoE layer has no expert width");
                }
                if (packed_expert_model) {
                    continue;
                }
                if (individual_expert_model) {
                    const std::string w13_name = layer_name(index,
                        "mlp.experts." + std::to_string(expert) + ".w13.weight");
                    layer.expert_w13[static_cast<size_t>(expert)] = load_concat(
                        source, reader.get(), writer.get(), w13_name,
                        {{tensor_name(weight_requests, TensorRole::MoeExpertGate,
                                      index, expert), {moe_inter, shape.hidden}},
                         {tensor_name(weight_requests, TensorRole::MoeExpertUp,
                                      index, expert), {moe_inter, shape.hidden}}});
                    layer.expert_w2[static_cast<size_t>(expert)] = load_matrix(
                        source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::MoeExpertDown,
                                    index, expert),
                        {shape.hidden, moe_inter});
                    continue;
                }
                const std::string prefix =
                    "feed_forward.experts." + std::to_string(expert);
                const std::string w13_name =
                    layer_name(index, prefix + ".w13.weight");
                const std::string w2_name =
                    layer_name(index, prefix + ".w2.weight");

                if (disk_cached_experts && reader) {
                    if (!reader->contains(w13_name) ||
                        !reader->contains(w2_name)) {
                        throw std::runtime_error(
                            "CPU pack is missing disk-backed expert entries for layer " +
                            std::to_string(index) + ", expert " +
                            std::to_string(expert));
                    }
                    continue;
                }

                const std::vector<std::pair<std::string, std::vector<int64_t>>> parts{
                    {layer_name(index, prefix + ".w1.weight"),
                     {moe_inter, shape.hidden}},
                    {layer_name(index, prefix + ".w3.weight"),
                     {moe_inter, shape.hidden}},
                };
                if (disk_cached_experts) {
                    (void)load_concat(source, nullptr, writer.get(),
                                      w13_name, parts);
                    (void)load_matrix(source, nullptr, writer.get(),
                                      w2_name, {shape.hidden, moe_inter});
                } else {
                    layer.expert_w13[static_cast<size_t>(expert)] = load_concat(
                        source, reader.get(), writer.get(), w13_name, parts);
                    layer.expert_w2[static_cast<size_t>(expert)] = load_matrix(
                        source, reader.get(), writer.get(), w2_name,
                        {shape.hidden, moe_inter});
                }
            }
            weight_store.layers.push_back(std::move(layer));
        }
    } else {
        weight_store.layers.reserve(static_cast<size_t>(shape.num_hidden_layers));
        for (int index = 0; index < shape.num_hidden_layers; ++index) {
            CommonWeights common =
                load_common(source, reader.get(), writer.get(), index);
            auto layer = load_operator(index, program.layers.at(static_cast<size_t>(index)));
            std::visit([&](auto& value) {
                value.common = std::move(common);
                weight_store.layers.emplace_back(std::move(value));
            }, layer);
        }
    }
    const auto require_matrix = [](const CpuLinearWeight& weight,
                                   const std::string& label) {
        if (weight.rows == 0 || weight.cols == 0 || weight.segments.empty()) {
            throw std::runtime_error("CPU loaded linear weight is empty: " + label);
        }
    };
    const auto check_attention = [&](const AttentionWeights& attention,
                                     const AttentionSpec& semantics,
                                     const std::string& label) {
        require_matrix(attention.out, label + ".out");
        if (const auto* latent = semantics.latent_state(); latent && latent->factorized) {
            require_matrix(attention.latent_q_projection, label + ".latent_q_projection");
            require_matrix(attention.latent_q_expansion, label + ".latent_q_expansion");
            require_matrix(attention.latent_k_projection, label + ".latent_k_projection");
            require_matrix(attention.latent_expansion, label + ".latent_expansion");
            require_matrix(attention.gate, label + ".gate");
        } else {
            require_matrix(attention.q, label + ".q");
            require_matrix(attention.k, label + ".k");
            require_matrix(attention.v, label + ".v");
            if (semantics.output_gate.enabled() && !semantics.output_gate.packed_with_query) {
                require_matrix(attention.gate, label + ".gate");
            }
        }
    };
    for (size_t layer = 0; layer < weight_store.layers.size(); ++layer) {
        std::visit([&](const auto& value) {
            using Value = std::decay_t<decltype(value)>;
            const auto check_common = [&](const CommonWeights& common) {
                require_matrix(common.w13, "layer " + std::to_string(layer) + ".w13");
                require_matrix(common.w2, "layer " + std::to_string(layer) + ".w2");
            };
            if constexpr (std::is_same_v<Value, MoeWeights>) {
                for (size_t expert = 0; expert < value.expert_w13.size(); ++expert) {
                    require_matrix(value.expert_w13[expert], "layer " +
                        std::to_string(layer) + ".expert_w13." + std::to_string(expert));
                    require_matrix(value.expert_w2[expert], "layer " +
                        std::to_string(layer) + ".expert_w2." + std::to_string(expert));
                }
                if (value.shared_w13.rows != 0) require_matrix(value.shared_w13, "shared_w13");
                if (value.shared_w2.rows != 0) require_matrix(value.shared_w2, "shared_w2");
                if (const auto* attention = std::get_if<AttentionWeights>(&value.operator_layer)) {
                    check_attention(*attention, program.layers[layer].attention.value(),
                                    "layer " + std::to_string(layer) + ".attention");
                }
            } else if constexpr (std::is_same_v<Value, AttentionWeights>) {
                check_common(value.common);
                check_attention(value, program.layers[layer].attention.value(),
                                "layer " + std::to_string(layer) + ".attention");
            } else if constexpr (std::is_same_v<Value, ConvolutionWeights>) {
                check_common(value.common);
                require_matrix(value.in, "layer " + std::to_string(layer) + ".conv.in");
                require_matrix(value.out, "layer " + std::to_string(layer) + ".conv.out");
            } else if constexpr (std::is_same_v<Value, GatedDeltaNetWeights>) {
                check_common(value.common);
                require_matrix(value.out, "layer " + std::to_string(layer) + ".kda.out");
            } else if constexpr (std::is_same_v<Value, Mamba2Weights>) {
                check_common(value.common);
                require_matrix(value.in, "layer " + std::to_string(layer) + ".mamba.in");
                require_matrix(value.out, "layer " + std::to_string(layer) + ".mamba.out");
            }
        }, weight_store.layers[layer]);
    }
    if (writer) writer->commit();
}

} // namespace celeg
