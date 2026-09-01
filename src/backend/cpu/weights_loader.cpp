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

int attention_norm_width(const NormSpec& norm, int heads, int head_dim) {
    return norm.granularity == NormGranularity::PerHead
        ? head_dim
        : heads * head_dim;
}

}

CpuCompiledModel::CommonWeights CpuCompiledModel::Shared::load_common(
    IWeightRepository* source, CpuPackReader* reader, CpuPackWriter* writer,
    int layer) {
    CommonWeights common;
    const CompiledLayerProgram& layer_program = program.layers.at(
        static_cast<size_t>(layer));
    const int hidden = program.hidden;
    const auto load_norm = [&](TensorRole role, const NormSpec& spec) {
        if (spec.weightless()) {
            return std::vector<float>(static_cast<size_t>(hidden), 1.0f);
        }
        std::vector<float> values = load_vector(source, reader, writer,
            tensor_name(weight_requests, role, layer), {hidden});
        if (spec.weight_kind == NormWeightKind::OnePlusScale) {
            for (float& value : values) value += 1.0f;
        }
        return values;
    };
    if (layer_program.mixer_norm.before) {
        common.operator_norm = load_norm(
            TensorRole::AttentionInputNorm, *layer_program.mixer_norm.before);
    }
    if (layer_program.mixer_norm.after) {
        common.post_attention_norm = load_norm(
            TensorRole::AttentionPostNorm, *layer_program.mixer_norm.after);
    }
    if (!std::holds_alternative<std::monostate>(layer_program.feed_forward)) {
        if (layer_program.feed_forward_norm.before) {
            common.ffn_norm = load_norm(
                TensorRole::FfnInputNorm, *layer_program.feed_forward_norm.before);
        }
        if (layer_program.feed_forward_norm.after) {
            common.post_feed_forward_norm = load_norm(
                TensorRole::FfnOutputNorm, *layer_program.feed_forward_norm.after);
        }
    }
    if (program.per_layer_input.enabled) {
        common.per_layer_input_norm = load_vector(source, reader, writer,
            tensor_name(weight_requests, TensorRole::PerLayerInputNorm, layer),
            {program.hidden});
        const std::vector<float> scalar = load_vector(source, reader, writer,
            tensor_name(weight_requests, TensorRole::LayerScalar, layer), {1});
        common.layer_scalar = scalar.front();
    }
    return common;
}

CpuCompiledModel::DenseFeedForwardWeights CpuCompiledModel::Shared::load_dense_feed_forward(
    IWeightRepository* source, CpuPackReader* reader, CpuPackWriter* writer,
    int layer) {
    DenseFeedForwardWeights dense;
    const CompiledLayerProgram& layer_program = program.layers.at(
        static_cast<size_t>(layer));
    const auto* compiled_dense =
        std::get_if<CompiledDenseFeedForwardProgram>(&layer_program.feed_forward);
    if (!compiled_dense || compiled_dense->intermediate_size <= 0) {
        throw std::runtime_error("compiled dense layer has no FFN width");
    }
    const int intermediate = compiled_dense->intermediate_size;
    dense.w13 = load_concat(source, reader, writer,
        layer_name(layer, "feed_forward.w13.weight"), {
            {tensor_name(weight_requests, TensorRole::FfnGate, layer),
             {intermediate, program.hidden}},
            {tensor_name(weight_requests, TensorRole::FfnUp, layer),
             {intermediate, program.hidden}},
        });
    dense.w2 = load_matrix(source, reader, writer,
        tensor_name(weight_requests, TensorRole::FfnDown, layer),
        {program.hidden, intermediate});
    if (program.per_layer_input.enabled) {
        dense.per_layer_input_gate = load_matrix(source, reader, writer,
            tensor_name(weight_requests, TensorRole::PerLayerInputGate, layer),
            {program.per_layer_input.input_size, program.hidden});
        dense.per_layer_projection = load_matrix(source, reader, writer,
            tensor_name(weight_requests, TensorRole::PerLayerProjection, layer),
            {program.hidden, program.per_layer_input.input_size});
    }
    return dense;
}

void CpuCompiledModel::Shared::load_weights() {
    std::unique_ptr<CpuPackReader> reader;
    if (!checkpoint.pack_file.empty() && std::filesystem::exists(checkpoint.pack_file)) {
        try {
            reader = std::make_unique<CpuPackReader>(checkpoint.pack_file);
            if (reader->metadata().source_id != checkpoint.source_id ||
                reader->metadata().group_size != group_size) {
                reader.reset();
            } else {
                checkpoint.loaded_pack = true;
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
        if (!checkpoint.pack_file.empty()) {
            CpuPackMetadata metadata;
            metadata.source_id = checkpoint.source_id;
            metadata.isa = cpu_isa_name(options.isa);
            metadata.group_size = group_size;
            writer = std::make_unique<CpuPackWriter>(checkpoint.pack_file, std::move(metadata));
        }
    }

    IWeightRepository* source = reader ? nullptr : repository.get();
    const bool disk_cached_experts =
        options.expert_backing == CpuExpertBacking::DiskCached &&
        !checkpoint.native_checkpoint;
    weight_store.embedding = load_matrix(source, reader.get(), writer.get(),
        tensor_name(weight_requests, TensorRole::TokenEmbedding),
        {dims.vocab_size, program.hidden});
    if (!tie_word_embeddings) {
        weight_store.lm_head = load_matrix(source, reader.get(), writer.get(),
            tensor_name(weight_requests, TensorRole::LanguageModelHead),
            {dims.vocab_size, program.hidden});
    }
    if (program.final_norm.weightless()) {
        weight_store.final_norm.assign(static_cast<size_t>(program.hidden), 1.0f);
    } else {
        weight_store.final_norm = load_vector(source, reader.get(), writer.get(),
            tensor_name(weight_requests, TensorRole::FinalNorm),
            {program.hidden});
        if (program.final_norm.weight_kind == NormWeightKind::OnePlusScale) {
            for (float& value : weight_store.final_norm) value += 1.0f;
        }
    }
    if (program.embedding_transform.post_norm) {
        const NormSpec& spec = *program.embedding_transform.post_norm;
        if (spec.weightless()) {
            weight_store.embedding_norm.assign(static_cast<size_t>(program.hidden), 1.0f);
        } else {
            weight_store.embedding_norm = load_vector(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::FinalNorm), {program.hidden});
            if (spec.weight_kind == NormWeightKind::OnePlusScale) {
                for (float& value : weight_store.embedding_norm) value += 1.0f;
            }
        }
    }
    if (program.per_layer_input.enabled) {
        weight_store.per_layer_embedding = load_matrix(source, reader.get(), writer.get(),
            tensor_name(weight_requests, TensorRole::PerLayerEmbedding),
                            {dims.vocab_size, program.per_layer_input.layer_count *
                                program.per_layer_input.input_size});
        weight_store.per_layer_context_projection = load_matrix(source, reader.get(), writer.get(),
            tensor_name(weight_requests, TensorRole::PerLayerContextProjection),
            {program.per_layer_input.layer_count * program.per_layer_input.input_size,
             program.hidden});
        weight_store.per_layer_projection_norm = load_vector(source, reader.get(), writer.get(),
            tensor_name(weight_requests, TensorRole::PerLayerProjectionNorm),
            {program.per_layer_input.input_size});
    }

    const auto load_operator = [&](int index,
                                   const CompiledLayerProgram& layer_program)
        -> CpuMixerWeights {
        if (const auto* compiled_attention =
                std::get_if<CompiledAttentionProgram>(&layer_program.mixer)) {
            AttentionWeights layer;
            const AttentionSpec& attention = compiled_attention->semantics;
            if (attention.uses_latent_state()) {
                const auto& latent = *attention.latent_state();
                if (const auto* factorized = latent.factorized_projection()) {
                    layer.latent_q_projection = load_matrix(
                        source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionLatentQueryProjection, index),
                        {factorized->query_rank, program.hidden});
                    layer.latent_q_expansion = load_matrix(
                        source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionLatentQueryExpansion, index),
                        {attention.query_heads * (latent.nope_head_dim + latent.rope_head_dim),
                         factorized->query_rank});
                    layer.latent_q_norm = load_vector(
                        source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionLatentQueryNorm, index),
                        {factorized->query_rank});
                    layer.latent_k_projection = load_matrix(
                        source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionLatentKeyProjection, index),
                        {latent.latent_rank + latent.rope_head_dim, program.hidden});
                    layer.latent_k_norm = load_vector(
                        source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionLatentKeyNorm, index),
                        {latent.latent_rank});
                    layer.latent_expansion = load_matrix(
                        source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionLatentExpansion, index),
                        {attention.query_heads * (latent.nope_head_dim + factorized->value_head_dim),
                         latent.latent_rank});
                    layer.out = load_matrix(source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionLatentOutput, index),
                        {program.hidden, attention.latent_output_width()});
                    layer.gate = load_matrix(source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionGate, index),
                        {attention.output_gate_width(), program.hidden});
                    return layer;
                }
                layer.q = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::AttentionLatentQuery, index),
                    {attention.latent_query_content_width(), program.hidden});
                if (attention.latent_query_rope_width() != 0) {
                    layer.latent_q_rope = load_matrix(
                        source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionLatentQueryRope, index),
                        {attention.latent_query_rope_width(), program.hidden});
                }
                layer.k = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::AttentionLatentKey, index),
                    {latent.latent_rank, program.hidden});
                layer.v = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::AttentionLatentValue, index),
                    {latent.latent_rank, program.hidden});
                if (latent.decoupled_rope && latent.rope_head_dim != 0) {
                    layer.latent_k_rope = load_matrix(
                        source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionLatentKeyRope, index),
                        {latent.rope_head_dim, program.hidden});
                }
                layer.out = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::AttentionLatentOutput, index),
                    {program.hidden, attention.latent_query_content_width()});
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
                {attention.query_projection_width(), program.hidden});
            if (!attention.uses_external_memory() &&
                !std::holds_alternative<SharedKvConsumer>(attention.kv_sharing)) {
                layer.k = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::AttentionKey, index),
                    {attention.key_value_width(), program.hidden});
                layer.v = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::AttentionValue, index),
                    {attention.key_value_width(), program.hidden});
            }
            layer.out = load_matrix(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::AttentionOutput, index),
                {program.hidden, attention.query_width()});
            const int query_norm_width = attention.query_norm
                ? attention_norm_width(*attention.query_norm,
                                       attention.query_heads, attention.head_dim)
                : attention.head_dim;
            const int key_norm_width = attention.key_norm
                ? attention_norm_width(*attention.key_norm,
                                       attention.key_value_heads, attention.head_dim)
                : attention.head_dim;
            layer.q_norm = attention.query_norm.has_value()
                ? (attention.query_norm->weightless()
                    ? std::vector<float>(static_cast<size_t>(query_norm_width), 1.0f)
                    : load_vector(source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionQueryNorm, index),
                        {query_norm_width}))
                : std::vector<float>(static_cast<size_t>(attention.head_dim), 1.0f);
            layer.k_norm = attention.key_norm.has_value()
                ? (attention.key_norm->weightless()
                    ? std::vector<float>(static_cast<size_t>(key_norm_width), 1.0f)
                    : load_vector(source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionKeyNorm, index),
                        {key_norm_width}))
                : std::vector<float>(static_cast<size_t>(attention.head_dim), 1.0f);
            if (attention.query_norm && attention.query_norm->weight_kind == NormWeightKind::OnePlusScale) {
                for (float& value : layer.q_norm) value += 1.0f;
            }
            if (attention.key_norm && attention.key_norm->weight_kind == NormWeightKind::OnePlusScale) {
                for (float& value : layer.k_norm) value += 1.0f;
            }
            if (attention.output_gate.has_value() && !attention.output_gate->packed_with_query) {
                layer.gate = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::AttentionGate, index),
                    {attention.output_gate_width(), program.hidden});
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

        if (const auto* gated_delta =
                std::get_if<GatedDeltaNetSpec>(&layer_program.mixer)) {
            GatedDeltaNetWeights layer;
            layer.spec = *gated_delta;
            const auto& spec = layer.spec;
            const int key_width = spec.key_heads * spec.key_head_dim;
            const int value_width = spec.value_heads * spec.value_head_dim;
            const int qkv_width = 2 * key_width + value_width;
            if (spec.factorized_projections) {
                layer.q = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::GatedDeltaNetQuery, index),
                    {key_width, program.hidden});
                layer.k = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::GatedDeltaNetKey, index),
                    {key_width, program.hidden});
                layer.v = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::GatedDeltaNetValue, index),
                    {value_width, program.hidden});
                layer.z = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::GatedDeltaNetOutputGate, index),
                    {value_width, program.hidden});
            } else {
                layer.qkv = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::GatedDeltaNetQkv, index),
                    {qkv_width, program.hidden});
                layer.z = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::GatedDeltaNetZ, index),
                    {value_width, program.hidden});
            }
            layer.b = load_matrix(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::GatedDeltaNetBeta, index),
                {spec.value_heads, program.hidden});
            layer.a = load_matrix(source, reader.get(), writer.get(),
                tensor_name(weight_requests, spec.factorized_projections
                    ? TensorRole::GatedDeltaNetDecay : TensorRole::GatedDeltaNetAlpha, index),
                {spec.decay_width(), program.hidden});
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
                {program.hidden, value_width});
            return layer;
        }

        if (const auto* mamba = std::get_if<Mamba2Spec>(&layer_program.mixer)) {
            Mamba2Weights layer;
            const auto& spec = *mamba;
            const int conv_dim = spec.intermediate_size + 2 * spec.group_count * spec.state_size;
            layer.in = load_matrix(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::Mamba2Input, index),
                {2 * spec.intermediate_size + 2 * spec.group_count * spec.state_size +
                 spec.num_heads, program.hidden});
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
                {program.hidden, spec.intermediate_size});
            return layer;
        }
        if (const auto* mlp = std::get_if<MlpBlockSpec>(&layer_program.mixer)) {
            MlpOnlyWeights layer;
            const int intermediate = mlp->intermediate_size;
            layer.mlp_up = load_matrix(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::FfnUp, index),
                {intermediate, program.hidden});
            layer.w2 = load_matrix(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::FfnDown, index),
                {program.hidden, intermediate});
            return layer;
        }

        const auto* convolution =
            std::get_if<ShortConvolutionSpec>(&layer_program.mixer);
        if (!convolution) {
            throw std::logic_error("CPU weight loader received unknown mixer alternative");
        }
        ConvolutionWeights layer;
        layer.spec = *convolution;
        layer.in = load_matrix(source, reader.get(), writer.get(),
            tensor_name(weight_requests, TensorRole::ShortConvInput, index),
            {3 * program.hidden, program.hidden});
        const std::vector<float> channel_major =
            load_vector(source, reader.get(), writer.get(),
            tensor_name(weight_requests, TensorRole::ShortConvKernel, index),
            {program.hidden, 1, convolution->cache_length});
        layer.weight_tap_major.resize(channel_major.size());
        for (int tap = 0; tap < convolution->cache_length; ++tap) {
            for (int channel = 0; channel < program.hidden; ++channel) {
                layer.weight_tap_major[static_cast<size_t>(tap) * program.hidden + channel] =
                    channel_major[static_cast<size_t>(channel) * convolution->cache_length + tap];
            }
        }
        layer.out = load_matrix(source, reader.get(), writer.get(),
            tensor_name(weight_requests, TensorRole::ShortConvOutput, index),
            {program.hidden, program.hidden});
        return layer;
    };

    if (program.has_moe()) {
        weight_store.layers.reserve(program.layers.size());
        for (int index = 0; index < static_cast<int>(program.layers.size()); ++index) {
            const CompiledLayerProgram& layer_program = program.layers.at(
                static_cast<size_t>(index));
            if (!std::holds_alternative<MoeLayerProgram>(layer_program.feed_forward)) {
                CpuLayerWeights layer;
                layer.common = load_common(source, reader.get(), writer.get(), index);
                layer.mixer = load_operator(index, layer_program);
                if (std::holds_alternative<std::monostate>(layer_program.feed_forward)) {
                    layer.feed_forward = std::monostate{};
                } else {
                    layer.feed_forward =
                        load_dense_feed_forward(source, reader.get(), writer.get(), index);
                }
                weight_store.layers.push_back(std::move(layer));
                continue;
            }

            CpuLayerWeights layer;
            layer.common = load_common(source, reader.get(), writer.get(), index);
            MoeWeights moe;
            const auto has_request = [&](TensorRole role, int expert = -1) {
                return std::any_of(weight_requests.begin(), weight_requests.end(),
                    [&](const TensorRequest& request) {
                        return request.role == role && request.layer == index &&
                               request.expert == expert;
                    });
            };
            const bool individual_expert_model =
                has_request(TensorRole::MoeExpertGate, 0);
            const bool packed_expert_model =
                has_request(TensorRole::MoePackedGateUp);
            const MoeLayerProgram& moe_semantics = std::get<MoeLayerProgram>(
                program.layers.at(static_cast<size_t>(index)).feed_forward);
            layer.mixer = load_operator(index, layer_program);
            moe.num_experts = moe_semantics.router.expert_count;
            moe.experts_per_token = moe_semantics.router.experts_per_token;
            moe.normalize_topk = moe_semantics.router.normalization ==
                MoeNormalizationKind::SumSelected;
            moe.use_expert_bias = moe_semantics.router.has_expert_bias;
            moe.routed_scaling_factor = moe_semantics.router.routed_scaling;
            moe.router = load_vector(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::MoeRouter, index),
                {moe_semantics.router.expert_count, program.hidden});

            if (packed_expert_model) {
                if (moe_semantics.shared.has_value()) {
                    const int shared_intermediate =
                        moe_semantics.shared->mlp.intermediate_size;
                    moe.shared_w13 = load_concat(source, reader.get(), writer.get(),
                        layer_name(index, "shared_expert.w13.weight"), {
                            {tensor_name(weight_requests, TensorRole::MoeSharedGate, index),
                             {shared_intermediate, program.hidden}},
                            {tensor_name(weight_requests, TensorRole::MoeSharedUp, index),
                             {shared_intermediate, program.hidden}},
                        });
                    moe.shared_w2 = load_matrix(source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::MoeSharedDown, index),
                        {program.hidden, shared_intermediate});
                    if (has_request(TensorRole::MoeSharedGateWeight)) {
                        moe.shared_gate = load_matrix(source, reader.get(), writer.get(),
                            tensor_name(weight_requests,
                                        TensorRole::MoeSharedGateWeight, index),
                            {1, program.hidden});
                        moe.has_shared_gate = true;
                    }
                }
                moe.expert_w13 = CpuWeightCodec(source, reader.get(), writer.get(), group_size)
                    .packed_matrices(
                        tensor_name(weight_requests, TensorRole::MoePackedGateUp, index),
                        {moe_semantics.router.expert_count,
                         2 * moe_semantics.routed.mlp.intermediate_size, program.hidden});
                moe.expert_w2 = CpuWeightCodec(source, reader.get(), writer.get(), group_size)
                    .packed_matrices(
                        tensor_name(weight_requests, TensorRole::MoePackedDown, index),
                        {moe_semantics.router.expert_count, program.hidden,
                         moe_semantics.routed.mlp.intermediate_size});
            } else if (individual_expert_model && moe_semantics.shared.has_value()) {
                const int shared_intermediate =
                    moe_semantics.shared->mlp.intermediate_size;
                moe.shared_w13 = load_concat(source, reader.get(), writer.get(),
                    layer_name(index, "mlp.shared_experts.w13.weight"), {
                        {tensor_name(weight_requests, TensorRole::MoeSharedGate, index),
                         {shared_intermediate, program.hidden}},
                        {tensor_name(weight_requests, TensorRole::MoeSharedUp, index),
                         {shared_intermediate, program.hidden}},
                    });
                moe.shared_w2 = load_matrix(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::MoeSharedDown, index),
                    {program.hidden, shared_intermediate});
            }

            if (has_request(TensorRole::MoeRouterBias)) {
                moe.router_bias = load_vector(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::MoeRouterBias, index),
                    {moe_semantics.router.expert_count});
            }

            if (!disk_cached_experts) {
                moe.expert_w13.resize(static_cast<size_t>(moe_semantics.router.expert_count));
                moe.expert_w2.resize(static_cast<size_t>(moe_semantics.router.expert_count));
            }
            for (int expert = 0; expert < moe_semantics.router.expert_count; ++expert) {
                const int moe_inter = moe_semantics.routed.mlp.intermediate_size;
                if (moe_inter <= 0) {
                    throw std::runtime_error("compiled MoE layer has no expert width");
                }
                if (packed_expert_model) {
                    continue;
                }
                if (individual_expert_model) {
                    const std::string w13_name = layer_name(index,
                        "mlp.experts." + std::to_string(expert) + ".w13.weight");
                    moe.expert_w13[static_cast<size_t>(expert)] = load_concat(
                        source, reader.get(), writer.get(), w13_name,
                        {{tensor_name(weight_requests, TensorRole::MoeExpertGate,
                                      index, expert), {moe_inter, program.hidden}},
                         {tensor_name(weight_requests, TensorRole::MoeExpertUp,
                                      index, expert), {moe_inter, program.hidden}}});
                    moe.expert_w2[static_cast<size_t>(expert)] = load_matrix(
                        source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::MoeExpertDown,
                                    index, expert),
                        {program.hidden, moe_inter});
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
                     {moe_inter, program.hidden}},
                    {layer_name(index, prefix + ".w3.weight"),
                     {moe_inter, program.hidden}},
                };
                if (disk_cached_experts) {
                    (void)load_concat(source, nullptr, writer.get(),
                                      w13_name, parts);
                    (void)load_matrix(source, nullptr, writer.get(),
                                      w2_name, {program.hidden, moe_inter});
                } else {
                    moe.expert_w13[static_cast<size_t>(expert)] = load_concat(
                        source, reader.get(), writer.get(), w13_name, parts);
                    moe.expert_w2[static_cast<size_t>(expert)] = load_matrix(
                        source, reader.get(), writer.get(), w2_name,
                        {program.hidden, moe_inter});
                }
            }
            layer.feed_forward = std::move(moe);
            weight_store.layers.push_back(std::move(layer));
        }
    } else {
        weight_store.layers.reserve(program.layers.size());
        for (int index = 0; index < static_cast<int>(program.layers.size()); ++index) {
            const CompiledLayerProgram& layer_program =
                program.layers.at(static_cast<size_t>(index));
            CpuLayerWeights layer;
            layer.common = load_common(source, reader.get(), writer.get(), index);
            layer.mixer = load_operator(index, layer_program);
            if (std::holds_alternative<std::monostate>(layer_program.feed_forward)) {
                layer.feed_forward = std::monostate{};
            } else {
                layer.feed_forward =
                    load_dense_feed_forward(source, reader.get(), writer.get(), index);
            }
            weight_store.layers.push_back(std::move(layer));
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
        if (const auto* latent = semantics.latent_state(); latent && latent->factorized()) {
            require_matrix(attention.latent_q_projection, label + ".latent_q_projection");
            require_matrix(attention.latent_q_expansion, label + ".latent_q_expansion");
            require_matrix(attention.latent_k_projection, label + ".latent_k_projection");
            require_matrix(attention.latent_expansion, label + ".latent_expansion");
            require_matrix(attention.gate, label + ".gate");
        } else {
            require_matrix(attention.q, label + ".q");
            const bool owns_key_value =
                !semantics.uses_external_memory() &&
                !std::holds_alternative<SharedKvConsumer>(semantics.kv_sharing);
            if (owns_key_value) {
                require_matrix(attention.k, label + ".k");
                require_matrix(attention.v, label + ".v");
            }
            if (semantics.output_gate.has_value() && !semantics.output_gate->packed_with_query) {
                require_matrix(attention.gate, label + ".gate");
            }
        }
    };
    for (size_t layer = 0; layer < weight_store.layers.size(); ++layer) {
        const CpuLayerWeights& entry = weight_store.layers[layer];
        std::visit([&](const auto& value) {
            using Value = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Value, AttentionWeights>) {
                check_attention(value,
                                std::get<CompiledAttentionProgram>(
                                    program.layers[layer].mixer).semantics,
                                "layer " + std::to_string(layer) + ".attention");
            } else if constexpr (std::is_same_v<Value, ConvolutionWeights>) {
                require_matrix(value.in, "layer " + std::to_string(layer) + ".conv.in");
                require_matrix(value.out, "layer " + std::to_string(layer) + ".conv.out");
            } else if constexpr (std::is_same_v<Value, GatedDeltaNetWeights>) {
                require_matrix(value.out, "layer " + std::to_string(layer) + ".kda.out");
            } else if constexpr (std::is_same_v<Value, Mamba2Weights>) {
                require_matrix(value.in, "layer " + std::to_string(layer) + ".mamba.in");
                require_matrix(value.out, "layer " + std::to_string(layer) + ".mamba.out");
            }
        }, entry.mixer);
        std::visit([&](const auto& value) {
            using Value = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Value, DenseFeedForwardWeights>) {
                require_matrix(value.w13, "layer " + std::to_string(layer) + ".w13");
                require_matrix(value.w2, "layer " + std::to_string(layer) + ".w2");
            } else if constexpr (std::is_same_v<Value, MoeWeights>) {
                for (size_t expert = 0; expert < value.expert_w13.size(); ++expert) {
                    require_matrix(value.expert_w13[expert], "layer " +
                        std::to_string(layer) + ".expert_w13." + std::to_string(expert));
                    require_matrix(value.expert_w2[expert], "layer " +
                        std::to_string(layer) + ".expert_w2." + std::to_string(expert));
                }
                if (value.shared_w13.rows != 0) require_matrix(value.shared_w13, "shared_w13");
                if (value.shared_w2.rows != 0) require_matrix(value.shared_w2, "shared_w2");
            }
        }, entry.feed_forward);
    }
    if (writer) writer->commit();
}

}
