#include "detail/model_internal.hpp"

#include "celeg/detail/checkpoint/bootstrap.hpp"
#include "celeg/backend/cpu/compiler.hpp"
#include "celeg/backend/cpu/weight_codec.hpp"
#include "celeg/checkpoint/weight_repository.hpp"
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
                        int layer = -1) {
    return resolved_tensor_name(requests, role, layer);
}

std::string source_identity(const std::filesystem::path& path) {
    std::ostringstream out;
    out << std::filesystem::weakly_canonical(path).string() << ':';
    if (std::filesystem::is_regular_file(path)) {
        out << std::filesystem::file_size(path) << ':';
    }
    out << std::filesystem::last_write_time(path).time_since_epoch().count();
    return out.str();
}

std::filesystem::path default_cache_directory() {
    if (const char* xdg = std::getenv("XDG_CACHE_HOME")) {
        return std::filesystem::path(xdg) / "celeg";
    }
    if (const char* home = std::getenv("HOME")) {
        return std::filesystem::path(home) / ".cache" / "celeg";
    }
    return std::filesystem::temp_directory_path() / "celeg-cache";
}
}

CpuCompiledModel::Shared::Shared(const std::string& path, int context,
                               CpuModelOptions requested,
                               std::shared_ptr<const RuntimeContext> runtime_context)
    : model_path(path), runtime(std::move(runtime_context)), max_context(context),
      options(std::move(requested)),
      capabilities(detect_cpu_capabilities()),
      pool(options.threads, options.affinity),
      linear(resolve_isa(options.isa), pool) {
    if (max_context <= 0) throw std::invalid_argument("max_context must be positive");
    if (options.kv_page_tokens == 0) {
        throw std::invalid_argument("CPU KV page size must be positive");
    }
    if (options.prefill_chunk_tokens == 0) {
        throw std::invalid_argument("CPU prefill chunk size must be positive");
    }
    if (options.attention_parallel_threshold == 0 ||
        options.attention_page_tile == 0) {
        throw std::invalid_argument("CPU paged attention limits must be positive");
    }
    if (options.numa_mode == CpuNumaMode::ReplicateWeights) {
        throw std::invalid_argument(
            "CPU NUMA replicate-weights is reserved for a later backend; use local");
    }
    options.isa = linear.isa();
    group_size = options.weight_format == CpuWeightFormat::Q4Group64 ? 64 : 32;
    const detail::ModelBootstrap bootstrap =
        detail::load_model_bootstrap(std::filesystem::path(model_path), *runtime);
    const auto* native_storage = dynamic_cast<const INativeBlockStorageRepository*>(
        bootstrap.checkpoint.repository.get());
    native_checkpoint = native_storage != nullptr &&
                        native_storage->has_native_block_storage();
    shape = bootstrap.model.topology;
    tie_word_embeddings = bootstrap.model.capabilities.tied_embeddings;
    final_logit_softcap = bootstrap.model.topology.numerical_policy.final_logit_softcap;
    program = CpuModelCompiler{}.compile(bootstrap.model);
    model_identity = bootstrap.model.provenance.identity;
    weight_requests = bootstrap.model.weight_plan.requests;
    repository = bootstrap.checkpoint.repository;
    prepare_pack_path();
    if (options.expert_backing == CpuExpertBacking::DiskCached &&
        !native_checkpoint && (!options.use_pack_cache || pack_file.empty())) {
        throw std::invalid_argument(
            "CPU disk-backed experts require the CPU pack cache");
    }
    load_weights();
    CpuKvTopology kv_topology = build_cpu_kv_topology(shape, options);
    kv_pools = std::move(kv_topology.pools);
    layer_to_kv_pool = std::move(kv_topology.layer_to_pool);
    layer_to_kv_owner = std::move(kv_topology.layer_to_owner);
}

CpuIsa CpuCompiledModel::Shared::resolve_isa(CpuIsa requested) {
    const CpuCapabilities caps = detect_cpu_capabilities();
    if (requested == CpuIsa::Auto) return caps.best_isa();
    if (requested != CpuIsa::Scalar && requested != CpuIsa::Avx2 &&
        requested != CpuIsa::AvxVnni && requested != CpuIsa::Avx512Vnni &&
        requested != CpuIsa::Neon) {
        throw std::invalid_argument(
            "requested ISA is detected by the API but its native kernel is not implemented in v0.0.20");
    }
    if (!cpu_isa_compiled(requested)) {
        throw std::invalid_argument("requested CPU ISA was not compiled into this binary");
    }
    if (!caps.supports(requested)) {
        throw std::invalid_argument("requested CPU ISA is not supported by this host");
    }
    if (requested == CpuIsa::Avx2 && !caps.fma) {
        throw std::invalid_argument("AVX2 CPU backend requires FMA");
    }
    return requested;
}

void CpuCompiledModel::Shared::prepare_pack_path() {
    if (native_checkpoint || !options.use_pack_cache) return;
    std::filesystem::path directory = options.pack_cache_directory.empty()
        ? default_cache_directory() : options.pack_cache_directory;
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) throw std::runtime_error("cannot create CPU pack cache: " + error.message());
    const std::string source = source_identity(model_path);
    const size_t id = std::hash<std::string>{}(source);
    // Keep the on-disk name short.  The full topology fingerprint is useful
    // for diagnostics but can exceed MAX_PATH for large LFM2.5 models whose
    // vocab/layer schedule is encoded in the identity string.
    const size_t model_hash = std::hash<std::string>{}(model_identity);
    std::ostringstream filename;
    filename << "celeg-" << std::hex << model_hash << '-' << id
             << "-q4g" << group_size
             << '-' << cpu_isa_name(options.isa) << ".lfmpack";
    pack_file = directory / filename.str();
    source_id = source;
}

CpuCompiledModel::CommonWeights CpuCompiledModel::Shared::load_common(
    IWeightRepository* source, CpuPackReader* reader, CpuPackWriter* writer,
    int layer) {
    CommonWeights common;
    common.operator_norm = load_vector(source, reader, writer,
        tensor_name(weight_requests, TensorRole::AttentionInputNorm, layer),
        {shape.hidden});
    if (shape.numerical_policy.rms_norm_add_one) {
        for (float& value : common.operator_norm) value += 1.0f;
    }
    const CompiledLayerProgram& layer_program = program.layers.at(
        static_cast<size_t>(layer));
    if (!layer_program.execute_feed_forward) {
        common.ffn_norm = common.operator_norm;
        if (layer_program.mixer == CompiledMixer::MlpOnly) {
            const int intermediate = shape.mlp_only_layouts.at(static_cast<size_t>(layer)).intermediate_size;
            common.mlp_up = load_matrix(source, reader, writer,
                tensor_name(weight_requests, TensorRole::FfnUp, layer),
                {intermediate, shape.hidden});
            common.w2 = load_matrix(source, reader, writer,
                tensor_name(weight_requests, TensorRole::FfnDown, layer),
                {shape.hidden, intermediate});
        }
        return common;
    }
    if (shape.has_split_attention_norms) {
        common.post_attention_norm = load_vector(source, reader, writer,
            tensor_name(weight_requests, TensorRole::AttentionPostNorm, layer),
            {shape.hidden});
    }
    common.ffn_norm = load_vector(source, reader, writer,
        tensor_name(weight_requests, TensorRole::FfnInputNorm, layer),
        {shape.hidden});
    if (shape.numerical_policy.rms_norm_add_one) {
        for (float& value : common.ffn_norm) value += 1.0f;
    }
    if (shape.has_split_attention_norms) {
        common.post_feed_forward_norm = load_vector(source, reader, writer,
            tensor_name(weight_requests, TensorRole::FfnOutputNorm, layer),
            {shape.hidden});
    }
    const int intermediate = shape.feed_forward_intermediates.empty()
        ? shape.intermediate : shape.feed_forward_intermediates.at(static_cast<size_t>(layer));
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
    if (shape.has_per_layer_input) {
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
        {shape.vocab_size, shape.hidden});
    if (!tie_word_embeddings) {
        weight_store.lm_head = load_matrix(source, reader.get(), writer.get(),
            tensor_name(weight_requests, TensorRole::LanguageModelHead),
            {shape.vocab_size, shape.hidden});
    }
    weight_store.final_norm = load_vector(source, reader.get(), writer.get(),
        tensor_name(weight_requests, TensorRole::FinalNorm),
        {shape.hidden});
    if (shape.numerical_policy.rms_norm_add_one) {
        for (float& value : weight_store.final_norm) value += 1.0f;
    }
    if (shape.has_per_layer_input) {
        weight_store.per_layer_embedding = load_matrix(source, reader.get(), writer.get(),
            tensor_name(weight_requests, TensorRole::PerLayerEmbedding),
                            {shape.vocab_size, shape.num_hidden_layers * shape.per_layer_input_size});
        weight_store.per_layer_context_projection = load_matrix(source, reader.get(), writer.get(),
            tensor_name(weight_requests, TensorRole::PerLayerContextProjection),
            {shape.num_hidden_layers * shape.per_layer_input_size, shape.hidden});
        weight_store.per_layer_projection_norm = load_vector(source, reader.get(), writer.get(),
            tensor_name(weight_requests, TensorRole::PerLayerProjectionNorm),
            {shape.per_layer_input_size});
    }

    const auto load_operator = [&](int index, MixerKind layer_type)
        -> std::variant<AttentionWeights, ConvolutionWeights, GatedDeltaNetWeights,
                        Mamba2Weights, MlpOnlyWeights> {
        if (layer_type == MixerKind::Attention) {
            AttentionWeights layer;
            const AttentionSpec& attention = shape.attention_layout(index);
            layer.q = load_matrix(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::AttentionQuery, index),
                {attention.query_projection_width(), shape.hidden});
            if (!attention.kv_sharing.shared() || attention.kv_sharing.publishes) {
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
            if (!attention.query_key_norm) {
                layer.q_norm.assign(static_cast<size_t>(attention.head_dim), 1.0f);
                layer.k_norm.assign(static_cast<size_t>(attention.head_dim), 1.0f);
            } else {
                layer.q_norm = load_vector(source, reader.get(), writer.get(),
                    tensor_name(weight_requests, TensorRole::AttentionQueryNorm, index),
                    {attention.head_dim});
                if (shape.numerical_policy.rms_norm_add_one) {
                    for (float& value : layer.q_norm) value += 1.0f;
                }
                if (!attention.kv_sharing.shared() || attention.kv_sharing.publishes) {
                    layer.k_norm = load_vector(source, reader.get(), writer.get(),
                        tensor_name(weight_requests, TensorRole::AttentionKeyNorm, index),
                        {attention.head_dim});
                    if (shape.numerical_policy.rms_norm_add_one) {
                        for (float& value : layer.k_norm) value += 1.0f;
                    }
                }
            }
            return layer;
        }

        if (layer_type == MixerKind::GatedDeltaNet) {
            GatedDeltaNetWeights layer;
            layer.spec = shape.gated_delta_net_layouts.at(static_cast<size_t>(index));
            const auto& spec = layer.spec;
            const int key_width = spec.key_heads * spec.key_head_dim;
            const int value_width = spec.value_heads * spec.value_head_dim;
            const int qkv_width = 2 * key_width + value_width;
            layer.qkv = load_matrix(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::GatedDeltaNetQkv, index),
                {qkv_width, shape.hidden});
            layer.z = load_matrix(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::GatedDeltaNetZ, index),
                {value_width, shape.hidden});
            layer.b = load_matrix(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::GatedDeltaNetBeta, index),
                {spec.value_heads, shape.hidden});
            layer.a = load_matrix(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::GatedDeltaNetAlpha, index),
                {spec.value_heads, shape.hidden});
            const int conv_dim = qkv_width;
            layer.conv_weight = load_vector(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::GatedDeltaNetConv, index),
                {conv_dim, 1, spec.conv_kernel});
            layer.dt_bias = load_vector(source, reader.get(), writer.get(),
                tensor_name(weight_requests, TensorRole::GatedDeltaNetDtBias, index),
                {spec.value_heads});
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

        if (layer_type == MixerKind::Mamba2) {
            Mamba2Weights layer;
            const auto& spec = shape.mamba2_layouts.at(static_cast<size_t>(index));
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
        if (layer_type == MixerKind::MlpOnly) return MlpOnlyWeights{};

        ConvolutionWeights layer;
        layer.in = load_matrix(source, reader.get(), writer.get(),
            layer_name(index, "conv.in_proj.weight"),
            {3 * shape.hidden, shape.hidden});
        const std::vector<float> channel_major =
            load_vector(source, reader.get(), writer.get(),
            layer_name(index, "conv.conv.weight"),
            {shape.hidden, 1, shape.conv_cache});
        layer.weight_tap_major.resize(channel_major.size());
        for (int tap = 0; tap < shape.conv_cache; ++tap) {
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
            const MixerKind layer_type = shape.mixer_kinds[static_cast<size_t>(index)];
            if (!shape.layer_uses_moe(index)) {
                CommonWeights common =
                    load_common(source, reader.get(), writer.get(), index);
                auto layer = load_operator(index, layer_type);
                std::visit([&](auto& value) {
                    value.common = std::move(common);
                    weight_store.layers.emplace_back(std::move(value));
                }, layer);
                continue;
            }

            MoeWeights layer;
            const bool qwen_moe = shape.numerical_policy.rms_norm_add_one;
            layer.common.operator_norm = load_vector(source, reader.get(), writer.get(),
                qwen_moe
                    ? tensor_name(weight_requests, TensorRole::AttentionInputNorm, index)
                    : layer_name(index, "operator_norm.weight"), {shape.hidden});
            layer.common.ffn_norm = load_vector(source, reader.get(), writer.get(),
                qwen_moe
                    ? tensor_name(weight_requests, TensorRole::FfnInputNorm, index)
                    : layer_name(index, "ffn_norm.weight"), {shape.hidden});
            if (qwen_moe) {
                for (float& value : layer.common.operator_norm) value += 1.0f;
                for (float& value : layer.common.ffn_norm) value += 1.0f;
            }
            layer.operator_layer = load_operator(index, layer_type);
            layer.num_experts = shape.num_experts;
            layer.experts_per_token = shape.experts_per_token;
            layer.normalize_topk = shape.normalize_topk;
            layer.use_expert_bias = shape.use_expert_bias;
            layer.routed_scaling_factor = shape.routed_scaling_factor;
            layer.router = load_vector(source, reader.get(), writer.get(),
                qwen_moe
                    ? tensor_name(weight_requests, TensorRole::MoeRouter, index)
                    : layer_name(index, "feed_forward.gate.weight"),
                {shape.num_experts, shape.hidden});

            if (qwen_moe) {
                const int shared_intermediate = shape.shared_expert_intermediate;
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
                layer.expert_w13 = CpuWeightCodec(source, reader.get(), writer.get(), group_size)
                    .packed_matrices(
                        tensor_name(weight_requests, TensorRole::MoePackedGateUp, index),
                        {shape.num_experts, 2 * shape.moe_intermediate, shape.hidden});
                layer.expert_w2 = CpuWeightCodec(source, reader.get(), writer.get(), group_size)
                    .packed_matrices(
                        tensor_name(weight_requests, TensorRole::MoePackedDown, index),
                        {shape.num_experts, shape.hidden, shape.moe_intermediate});
            }

            const std::string bias_name =
                layer_name(index, "feed_forward.expert_bias.weight");
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
                const int moe_inter = shape.moe_intermediate > 0
                    ? shape.moe_intermediate : shape.intermediate;
                if (qwen_moe) {
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
            auto layer = load_operator(
                index, shape.mixer_kinds[static_cast<size_t>(index)]);
            std::visit([&](auto& value) {
                value.common = std::move(common);
                weight_store.layers.emplace_back(std::move(value));
            }, layer);
        }
    }
    if (writer) writer->commit();
}

CpuLinearWeight CpuCompiledModel::Shared::load_matrix(
    IWeightRepository* source, CpuPackReader* reader, CpuPackWriter* writer,
    const std::string& name, const std::vector<int64_t>& expected) {
    return CpuWeightCodec(source, reader, writer, group_size).matrix(name, expected);
}

CpuLinearWeight CpuCompiledModel::Shared::load_concat(
    IWeightRepository* source, CpuPackReader* reader, CpuPackWriter* writer,
    const std::string& synthetic,
    const std::vector<std::pair<std::string, std::vector<int64_t>>>& parts) {
    return CpuWeightCodec(source, reader, writer, group_size).concat(synthetic, parts);
}

std::vector<float> CpuCompiledModel::Shared::load_vector(
    IWeightRepository* source, CpuPackReader* reader, CpuPackWriter* writer,
    const std::string& name, const std::vector<int64_t>& expected) {
    return CpuWeightCodec(source, reader, writer, group_size).vector(name, expected);
}

size_t CpuCompiledModel::Shared::weights_memory_bytes() const {
    size_t bytes = weight_store.embedding.memory_bytes() +
        weight_store.final_norm.size() * sizeof(float);
    for (const WeightLayer& layer : weight_store.layers) {
        std::visit([&](const auto& value) {
            bytes += value.common.operator_norm.size() * sizeof(float) +
                     value.common.ffn_norm.size() * sizeof(float) +
                     value.common.w13.memory_bytes() +
                     value.common.w2.memory_bytes();
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, AttentionWeights>) {
                bytes += value.q.memory_bytes() + value.k.memory_bytes() +
                         value.v.memory_bytes() + value.out.memory_bytes() +
                         value.q_norm.size() * sizeof(float) +
                         value.k_norm.size() * sizeof(float);
            } else if constexpr (std::is_same_v<T, ConvolutionWeights>) {
                bytes += value.in.memory_bytes() + value.out.memory_bytes() +
                         value.weight_tap_major.size() * sizeof(float);
            } else if constexpr (std::is_same_v<T, Mamba2Weights>) {
                bytes += value.in.memory_bytes() + value.out.memory_bytes() +
                         (value.conv_weight.size() + value.conv_bias.size() +
                          value.dt_bias.size() + value.a_log.size() + value.d.size() +
                          value.norm.size()) * sizeof(float);
            } else if constexpr (std::is_same_v<T, GatedDeltaNetWeights>) {
                bytes += value.qkv.memory_bytes() + value.z.memory_bytes() +
                         value.a.memory_bytes() + value.b.memory_bytes() +
                         value.out.memory_bytes() +
                         (value.conv_weight.size() + value.dt_bias.size() +
                          value.a_log.size() + value.norm.size()) * sizeof(float);
            } else if constexpr (std::is_same_v<T, MlpOnlyWeights>) {
                bytes += value.common.mlp_up.memory_bytes();
            } else {
                bytes += (value.router.size() + value.router_bias.size()) * sizeof(float);
                std::visit([&](const auto& operator_weights) {
                    using Operator = std::decay_t<decltype(operator_weights)>;
                    if constexpr (std::is_same_v<Operator, AttentionWeights>) {
                        bytes += operator_weights.q.memory_bytes() +
                                 operator_weights.k.memory_bytes() +
                                 operator_weights.v.memory_bytes() +
                                 operator_weights.out.memory_bytes() +
                                 (operator_weights.q_norm.size() +
                                  operator_weights.k_norm.size()) * sizeof(float);
                    } else if constexpr (std::is_same_v<Operator, ConvolutionWeights>) {
                        bytes += operator_weights.in.memory_bytes() +
                                 operator_weights.out.memory_bytes() +
                                 operator_weights.weight_tap_major.size() * sizeof(float);
                    } else if constexpr (std::is_same_v<Operator, Mamba2Weights>) {
                        bytes += operator_weights.in.memory_bytes() +
                                 operator_weights.out.memory_bytes();
                    } else if constexpr (std::is_same_v<Operator, GatedDeltaNetWeights>) {
                        bytes += operator_weights.qkv.memory_bytes() +
                                 operator_weights.z.memory_bytes() +
                                 operator_weights.a.memory_bytes() +
                                 operator_weights.b.memory_bytes() +
                                 operator_weights.out.memory_bytes();
                    }
                }, value.operator_layer);
                for (const CpuLinearWeight& weight : value.expert_w13) {
                    bytes += weight.memory_bytes();
                }
                for (const CpuLinearWeight& weight : value.expert_w2) {
                    bytes += weight.memory_bytes();
                }
                bytes += value.shared_w13.memory_bytes() + value.shared_w2.memory_bytes() +
                         value.shared_gate.memory_bytes();
            }
        }, layer);
    }
    return bytes;
}

} // namespace celeg
