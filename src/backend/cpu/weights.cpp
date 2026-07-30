#include "detail/model_internal.hpp"

#include "lfm/detail/checkpoint/bootstrap.hpp"
#include "lfm/checkpoint/repositories/gguf.hpp"
#include "lfm/checkpoint/repositories/safetensors.hpp"
#include "lfm/checkpoint/tensor_names.hpp"
#include "lfm/model/config/variant.hpp"
#include "lfm/model/weights/quantization.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace lfm {
namespace {
size_t checked_elements(const std::vector<int64_t>& shape) {
    size_t result = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) throw std::runtime_error("invalid tensor dimension");
        if (result > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
            throw std::overflow_error("tensor dimensions overflow");
        }
        result *= static_cast<size_t>(dim);
    }
    return result;
}

bool tensor_shape_matches(const std::vector<int64_t>& actual,
                          const std::vector<int64_t>& expected) {
    if (actual == expected) return true;
    return actual.size() == 2 && expected.size() == 3 &&
           expected[1] == 1 && actual[0] == expected[0] &&
           actual[1] == expected[2];
}

float fp16_to_float(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fu;
    uint32_t mantissa = bits & 0x03ffu;
    uint32_t result = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x03ffu;
            result = sign |
                static_cast<uint32_t>(127 - 15 - shift) << 23 |
                mantissa << 13;
        }
    } else if (exponent == 31) {
        result = sign | 0x7f800000u | mantissa << 13;
    } else {
        result = sign | (exponent + (127 - 15)) << 23 | mantissa << 13;
    }
    return std::bit_cast<float>(result);
}

std::vector<float> host_vector(const HostTensorView& tensor,
                               const std::vector<int64_t>& expected,
                               const std::string& name) {
    if (!tensor_shape_matches(tensor.shape, expected)) {
        throw std::runtime_error("unexpected CPU tensor: " + name);
    }
    const size_t count = checked_elements(expected);
    std::vector<float> result(count);
    if (tensor.dtype == TensorDType::BF16 ||
        tensor.dtype == TensorDType::F16) {
        if (tensor.bytes != count * sizeof(uint16_t)) {
            throw std::runtime_error("invalid 16-bit tensor bytes for " + name);
        }
        for (size_t i = 0; i < count; ++i) {
            uint16_t bits = 0;
            std::memcpy(&bits, tensor.data + i * sizeof(uint16_t), sizeof(bits));
            result[i] = tensor.dtype == TensorDType::BF16
                ? bf16_bits_to_float(bits) : fp16_to_float(bits);
        }
        return result;
    }
    if (tensor.dtype == TensorDType::F32) {
        if (tensor.bytes != count * sizeof(float)) {
            throw std::runtime_error("invalid F32 tensor bytes for " + name);
        }
        std::memcpy(result.data(), tensor.data, tensor.bytes);
        return result;
    }
    throw std::runtime_error(
        "CPU vector requires F32, F16 or BF16 source: " + name);
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
        return std::filesystem::path(xdg) / "lfm25";
    }
    if (const char* home = std::getenv("HOME")) {
        return std::filesystem::path(home) / ".cache" / "lfm25";
    }
    return std::filesystem::temp_directory_path() / "lfm25-cache";
}
}

CpuModel::Impl::Shared::Shared(const std::string& path, int context,
                               CpuModelOptions requested)
    : model_path(path), max_context(context), options(std::move(requested)),
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
        detail::load_model_bootstrap(std::filesystem::path(model_path));
    is_gguf = bootstrap.is_gguf;
    shape = bootstrap.shape;
    variant = bootstrap.variant;
    if (is_gguf) {
        repository = std::make_shared<GgufRepository>(bootstrap.gguf_file);
    } else {
        repository = std::make_shared<SafeTensorRepository>(
            std::filesystem::path(model_path));
    }
    prepare_pack_path();
    load_weights();
    layer_to_kv_pool.assign(layers.size(), -1);
    for (size_t layer = 0; layer < layers.size(); ++layer) {
        if (Impl::attention_operator(layers[layer])) {
            layer_to_kv_pool[layer] = static_cast<int>(kv_pools.size());
            kv_pools.push_back(std::make_shared<CpuKvPagePool>(
                options.kv_cache_mode, options.kv_page_tokens,
                static_cast<size_t>(shape.kv_width)));
        }
    }
}

CpuIsa CpuModel::Impl::Shared::resolve_isa(CpuIsa requested) {
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

void CpuModel::Impl::Shared::prepare_pack_path() {
    if (is_gguf || !options.use_pack_cache) return;
    std::filesystem::path directory = options.pack_cache_directory.empty()
        ? default_cache_directory() : options.pack_cache_directory;
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) throw std::runtime_error("cannot create CPU pack cache: " + error.message());
    const std::string source = source_identity(model_path);
    const size_t id = std::hash<std::string>{}(source);
    const std::string variant_id = variant ? std::string(variant->id()) : "unknown";
    std::ostringstream filename;
    filename << variant_id << '-' << std::hex << id << "-q4g" << group_size
             << '-' << cpu_isa_name(options.isa) << ".lfmpack";
    pack_file = directory / filename.str();
    source_id = source;
}

CpuModel::Impl::CommonWeights CpuModel::Impl::Shared::load_common(
    IWeightRepository* source, CpuPackReader* reader, CpuPackWriter* writer,
    int layer) {
    CommonWeights common;
    common.operator_norm = load_vector(source, reader, writer,
        layer_name(layer, "operator_norm.weight"), {shape.hidden});
    common.ffn_norm = load_vector(source, reader, writer,
        layer_name(layer, "ffn_norm.weight"), {shape.hidden});
    common.w13 = load_concat(source, reader, writer,
        layer_name(layer, "feed_forward.w13.weight"), {
            {layer_name(layer, "feed_forward.w1.weight"),
             {shape.intermediate, shape.hidden}},
            {layer_name(layer, "feed_forward.w3.weight"),
             {shape.intermediate, shape.hidden}},
        });
    common.w2 = load_matrix(source, reader, writer,
        layer_name(layer, "feed_forward.w2.weight"),
        {shape.hidden, shape.intermediate});
    return common;
}

void CpuModel::Impl::Shared::load_weights() {
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
    embedding = load_matrix(source, reader.get(), writer.get(),
        "model.embed_tokens.weight", {shape.vocab_size, shape.hidden});
    final_norm = load_vector(source, reader.get(), writer.get(),
        "model.embedding_norm.weight", {shape.hidden});

    // Attention and short-convolution are identical between dense and MoE
    // layers. Keeping their loader in one place prevents the two checkpoint
    // paths from drifting when an operator tensor changes.
    const auto load_operator = [&](int index, LayerType layer_type)
        -> std::variant<AttentionWeights, ConvolutionWeights> {
        if (layer_type == LayerType::FullAttention) {
            AttentionWeights layer;
            layer.qkv = load_concat(source, reader.get(), writer.get(),
                layer_name(index, "self_attn.qkv.weight"), {
                    {layer_name(index, "self_attn.q_proj.weight"),
                     {shape.q_width, shape.hidden}},
                    {layer_name(index, "self_attn.k_proj.weight"),
                     {shape.kv_width, shape.hidden}},
                    {layer_name(index, "self_attn.v_proj.weight"),
                     {shape.kv_width, shape.hidden}},
                });
            layer.out = load_matrix(source, reader.get(), writer.get(),
                layer_name(index, "self_attn.out_proj.weight"),
                {shape.hidden, shape.hidden});
            layer.q_norm = load_vector(source, reader.get(), writer.get(),
                layer_name(index, "self_attn.q_layernorm.weight"),
                {shape.head_dim});
            layer.k_norm = load_vector(source, reader.get(), writer.get(),
                layer_name(index, "self_attn.k_layernorm.weight"),
                {shape.head_dim});
            return layer;
        }

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

    if (shape.architecture == ArchitectureKind::MoeLfm2) {
        // MoE checkpoints begin with ordinary dense FFN blocks. Keep them as
        // dense layers, then load only the remaining routed blocks as MoE.
        layers.reserve(static_cast<size_t>(shape.num_hidden_layers));
        for (int index = 0; index < shape.num_hidden_layers; ++index) {
            const LayerType layer_type = shape.layer_types[static_cast<size_t>(index)];
            if (!shape.layer_uses_moe(index)) {
                CommonWeights common =
                    load_common(source, reader.get(), writer.get(), index);
                auto layer = load_operator(index, layer_type);
                std::visit([&](auto& value) {
                    value.common = std::move(common);
                    layers.emplace_back(std::move(value));
                }, layer);
                continue;
            }

            MoeWeights layer;
            layer.common.operator_norm = load_vector(source, reader.get(), writer.get(),
                layer_name(index, "operator_norm.weight"), {shape.hidden});
            layer.common.ffn_norm = load_vector(source, reader.get(), writer.get(),
                layer_name(index, "ffn_norm.weight"), {shape.hidden});
            layer.operator_layer = load_operator(index, layer_type);

            // Load MoE-specific weights: router + per-expert gate_up (w1+w3) and down (w2).
            layer.num_experts = shape.num_experts;
            layer.experts_per_token = shape.experts_per_token;
            layer.normalize_topk = shape.normalize_topk;
            layer.use_expert_bias = shape.use_expert_bias;
            layer.routed_scaling_factor = shape.routed_scaling_factor;

            // Router weight: [num_experts, hidden]
            layer.router = load_vector(source, reader.get(), writer.get(),
                layer_name(index, "feed_forward.gate.weight"),
                {shape.num_experts, shape.hidden});

            // Optional expert bias.
            if (source && source->contains(layer_name(index, "feed_forward.expert_bias.weight"))) {
                layer.router_bias = load_vector(source, reader.get(), writer.get(),
                    layer_name(index, "feed_forward.expert_bias.weight"),
                    {shape.num_experts});
            }

            layer.expert_w13.resize(static_cast<size_t>(shape.num_experts));
            layer.expert_w2.resize(static_cast<size_t>(shape.num_experts));
            for (int e = 0; e < shape.num_experts; ++e) {
                const int moe_inter = shape.moe_intermediate > 0
                    ? shape.moe_intermediate : shape.intermediate;
                layer.expert_w13[static_cast<size_t>(e)] = load_concat(
                    source, reader.get(), writer.get(),
                    layer_name(index, "feed_forward.experts." + std::to_string(e) + ".w13.weight"),
                    {
                        {layer_name(index, "feed_forward.experts." + std::to_string(e) + ".w1.weight"),
                         {moe_inter, shape.hidden}},
                        {layer_name(index, "feed_forward.experts." + std::to_string(e) + ".w3.weight"),
                         {moe_inter, shape.hidden}},
                    });
                layer.expert_w2[static_cast<size_t>(e)] = load_matrix(
                    source, reader.get(), writer.get(),
                    layer_name(index, "feed_forward.experts." + std::to_string(e) + ".w2.weight"),
                    {shape.hidden, moe_inter});
            }
            layers.push_back(std::move(layer));
        }
    } else {
        layers.reserve(static_cast<size_t>(shape.num_hidden_layers));
        for (int index = 0; index < shape.num_hidden_layers; ++index) {
            CommonWeights common =
                load_common(source, reader.get(), writer.get(), index);
            auto layer = load_operator(
                index, shape.layer_types[static_cast<size_t>(index)]);
            std::visit([&](auto& value) {
                value.common = std::move(common);
                layers.emplace_back(std::move(value));
            }, layer);
        }
    }
    if (writer) writer->commit();
}

CpuLinearWeight CpuModel::Impl::Shared::load_matrix(
    IWeightRepository* source, CpuPackReader* reader, CpuPackWriter* writer,
    const std::string& name, const std::vector<int64_t>& expected) {
    if (reader) {
        return CpuLinearWeight::from_q4(reader->read_q4_matrix(name));
    }
    if (!source) throw std::logic_error("CPU weight source is missing");
    const HostTensorView tensor = source->tensor(name);
    if (tensor.shape != expected || expected.size() != 2) {
        throw std::runtime_error("unexpected CPU linear tensor: " + name);
    }
    if (tensor.dtype == TensorDType::Quantized) {
        if (tensor.ggml_type != GgmlType::Q4_K &&
            tensor.ggml_type != GgmlType::Q6_K) {
            throw std::runtime_error(
                "CPU GGUF supports only Q4_K/Q6_K linear tensor: " + name);
        }
        CpuGgufMatrix matrix;
        matrix.type = tensor.ggml_type;
        matrix.rows = static_cast<uint32_t>(expected[0]);
        matrix.cols = static_cast<uint32_t>(expected[1]);
        matrix.data = tensor.data;
        matrix.bytes = tensor.bytes;
        return CpuLinearWeight::from_gguf(matrix);
    }
    if (tensor.dtype != TensorDType::BF16) {
        throw std::runtime_error(
            "Safetensors CPU linear tensor must be BF16: " + name);
    }
    Q4GroupMatrix matrix = quantize_bf16_groupwise_q4(
        tensor.data, static_cast<size_t>(expected[0]),
        static_cast<size_t>(expected[1]), group_size);
    if (writer) writer->add_q4_matrix(name, matrix);
    return CpuLinearWeight::from_q4(std::move(matrix));
}

CpuLinearWeight CpuModel::Impl::Shared::load_concat(
    IWeightRepository* source, CpuPackReader* reader, CpuPackWriter* writer,
    const std::string& synthetic,
    const std::vector<std::pair<std::string, std::vector<int64_t>>>& parts) {
    if (reader) {
        return CpuLinearWeight::from_q4(reader->read_q4_matrix(synthetic));
    }
    if (!source || parts.empty()) {
        throw std::logic_error("invalid CPU concat source");
    }
    const int64_t cols = parts.front().second[1];
    size_t total_rows = 0;
    std::vector<HostTensorView> tensors;
    tensors.reserve(parts.size());
    bool quantized = true;
    for (const auto& [name, expected] : parts) {
        const HostTensorView tensor = source->tensor(name);
        if (tensor.shape != expected || expected.size() != 2 ||
            expected[1] != cols) {
            throw std::runtime_error("unexpected CPU concat tensor: " + name);
        }
        total_rows += static_cast<size_t>(expected[0]);
        quantized = quantized && tensor.dtype == TensorDType::Quantized;
        tensors.push_back(tensor);
    }

    if (quantized) {
        CpuLinearWeight result;
        result.rows = static_cast<uint32_t>(total_rows);
        result.cols = static_cast<uint32_t>(cols);
        for (size_t index = 0; index < tensors.size(); ++index) {
            const HostTensorView& tensor = tensors[index];
            if (tensor.ggml_type != GgmlType::Q4_K &&
                tensor.ggml_type != GgmlType::Q6_K) {
                throw std::runtime_error(
                    "CPU GGUF concat supports only Q4_K/Q6_K: " +
                    parts[index].first);
            }
            CpuGgufMatrix matrix;
            matrix.type = tensor.ggml_type;
            matrix.rows = static_cast<uint32_t>(tensor.shape[0]);
            matrix.cols = static_cast<uint32_t>(tensor.shape[1]);
            matrix.data = tensor.data;
            matrix.bytes = tensor.bytes;
            matrix.validate();
            result.segments.emplace_back(matrix);
        }
        result.validate();
        return result;
    }
    if (std::any_of(tensors.begin(), tensors.end(),
                    [](const HostTensorView& tensor) {
                        return tensor.dtype == TensorDType::Quantized;
                    })) {
        throw std::runtime_error(
            "CPU GGUF concat cannot mix quantized and dense tensors: " +
            synthetic);
    }

    std::vector<float> joined(total_rows * static_cast<size_t>(cols));
    size_t row_offset = 0;
    for (size_t index = 0; index < parts.size(); ++index) {
        const auto& [name, expected] = parts[index];
        if (tensors[index].dtype != TensorDType::BF16) {
            throw std::runtime_error(
                "Safetensors CPU concat tensor must be BF16: " + name);
        }
        const std::vector<float> values =
            host_vector(tensors[index], expected, name);
        std::copy(values.begin(), values.end(),
                  joined.begin() + static_cast<ptrdiff_t>(
                      row_offset * static_cast<size_t>(cols)));
        row_offset += static_cast<size_t>(expected[0]);
    }
    Q4GroupMatrix matrix = quantize_float_groupwise_q4(
        joined.data(), total_rows, static_cast<size_t>(cols), group_size);
    if (writer) writer->add_q4_matrix(synthetic, matrix);
    return CpuLinearWeight::from_q4(std::move(matrix));
}

std::vector<float> CpuModel::Impl::Shared::load_vector(
    IWeightRepository* source, CpuPackReader* reader, CpuPackWriter* writer,
    const std::string& name, const std::vector<int64_t>& expected) {
    if (reader) return reader->read_bf16_vector(name);
    if (!source) throw std::logic_error("CPU vector source is missing");
    const HostTensorView tensor = source->tensor(name);
    std::vector<float> result = host_vector(tensor, expected, name);
    if (writer) {
        if (tensor.dtype != TensorDType::BF16) {
            throw std::runtime_error(
                "CPU pack vectors require BF16 source: " + name);
        }
        writer->add_bf16_vector(name, tensor.data, result.size());
    }
    return result;
}

size_t CpuModel::Impl::Shared::weights_memory_bytes() const {
    size_t bytes = embedding.memory_bytes() + final_norm.size() * sizeof(float);
    for (const WeightLayer& layer : layers) {
        std::visit([&](const auto& value) {
            bytes += value.common.operator_norm.size() * sizeof(float) +
                     value.common.ffn_norm.size() * sizeof(float) +
                     value.common.w13.memory_bytes() +
                     value.common.w2.memory_bytes();
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, AttentionWeights>) {
                bytes += value.qkv.memory_bytes() + value.out.memory_bytes() +
                         value.q_norm.size() * sizeof(float) +
                         value.k_norm.size() * sizeof(float);
            } else if constexpr (std::is_same_v<T, ConvolutionWeights>) {
                bytes += value.in.memory_bytes() + value.out.memory_bytes() +
                         value.weight_tap_major.size() * sizeof(float);
            } else {
                bytes += (value.router.size() + value.router_bias.size()) * sizeof(float);
                std::visit([&](const auto& operator_weights) {
                    using Operator = std::decay_t<decltype(operator_weights)>;
                    if constexpr (std::is_same_v<Operator, AttentionWeights>) {
                        bytes += operator_weights.qkv.memory_bytes() +
                                 operator_weights.out.memory_bytes() +
                                 (operator_weights.q_norm.size() +
                                  operator_weights.k_norm.size()) * sizeof(float);
                    } else {
                        bytes += operator_weights.in.memory_bytes() +
                                 operator_weights.out.memory_bytes() +
                                 operator_weights.weight_tap_major.size() * sizeof(float);
                    }
                }, value.operator_layer);
                for (const CpuLinearWeight& weight : value.expert_w13) {
                    bytes += weight.memory_bytes();
                }
                for (const CpuLinearWeight& weight : value.expert_w2) {
                    bytes += weight.memory_bytes();
                }
            }
        }, layer);
    }
    return bytes;
}

} // namespace lfm
