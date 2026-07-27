#include "detail/model_internal.hpp"

#include "lfm/detail/checkpoint/bootstrap.hpp"
#include "lfm/model/config/variant.hpp"
#include "lfm/model/weights/quantization.hpp"
#include "lfm/checkpoint/formats/safetensors.hpp"

#include <algorithm>
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
std::string layer_name(int index, const std::string& suffix) {
    return "model.layers." + std::to_string(index) + "." + suffix;
}

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

std::vector<float> bf16_vector(const HostTensorView& tensor,
                               const std::vector<int64_t>& expected,
                               const std::string& name) {
    if (tensor.dtype != TensorDType::BF16 || tensor.shape != expected) {
        throw std::runtime_error("unexpected CPU tensor: " + name);
    }
    const size_t count = checked_elements(expected);
    if (tensor.bytes != count * sizeof(uint16_t)) {
        throw std::runtime_error("invalid BF16 bytes for " + name);
    }
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) {
        uint16_t bits = 0;
        std::memcpy(&bits, tensor.data + i * sizeof(uint16_t), sizeof(bits));
        result[i] = bf16_bits_to_float(bits);
    }
    return result;
}

std::string source_identity(const std::filesystem::path& path) {
    std::ostringstream out;
    out << std::filesystem::weakly_canonical(path).string() << ':'
        << std::filesystem::file_size(path) << ':'
        << std::filesystem::last_write_time(path).time_since_epoch().count();
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
    : safetensors_path(path), max_context(context), options(std::move(requested)),
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
    // Load model topology from config.json next to the safetensors file so the
    // CPU runtime no longer depends on the 230M constexpr shape.
    const detail::ModelBootstrap bootstrap =
        detail::load_model_bootstrap(std::filesystem::path(safetensors_path));
    shape = bootstrap.shape;
    variant = bootstrap.variant;
    prepare_pack_path();
    load_weights();
    layer_to_kv_pool.assign(layers.size(), -1);
    for (size_t layer = 0; layer < layers.size(); ++layer) {
        if (std::holds_alternative<AttentionWeights>(layers[layer])) {
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
    if (!options.use_pack_cache) return;
    std::filesystem::path directory = options.pack_cache_directory.empty()
        ? default_cache_directory() : options.pack_cache_directory;
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) throw std::runtime_error("cannot create CPU pack cache: " + error.message());
    const std::string source = source_identity(safetensors_path);
    const size_t id = std::hash<std::string>{}(source);
    const std::string variant_id = variant ? std::string(variant->id()) : "unknown";
    std::ostringstream filename;
    filename << variant_id << '-' << std::hex << id << "-q4g" << group_size
             << '-' << cpu_isa_name(options.isa) << ".lfmpack";
    pack_file = directory / filename.str();
    source_id = source;
}

CpuModel::Impl::CommonWeights CpuModel::Impl::Shared::load_common(
    SafeTensorFile* file, CpuPackReader* reader, CpuPackWriter* writer,
    int layer) {
    CommonWeights common;
    common.operator_norm = load_vector(file, reader, writer,
        layer_name(layer, "operator_norm.weight"), {shape.hidden});
    common.ffn_norm = load_vector(file, reader, writer,
        layer_name(layer, "ffn_norm.weight"), {shape.hidden});
    common.w13 = load_concat(file, reader, writer,
        layer_name(layer, "feed_forward.w13.weight"), {
            {layer_name(layer, "feed_forward.w1.weight"),
             {shape.intermediate, shape.hidden}},
            {layer_name(layer, "feed_forward.w3.weight"),
             {shape.intermediate, shape.hidden}},
        });
    common.w2 = load_matrix(file, reader, writer,
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

    std::unique_ptr<SafeTensorFile> file;
    std::unique_ptr<CpuPackWriter> writer;
    if (!reader) {
        file = std::make_unique<SafeTensorFile>(safetensors_path);
        if (!pack_file.empty()) {
            CpuPackMetadata metadata;
            metadata.source_id = source_id;
            metadata.isa = cpu_isa_name(options.isa);
            metadata.group_size = static_cast<uint32_t>(group_size);
            writer = std::make_unique<CpuPackWriter>(pack_file, std::move(metadata));
        }
    }

    embedding = load_matrix(file.get(), reader.get(), writer.get(),
        "model.embed_tokens.weight", {shape.vocab_size, shape.hidden});
    final_norm = load_vector(file.get(), reader.get(), writer.get(),
        "model.embedding_norm.weight", {shape.hidden});

    if (shape.architecture == ArchitectureKind::MoeLfm2) {
        throw std::runtime_error(
            "LFM2 MoE CPU execution is not implemented in this release. "
            "Use the CUDA backend to run MoE checkpoints such as LFM2.5-8B-A1B.");
    }

    layers.reserve(static_cast<size_t>(shape.num_hidden_layers));
    for (int index = 0; index < shape.num_hidden_layers; ++index) {
        CommonWeights common = load_common(file.get(), reader.get(), writer.get(), index);
        const LayerType layer_type = shape.layer_types[static_cast<size_t>(index)];
        if (layer_type == LayerType::FullAttention) {
            AttentionWeights layer;
            layer.common = std::move(common);
            layer.qkv = load_concat(file.get(), reader.get(), writer.get(),
                layer_name(index, "self_attn.qkv.weight"), {
                    {layer_name(index, "self_attn.q_proj.weight"),
                     {shape.q_width, shape.hidden}},
                    {layer_name(index, "self_attn.k_proj.weight"),
                     {shape.kv_width, shape.hidden}},
                    {layer_name(index, "self_attn.v_proj.weight"),
                     {shape.kv_width, shape.hidden}},
                });
            layer.out = load_matrix(file.get(), reader.get(), writer.get(),
                layer_name(index, "self_attn.out_proj.weight"),
                {shape.hidden, shape.hidden});
            layer.q_norm = load_vector(file.get(), reader.get(), writer.get(),
                layer_name(index, "self_attn.q_layernorm.weight"),
                {shape.head_dim});
            layer.k_norm = load_vector(file.get(), reader.get(), writer.get(),
                layer_name(index, "self_attn.k_layernorm.weight"),
                {shape.head_dim});
            layers.emplace_back(std::move(layer));
        } else {
            ConvolutionWeights layer;
            layer.common = std::move(common);
            layer.in = load_matrix(file.get(), reader.get(), writer.get(),
                layer_name(index, "conv.in_proj.weight"),
                {3 * shape.hidden, shape.hidden});
            layer.weight = load_vector(file.get(), reader.get(), writer.get(),
                layer_name(index, "conv.conv.weight"),
                {shape.hidden, 1, shape.conv_cache});
            layer.out = load_matrix(file.get(), reader.get(), writer.get(),
                layer_name(index, "conv.out_proj.weight"),
                {shape.hidden, shape.hidden});
            layers.emplace_back(std::move(layer));
        }
    }
    if (writer) writer->commit();
}

Q4GroupMatrix CpuModel::Impl::Shared::load_matrix(
    SafeTensorFile* file, CpuPackReader* reader, CpuPackWriter* writer,
    const std::string& name, const std::vector<int64_t>& expected) {
    if (reader) return reader->read_q4_matrix(name);
    if (!file) throw std::logic_error("CPU weight source is missing");
    const HostTensorView tensor = file->tensor(name);
    if (tensor.dtype != TensorDType::BF16 || tensor.shape != expected ||
        expected.size() != 2) {
        throw std::runtime_error("unexpected CPU linear tensor: " + name);
    }
    Q4GroupMatrix matrix = quantize_bf16_groupwise_q4(
        tensor.data, static_cast<size_t>(expected[0]),
        static_cast<size_t>(expected[1]), group_size);
    if (writer) writer->add_q4_matrix(name, matrix);
    return matrix;
}

Q4GroupMatrix CpuModel::Impl::Shared::load_concat(
    SafeTensorFile* file, CpuPackReader* reader, CpuPackWriter* writer,
    const std::string& synthetic,
    const std::vector<std::pair<std::string, std::vector<int64_t>>>& parts) {
    if (reader) return reader->read_q4_matrix(synthetic);
    if (!file || parts.empty()) throw std::logic_error("invalid CPU concat source");
    const int64_t cols = parts.front().second[1];
    size_t total_rows = 0;
    for (const auto& part : parts) total_rows += static_cast<size_t>(part.second[0]);
    std::vector<float> joined(total_rows * static_cast<size_t>(cols));
    size_t row_offset = 0;
    for (const auto& [name, expected] : parts) {
        const std::vector<float> values = bf16_vector(file->tensor(name), expected, name);
        std::copy(values.begin(), values.end(),
                  joined.begin() + static_cast<ptrdiff_t>(
                      row_offset * static_cast<size_t>(cols)));
        row_offset += static_cast<size_t>(expected[0]);
    }
    Q4GroupMatrix matrix = quantize_float_groupwise_q4(
        joined.data(), total_rows, static_cast<size_t>(cols), group_size);
    if (writer) writer->add_q4_matrix(synthetic, matrix);
    return matrix;
}

std::vector<float> CpuModel::Impl::Shared::load_vector(
    SafeTensorFile* file, CpuPackReader* reader, CpuPackWriter* writer,
    const std::string& name, const std::vector<int64_t>& expected) {
    if (reader) return reader->read_bf16_vector(name);
    if (!file) throw std::logic_error("CPU vector source is missing");
    const HostTensorView tensor = file->tensor(name);
    std::vector<float> result = bf16_vector(tensor, expected, name);
    if (writer) writer->add_bf16_vector(name, tensor.data, result.size());
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
            } else {
                bytes += value.in.memory_bytes() + value.out.memory_bytes() +
                         value.weight.size() * sizeof(float);
            }
        }, layer);
    }
    return bytes;
}

} // namespace lfm
