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
      linear(options.isa, pool),
      shape(runtime_topology.exec), dims(runtime_topology.dims) {
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
    checkpoint.native_checkpoint = native_storage != nullptr &&
                         native_storage->has_native_block_storage();
    runtime_topology = bootstrap.model.topology;
    tie_word_embeddings = bootstrap.model.graph.tied_embeddings;
    program = CpuModelCompiler{}.compile(bootstrap.model);
    workspace_plan = CpuWorkspacePlan::from_program(program);
    model_identity = bootstrap.model.provenance.identity;
    weight_requests = bootstrap.model.weight_plan.requests;
    repository = bootstrap.checkpoint.repository;
    const std::vector<std::string> repository_names = repository->names();
    checkpoint.compressed_checkpoint = std::any_of(
        repository_names.begin(), repository_names.end(),
        [](const std::string& name) { return name.ends_with("_packed"); });
    prepare_pack_path();
    if (options.expert_backing == CpuExpertBacking::DiskCached &&
        !checkpoint.native_checkpoint && !checkpoint.compressed_checkpoint &&
        (!options.use_pack_cache || checkpoint.pack_file.empty())) {
        throw std::invalid_argument(
            "CPU disk-backed experts require the CPU pack cache");
    }
    load_weights();
    CpuKvTopology kv_topology = build_cpu_kv_topology(shape, program, options);
    kv_pools = std::move(kv_topology.pools);
    layer_to_kv_pool = std::move(kv_topology.layer_to_pool);
    layer_to_kv_owner = std::move(kv_topology.layer_to_owner);
}

void CpuCompiledModel::Shared::prepare_pack_path() {
    if (checkpoint.native_checkpoint || checkpoint.compressed_checkpoint ||
        !options.use_pack_cache) {
        return;
    }
    std::filesystem::path directory = options.pack_cache_directory.empty()
        ? default_cache_directory() : options.pack_cache_directory;
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) throw std::runtime_error("cannot create CPU pack cache: " + error.message());
    const std::string source = source_identity(model_path);
    const size_t id = std::hash<std::string>{}(source);
    const size_t model_hash = std::hash<std::string>{}(model_identity);
    std::ostringstream filename;
    filename << "celeg-" << std::hex << model_hash << '-' << id
             << "-q4g" << group_size
             << '-' << cpu_isa_name(options.isa) << ".lfmpack";
    checkpoint.pack_file = directory / filename.str();
    checkpoint.source_id = source;
}



CpuLinearWeight CpuCompiledModel::Shared::load_matrix(
    IWeightRepository* source, CpuPackReader* reader, CpuPackWriter* writer,
    const std::string& name, const std::vector<int64_t>& expected) {
    try {
        if (checkpoint.compressed_checkpoint) {
            if (const auto cached = compressed_linear_cache.find(name);
                cached != compressed_linear_cache.end()) return cached->second;
            CpuLinearWeight loaded = CpuWeightCodec(source, reader, writer, group_size).matrix(name, expected);
            auto [it, inserted] = compressed_linear_cache.emplace(name, std::move(loaded));
            return it->second;
        }
        return CpuWeightCodec(source, reader, writer, group_size).matrix(name, expected);
    } catch (const std::exception& error) {
        throw std::runtime_error("CPU load_matrix '" + name + "': " + error.what());
    }
}

CpuLinearWeight CpuCompiledModel::Shared::load_concat(
    IWeightRepository* source, CpuPackReader* reader, CpuPackWriter* writer,
    const std::string& synthetic,
    const std::vector<std::pair<std::string, std::vector<int64_t>>>& parts) {
    try {
    if (checkpoint.compressed_checkpoint) {
        if (const auto cached = compressed_linear_cache.find(synthetic);
            cached != compressed_linear_cache.end()) return cached->second;
        if (parts.empty()) throw std::invalid_argument("compressed CPU concat has no parts");
        CpuLinearWeight result;
        result.cols = static_cast<uint32_t>(parts.front().second.at(1));
        for (const auto& [name, expected] : parts) {
            CpuLinearWeight part = load_matrix(source, nullptr, nullptr, name, expected);
            if (part.cols != result.cols) throw std::runtime_error("compressed CPU concat width mismatch");
            result.rows += part.rows;
            for (const CpuLinearMatrix& segment : part.segments) result.segments.push_back(segment);
        }
        result.validate();
        auto [it, inserted] = compressed_linear_cache.emplace(synthetic, std::move(result));
        return it->second;
    }
    return CpuWeightCodec(source, reader, writer, group_size).concat(synthetic, parts);
    } catch (const std::exception& error) {
        throw std::runtime_error("CPU load_concat '" + synthetic + "': " + error.what());
    }
}

std::vector<float> CpuCompiledModel::Shared::load_vector(
    IWeightRepository* source, CpuPackReader* reader, CpuPackWriter* writer,
    const std::string& name, const std::vector<int64_t>& expected) {
    return CpuWeightCodec(source, reader, writer, group_size).vector(name, expected);
}

size_t CpuCompiledModel::Shared::weights_memory_bytes() const {
    size_t bytes = weight_store.embedding.memory_bytes() +
        weight_store.final_norm.size() * sizeof(float);
    for (const CpuLayerWeights& layer : weight_store.layers) {
        bytes += layer.common.mixer_norm_before.size() * sizeof(float) +
                 layer.common.feed_forward_norm_before.size() * sizeof(float);
        std::visit([&](const auto& value) {
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
                bytes += value.mlp_up.memory_bytes() + value.w2.memory_bytes();
            }
        }, layer.mixer);
        std::visit([&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, DenseFeedForwardWeights>) {
                bytes += value.w13.memory_bytes() + value.w2.memory_bytes() +
                         value.per_layer_input_gate.memory_bytes() +
                         value.per_layer_projection.memory_bytes();
            } else if constexpr (std::is_same_v<T, MoeWeights>) {
                bytes += (value.router.size() + value.router_bias.size()) * sizeof(float);
                for (const CpuLinearWeight& weight : value.expert_w13) {
                    bytes += weight.memory_bytes();
                }
                for (const CpuLinearWeight& weight : value.expert_w2) {
                    bytes += weight.memory_bytes();
                }
                bytes += value.shared_w13.memory_bytes() + value.shared_w2.memory_bytes() +
                         value.shared_gate.memory_bytes();
            }
        }, layer.feed_forward);
    }
    return bytes;
}

}
