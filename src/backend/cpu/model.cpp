#include "lfm/backend/cpu/model.hpp"

#include "detail/model_internal.hpp"

#include "lfm/model/config/config.hpp"
#include "lfm/detail/checkpoint/bootstrap.hpp"
#include "lfm/model/weights/quantization.hpp"
#include "lfm/checkpoint/formats/safetensors.hpp"
#include "lfm/model/config/variant.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
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

uint64_t xorshift64star(uint64_t& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 2685821657736338717ULL;
}

float random_unit(uint64_t& state) {
    return static_cast<float>((xorshift64star(state) >> 40) *
                              (1.0 / 16777216.0));
}

} // namespace

const char* cpu_kv_cache_mode_name(CpuKvCacheMode mode) {
    switch (mode) {
        case CpuKvCacheMode::Fp32: return "fp32";
        case CpuKvCacheMode::Bf16: return "bf16";
    }
    return "unknown";
}

CpuKvCacheMode parse_cpu_kv_cache_mode(const std::string& text) {
    if (text == "fp32") return CpuKvCacheMode::Fp32;
    if (text == "bf16") return CpuKvCacheMode::Bf16;
    throw std::invalid_argument("CPU KV cache mode must be fp32 or bf16");
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

CpuModel::Impl::Impl(std::shared_ptr<Shared> shared_weights,
                     GenerationConfig generation_config,
                     int requested_numa_node)
    : shared(std::move(shared_weights)), generation(generation_config),
      preferred_numa_node(requested_numa_node) {
    if (!shared) throw std::invalid_argument("shared CPU model weights are required");
    generation.validate();
    allocate_state();
    allocate_activations();
    reset();
}

CpuModel::Impl::~Impl() {
    for (LayerState& layer : states) {
        if (auto* attention = std::get_if<AttentionState>(&layer)) {
            release_attention_pages(*attention);
        }
    }
}

void CpuModel::Impl::allocate_state() {
    states.reserve(shared->layers.size());
    for (size_t index = 0; index < shared->layers.size(); ++index) {
        const WeightLayer& layer = shared->layers[index];
        if (std::holds_alternative<AttentionWeights>(layer)) {
            const int pool = shared->layer_to_kv_pool.at(index);
            if (pool < 0) throw std::logic_error("attention layer has no CPU KV page pool");
            AttentionState state;
            state.pool_index = static_cast<size_t>(pool);
            states.emplace_back(std::move(state));
        } else {
            ConvolutionState state;
            state.state.resize(static_cast<size_t>(shared->shape.conv_cache) *
                               shared->shape.hidden);
            states.emplace_back(std::move(state));
        }
    }
}

void CpuModel::Impl::allocate_activations() {
    hidden.resize(shared->shape.hidden);
    residual.resize(shared->shape.hidden);
    normed.resize(shared->shape.hidden);
    op_output.resize(shared->shape.hidden);
    qkv.resize(shared->shape.qkv_width);
    conv_projected.resize(3 * shared->shape.hidden);
    gate_up.resize(2 * shared->shape.intermediate);
    activated.resize(shared->shape.intermediate);
    mlp_output.resize(shared->shape.hidden);
    logits.resize(shared->shape.vocab_size);
    seen.resize(shared->shape.vocab_size);
}

const CpuModel::Impl::CommonWeights& CpuModel::Impl::common_weights(
    size_t layer) const {
    return std::visit([](const auto& value) -> const CommonWeights& {
        return value.common;
    }, shared->layers.at(layer));
}

CpuModel::Impl::AttentionState& CpuModel::Impl::attention_state(size_t layer) {
    return std::get<AttentionState>(states.at(layer));
}

const CpuModel::Impl::AttentionState& CpuModel::Impl::attention_state(
    size_t layer) const {
    return std::get<AttentionState>(states.at(layer));
}

CpuModel::Impl::ConvolutionState& CpuModel::Impl::convolution_state(
    size_t layer) {
    return std::get<ConvolutionState>(states.at(layer));
}

const CpuModel::Impl::ConvolutionState& CpuModel::Impl::convolution_state(
    size_t layer) const {
    return std::get<ConvolutionState>(states.at(layer));
}

void CpuModel::Impl::release_attention_pages(AttentionState& state) noexcept {
    if (!shared || state.pool_index >= shared->kv_pools.size()) return;
    auto& pool = *shared->kv_pools[state.pool_index];
    for (CpuKvPageId page : state.pages) {
        try {
            pool.release(page);
        } catch (const std::exception& error) {
            std::clog << "CPU KV page cleanup failed: " << error.what() << '\n';
        } catch (...) {
            std::clog << "CPU KV page cleanup failed: unknown exception\n";
        }
    }
    state.pages.clear();
    state.token_count = 0;
}

void CpuModel::Impl::store_kv(AttentionState& state, int position,
                              const float* key, const float* value) {
    if (position < 0) throw std::invalid_argument("negative CPU KV position");
    CpuKvPagePool& pool = *shared->kv_pools.at(state.pool_index);
    const size_t position_value = static_cast<size_t>(position);
    const size_t page_index = position_value / pool.page_tokens();
    const size_t token_offset = position_value % pool.page_tokens();
    while (state.pages.size() <= page_index) {
        state.pages.push_back(pool.allocate(preferred_numa_node));
    }
    if (token_offset != 0 && pool.reference_count(state.pages[page_index]) > 1) {
        const CpuKvPageId shared_page = state.pages[page_index];
        const CpuKvPageId private_page = pool.clone_prefix(
            shared_page, token_offset, preferred_numa_node);
        state.pages[page_index] = private_page;
        pool.release(shared_page);
    }
    pool.write(state.pages[page_index], token_offset, key, value);
    state.token_count = std::max(state.token_count, position_value + 1);
}

void CpuModel::Impl::run_attention(const AttentionState& state,
                                   const float* q, float* output,
                                   int sequence_length) const {
    if (sequence_length <= 0 || static_cast<size_t>(sequence_length) > state.token_count) {
        throw std::invalid_argument("CPU paged attention sequence length is invalid");
    }
    const CpuKvPagePool& pool = *shared->kv_pools.at(state.pool_index);
    CpuPagedAttentionStats attention_stats;
    cpu_gqa_decode_paged_parallel(
        q, pool, state.pages, output, sequence_length,
        shared->shape.num_attention_heads, shared->shape.num_key_value_heads, shared->shape.head_dim,
        shared->pool,
        CpuPagedAttentionOptions{
            shared->options.attention_parallel_threshold,
            shared->options.attention_page_tile},
        &attention_stats);
    if (attention_stats.parallel) {
        ++const_cast<Impl*>(this)->attention_parallel_calls;
    }
}

CpuPrefixSnapshot CpuModel::Impl::export_prefix_snapshot() const {
    CpuPrefixSnapshot snapshot;
    snapshot.position = static_cast<size_t>(position_value);
    snapshot.numa_node = preferred_numa_node;
    snapshot.logits = logits;
    snapshot.seen_tokens = seen;
    for (const LayerState& layer : states) {
        if (const auto* attention = std::get_if<AttentionState>(&layer)) {
            snapshot.attention_pages.push_back(attention->pages);
            snapshot.attention_token_counts.push_back(attention->token_count);
        } else {
            snapshot.convolution_states.push_back(
                std::get<ConvolutionState>(layer).state);
        }
    }
    return snapshot;
}

void CpuModel::Impl::restore_prefix_snapshot(CpuPrefixSnapshot snapshot,
                                             bool ready_for_decode) {
    size_t expected_attention = 0;
    size_t expected_convolution = 0;
    for (const LayerState& layer : states) {
        if (std::holds_alternative<AttentionState>(layer)) ++expected_attention;
        else ++expected_convolution;
    }
    if (snapshot.attention_pages.size() != expected_attention ||
        snapshot.attention_token_counts.size() != expected_attention ||
        snapshot.convolution_states.size() != expected_convolution ||
        snapshot.logits.size() != logits.size() ||
        snapshot.seen_tokens.size() != seen.size() ||
        snapshot.position > static_cast<size_t>(shared->max_context)) {
        throw std::invalid_argument("CPU prefix snapshot shape is invalid");
    }
    for (size_t index = 0; index < snapshot.attention_token_counts.size(); ++index) {
        const size_t required_pages = snapshot.attention_token_counts[index] == 0 ? 0 :
            (snapshot.attention_token_counts[index] +
             shared->options.kv_page_tokens - 1) /
            shared->options.kv_page_tokens;
        if (snapshot.attention_pages[index].size() < required_pages) {
            throw std::invalid_argument("CPU prefix snapshot page table is incomplete");
        }
    }

    reset();
    size_t attention_index = 0;
    size_t convolution_index = 0;
    try {
        for (LayerState& layer : states) {
            if (auto* attention = std::get_if<AttentionState>(&layer)) {
                attention->pages = std::move(snapshot.attention_pages[attention_index]);
                attention->token_count = snapshot.attention_token_counts[attention_index];
                ++attention_index;
            } else {
                std::get<ConvolutionState>(layer).state =
                    std::move(snapshot.convolution_states[convolution_index]);
                ++convolution_index;
            }
        }
        logits = std::move(snapshot.logits);
        seen = std::move(snapshot.seen_tokens);
        position_value = static_cast<int>(snapshot.position);
        preferred_numa_node = snapshot.numa_node;
        phase = ready_for_decode ? SessionPhase::Ready : SessionPhase::Prefilling;
        rng_state = generation.seed;
    } catch (...) {
        reset();
        // Pages not yet transferred remain owned by snapshot. Release them.
        for (size_t index = attention_index;
             index < snapshot.attention_pages.size(); ++index) {
            auto& pool = *shared->kv_pools.at(index);
            for (CpuKvPageId page : snapshot.attention_pages[index]) {
                try {
                    pool.release(page);
                } catch (const std::exception& error) {
                    std::clog << "CPU prefix snapshot rollback failed: "
                              << error.what() << '\n';
                } catch (...) {
                    std::clog << "CPU prefix snapshot rollback failed: unknown exception\n";
                }
            }
        }
        throw;
    }
}

void CpuModel::Impl::forward_token(int32_t token, bool compute_logits) {
    if (position_value >= shared->max_context) {
        throw std::runtime_error("CPU context limit reached");
    }
    shared->linear.embedding(shared->embedding, token, hidden.data());
    for (size_t index = 0; index < shared->layers.size(); ++index) {
        const WeightLayer& layer_variant = shared->layers[index];
        const CommonWeights& common = common_weights(index);
        std::copy(hidden.begin(), hidden.end(), residual.begin());
        cpu_rmsnorm(hidden.data(), common.operator_norm.data(), normed.data(),
                    shared->shape.hidden, shared->shape.norm_eps);
        if (const auto* attention = std::get_if<AttentionWeights>(&layer_variant)) {
            shared->linear.gemv(attention->qkv, normed.data(), qkv.data());
            float* q = qkv.data();
            float* k = q + shared->shape.q_width;
            float* v = k + shared->shape.kv_width;
            cpu_qk_norm_rope(q, attention->q_norm.data(), shared->shape.num_attention_heads,
                shared->shape.head_dim, position_value, shared->shape.rope_theta,
                shared->shape.norm_eps);
            cpu_qk_norm_rope(k, attention->k_norm.data(), shared->shape.num_key_value_heads,
                shared->shape.head_dim, position_value, shared->shape.rope_theta,
                shared->shape.norm_eps);
            AttentionState& state = attention_state(index);
            store_kv(state, position_value, k, v);
            run_attention(state, q, op_output.data(), position_value + 1);
            shared->linear.gemv(attention->out, op_output.data(), hidden.data());
        } else {
            const auto& convolution = std::get<ConvolutionWeights>(layer_variant);
            ConvolutionState& state = convolution_state(index);
            shared->linear.gemv(convolution.in, normed.data(), conv_projected.data());
            cpu_conv_decode(conv_projected.data(), convolution.weight.data(),
                state.state.data(), op_output.data(), shared->shape.hidden,
                shared->shape.conv_cache, position_value);
            shared->linear.gemv(convolution.out, op_output.data(), hidden.data());
        }
        cpu_residual_add(hidden.data(), residual.data(), shared->shape.hidden);

        cpu_rmsnorm(hidden.data(), common.ffn_norm.data(), normed.data(),
                    shared->shape.hidden, shared->shape.norm_eps);
        shared->linear.gemv(common.w13, normed.data(), gate_up.data());
        cpu_swiglu(gate_up.data(), activated.data(), shared->shape.intermediate);
        shared->linear.gemv(common.w2, activated.data(), mlp_output.data());
        cpu_residual_add(hidden.data(), mlp_output.data(), shared->shape.hidden);
    }
    if (compute_logits) {
        cpu_rmsnorm(hidden.data(), shared->final_norm.data(), normed.data(),
                    shared->shape.hidden, shared->shape.norm_eps);
        shared->linear.gemv(shared->embedding, normed.data(), logits.data());
    }
    ++position_value;
}

void CpuModel::Impl::forward_chunk(std::span<const int32_t> tokens,
                                   bool compute_logits) {
    if (tokens.empty()) return;
    if (position_value < 0 ||
        tokens.size() > static_cast<size_t>(shared->max_context - position_value)) {
        throw std::runtime_error("CPU chunked prefill exceeds context limit");
    }
    const size_t rows = tokens.size();
    chunk_hidden.resize(rows * shared->shape.hidden);
    chunk_residual.resize(rows * shared->shape.hidden);
    chunk_normed.resize(rows * shared->shape.hidden);
    chunk_op.resize(rows * shared->shape.hidden);
    chunk_qkv.resize(rows * shared->shape.qkv_width);
    chunk_conv.resize(rows * 3ULL * shared->shape.hidden);
    chunk_gate_up.resize(rows * 2ULL * shared->shape.intermediate);
    chunk_activated.resize(rows * shared->shape.intermediate);
    chunk_mlp.resize(rows * shared->shape.hidden);

    for (size_t row = 0; row < rows; ++row) {
        const int32_t token = tokens[row];
        if (token < 0 || token >= shared->shape.vocab_size) {
            throw std::invalid_argument("CPU chunked prefill token out of range");
        }
        shared->linear.embedding(shared->embedding, token,
            chunk_hidden.data() + row * shared->shape.hidden);
    }

    const int base_position = position_value;
    for (size_t layer_index = 0; layer_index < shared->layers.size(); ++layer_index) {
        const WeightLayer& layer = shared->layers[layer_index];
        const CommonWeights& common = common_weights(layer_index);
        std::copy(chunk_hidden.begin(), chunk_hidden.end(), chunk_residual.begin());
        for (size_t row = 0; row < rows; ++row) {
            cpu_rmsnorm(chunk_hidden.data() + row * shared->shape.hidden,
                        common.operator_norm.data(),
                        chunk_normed.data() + row * shared->shape.hidden,
                        shared->shape.hidden, shared->shape.norm_eps);
        }

        if (const auto* attention = std::get_if<AttentionWeights>(&layer)) {
            shared->linear.gemm(attention->qkv, chunk_normed.data(),
                                chunk_qkv.data(), rows);
            AttentionState& cache = attention_state(layer_index);
            // Normalize/rotate and commit the complete chunk first. Each query
            // still observes only [0, base + row], preserving causality.
            for (size_t row = 0; row < rows; ++row) {
                float* q = chunk_qkv.data() + row * shared->shape.qkv_width;
                float* k = q + shared->shape.q_width;
                float* v = k + shared->shape.kv_width;
                const int absolute_position = base_position + static_cast<int>(row);
                cpu_qk_norm_rope(q, attention->q_norm.data(),
                    shared->shape.num_attention_heads, shared->shape.head_dim,
                    absolute_position, shared->shape.rope_theta,
                    shared->shape.norm_eps);
                cpu_qk_norm_rope(k, attention->k_norm.data(),
                    shared->shape.num_key_value_heads, shared->shape.head_dim,
                    absolute_position, shared->shape.rope_theta,
                    shared->shape.norm_eps);
                store_kv(cache, absolute_position, k, v);
            }
            for (size_t row = 0; row < rows; ++row) {
                const float* q = chunk_qkv.data() + row * shared->shape.qkv_width;
                run_attention(cache, q,
                    chunk_op.data() + row * shared->shape.hidden,
                    base_position + static_cast<int>(row) + 1);
            }
            shared->linear.gemm(attention->out, chunk_op.data(),
                                chunk_hidden.data(), rows);
        } else {
            const auto& convolution = std::get<ConvolutionWeights>(layer);
            shared->linear.gemm(convolution.in, chunk_normed.data(),
                                chunk_conv.data(), rows);
            ConvolutionState& conv_state = convolution_state(layer_index);
            for (size_t row = 0; row < rows; ++row) {
                cpu_conv_decode(
                    chunk_conv.data() + row * 3ULL * shared->shape.hidden,
                    convolution.weight.data(), conv_state.state.data(),
                    chunk_op.data() + row * shared->shape.hidden,
                    shared->shape.hidden, shared->shape.conv_cache,
                    base_position + static_cast<int>(row));
            }
            shared->linear.gemm(convolution.out, chunk_op.data(),
                                chunk_hidden.data(), rows);
        }

        for (size_t row = 0; row < rows; ++row) {
            float* row_hidden = chunk_hidden.data() + row * shared->shape.hidden;
            cpu_residual_add(row_hidden,
                chunk_residual.data() + row * shared->shape.hidden,
                shared->shape.hidden);
            cpu_rmsnorm(row_hidden, common.ffn_norm.data(),
                chunk_normed.data() + row * shared->shape.hidden,
                shared->shape.hidden, shared->shape.norm_eps);
        }
        shared->linear.gemm(common.w13, chunk_normed.data(),
                            chunk_gate_up.data(), rows);
        for (size_t row = 0; row < rows; ++row) {
            cpu_swiglu(chunk_gate_up.data() + row * 2ULL * shared->shape.intermediate,
                       chunk_activated.data() + row * shared->shape.intermediate,
                       shared->shape.intermediate);
        }
        shared->linear.gemm(common.w2, chunk_activated.data(),
                            chunk_mlp.data(), rows);
        for (size_t row = 0; row < rows; ++row) {
            cpu_residual_add(chunk_hidden.data() + row * shared->shape.hidden,
                             chunk_mlp.data() + row * shared->shape.hidden,
                             shared->shape.hidden);
        }
    }

    if (compute_logits) {
        const float* last = chunk_hidden.data() + (rows - 1) * shared->shape.hidden;
        cpu_rmsnorm(last, shared->final_norm.data(), normed.data(),
                    shared->shape.hidden, shared->shape.norm_eps);
        shared->linear.gemv(shared->embedding, normed.data(), logits.data());
    }
    position_value += static_cast<int>(rows);
}

int32_t CpuModel::Impl::sample() {
    const float temperature = generation.temperature;
    auto penalized = [&](int32_t token) {
        float value = logits[static_cast<size_t>(token)];
        if (seen[static_cast<size_t>(token)] &&
            generation.repetition_penalty > 1.0f) {
            value = value >= 0.0f ? value / generation.repetition_penalty
                                  : value * generation.repetition_penalty;
        }
        return value;
    };
    if (temperature <= 0.0f || generation.top_k == 1) {
        int32_t best = 0;
        float best_value = penalized(0);
        for (int32_t token = 1; token < shared->shape.vocab_size; ++token) {
            const float value = penalized(token);
            if (value > best_value) {
                best = token;
                best_value = value;
            }
        }
        return best;
    }
    const int top_k = std::min(generation.top_k, shared->shape.vocab_size);
    std::vector<int32_t> indices(static_cast<size_t>(shared->shape.vocab_size));
    std::iota(indices.begin(), indices.end(), 0);
    auto adjusted = [&](int32_t token) { return penalized(token) / temperature; };
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
        [&](int32_t a, int32_t b) {
            const float av = adjusted(a);
            const float bv = adjusted(b);
            return av == bv ? a < b : av > bv;
        });
    indices.resize(static_cast<size_t>(top_k));
    const float maximum = adjusted(indices.front());
    std::vector<float> probabilities(indices.size());
    float total = 0.0f;
    for (size_t i = 0; i < indices.size(); ++i) {
        probabilities[i] = std::exp(adjusted(indices[i]) - maximum);
        total += probabilities[i];
    }
    for (float& probability : probabilities) probability /= total;
    size_t active = probabilities.size();
    if (generation.top_p < 1.0f) {
        float cumulative = 0.0f;
        active = 0;
        do {
            cumulative += probabilities[active++];
        } while (active < probabilities.size() && cumulative < generation.top_p);
        total = std::accumulate(
            probabilities.begin(),
            probabilities.begin() + static_cast<ptrdiff_t>(active), 0.0f);
    } else {
        total = 1.0f;
    }
    float target = random_unit(rng_state) * total;
    for (size_t i = 0; i < active; ++i) {
        target -= probabilities[i];
        if (target <= 0.0f) return indices[i];
    }
    return indices[active - 1];
}

void CpuModel::Impl::set_generation(GenerationConfig config) {
    config.validate();
    generation = config;
    rng_state = generation.seed;
}

void CpuModel::Impl::reset() {
    position_value = 0;
    phase = SessionPhase::Empty;
    metrics = {};
    rng_state = generation.seed;
    std::fill(seen.begin(), seen.end(), uint8_t{0});
    std::fill(logits.begin(), logits.end(), 0.0f);
    for (LayerState& state : states) {
        if (auto* attention = std::get_if<AttentionState>(&state)) {
            release_attention_pages(*attention);
        } else {
            auto& convolution = std::get<ConvolutionState>(state);
            std::fill(convolution.state.begin(), convolution.state.end(), 0.0f);
        }
    }
}

CpuModelMemoryStats CpuModel::Impl::memory_stats() const {
    CpuModelMemoryStats stats;
    stats.weights = shared->weights_memory_bytes();
    for (const LayerState& state : states) {
        std::visit([&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, AttentionState>) {
                const auto& pool = *shared->kv_pools.at(value.pool_index);
                stats.kv_cache += value.pages.size() * pool.page_bytes();
                stats.kv_pages_used += value.pages.size();
                stats.kv_pages_total += pool.stats().total_pages;
            } else {
                stats.conv_state += value.state.size() * sizeof(float);
            }
        }, state);
    }
    stats.activations =
        (hidden.size() + residual.size() + normed.size() + op_output.size() +
         qkv.size() + conv_projected.size() + gate_up.size() + activated.size() +
         mlp_output.size() + logits.size() + chunk_hidden.size() +
         chunk_residual.size() + chunk_normed.size() + chunk_op.size() +
         chunk_qkv.size() + chunk_conv.size() + chunk_gate_up.size() +
         chunk_activated.size() + chunk_mlp.size()) * sizeof(float) + seen.size();
    return stats;
}

CpuModel::CpuModel(const std::string& path, int context,
                   CpuModelOptions options, GenerationConfig generation)
    : impl_(std::make_unique<Impl>(
          std::make_shared<Impl::Shared>(path, context, std::move(options)),
          generation, -1)),
      session_view_(*this),
      diagnostics_view_(*this),
      persistence_view_(*this) {}

CpuModel::CpuModel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)),
      session_view_(*this),
      diagnostics_view_(*this),
      persistence_view_(*this) {
    if (!impl_) throw std::invalid_argument("CPU model implementation is required");
}

CpuModel::~CpuModel() = default;

CpuModel::CpuModel(CpuModel&& other) noexcept
    : impl_(std::move(other.impl_)),
      session_view_(*this),
      diagnostics_view_(*this),
      persistence_view_(*this) {}

CpuModel& CpuModel::operator=(CpuModel&& other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
    }
    return *this;
}

std::unique_ptr<CpuModel> CpuModel::clone_session() const {
    return clone_session_on_node(-1);
}

std::unique_ptr<CpuModel> CpuModel::clone_session_on_node(int numa_node) const {
    return std::unique_ptr<CpuModel>(new CpuModel(
        std::make_unique<Impl>(impl_->shared, impl_->generation, numa_node)));
}

std::vector<std::shared_ptr<CpuKvPagePool>> CpuModel::shared_kv_pools() const {
    return impl_->shared->kv_pools;
}

void CpuModel::reset_session() { impl_->reset(); }

void CpuModel::prefill_session(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) throw std::invalid_argument("CPU prefill needs at least one token");
    if (tokens.size() > static_cast<size_t>(impl_->shared->max_context)) {
        throw std::invalid_argument("CPU prefill exceeds context");
    }
    for (int32_t token : tokens) {
        if (token < 0 || token >= impl_->shared->shape.vocab_size) {
            throw std::invalid_argument("CPU token out of range");
        }
    }
    impl_->reset();
    impl_->phase = SessionPhase::Prefilling;
    const auto started = std::chrono::steady_clock::now();
    if (tokens.size() < impl_->shared->options.prefill_chunk_threshold) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            impl_->seen[static_cast<size_t>(tokens[i])] = 1;
            impl_->forward_token(tokens[i], i + 1 == tokens.size());
        }
    } else {
        const size_t chunk = impl_->shared->options.prefill_chunk_tokens;
        for (size_t begin = 0; begin < tokens.size(); begin += chunk) {
            const size_t end = std::min(tokens.size(), begin + chunk);
            for (size_t i = begin; i < end; ++i) {
                impl_->seen[static_cast<size_t>(tokens[i])] = 1;
            }
            impl_->forward_chunk(
                std::span<const int32_t>(tokens.data() + begin, end - begin),
                end == tokens.size());
        }
    }
    const auto ended = std::chrono::steady_clock::now();
    impl_->metrics.last_prefill_ms =
        std::chrono::duration<double, std::milli>(ended - started).count();
    impl_->metrics.prefill_tokens = tokens.size();
    impl_->phase = SessionPhase::Ready;
}

int32_t CpuModel::decode_session() {
    if (impl_->phase != SessionPhase::Ready) {
        throw std::runtime_error("CPU model is not ready for decode");
    }
    const auto started = std::chrono::steady_clock::now();
    const int32_t token = impl_->sample();
    impl_->seen[static_cast<size_t>(token)] = 1;
    impl_->forward_token(token, true);
    const auto ended = std::chrono::steady_clock::now();
    impl_->metrics.cumulative_decode_ms +=
        std::chrono::duration<double, std::milli>(ended - started).count();
    ++impl_->metrics.decoded_tokens;
    return token;
}

void CpuModel::set_session_generation(GenerationConfig generation) {
    impl_->set_generation(generation);
}

int CpuModel::session_position() const { return impl_->position_value; }

bool CpuModel::session_ready_for_decode() const {
    return impl_->phase == SessionPhase::Ready;
}

std::vector<float> CpuModel::session_logits() const { return impl_->logits; }
RuntimeMetrics CpuModel::session_metrics() const { return impl_->metrics; }
void CpuModel::clear_session_metrics() { impl_->metrics = {}; }
CpuModelMemoryStats CpuModel::session_memory_stats() const { return impl_->memory_stats(); }
int CpuModel::session_vocab_size() const { return impl_->shared->shape.vocab_size; }
CpuIsa CpuModel::session_isa() const { return impl_->shared->options.isa; }
CpuKvCacheMode CpuModel::session_kv_cache_mode() const {
    return impl_->shared->options.kv_cache_mode;
}

std::string CpuModel::session_backend_description() const {
    std::ostringstream out;
    out << "cpu-native isa=" << cpu_isa_name(impl_->shared->options.isa)
        << " q4-group=" << impl_->shared->group_size
        << " kv=" << cpu_kv_cache_mode_name(impl_->shared->options.kv_cache_mode)
        << " kv-page=" << impl_->shared->options.kv_page_tokens
        << " prefill-chunk=" << impl_->shared->options.prefill_chunk_tokens
        << " threads=" << (impl_->shared->pool.size() + 1)
        << " affinity=" << cpu_affinity_name(impl_->shared->options.affinity)
        << " numa=" << cpu_numa_mode_name(impl_->shared->options.numa_mode)
        << " attention-threshold=" << impl_->shared->options.attention_parallel_threshold
        << " attention-page-tile=" << impl_->shared->options.attention_page_tile
        << " pinned-workers=" << impl_->shared->pool.pinned_workers()
        << " pack=" << (impl_->shared->loaded_pack ? "hit" : "built")
        << " hardware-best="
        << cpu_isa_name(impl_->shared->capabilities.best_isa());
    return out.str();
}

const std::filesystem::path& CpuModel::session_pack_path() const {
    return impl_->shared->pack_file;
}

bool CpuModel::session_loaded_from_pack() const { return impl_->shared->loaded_pack; }

uint64_t CpuModel::session_attention_parallel_calls() const {
    return impl_->attention_parallel_calls;
}

CpuPrefixSnapshot CpuModel::export_session_prefix() const {
    return impl_->export_prefix_snapshot();
}

void CpuModel::restore_session_prefix(CpuPrefixSnapshot snapshot,
                                      bool ready_for_decode) {
    impl_->restore_prefix_snapshot(std::move(snapshot), ready_for_decode);
}

void CpuInferenceSession::reset() { owner_->reset_session(); }
void CpuInferenceSession::prefill(const std::vector<int32_t>& tokens) {
    owner_->prefill_session(tokens);
}
int32_t CpuInferenceSession::decode() { return owner_->decode_session(); }
void CpuInferenceSession::set_generation_config(GenerationConfig generation) {
    owner_->set_session_generation(generation);
}
int CpuInferenceSession::position() const { return owner_->session_position(); }
bool CpuInferenceSession::ready_for_decode() const {
    return owner_->session_ready_for_decode();
}

std::vector<float> CpuDiagnostics::copy_logits() const { return owner_->session_logits(); }
RuntimeMetrics CpuDiagnostics::runtime_metrics() const { return owner_->session_metrics(); }
void CpuDiagnostics::clear_runtime_metrics() { owner_->clear_session_metrics(); }
CpuModelMemoryStats CpuDiagnostics::memory_stats() const { return owner_->session_memory_stats(); }
int CpuDiagnostics::vocab_size() const { return owner_->session_vocab_size(); }
CpuIsa CpuDiagnostics::isa() const { return owner_->session_isa(); }
CpuKvCacheMode CpuDiagnostics::kv_cache_mode() const { return owner_->session_kv_cache_mode(); }
std::string CpuDiagnostics::backend_description() const {
    return owner_->session_backend_description();
}
const std::filesystem::path& CpuDiagnostics::pack_path() const {
    return owner_->session_pack_path();
}
bool CpuDiagnostics::loaded_from_pack() const { return owner_->session_loaded_from_pack(); }
uint64_t CpuDiagnostics::attention_parallel_calls() const {
    return owner_->session_attention_parallel_calls();
}

CpuPrefixSnapshot CpuPersistence::export_prefix_snapshot() const {
    return owner_->export_session_prefix();
}
void CpuPersistence::restore_prefix_snapshot(CpuPrefixSnapshot snapshot,
                                             bool ready_for_decode) {
    owner_->restore_session_prefix(std::move(snapshot), ready_for_decode);
}

} // namespace lfm
