#include "lfm/detail/model_impl.hpp"
#include "lfm/kernels.cuh"
#include "lfm/quantization.hpp"
#include "lfm/paged_kv.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <utility>

namespace lfm {
namespace {

size_t checked_element_count(const std::vector<int64_t>& shape) {
    size_t count = 1;
    for (int64_t dim : shape) {
        if (dim < 0) throw std::runtime_error("negative tensor dimension");
        if (dim == 0) return 0;
        const size_t value = static_cast<size_t>(dim);
        if (count > std::numeric_limits<size_t>::max() / value) {
            throw std::overflow_error("tensor element count overflow");
        }
        count *= value;
    }
    return count;
}

std::string layer_name(int index, const std::string& suffix) {
    return "model.layers." + std::to_string(index) + "." + suffix;
}

struct SessionHeader {
    std::array<char, 8> magic{{'L', 'F', 'M', 'S', 'E', 'S', 'S', '1'}};
    uint32_t version = 1;
    uint32_t kv_cache_mode = 0;
    int32_t position = 0;
    int32_t max_context = 0;
    int32_t layers = LfmConfig::layers;
    int32_t kv_width = LfmConfig::kv_width;
    int32_t kv_heads = LfmConfig::kv_heads;
    int32_t vocab = LfmConfig::vocab;
    uint64_t rng_state = 0;
};

template <typename T>
void write_scalar(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("failed writing session");
}

template <typename T>
void read_scalar(std::ifstream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("truncated session file");
}

} // namespace

struct LfmModel::Impl::LtPlan {
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr;
    cublasLtMatrixLayout_t b = nullptr;
    cublasLtMatrixLayout_t c = nullptr;
    cublasLtMatrixLayout_t d = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    size_t workspace_size = 0;
    bool available = false;

    ~LtPlan() {
        if (d) cublasLtMatrixLayoutDestroy(d);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (operation) cublasLtMatmulDescDestroy(operation);
    }
};

void LfmModel::Impl::LinearWeight::validate_storage() const {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("linear weight dimensions must be positive");
    }
    switch (kind) {
        case LinearStorageKind::Bf16:
            if (!bf16 || int8 || int4 || scales) {
                throw std::runtime_error("invalid BF16 linear weight storage");
            }
            break;
        case LinearStorageKind::Int8:
            if (bf16 || !int8 || int4 || !scales) {
                throw std::runtime_error("invalid INT8 linear weight storage");
            }
            break;
        case LinearStorageKind::Int4:
            if (bf16 || int8 || !int4 || !scales) {
                throw std::runtime_error("invalid INT4 linear weight storage");
            }
            break;
    }
}

size_t LfmModel::Impl::SharedModelWeights::memory_bytes() const {
    size_t total = 0;
    for (const auto& [unused_name, weight] : tensors) {
        (void)unused_name;
        total += weight.bf16_storage.bytes() +
                 weight.int8_storage.bytes() +
                 weight.int4_storage.bytes() +
                 weight.scales_storage.bytes();
    }
    return total;
}

LfmModel::Impl::Impl(const std::string& safetensors_path,
                   int max_context,
                   ModelOptions options,
                   GenerationConfig generation)
    : plan_(ExecutionPlan::compile(options, max_context)),
      options_(plan_.options()),
      generation_(generation),
      stream_(),
      cublas_(stream_.get()),
      max_context_(max_context) {
    generation_.validate();
    if (max_context_ <= 0) {
        throw std::invalid_argument("max_context must be positive");
    }
    attention_chunks_ = plan_.attention_chunks();
    if (attention_chunks_ > 0) {
        const size_t partials =
            static_cast<size_t>(LfmConfig::q_heads) * attention_chunks_;
        attention_partial_max_.reset(partials);
        attention_partial_denom_.reset(partials);
        attention_partial_accum_.reset(partials * LfmConfig::head_dim);
    }
    if (options_.gemm_backend == GemmBackend::CublasLt &&
        options_.lt_workspace_bytes > 0) {
        lt_workspace_.reset(options_.lt_workspace_bytes);
    }
    initialize_rope_tables();

    // Immutable device weights are shared across all request lanes that use
    // the same checkpoint, device and quantization mode. Session state remains
    // private to each LfmModel instance.
    static std::mutex weight_cache_mutex;
    static std::unordered_map<std::string, std::weak_ptr<SharedModelWeights>> weight_cache;
    int device_id = 0;
    LFM_CUDA(cudaGetDevice(&device_id));
    std::ostringstream key_builder;
    key_builder << device_id << ':' << static_cast<int>(options_.weight_mode)
                << ':' << safetensors_path;
    const std::string weight_cache_key = key_builder.str();
    {
        std::lock_guard<std::mutex> lock(weight_cache_mutex);
        const auto found = weight_cache.find(weight_cache_key);
        if (found != weight_cache.end()) weights_ = found->second.lock();
        if (!weights_) {
            weights_ = std::make_shared<SharedModelWeights>();
            weight_cache[weight_cache_key] = weights_;
        }
    }

    // Only one constructor populates a shared checkpoint at a time. Other
    // sessions wait, then reuse the immutable device buffers.
    std::unique_lock<std::mutex> shared_weights_lock(weights_->mutex);
    SafeTensorFile file(safetensors_path);
    embedding_ = load_linear_weight(
        file, "model.embed_tokens.weight",
        {LfmConfig::vocab, LfmConfig::hidden});
    final_norm_ = load_weight(
        file, "model.embedding_norm.weight", {LfmConfig::hidden});

    static constexpr bool attention_map[LfmConfig::layers] = {
        false, false, true, false, true, false, true,
        false, true, false, true, false, true, false};

    layers_.reserve(LfmConfig::layers);
    for (int i = 0; i < LfmConfig::layers; ++i) {
        LayerCommon common_layer;
        common_layer.operator_norm = load_weight(
            file, layer_name(i, "operator_norm.weight"), {LfmConfig::hidden});
        common_layer.ffn_norm = load_weight(
            file, layer_name(i, "ffn_norm.weight"), {LfmConfig::hidden});
        common_layer.w13 = load_concat_linear_weight(
            file, layer_name(i, "feed_forward.w13.weight"),
            {
                {layer_name(i, "feed_forward.w1.weight"),
                 {LfmConfig::intermediate, LfmConfig::hidden}},
                {layer_name(i, "feed_forward.w3.weight"),
                 {LfmConfig::intermediate, LfmConfig::hidden}},
            });
        common_layer.w2 = load_linear_weight(
            file, layer_name(i, "feed_forward.w2.weight"),
            {LfmConfig::hidden, LfmConfig::intermediate});

        if (attention_map[i]) {
            AttentionLayer attention_layer;
            attention_layer.common = common_layer;
            attention_layer.qkv = load_concat_linear_weight(
                file, layer_name(i, "self_attn.qkv.weight"),
                {
                    {layer_name(i, "self_attn.q_proj.weight"),
                     {LfmConfig::q_width, LfmConfig::hidden}},
                    {layer_name(i, "self_attn.k_proj.weight"),
                     {LfmConfig::kv_width, LfmConfig::hidden}},
                    {layer_name(i, "self_attn.v_proj.weight"),
                     {LfmConfig::kv_width, LfmConfig::hidden}},
                });
            attention_layer.out = load_linear_weight(
                file, layer_name(i, "self_attn.out_proj.weight"),
                {LfmConfig::hidden, LfmConfig::hidden});
            attention_layer.q_norm = load_weight(
                file, layer_name(i, "self_attn.q_layernorm.weight"),
                {LfmConfig::head_dim});
            attention_layer.k_norm = load_weight(
                file, layer_name(i, "self_attn.k_layernorm.weight"),
                {LfmConfig::head_dim});

            if (options_.allocate_local_kv_cache) {
                const size_t cache_elements =
                    static_cast<size_t>(max_context_) * LfmConfig::kv_width;
                if (options_.kv_cache_mode == KvCacheMode::Int8) {
                    attention_layer.key_cache_int8.reset(cache_elements);
                    attention_layer.value_cache_int8.reset(cache_elements);
                    const size_t scale_elements =
                        static_cast<size_t>(max_context_) * LfmConfig::kv_heads;
                    attention_layer.key_cache_scales.reset(scale_elements);
                    attention_layer.value_cache_scales.reset(scale_elements);
                } else {
                    attention_layer.key_cache.reset(cache_elements);
                    attention_layer.value_cache.reset(cache_elements);
                }
            }
            layers_.emplace_back(std::move(attention_layer));
        } else {
            ConvolutionLayer convolution_layer;
            convolution_layer.common = common_layer;
            convolution_layer.conv_in = load_linear_weight(
                file, layer_name(i, "conv.in_proj.weight"),
                {3 * LfmConfig::hidden, LfmConfig::hidden});
            convolution_layer.conv_weight = load_weight(
                file, layer_name(i, "conv.conv.weight"),
                {LfmConfig::hidden, 1, LfmConfig::conv_cache});
            convolution_layer.conv_out = load_linear_weight(
                file, layer_name(i, "conv.out_proj.weight"),
                {LfmConfig::hidden, LfmConfig::hidden});
            convolution_layer.conv_state.reset(
                static_cast<size_t>(LfmConfig::conv_cache) * LfmConfig::hidden);
            layers_.emplace_back(std::move(convolution_layer));
        }
    }

    shared_weights_lock.unlock();

    if (options_.cuda_graph ||
        options_.gemm_backend == GemmBackend::CublasLt) {
        warmup_decode_gemms();
    }
    local_kv_cache_available_ = options_.allocate_local_kv_cache;
    reset(options_.allocate_local_kv_cache);
}

LfmModel::Impl::~Impl() = default;

const __nv_bfloat16* LfmModel::Impl::load_weight(
    const SafeTensorFile& file,
    const std::string& name,
    std::vector<int64_t> expected) {
    if (const auto cached = weights_->tensors.find(name); cached != weights_->tensors.end()) {
        if (!expected.empty() && cached->second.shape != expected) {
            throw std::runtime_error("cached weight shape mismatch for " + name);
        }
        return cached->second.bf16_storage.data();
    }
    const HostTensorView tensor = file.tensor(name);
    if (tensor.dtype != TensorDType::BF16) {
        throw std::runtime_error(
            "only BF16 source weights are supported; incompatible tensor: " + name);
    }
    if (!expected.empty() && tensor.shape != expected) {
        throw std::runtime_error("unexpected shape for " + name);
    }
    const size_t count = checked_element_count(tensor.shape);
    if (tensor.bytes != count * sizeof(__nv_bfloat16)) {
        throw std::runtime_error("invalid BF16 byte count for " + name);
    }

    DeviceWeight weight;
    weight.shape = tensor.shape;
    weight.bf16_storage.reset(count);
    LFM_CUDA(cudaMemcpy(weight.bf16_storage.data(), tensor.data, tensor.bytes,
                        cudaMemcpyHostToDevice));
    auto [it, inserted] = weights_->tensors.emplace(name, std::move(weight));
    if (!inserted) throw std::runtime_error("duplicate weight: " + name);
    return it->second.bf16_storage.data();
}

const LfmModel::Impl::LinearWeight* LfmModel::Impl::load_linear_weight(
    const SafeTensorFile& file,
    const std::string& name,
    std::vector<int64_t> expected) {
    if (const auto cached = weights_->tensors.find(name); cached != weights_->tensors.end()) {
        if (cached->second.shape != expected) {
            throw std::runtime_error("cached linear shape mismatch for " + name);
        }
        return &cached->second.linear;
    }
    const HostTensorView tensor = file.tensor(name);
    if (tensor.dtype != TensorDType::BF16 || tensor.shape != expected ||
        tensor.shape.size() != 2) {
        throw std::runtime_error("unexpected linear tensor: " + name);
    }
    const size_t count = checked_element_count(tensor.shape);
    if (tensor.bytes != count * sizeof(__nv_bfloat16)) {
        throw std::runtime_error("invalid linear tensor byte count: " + name);
    }

    DeviceWeight weight;
    weight.shape = tensor.shape;
    const int rows = static_cast<int>(tensor.shape[0]);
    const int cols = static_cast<int>(tensor.shape[1]);
    if (options_.weight_mode == WeightMode::Int8) {
        Int8RowwisePack pack = quantize_bf16_rows(
            tensor.data, static_cast<size_t>(rows), static_cast<size_t>(cols));
        std::vector<int8_t>& quantized = pack.values;
        std::vector<float>& scales = pack.scales;
        weight.int8_storage.reset(count);
        weight.scales_storage.reset(scales.size());
        LFM_CUDA(cudaMemcpy(weight.int8_storage.data(), quantized.data(),
                            quantized.size() * sizeof(int8_t),
                            cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(weight.scales_storage.data(), scales.data(),
                            scales.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
        weight.linear.kind = LinearStorageKind::Int8;
        weight.linear.int8 = weight.int8_storage.data();
        weight.linear.scales = weight.scales_storage.data();
    } else if (options_.weight_mode == WeightMode::Int4) {
        Int4RowwisePack pack = quantize_bf16_rows_int4(
            tensor.data, static_cast<size_t>(rows), static_cast<size_t>(cols));
        weight.int4_storage.reset(pack.values.size());
        weight.scales_storage.reset(pack.scales.size());
        LFM_CUDA(cudaMemcpy(weight.int4_storage.data(), pack.values.data(),
                            pack.values.size() * sizeof(uint8_t),
                            cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(weight.scales_storage.data(), pack.scales.data(),
                            pack.scales.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
        weight.linear.kind = LinearStorageKind::Int4;
        weight.linear.int4 = weight.int4_storage.data();
        weight.linear.scales = weight.scales_storage.data();
    } else {
        weight.bf16_storage.reset(count);
        LFM_CUDA(cudaMemcpy(weight.bf16_storage.data(), tensor.data, tensor.bytes,
                            cudaMemcpyHostToDevice));
        weight.linear.kind = LinearStorageKind::Bf16;
        weight.linear.bf16 = weight.bf16_storage.data();
    }
    weight.linear.rows = rows;
    weight.linear.cols = cols;
    weight.linear.validate_storage();

    auto [it, inserted] = weights_->tensors.emplace(name, std::move(weight));
    if (!inserted) throw std::runtime_error("duplicate linear weight: " + name);
    return &it->second.linear;
}

const LfmModel::Impl::LinearWeight* LfmModel::Impl::load_concat_linear_weight(
    const SafeTensorFile& file,
    const std::string& synthetic_name,
    const std::vector<std::pair<std::string, std::vector<int64_t>>>& parts) {
    if (const auto cached = weights_->tensors.find(synthetic_name);
        cached != weights_->tensors.end()) {
        return &cached->second.linear;
    }
    if (parts.empty()) throw std::invalid_argument("concat weight requires parts");
    int64_t common_width = -1;
    int64_t total_rows = 0;
    size_t total_count = 0;
    std::vector<HostTensorView> views;
    views.reserve(parts.size());
    for (const auto& [name, expected] : parts) {
        const HostTensorView tensor = file.tensor(name);
        if (tensor.dtype != TensorDType::BF16 || tensor.shape != expected ||
            tensor.shape.size() != 2) {
            throw std::runtime_error("unexpected concatenated tensor: " + name);
        }
        if (common_width < 0) common_width = tensor.shape[1];
        if (tensor.shape[1] != common_width) {
            throw std::runtime_error("concatenated tensors have different widths");
        }
        const size_t count = checked_element_count(tensor.shape);
        if (tensor.bytes != count * sizeof(__nv_bfloat16)) {
            throw std::runtime_error("invalid concatenated tensor bytes: " + name);
        }
        total_count += count;
        total_rows += tensor.shape[0];
        views.push_back(tensor);
    }

    DeviceWeight weight;
    weight.shape = {total_rows, common_width};
    if (options_.weight_mode == WeightMode::Int8) {
        std::vector<int8_t> quantized(total_count);
        std::vector<float> scales(static_cast<size_t>(total_rows));
        size_t row_offset = 0;
        for (const HostTensorView& tensor : views) {
            quantize_bf16_rows_into(
                tensor.data, static_cast<size_t>(tensor.shape[0]),
                static_cast<size_t>(tensor.shape[1]), quantized, scales,
                row_offset);
            row_offset += static_cast<size_t>(tensor.shape[0]);
        }
        weight.int8_storage.reset(total_count);
        weight.scales_storage.reset(scales.size());
        LFM_CUDA(cudaMemcpy(weight.int8_storage.data(), quantized.data(),
                            quantized.size() * sizeof(int8_t),
                            cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(weight.scales_storage.data(), scales.data(),
                            scales.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
        weight.linear.kind = LinearStorageKind::Int8;
        weight.linear.int8 = weight.int8_storage.data();
        weight.linear.scales = weight.scales_storage.data();
    } else if (options_.weight_mode == WeightMode::Int4) {
        const size_t packed_cols =
            (static_cast<size_t>(common_width) + 1) / 2;
        std::vector<uint8_t> quantized(
            static_cast<size_t>(total_rows) * packed_cols);
        std::vector<float> scales(static_cast<size_t>(total_rows));
        size_t row_offset = 0;
        for (const HostTensorView& tensor : views) {
            quantize_bf16_rows_int4_into(
                tensor.data, static_cast<size_t>(tensor.shape[0]),
                static_cast<size_t>(tensor.shape[1]), quantized, scales,
                row_offset);
            row_offset += static_cast<size_t>(tensor.shape[0]);
        }
        weight.int4_storage.reset(quantized.size());
        weight.scales_storage.reset(scales.size());
        LFM_CUDA(cudaMemcpy(weight.int4_storage.data(), quantized.data(),
                            quantized.size() * sizeof(uint8_t),
                            cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(weight.scales_storage.data(), scales.data(),
                            scales.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
        weight.linear.kind = LinearStorageKind::Int4;
        weight.linear.int4 = weight.int4_storage.data();
        weight.linear.scales = weight.scales_storage.data();
    } else {
        weight.bf16_storage.reset(total_count);
        size_t offset = 0;
        for (const HostTensorView& tensor : views) {
            LFM_CUDA(cudaMemcpy(weight.bf16_storage.data() + offset,
                                tensor.data, tensor.bytes,
                                cudaMemcpyHostToDevice));
            offset += tensor.bytes / sizeof(__nv_bfloat16);
        }
        weight.linear.kind = LinearStorageKind::Bf16;
        weight.linear.bf16 = weight.bf16_storage.data();
    }
    weight.linear.rows = static_cast<int>(total_rows);
    weight.linear.cols = static_cast<int>(common_width);
    weight.linear.validate_storage();

    auto [it, inserted] = weights_->tensors.emplace(synthetic_name, std::move(weight));
    if (!inserted) {
        throw std::runtime_error("duplicate concatenated weight: " + synthetic_name);
    }
    return &it->second.linear;
}

LfmModel::Impl::LinearWeight LfmModel::Impl::slice_rows(
    const LinearWeight& weight, int row_offset, int rows) const {
    if (row_offset < 0 || rows <= 0 || row_offset + rows > weight.rows) {
        throw std::out_of_range("linear weight row slice is out of range");
    }
    LinearWeight result = weight;
    result.rows = rows;
    if (weight.bf16) {
        result.bf16 += static_cast<size_t>(row_offset) * weight.cols;
    }
    if (weight.int8) {
        result.int8 += static_cast<size_t>(row_offset) * weight.cols;
        result.scales += row_offset;
    }
    if (weight.int4) {
        const size_t packed_cols =
            (static_cast<size_t>(weight.cols) + 1) / 2;
        result.int4 += static_cast<size_t>(row_offset) * packed_cols;
        result.scales += row_offset;
    }
    return result;
}

void LfmModel::Impl::initialize_rope_tables() {
    const size_t table_elements =
        static_cast<size_t>(max_context_) * LfmConfig::rope_pairs;
    std::vector<__nv_bfloat16> cos_table(table_elements);
    std::vector<__nv_bfloat16> sin_table(table_elements);

    for (int position = 0; position < max_context_; ++position) {
        for (int pair = 0; pair < LfmConfig::rope_pairs; ++pair) {
            const float exponent =
                -2.0f * static_cast<float>(pair) /
                static_cast<float>(LfmConfig::head_dim);
            const float frequency = std::pow(LfmConfig::rope_theta, exponent);
            const float angle = static_cast<float>(position) * frequency;
            const size_t index =
                static_cast<size_t>(position) * LfmConfig::rope_pairs + pair;
            cos_table[index] = __float2bfloat16(std::cos(angle));
            sin_table[index] = __float2bfloat16(std::sin(angle));
        }
    }

    rope_cos_.reset(table_elements);
    rope_sin_.reset(table_elements);
    LFM_CUDA(cudaMemcpy(rope_cos_.data(), cos_table.data(),
                        cos_table.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
    LFM_CUDA(cudaMemcpy(rope_sin_.data(), sin_table.data(),
                        sin_table.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
}

void LfmModel::Impl::warmup_decode_gemms() {
    hidden_.zero_async(stream_.get());
    normed_.zero_async(stream_.get());
    op_output_.zero_async(stream_.get());
    qkv_output_.zero_async(stream_.get());
    conv_projected_.zero_async(stream_.get());
    gate_up_.zero_async(stream_.get());
    activated_.zero_async(stream_.get());
    logits_.zero_async(stream_.get());

    const AttentionLayer* attention_layer = nullptr;
    const ConvolutionLayer* convolution_layer = nullptr;
    for (const Layer& layer : layers_) {
        if (!attention_layer) attention_layer = as_attention(layer);
        if (!convolution_layer) convolution_layer = as_convolution(layer);
    }
    if (!attention_layer || !convolution_layer || layers_.empty()) {
        throw std::runtime_error("compiled layer map is incomplete");
    }
    const LayerCommon& first_common = common(layers_.front());

    if (options_.fused_projections) {
        linear(normed_.data(), *attention_layer->qkv, qkv_output_.data(),
               1, LfmConfig::qkv_width, LfmConfig::hidden);
        linear(normed_.data(), *first_common.w13, gate_up_.data(),
               1, 2 * LfmConfig::intermediate, LfmConfig::hidden);
    } else {
        const LinearWeight q_weight =
            slice_rows(*attention_layer->qkv, 0, LfmConfig::q_width);
        const LinearWeight k_weight = slice_rows(
            *attention_layer->qkv, LfmConfig::q_width, LfmConfig::kv_width);
        const LinearWeight v_weight = slice_rows(
            *attention_layer->qkv, LfmConfig::q_width + LfmConfig::kv_width,
            LfmConfig::kv_width);
        linear(normed_.data(), q_weight, qkv_output_.data(),
               1, LfmConfig::q_width, LfmConfig::hidden);
        linear(normed_.data(), k_weight, qkv_output_.data() + LfmConfig::q_width,
               1, LfmConfig::kv_width, LfmConfig::hidden);
        linear(normed_.data(), v_weight,
               qkv_output_.data() + LfmConfig::q_width + LfmConfig::kv_width,
               1, LfmConfig::kv_width, LfmConfig::hidden);

        const LinearWeight w1 =
            slice_rows(*first_common.w13, 0, LfmConfig::intermediate);
        const LinearWeight w3 = slice_rows(
            *first_common.w13, LfmConfig::intermediate,
            LfmConfig::intermediate);
        linear(normed_.data(), w1, gate_up_.data(),
               1, LfmConfig::intermediate, LfmConfig::hidden);
        linear(normed_.data(), w3, gate_up_.data() + LfmConfig::intermediate,
               1, LfmConfig::intermediate, LfmConfig::hidden);
    }

    linear(op_output_.data(), *attention_layer->out, hidden_.data(),
           1, LfmConfig::hidden, LfmConfig::hidden,
           options_.fused_residuals ? 1.0f : 0.0f);
    linear(normed_.data(), *convolution_layer->conv_in, conv_projected_.data(),
           1, 3 * LfmConfig::hidden, LfmConfig::hidden);
    linear(activated_.data(), *first_common.w2, hidden_.data(),
           1, LfmConfig::hidden, LfmConfig::intermediate,
           options_.fused_residuals ? 1.0f : 0.0f);
    linear(normed_.data(), *embedding_, logits_.data(),
           1, LfmConfig::vocab, LfmConfig::hidden);
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
}

void LfmModel::Impl::reset(bool allocate_local_kv) {
    allocate_local_kv = allocate_local_kv && options_.allocate_local_kv_cache;
    if (allocate_local_kv && !local_kv_cache_available_) {
        const size_t cache_elements =
            static_cast<size_t>(max_context_) * LfmConfig::kv_width;
        const size_t scale_elements =
            static_cast<size_t>(max_context_) * LfmConfig::kv_heads;
        for (Layer& layer : layers_) {
            AttentionLayer* attention = as_attention(layer);
            if (!attention) continue;
            if (options_.kv_cache_mode == KvCacheMode::Int8) {
                attention->key_cache_int8.reset(cache_elements);
                attention->value_cache_int8.reset(cache_elements);
                attention->key_cache_scales.reset(scale_elements);
                attention->value_cache_scales.reset(scale_elements);
            } else {
                attention->key_cache.reset(cache_elements);
                attention->value_cache.reset(cache_elements);
            }
        }
        local_kv_cache_available_ = true;
    }
    position_ = 0;
    phase_ = SessionPhase::Empty;
    const int32_t zero = 0;
    uint64_t seed = generation_.seed;
    LFM_CUDA(cudaMemcpyAsync(position_device_.data(), &zero, sizeof(zero),
                             cudaMemcpyHostToDevice, stream_.get()));
    LFM_CUDA(cudaMemcpyAsync(rng_state_.data(), &seed, sizeof(seed),
                             cudaMemcpyHostToDevice, stream_.get()));
    seen_tokens_.zero_async(stream_.get());
    for (Layer& layer : layers_) {
        if (AttentionLayer* attention = as_attention(layer)) {
            if (!local_kv_cache_available_) continue;
            if (options_.kv_cache_mode == KvCacheMode::Int8) {
                attention->key_cache_int8.zero_async(stream_.get());
                attention->value_cache_int8.zero_async(stream_.get());
                attention->key_cache_scales.zero_async(stream_.get());
                attention->value_cache_scales.zero_async(stream_.get());
            } else {
                attention->key_cache.zero_async(stream_.get());
                attention->value_cache.zero_async(stream_.get());
            }
        } else {
            as_convolution(layer)->conv_state.zero_async(stream_.get());
        }
    }
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
}

void LfmModel::Impl::linear_cublas(const __nv_bfloat16* x,
                             const __nv_bfloat16* weight,
                             __nv_bfloat16* y,
                             int m,
                             int n,
                             int k,
                             float beta) {
    const float alpha = 1.0f;
    LFM_CUBLAS(cublasGemmEx(
        cublas_.get(), CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k, &alpha,
        weight, CUDA_R_16BF, k,
        x, CUDA_R_16BF, k,
        &beta,
        y, CUDA_R_16BF, n,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

LfmModel::Impl::LtPlan& LfmModel::Impl::get_or_create_lt_plan(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    int m,
    int n,
    int k) {
    const MatmulKey key{m, n, k};
    if (auto it = lt_plans_.find(key); it != lt_plans_.end()) {
        return *it->second;
    }

    auto plan = std::make_unique<LtPlan>();
    LFM_CUBLAS(cublasLtMatmulDescCreate(
        &plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t transa = CUBLAS_OP_T;
    const cublasOperation_t transb = CUBLAS_OP_N;
    LFM_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_TRANSA,
        &transa, sizeof(transa)));
    LFM_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_TRANSB,
        &transb, sizeof(transb)));

    // Buffers are row-major in the runtime. Interpreting them as column-major
    // transposes the logical matrices: W[n,k] -> A[k,n], X[m,k] -> B[k,m],
    // and Y[m,n] -> D[n,m]. TRANSA restores W to [n,k].
    LFM_CUBLAS(cublasLtMatrixLayoutCreate(
        &plan->a, CUDA_R_16BF, k, n, k));
    LFM_CUBLAS(cublasLtMatrixLayoutCreate(
        &plan->b, CUDA_R_16BF, k, m, k));
    LFM_CUBLAS(cublasLtMatrixLayoutCreate(
        &plan->c, CUDA_R_16BF, n, m, n));
    LFM_CUBLAS(cublasLtMatrixLayoutCreate(
        &plan->d, CUDA_R_16BF, n, m, n));

    cublasLtMatmulPreference_t preference = nullptr;
    LFM_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    try {
        const uint64_t workspace_limit =
            static_cast<uint64_t>(options_.lt_workspace_bytes);
        LFM_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_limit, sizeof(workspace_limit)));

        std::vector<cublasLtMatmulHeuristicResult_t> results(
            static_cast<size_t>(options_.lt_heuristics));
        int returned = 0;
        LFM_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
            cublas_lt_.get(), plan->operation,
            plan->a, plan->b, plan->c, plan->d,
            preference, options_.lt_heuristics,
            results.data(), &returned));

        std::vector<int> valid;
        for (int i = 0; i < returned; ++i) {
            if (results[static_cast<size_t>(i)].state == CUBLAS_STATUS_SUCCESS &&
                results[static_cast<size_t>(i)].workspaceSize <=
                    options_.lt_workspace_bytes) {
                valid.push_back(i);
            }
        }

        if (!valid.empty()) {
            int selected = valid.front();
            // Autotuning is intentionally restricted to decode shapes. Large
            // prefill outputs would require an excessive temporary buffer.
            if (options_.lt_autotune && m == 1 && valid.size() > 1) {
                DeviceBuffer<__nv_bfloat16> scratch(static_cast<size_t>(n));
                const float alpha = 1.0f;
                const float beta = 0.0f;
                float best_ms = std::numeric_limits<float>::infinity();
                for (const int candidate : valid) {
                    const auto& result = results[static_cast<size_t>(candidate)];
                    LFM_CUBLAS(cublasLtMatmul(
                        cublas_lt_.get(), plan->operation,
                        &alpha, weight, plan->a, x, plan->b,
                        &beta, scratch.data(), plan->c,
                        scratch.data(), plan->d, &result.algo,
                        lt_workspace_.data(), result.workspaceSize,
                        stream_.get()));
                    CudaEvent begin;
                    CudaEvent end;
                    begin.record(stream_.get());
                    constexpr int iterations = 3;
                    for (int iteration = 0; iteration < iterations; ++iteration) {
                        LFM_CUBLAS(cublasLtMatmul(
                            cublas_lt_.get(), plan->operation,
                            &alpha, weight, plan->a, x, plan->b,
                            &beta, scratch.data(), plan->c,
                            scratch.data(), plan->d, &result.algo,
                            lt_workspace_.data(), result.workspaceSize,
                            stream_.get()));
                    }
                    end.record(stream_.get());
                    end.synchronize();
                    const float elapsed = CudaEvent::elapsed_ms(begin, end);
                    if (elapsed < best_ms) {
                        best_ms = elapsed;
                        selected = candidate;
                    }
                }
            }
            const auto& chosen = results[static_cast<size_t>(selected)];
            plan->algorithm = chosen.algo;
            plan->workspace_size = chosen.workspaceSize;
            plan->available = true;
        }
    } catch (...) {
        cublasLtMatmulPreferenceDestroy(preference);
        throw;
    }
    LFM_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));

    auto [it, inserted] = lt_plans_.emplace(key, std::move(plan));
    if (!inserted) throw std::runtime_error("duplicate cuBLASLt plan");
    return *it->second;
}

void LfmModel::Impl::linear_cublaslt(const __nv_bfloat16* x,
                               const __nv_bfloat16* weight,
                               __nv_bfloat16* y,
                               int m,
                               int n,
                               int k,
                               float beta) {
    LtPlan& plan = get_or_create_lt_plan(x, weight, m, n, k);
    if (!plan.available) {
        linear_cublas(x, weight, y, m, n, k, beta);
        return;
    }
    const float alpha = 1.0f;
    LFM_CUBLAS(cublasLtMatmul(
        cublas_lt_.get(), plan.operation,
        &alpha, weight, plan.a, x, plan.b,
        &beta, y, plan.c, y, plan.d,
        &plan.algorithm, lt_workspace_.data(), plan.workspace_size,
        stream_.get()));
}

void LfmModel::Impl::linear(const __nv_bfloat16* x,
                      const LinearWeight& weight,
                      __nv_bfloat16* y,
                      int m,
                      int n,
                      int k,
                      float beta) {
    if (weight.rows != n || weight.cols != k) {
        throw std::runtime_error("linear weight shape does not match the requested GEMM");
    }
    weight.validate_storage();
    switch (plan_.linear_kernel()) {
        case LinearKernelKind::W4A16:
            if (!weight.int4_quantized()) {
                throw std::runtime_error("execution plan requires INT4 weights");
            }
            launch_w4a16_linear(x, weight.int4, weight.scales, y,
                                m, n, k, beta, stream_.get());
            return;
        case LinearKernelKind::W8A16:
            if (!weight.int8_quantized()) {
                throw std::runtime_error("execution plan requires INT8 weights");
            }
            launch_w8a16_linear(x, weight.int8, weight.scales, y,
                                m, n, k, beta, stream_.get());
            return;
        case LinearKernelKind::Bf16CublasLt:
            if (weight.kind != LinearStorageKind::Bf16) {
                throw std::runtime_error("execution plan requires BF16 weights");
            }
            linear_cublaslt(x, weight.bf16, y, m, n, k, beta);
            return;
        case LinearKernelKind::Bf16Cublas:
            if (weight.kind != LinearStorageKind::Bf16) {
                throw std::runtime_error("execution plan requires BF16 weights");
            }
            linear_cublas(x, weight.bf16, y, m, n, k, beta);
            return;
    }
    throw std::runtime_error("unknown linear execution plan");
}

void LfmModel::Impl::allocate_prefill_workspace(int rows) {
    const size_t r = static_cast<size_t>(rows);
    prefill_tokens_.reset(r);
    prefill_hidden_.reset(r * LfmConfig::hidden);
    prefill_residual_.reset(r * LfmConfig::hidden);
    prefill_normed_.reset(r * LfmConfig::hidden);
    prefill_op_output_.reset(r * LfmConfig::hidden);
    prefill_q_.reset(r * LfmConfig::q_width);
    prefill_k_.reset(r * LfmConfig::kv_width);
    prefill_v_.reset(r * LfmConfig::kv_width);
    prefill_conv_projected_.reset(r * 3 * LfmConfig::hidden);
    prefill_gate_up_.reset(r * 2 * LfmConfig::intermediate);
    prefill_activated_.reset(r * LfmConfig::intermediate);
    prefill_mlp_output_.reset(r * LfmConfig::hidden);
}

void LfmModel::Impl::release_prefill_workspace() {
    prefill_tokens_.reset(0);
    prefill_hidden_.reset(0);
    prefill_residual_.reset(0);
    prefill_normed_.reset(0);
    prefill_op_output_.reset(0);
    prefill_q_.reset(0);
    prefill_k_.reset(0);
    prefill_v_.reset(0);
    prefill_conv_projected_.reset(0);
    prefill_gate_up_.reset(0);
    prefill_activated_.reset(0);
    prefill_mlp_output_.reset(0);
}

void LfmModel::Impl::run_mlp_decode(const LayerCommon& common_layer) {
    launch_rmsnorm(hidden_.data(), common_layer.ffn_norm, normed_.data(),
                   1, LfmConfig::hidden, LfmConfig::norm_eps,
                   stream_.get());
    if (options_.fused_projections) {
        linear(normed_.data(), *common_layer.w13, gate_up_.data(),
               1, 2 * LfmConfig::intermediate, LfmConfig::hidden);
    } else {
        const LinearWeight w1 =
            slice_rows(*common_layer.w13, 0, LfmConfig::intermediate);
        const LinearWeight w3 = slice_rows(
            *common_layer.w13, LfmConfig::intermediate, LfmConfig::intermediate);
        linear(normed_.data(), w1, gate_up_.data(),
               1, LfmConfig::intermediate, LfmConfig::hidden);
        linear(normed_.data(), w3, gate_up_.data() + LfmConfig::intermediate,
               1, LfmConfig::intermediate, LfmConfig::hidden);
    }
    launch_swiglu_fused(gate_up_.data(), activated_.data(),
                        LfmConfig::intermediate, stream_.get());
    if (options_.fused_residuals) {
        linear(activated_.data(), *common_layer.w2, hidden_.data(),
               1, LfmConfig::hidden, LfmConfig::intermediate, 1.0f);
    } else {
        linear(activated_.data(), *common_layer.w2, mlp_output_.data(),
               1, LfmConfig::hidden, LfmConfig::intermediate);
        launch_residual_add(hidden_.data(), mlp_output_.data(),
                            LfmConfig::hidden, stream_.get());
    }
}

void LfmModel::Impl::run_mlp_prefill(const LayerCommon& common_layer, int rows) {
    const size_t matrix_elements =
        static_cast<size_t>(rows) * LfmConfig::intermediate;
    launch_rmsnorm(prefill_hidden_.data(), common_layer.ffn_norm,
                   prefill_normed_.data(), rows, LfmConfig::hidden,
                   LfmConfig::norm_eps, stream_.get());
    if (options_.fused_projections) {
        linear(prefill_normed_.data(), *common_layer.w13, prefill_gate_up_.data(),
               rows, 2 * LfmConfig::intermediate, LfmConfig::hidden);
    } else {
        const LinearWeight w1 =
            slice_rows(*common_layer.w13, 0, LfmConfig::intermediate);
        const LinearWeight w3 = slice_rows(
            *common_layer.w13, LfmConfig::intermediate, LfmConfig::intermediate);
        linear(prefill_normed_.data(), w1, prefill_gate_up_.data(),
               rows, LfmConfig::intermediate, LfmConfig::hidden);
        linear(prefill_normed_.data(), w3,
               prefill_gate_up_.data() + matrix_elements,
               rows, LfmConfig::intermediate, LfmConfig::hidden);
    }
    launch_swiglu_fused(prefill_gate_up_.data(), prefill_activated_.data(),
                        static_cast<int>(matrix_elements), stream_.get());
    if (options_.fused_residuals) {
        linear(prefill_activated_.data(), *common_layer.w2, prefill_hidden_.data(),
               rows, LfmConfig::hidden, LfmConfig::intermediate, 1.0f);
    } else {
        linear(prefill_activated_.data(), *common_layer.w2, prefill_mlp_output_.data(),
               rows, LfmConfig::hidden, LfmConfig::intermediate);
        launch_residual_add(prefill_hidden_.data(), prefill_mlp_output_.data(),
                            rows * LfmConfig::hidden, stream_.get());
    }
}

void LfmModel::Impl::prefill_batched(const std::vector<int32_t>& tokens) {
    reset();
    const int rows = static_cast<int>(tokens.size());
    allocate_prefill_workspace(rows);

    LFM_CUDA(cudaMemcpyAsync(prefill_tokens_.data(), tokens.data(),
                             tokens.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    launch_mark_seen_batch(prefill_tokens_.data(), rows, seen_tokens_.data(),
                           LfmConfig::vocab, stream_.get());
    if (embedding_->int4_quantized()) {
        launch_embedding_int4_batch(
            prefill_tokens_.data(), rows, embedding_->int4, embedding_->scales,
            prefill_hidden_.data(), LfmConfig::hidden, stream_.get());
    } else if (embedding_->int8) {
        launch_embedding_int8_batch(
            prefill_tokens_.data(), rows, embedding_->int8, embedding_->scales,
            prefill_hidden_.data(), LfmConfig::hidden, stream_.get());
    } else {
        launch_embedding_batch(prefill_tokens_.data(), rows, embedding_->bf16,
                               prefill_hidden_.data(), LfmConfig::hidden,
                               stream_.get());
    }

    for (Layer& layer : layers_) {
        LayerCommon& common_layer = common(layer);
        if (!options_.fused_residuals) {
            LFM_CUDA(cudaMemcpyAsync(
                prefill_residual_.data(), prefill_hidden_.data(),
                prefill_hidden_.bytes(), cudaMemcpyDeviceToDevice,
                stream_.get()));
        }
        launch_rmsnorm(prefill_hidden_.data(), common_layer.operator_norm,
                       prefill_normed_.data(), rows, LfmConfig::hidden,
                       LfmConfig::norm_eps, stream_.get());

        if (AttentionLayer* attention = as_attention(layer)) {
            const LinearWeight q_weight =
                slice_rows(*attention->qkv, 0, LfmConfig::q_width);
            const LinearWeight k_weight = slice_rows(
                *attention->qkv, LfmConfig::q_width, LfmConfig::kv_width);
            const LinearWeight v_weight = slice_rows(
                *attention->qkv, LfmConfig::q_width + LfmConfig::kv_width,
                LfmConfig::kv_width);
            linear(prefill_normed_.data(), q_weight, prefill_q_.data(),
                   rows, LfmConfig::q_width, LfmConfig::hidden);
            linear(prefill_normed_.data(), k_weight, prefill_k_.data(),
                   rows, LfmConfig::kv_width, LfmConfig::hidden);
            linear(prefill_normed_.data(), v_weight, prefill_v_.data(),
                   rows, LfmConfig::kv_width, LfmConfig::hidden);

            if (options_.fast_attention) {
                launch_qk_norm_rope_prefill_fast(
                    prefill_q_.data(), prefill_k_.data(),
                    attention->q_norm, attention->k_norm,
                    rope_cos_.data(), rope_sin_.data(), rows,
                    LfmConfig::q_heads, LfmConfig::kv_heads,
                    LfmConfig::head_dim, LfmConfig::norm_eps,
                    stream_.get());
            } else {
                launch_qk_norm_rope_prefill_strict(
                    prefill_q_.data(), prefill_k_.data(),
                    attention->q_norm, attention->k_norm,
                    rope_cos_.data(), rope_sin_.data(), rows,
                    LfmConfig::q_heads, LfmConfig::kv_heads,
                    LfmConfig::head_dim, LfmConfig::norm_eps,
                    stream_.get());
            }
            if (options_.kv_cache_mode == KvCacheMode::Int8) {
                launch_store_kv_int8_prefill(
                    prefill_k_.data(), prefill_v_.data(),
                    attention->key_cache_int8.data(), attention->value_cache_int8.data(),
                    attention->key_cache_scales.data(), attention->value_cache_scales.data(),
                    rows, LfmConfig::kv_heads, LfmConfig::head_dim,
                    stream_.get());
                if (options_.fast_attention) {
                    launch_gqa_prefill_online_int8(
                        prefill_q_.data(), attention->key_cache_int8.data(),
                        attention->value_cache_int8.data(),
                        attention->key_cache_scales.data(),
                        attention->value_cache_scales.data(),
                        prefill_op_output_.data(), rows, LfmConfig::q_heads,
                        LfmConfig::kv_heads, LfmConfig::head_dim, stream_.get());
                } else {
                    launch_gqa_prefill_strict_int8(
                        prefill_q_.data(), attention->key_cache_int8.data(),
                        attention->value_cache_int8.data(),
                        attention->key_cache_scales.data(),
                        attention->value_cache_scales.data(),
                        prefill_op_output_.data(), rows, LfmConfig::q_heads,
                        LfmConfig::kv_heads, LfmConfig::head_dim, stream_.get());
                }
            } else {
                launch_store_kv_prefill(
                    prefill_k_.data(), prefill_v_.data(),
                    attention->key_cache.data(), attention->value_cache.data(),
                    rows, LfmConfig::kv_width, stream_.get());
                if (options_.fast_attention) {
                    launch_gqa_prefill_online(
                        prefill_q_.data(), attention->key_cache.data(),
                        attention->value_cache.data(), prefill_op_output_.data(),
                        rows, LfmConfig::q_heads, LfmConfig::kv_heads,
                        LfmConfig::head_dim, stream_.get());
                } else {
                    launch_gqa_prefill_strict(
                        prefill_q_.data(), attention->key_cache.data(),
                        attention->value_cache.data(), prefill_op_output_.data(),
                        rows, LfmConfig::q_heads, LfmConfig::kv_heads,
                        LfmConfig::head_dim, stream_.get());
                }
            }
            linear(prefill_op_output_.data(), *attention->out,
                   prefill_hidden_.data(), rows, LfmConfig::hidden,
                   LfmConfig::hidden, options_.fused_residuals ? 1.0f : 0.0f);
        } else {
            ConvolutionLayer& convolution = *as_convolution(layer);
            linear(prefill_normed_.data(), *convolution.conv_in,
                   prefill_conv_projected_.data(), rows,
                   3 * LfmConfig::hidden, LfmConfig::hidden);
            launch_conv_prefill(
                prefill_conv_projected_.data(), convolution.conv_weight,
                convolution.conv_state.data(), prefill_op_output_.data(),
                rows, LfmConfig::hidden, LfmConfig::conv_cache,
                stream_.get());
            linear(prefill_op_output_.data(), *convolution.conv_out,
                   prefill_hidden_.data(), rows, LfmConfig::hidden,
                   LfmConfig::hidden, options_.fused_residuals ? 1.0f : 0.0f);
        }

        if (!options_.fused_residuals) {
            launch_residual_add(prefill_hidden_.data(), prefill_residual_.data(),
                                rows * LfmConfig::hidden, stream_.get());
        }
        run_mlp_prefill(common_layer, rows);
    }

    const __nv_bfloat16* last_hidden = prefill_hidden_.data() +
        static_cast<size_t>(rows - 1) * LfmConfig::hidden;
    launch_rmsnorm(last_hidden, final_norm_, normed_.data(),
                   1, LfmConfig::hidden, LfmConfig::norm_eps,
                   stream_.get());
    linear(normed_.data(), *embedding_, logits_.data(),
           1, LfmConfig::vocab, LfmConfig::hidden);

    position_ = rows;
    LFM_CUDA(cudaMemcpyAsync(position_device_.data(), &position_,
                             sizeof(position_), cudaMemcpyHostToDevice,
                             stream_.get()));
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    release_prefill_workspace();
    phase_ = SessionPhase::Ready;
}

void LfmModel::Impl::forward_token_host(int32_t token, bool compute_logits) {
    if (position_ >= max_context_) {
        throw std::runtime_error("context limit reached");
    }
    if (embedding_->int4_quantized()) {
        launch_embedding_int4(token, embedding_->int4, embedding_->scales,
                              hidden_.data(), LfmConfig::hidden, stream_.get());
    } else if (embedding_->int8) {
        launch_embedding_int8(token, embedding_->int8, embedding_->scales,
                              hidden_.data(), LfmConfig::hidden, stream_.get());
    } else {
        launch_embedding(token, embedding_->bf16, hidden_.data(),
                         LfmConfig::hidden, stream_.get());
    }

    for (Layer& layer : layers_) {
        LayerCommon& common_layer = common(layer);
        if (!options_.fused_residuals) {
            LFM_CUDA(cudaMemcpyAsync(
                residual_.data(), hidden_.data(), hidden_.bytes(),
                cudaMemcpyDeviceToDevice, stream_.get()));
        }
        launch_rmsnorm(hidden_.data(), common_layer.operator_norm, normed_.data(),
                       1, LfmConfig::hidden, LfmConfig::norm_eps,
                       stream_.get());
        if (AttentionLayer* attention = as_attention(layer)) {
            __nv_bfloat16* q = qkv_output_.data();
            __nv_bfloat16* k = q + LfmConfig::q_width;
            __nv_bfloat16* v = k + LfmConfig::kv_width;
            if (options_.fused_projections) {
                linear(normed_.data(), *attention->qkv, qkv_output_.data(),
                       1, LfmConfig::qkv_width, LfmConfig::hidden);
            } else {
                const LinearWeight q_weight =
                    slice_rows(*attention->qkv, 0, LfmConfig::q_width);
                const LinearWeight k_weight = slice_rows(
                    *attention->qkv, LfmConfig::q_width, LfmConfig::kv_width);
                const LinearWeight v_weight = slice_rows(
                    *attention->qkv, LfmConfig::q_width + LfmConfig::kv_width,
                    LfmConfig::kv_width);
                linear(normed_.data(), q_weight, q,
                       1, LfmConfig::q_width, LfmConfig::hidden);
                linear(normed_.data(), k_weight, k,
                       1, LfmConfig::kv_width, LfmConfig::hidden);
                linear(normed_.data(), v_weight, v,
                       1, LfmConfig::kv_width, LfmConfig::hidden);
            }
            if (options_.fast_attention) {
                launch_qk_norm_rope_fast(
                    q, k, attention->q_norm, attention->k_norm,
                    rope_cos_.data(), rope_sin_.data(),
                    LfmConfig::q_heads, LfmConfig::kv_heads,
                    LfmConfig::head_dim, position_,
                    LfmConfig::norm_eps, stream_.get());
            } else {
                launch_qk_norm_rope_strict(
                    q, k, attention->q_norm, attention->k_norm,
                    rope_cos_.data(), rope_sin_.data(),
                    LfmConfig::q_heads, LfmConfig::kv_heads,
                    LfmConfig::head_dim, position_,
                    LfmConfig::norm_eps, stream_.get());
            }
            if (options_.kv_cache_mode == KvCacheMode::Int8) {
                launch_store_kv_int8(
                    k, v, attention->key_cache_int8.data(),
                    attention->value_cache_int8.data(),
                    attention->key_cache_scales.data(),
                    attention->value_cache_scales.data(), position_,
                    LfmConfig::kv_heads, LfmConfig::head_dim, stream_.get());
                if (options_.fast_attention) {
                    launch_gqa_decode_online_int8(
                        q, attention->key_cache_int8.data(),
                        attention->value_cache_int8.data(),
                        attention->key_cache_scales.data(),
                        attention->value_cache_scales.data(), op_output_.data(),
                        position_ + 1, LfmConfig::q_heads,
                        LfmConfig::kv_heads, LfmConfig::head_dim, stream_.get());
                } else {
                    launch_gqa_decode_strict_int8(
                        q, attention->key_cache_int8.data(),
                        attention->value_cache_int8.data(),
                        attention->key_cache_scales.data(),
                        attention->value_cache_scales.data(), op_output_.data(),
                        position_ + 1, LfmConfig::q_heads,
                        LfmConfig::kv_heads, LfmConfig::head_dim, stream_.get());
                }
            } else {
                launch_store_kv(k, v, attention->key_cache.data(),
                                attention->value_cache.data(), position_,
                                LfmConfig::kv_width, stream_.get());
                if (options_.fast_attention) {
                    launch_gqa_decode_online(
                        q, attention->key_cache.data(), attention->value_cache.data(),
                        op_output_.data(), position_ + 1,
                        LfmConfig::q_heads, LfmConfig::kv_heads,
                        LfmConfig::head_dim, stream_.get());
                } else {
                    launch_gqa_decode_strict(
                        q, attention->key_cache.data(), attention->value_cache.data(),
                        op_output_.data(), position_ + 1,
                        LfmConfig::q_heads, LfmConfig::kv_heads,
                        LfmConfig::head_dim, stream_.get());
                }
            }
            linear(op_output_.data(), *attention->out, hidden_.data(),
                   1, LfmConfig::hidden, LfmConfig::hidden,
                   options_.fused_residuals ? 1.0f : 0.0f);
        } else {
            ConvolutionLayer& convolution = *as_convolution(layer);
            linear(normed_.data(), *convolution.conv_in, conv_projected_.data(),
                   1, 3 * LfmConfig::hidden, LfmConfig::hidden);
            launch_conv_decode(
                conv_projected_.data(), convolution.conv_weight,
                convolution.conv_state.data(), op_output_.data(),
                LfmConfig::hidden, LfmConfig::conv_cache, position_,
                stream_.get());
            linear(op_output_.data(), *convolution.conv_out, hidden_.data(),
                   1, LfmConfig::hidden, LfmConfig::hidden,
                   options_.fused_residuals ? 1.0f : 0.0f);
        }
        if (!options_.fused_residuals) {
            launch_residual_add(hidden_.data(), residual_.data(),
                                LfmConfig::hidden, stream_.get());
        }
        run_mlp_decode(common_layer);
    }
    if (compute_logits) {
        launch_rmsnorm(hidden_.data(), final_norm_, normed_.data(),
                       1, LfmConfig::hidden, LfmConfig::norm_eps,
                       stream_.get());
        linear(normed_.data(), *embedding_, logits_.data(),
               1, LfmConfig::vocab, LfmConfig::hidden);
    }
    ++position_;
}


void LfmModel::Impl::forward_token_paged_host(
    int32_t token, bool compute_logits, PhysicalPagedKvCache& paged_kv,
    const uint32_t* device_page_table, int page_table_stride) {
    if (position_ >= max_context_) {
        throw std::runtime_error("context limit reached");
    }
    if (paged_kv.mode() != options_.kv_cache_mode) {
        throw std::invalid_argument("model and physical paged KV modes differ");
    }
    if (embedding_->int4_quantized()) {
        launch_embedding_int4(token, embedding_->int4, embedding_->scales,
                              hidden_.data(), LfmConfig::hidden, stream_.get());
    } else if (embedding_->int8) {
        launch_embedding_int8(token, embedding_->int8, embedding_->scales,
                              hidden_.data(), LfmConfig::hidden, stream_.get());
    } else {
        launch_embedding(token, embedding_->bf16, hidden_.data(),
                         LfmConfig::hidden, stream_.get());
    }

    for (int layer_index = 0; layer_index < LfmConfig::layers; ++layer_index) {
        Layer& layer = layers_[static_cast<size_t>(layer_index)];
        LayerCommon& common_layer = common(layer);
        if (!options_.fused_residuals) {
            LFM_CUDA(cudaMemcpyAsync(residual_.data(), hidden_.data(), hidden_.bytes(),
                                     cudaMemcpyDeviceToDevice, stream_.get()));
        }
        launch_rmsnorm(hidden_.data(), common_layer.operator_norm, normed_.data(),
                       1, LfmConfig::hidden, LfmConfig::norm_eps, stream_.get());
        if (AttentionLayer* attention = as_attention(layer)) {
            __nv_bfloat16* q = qkv_output_.data();
            __nv_bfloat16* k = q + LfmConfig::q_width;
            __nv_bfloat16* v = k + LfmConfig::kv_width;
            if (options_.fused_projections) {
                linear(normed_.data(), *attention->qkv, qkv_output_.data(), 1,
                       LfmConfig::qkv_width, LfmConfig::hidden);
            } else {
                const LinearWeight q_weight =
                    slice_rows(*attention->qkv, 0, LfmConfig::q_width);
                const LinearWeight k_weight = slice_rows(
                    *attention->qkv, LfmConfig::q_width, LfmConfig::kv_width);
                const LinearWeight v_weight = slice_rows(
                    *attention->qkv, LfmConfig::q_width + LfmConfig::kv_width,
                    LfmConfig::kv_width);
                linear(normed_.data(), q_weight, q, 1, LfmConfig::q_width,
                       LfmConfig::hidden);
                linear(normed_.data(), k_weight, k, 1, LfmConfig::kv_width,
                       LfmConfig::hidden);
                linear(normed_.data(), v_weight, v, 1, LfmConfig::kv_width,
                       LfmConfig::hidden);
            }
            if (options_.fast_attention) {
                launch_qk_norm_rope_fast(
                    q, k, attention->q_norm, attention->k_norm, rope_cos_.data(),
                    rope_sin_.data(), LfmConfig::q_heads, LfmConfig::kv_heads,
                    LfmConfig::head_dim, position_, LfmConfig::norm_eps,
                    stream_.get());
            } else {
                launch_qk_norm_rope_strict(
                    q, k, attention->q_norm, attention->k_norm, rope_cos_.data(),
                    rope_sin_.data(), LfmConfig::q_heads, LfmConfig::kv_heads,
                    LfmConfig::head_dim, position_, LfmConfig::norm_eps,
                    stream_.get());
            }
            const int slot = PhysicalPagedKvCache::attention_slot(layer_index);
            if (slot < 0) throw std::logic_error("attention layer has no page slot");
            if (options_.kv_cache_mode == KvCacheMode::Int8) {
                launch_store_kv_int8_paged_batch(
                    k, v, paged_kv.key_int8(), paged_kv.value_int8(),
                    paged_kv.key_scales(), paged_kv.value_scales(),
                    device_page_table, page_table_stride, position_device_.data(),
                    1, slot, paged_kv.page_tokens(),
                    PhysicalPagedKvCache::attention_layers, LfmConfig::kv_heads,
                    LfmConfig::head_dim, stream_.get());
                if (use_segmented_attention(position_)) {
                    const int chunks = (position_ + 1 +
                        options_.attention_chunk_tokens - 1) /
                        options_.attention_chunk_tokens;
                    launch_gqa_decode_int8_paged_segmented_batch(
                        q, paged_kv.key_int8(), paged_kv.value_int8(),
                        paged_kv.key_scales(), paged_kv.value_scales(),
                        device_page_table, page_table_stride, op_output_.data(),
                        position_device_.data(), 1, slot,
                        paged_kv.page_tokens(),
                        PhysicalPagedKvCache::attention_layers,
                        LfmConfig::q_heads, LfmConfig::kv_heads,
                        LfmConfig::head_dim, options_.attention_chunk_tokens,
                        chunks, attention_partial_max_.data(),
                        attention_partial_denom_.data(),
                        attention_partial_accum_.data(), stream_.get());
                } else {
                    launch_gqa_decode_int8_paged_batch(
                        q, paged_kv.key_int8(), paged_kv.value_int8(),
                        paged_kv.key_scales(), paged_kv.value_scales(),
                        device_page_table, page_table_stride, op_output_.data(),
                        position_device_.data(), 1, slot,
                        paged_kv.page_tokens(),
                        PhysicalPagedKvCache::attention_layers,
                        LfmConfig::q_heads, LfmConfig::kv_heads,
                        LfmConfig::head_dim, options_.fast_attention,
                        stream_.get());
                }
            } else {
                launch_store_kv_paged_batch(
                    k, v, paged_kv.key_bf16(), paged_kv.value_bf16(),
                    device_page_table, page_table_stride, position_device_.data(),
                    1, slot, paged_kv.page_tokens(),
                    PhysicalPagedKvCache::attention_layers, LfmConfig::kv_heads,
                    LfmConfig::head_dim, stream_.get());
                if (use_segmented_attention(position_)) {
                    const int chunks = (position_ + 1 +
                        options_.attention_chunk_tokens - 1) /
                        options_.attention_chunk_tokens;
                    launch_gqa_decode_paged_segmented_batch(
                        q, paged_kv.key_bf16(), paged_kv.value_bf16(),
                        device_page_table, page_table_stride,
                        op_output_.data(), position_device_.data(), 1, slot,
                        paged_kv.page_tokens(),
                        PhysicalPagedKvCache::attention_layers,
                        LfmConfig::q_heads, LfmConfig::kv_heads,
                        LfmConfig::head_dim, options_.attention_chunk_tokens,
                        chunks, attention_partial_max_.data(),
                        attention_partial_denom_.data(),
                        attention_partial_accum_.data(), stream_.get());
                } else {
                    launch_gqa_decode_paged_batch(
                        q, paged_kv.key_bf16(), paged_kv.value_bf16(),
                        device_page_table, page_table_stride,
                        op_output_.data(), position_device_.data(), 1, slot,
                        paged_kv.page_tokens(),
                        PhysicalPagedKvCache::attention_layers,
                        LfmConfig::q_heads, LfmConfig::kv_heads,
                        LfmConfig::head_dim, options_.fast_attention,
                        stream_.get());
                }
            }
            linear(op_output_.data(), *attention->out, hidden_.data(), 1,
                   LfmConfig::hidden, LfmConfig::hidden,
                   options_.fused_residuals ? 1.0f : 0.0f);
        } else {
            ConvolutionLayer& convolution = *as_convolution(layer);
            linear(normed_.data(), *convolution.conv_in, conv_projected_.data(), 1,
                   3 * LfmConfig::hidden, LfmConfig::hidden);
            launch_conv_decode(conv_projected_.data(), convolution.conv_weight,
                               convolution.conv_state.data(), op_output_.data(),
                               LfmConfig::hidden, LfmConfig::conv_cache,
                               position_, stream_.get());
            linear(op_output_.data(), *convolution.conv_out, hidden_.data(), 1,
                   LfmConfig::hidden, LfmConfig::hidden,
                   options_.fused_residuals ? 1.0f : 0.0f);
        }
        if (!options_.fused_residuals) {
            launch_residual_add(hidden_.data(), residual_.data(),
                                LfmConfig::hidden, stream_.get());
        }
        run_mlp_decode(common_layer);
    }
    if (compute_logits) {
        launch_rmsnorm(hidden_.data(), final_norm_, normed_.data(), 1,
                       LfmConfig::hidden, LfmConfig::norm_eps, stream_.get());
        linear(normed_.data(), *embedding_, logits_.data(), 1,
               LfmConfig::vocab, LfmConfig::hidden);
    }
    ++position_;
    LFM_CUDA(cudaMemcpyAsync(position_device_.data(), &position_, sizeof(position_),
                             cudaMemcpyHostToDevice, stream_.get()));
}

void LfmModel::Impl::prefill_chunk_paged(
    const std::vector<int32_t>& tokens, bool begin, bool finalize,
    PhysicalPagedKvCache& paged_kv,
    const std::vector<uint32_t>& page_table) {
    if (tokens.empty()) {
        throw std::invalid_argument("prefill_chunk_paged needs at least one token");
    }
    if (paged_kv.mode() != options_.kv_cache_mode) {
        throw std::invalid_argument("model and physical paged KV modes differ");
    }
    if (begin) {
        release_local_kv_cache();
        reset(false);
        metrics_ = {};
    } else if (position_ == 0) {
        throw std::runtime_error(
            "paged prefill continuation requires an initial chunk or prefix state");
    }
    if (position_ + static_cast<int>(tokens.size()) > max_context_) {
        throw std::invalid_argument("paged prefill chunks exceed max_context");
    }
    const size_t final_position =
        static_cast<size_t>(position_) + tokens.size();
    const size_t pages_needed =
        (final_position + static_cast<size_t>(paged_kv.page_tokens()) - 1) /
        static_cast<size_t>(paged_kv.page_tokens());
    if (page_table.size() < pages_needed ||
        page_table.size() > static_cast<size_t>(paged_kv.max_pages_per_request())) {
        throw std::invalid_argument("paged prefill page table has invalid length");
    }
    if (paged_page_table_.size() < page_table.size()) {
        paged_page_table_.reset(page_table.size());
    }
    LFM_CUDA(cudaMemcpyAsync(paged_page_table_.data(), page_table.data(),
                             page_table.size() * sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    if (paged_prefill_tokens_.size() < tokens.size()) {
        paged_prefill_tokens_.reset(tokens.size());
    }
    LFM_CUDA(cudaMemcpyAsync(paged_prefill_tokens_.data(), tokens.data(),
                             tokens.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    launch_mark_seen_batch(paged_prefill_tokens_.data(),
                           static_cast<int>(tokens.size()), seen_tokens_.data(),
                           LfmConfig::vocab, stream_.get());
    phase_ = SessionPhase::Prefilling;
    const auto started = std::chrono::steady_clock::now();
    for (size_t i = 0; i < tokens.size(); ++i) {
        forward_token_paged_host(tokens[i], finalize && i + 1 == tokens.size(),
                                 paged_kv, paged_page_table_.data(),
                                 static_cast<int>(page_table.size()));
    }
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    const auto ended = std::chrono::steady_clock::now();
    metrics_.last_prefill_ms +=
        std::chrono::duration<double, std::milli>(ended - started).count();
    metrics_.prefill_tokens += tokens.size();
    if (finalize) {
        phase_ = SessionPhase::Ready;
        active_segmented_attention_ = use_segmented_attention(position_);
    }
}

void LfmModel::Impl::prefill_legacy(const std::vector<int32_t>& tokens) {
    reset();
    DeviceBuffer<int32_t> input(tokens.size());
    LFM_CUDA(cudaMemcpyAsync(input.data(), tokens.data(),
                             tokens.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    launch_mark_seen_batch(input.data(), static_cast<int>(tokens.size()),
                           seen_tokens_.data(), LfmConfig::vocab,
                           stream_.get());
    for (size_t i = 0; i < tokens.size(); ++i) {
        forward_token_host(tokens[i], i + 1 == tokens.size());
    }
    LFM_CUDA(cudaMemcpyAsync(position_device_.data(), &position_,
                             sizeof(position_), cudaMemcpyHostToDevice,
                             stream_.get()));
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    phase_ = SessionPhase::Ready;
}

void LfmModel::Impl::prefill(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) {
        throw std::invalid_argument("prefill needs at least one token");
    }
    if (tokens.size() > static_cast<size_t>(max_context_)) {
        throw std::invalid_argument("prefill exceeds max_context");
    }
    const auto begin = std::chrono::steady_clock::now();
    if (options_.legacy_prefill) {
        prefill_legacy(tokens);
    } else {
        prefill_batched(tokens);
    }
    const auto end = std::chrono::steady_clock::now();
    metrics_.last_prefill_ms =
        std::chrono::duration<double, std::milli>(end - begin).count();
    metrics_.prefill_tokens = tokens.size();
    metrics_.cumulative_decode_ms = 0.0;
    metrics_.decoded_tokens = 0;
}

void LfmModel::Impl::prefill_chunk(const std::vector<int32_t>& tokens,
                              bool begin, bool finalize) {
    if (tokens.empty()) {
        throw std::invalid_argument("prefill_chunk needs at least one token");
    }
    if (begin) {
        reset();
        metrics_ = {};
    } else if (position_ == 0) {
        throw std::runtime_error(
            "prefill_chunk continuation requires an initial chunk");
    }
    if (phase_ == SessionPhase::Ready) {
        throw std::runtime_error("cannot append prefill after finalization");
    }
    if (position_ + static_cast<int>(tokens.size()) > max_context_) {
        throw std::invalid_argument("prefill chunks exceed max_context");
    }
    phase_ = SessionPhase::Prefilling;

    DeviceBuffer<int32_t> input(tokens.size());
    LFM_CUDA(cudaMemcpyAsync(input.data(), tokens.data(),
                             tokens.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream_.get()));
    launch_mark_seen_batch(input.data(), static_cast<int>(tokens.size()),
                           seen_tokens_.data(), LfmConfig::vocab,
                           stream_.get());
    const auto started = std::chrono::steady_clock::now();
    for (size_t i = 0; i < tokens.size(); ++i) {
        forward_token_host(tokens[i], finalize && i + 1 == tokens.size());
    }
    LFM_CUDA(cudaMemcpyAsync(position_device_.data(), &position_,
                             sizeof(position_), cudaMemcpyHostToDevice,
                             stream_.get()));
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    const auto ended = std::chrono::steady_clock::now();
    metrics_.last_prefill_ms +=
        std::chrono::duration<double, std::milli>(ended - started).count();
    metrics_.prefill_tokens += tokens.size();
    if (finalize) {
        phase_ = SessionPhase::Ready;
        active_segmented_attention_ = use_segmented_attention(position_);
    }
}

void LfmModel::Impl::set_generation_config(GenerationConfig generation) {
    generation.validate();
    if (phase_ == SessionPhase::DecodePending) {
        throw std::runtime_error(
            "cannot change generation configuration during decode");
    }
    generation_ = generation;
    decode_graph_.reset();
    segmented_decode_graph_.reset();
}

void LfmModel::Impl::enqueue_sampling() {
    const int effective_top_k = generation_.greedy() ? 1 : generation_.top_k;
    const float effective_temperature =
        generation_.temperature > 0.0f ? generation_.temperature : 1.0f;
    if (plan_.sampling_kernel() == SamplingKernelKind::Fused) {
        launch_fused_sample_topk(
            logits_.data(), seen_tokens_.data(), sampling_scores_.data(),
            topk_values_.data(), topk_indices_.data(), LfmConfig::vocab,
            effective_temperature, generation_.repetition_penalty,
            effective_top_k,
            generation_.greedy() ? 1.0f : generation_.top_p,
            rng_state_.data(), sampled_device_.data(), stream_.get());
    } else {
        launch_prepare_sampling_scores(
            logits_.data(), seen_tokens_.data(), sampling_scores_.data(),
            LfmConfig::vocab, effective_temperature,
            generation_.repetition_penalty, stream_.get());
        for (int rank = 0; rank < effective_top_k; ++rank) {
            launch_select_topk(sampling_scores_.data(), topk_values_.data(),
                               topk_indices_.data(), rank, LfmConfig::vocab,
                               stream_.get());
        }
        launch_sample_topk(topk_values_.data(), topk_indices_.data(),
                           effective_top_k,
                           generation_.greedy() ? 1.0f : generation_.top_p,
                           rng_state_.data(), sampled_device_.data(),
                           stream_.get());
        launch_mark_seen(sampled_device_.data(), seen_tokens_.data(),
                         LfmConfig::vocab, stream_.get());
    }
}

void LfmModel::Impl::enqueue_decode_forward() {
    if (embedding_->int4_quantized()) {
        launch_embedding_int4_device(
            sampled_device_.data(), embedding_->int4, embedding_->scales,
            hidden_.data(), LfmConfig::hidden, stream_.get());
    } else if (embedding_->int8) {
        launch_embedding_int8_device(
            sampled_device_.data(), embedding_->int8, embedding_->scales,
            hidden_.data(), LfmConfig::hidden, stream_.get());
    } else {
        launch_embedding_device(sampled_device_.data(), embedding_->bf16,
                                hidden_.data(), LfmConfig::hidden,
                                stream_.get());
    }

    for (Layer& layer : layers_) {
        LayerCommon& common_layer = common(layer);
        if (!options_.fused_residuals) {
            LFM_CUDA(cudaMemcpyAsync(
                residual_.data(), hidden_.data(), hidden_.bytes(),
                cudaMemcpyDeviceToDevice, stream_.get()));
        }
        launch_rmsnorm(hidden_.data(), common_layer.operator_norm, normed_.data(),
                       1, LfmConfig::hidden, LfmConfig::norm_eps,
                       stream_.get());
        if (AttentionLayer* attention = as_attention(layer)) {
            __nv_bfloat16* q = qkv_output_.data();
            __nv_bfloat16* k = q + LfmConfig::q_width;
            __nv_bfloat16* v = k + LfmConfig::kv_width;
            if (options_.fused_projections) {
                linear(normed_.data(), *attention->qkv, qkv_output_.data(),
                       1, LfmConfig::qkv_width, LfmConfig::hidden);
            } else {
                const LinearWeight q_weight =
                    slice_rows(*attention->qkv, 0, LfmConfig::q_width);
                const LinearWeight k_weight = slice_rows(
                    *attention->qkv, LfmConfig::q_width, LfmConfig::kv_width);
                const LinearWeight v_weight = slice_rows(
                    *attention->qkv, LfmConfig::q_width + LfmConfig::kv_width,
                    LfmConfig::kv_width);
                linear(normed_.data(), q_weight, q,
                       1, LfmConfig::q_width, LfmConfig::hidden);
                linear(normed_.data(), k_weight, k,
                       1, LfmConfig::kv_width, LfmConfig::hidden);
                linear(normed_.data(), v_weight, v,
                       1, LfmConfig::kv_width, LfmConfig::hidden);
            }
            if (options_.fast_attention) {
                launch_qk_norm_rope_fast_device(
                    q, k, attention->q_norm, attention->k_norm,
                    rope_cos_.data(), rope_sin_.data(),
                    LfmConfig::q_heads, LfmConfig::kv_heads,
                    LfmConfig::head_dim, position_device_.data(),
                    LfmConfig::norm_eps, stream_.get());
            } else {
                launch_qk_norm_rope_strict_device(
                    q, k, attention->q_norm, attention->k_norm,
                    rope_cos_.data(), rope_sin_.data(),
                    LfmConfig::q_heads, LfmConfig::kv_heads,
                    LfmConfig::head_dim, position_device_.data(),
                    LfmConfig::norm_eps, stream_.get());
            }
            if (options_.kv_cache_mode == KvCacheMode::Int8) {
                launch_store_kv_int8_device(
                    k, v, attention->key_cache_int8.data(),
                    attention->value_cache_int8.data(),
                    attention->key_cache_scales.data(),
                    attention->value_cache_scales.data(), position_device_.data(),
                    LfmConfig::kv_heads, LfmConfig::head_dim, stream_.get());
                if (active_segmented_attention_) {
                    launch_gqa_decode_segmented_int8_device(
                        q, attention->key_cache_int8.data(),
                        attention->value_cache_int8.data(),
                        attention->key_cache_scales.data(),
                        attention->value_cache_scales.data(), op_output_.data(),
                        position_device_.data(), LfmConfig::q_heads,
                        LfmConfig::kv_heads, LfmConfig::head_dim,
                        options_.attention_chunk_tokens, attention_chunks_,
                        attention_partial_max_.data(),
                        attention_partial_denom_.data(),
                        attention_partial_accum_.data(), stream_.get());
                } else if (options_.fast_attention) {
                    launch_gqa_decode_online_int8_device(
                        q, attention->key_cache_int8.data(),
                        attention->value_cache_int8.data(),
                        attention->key_cache_scales.data(),
                        attention->value_cache_scales.data(), op_output_.data(),
                        position_device_.data(), LfmConfig::q_heads,
                        LfmConfig::kv_heads, LfmConfig::head_dim, stream_.get());
                } else {
                    launch_gqa_decode_strict_int8_device(
                        q, attention->key_cache_int8.data(),
                        attention->value_cache_int8.data(),
                        attention->key_cache_scales.data(),
                        attention->value_cache_scales.data(), op_output_.data(),
                        position_device_.data(), LfmConfig::q_heads,
                        LfmConfig::kv_heads, LfmConfig::head_dim, stream_.get());
                }
            } else {
                launch_store_kv_device(
                    k, v, attention->key_cache.data(), attention->value_cache.data(),
                    position_device_.data(), LfmConfig::kv_width, stream_.get());
                if (active_segmented_attention_) {
                    launch_gqa_decode_segmented_device(
                        q, attention->key_cache.data(), attention->value_cache.data(),
                        op_output_.data(), position_device_.data(),
                        LfmConfig::q_heads, LfmConfig::kv_heads,
                        LfmConfig::head_dim, options_.attention_chunk_tokens,
                        attention_chunks_, attention_partial_max_.data(),
                        attention_partial_denom_.data(),
                        attention_partial_accum_.data(), stream_.get());
                } else if (options_.fast_attention) {
                    launch_gqa_decode_online_device(
                        q, attention->key_cache.data(), attention->value_cache.data(),
                        op_output_.data(), position_device_.data(),
                        LfmConfig::q_heads, LfmConfig::kv_heads,
                        LfmConfig::head_dim, stream_.get());
                } else {
                    launch_gqa_decode_strict_device(
                        q, attention->key_cache.data(), attention->value_cache.data(),
                        op_output_.data(), position_device_.data(),
                        LfmConfig::q_heads, LfmConfig::kv_heads,
                        LfmConfig::head_dim, stream_.get());
                }
            }
            linear(op_output_.data(), *attention->out, hidden_.data(),
                   1, LfmConfig::hidden, LfmConfig::hidden,
                   options_.fused_residuals ? 1.0f : 0.0f);
        } else {
            ConvolutionLayer& convolution = *as_convolution(layer);
            linear(normed_.data(), *convolution.conv_in, conv_projected_.data(),
                   1, 3 * LfmConfig::hidden, LfmConfig::hidden);
            launch_conv_decode_device(
                conv_projected_.data(), convolution.conv_weight,
                convolution.conv_state.data(), op_output_.data(),
                LfmConfig::hidden, LfmConfig::conv_cache,
                position_device_.data(), stream_.get());
            linear(op_output_.data(), *convolution.conv_out, hidden_.data(),
                   1, LfmConfig::hidden, LfmConfig::hidden,
                   options_.fused_residuals ? 1.0f : 0.0f);
        }
        if (!options_.fused_residuals) {
            launch_residual_add(hidden_.data(), residual_.data(),
                                LfmConfig::hidden, stream_.get());
        }
        run_mlp_decode(common_layer);
    }
    launch_rmsnorm(hidden_.data(), final_norm_, normed_.data(),
                   1, LfmConfig::hidden, LfmConfig::norm_eps,
                   stream_.get());
    linear(normed_.data(), *embedding_, logits_.data(),
           1, LfmConfig::vocab, LfmConfig::hidden);
}

void LfmModel::Impl::enqueue_decode_step() {
    enqueue_sampling();
    enqueue_decode_forward();
    launch_increment_position(position_device_.data(), stream_.get());
}

bool LfmModel::Impl::use_segmented_attention(int host_position) const {
    return plan_.segmented_attention(host_position);
}

CudaGraphExec& LfmModel::Impl::graph_for_attention(bool segmented) {
    return segmented ? segmented_decode_graph_ : decode_graph_;
}

void LfmModel::Impl::capture_decode_graph(bool segmented) {
    if (!options_.cuda_graph) return;
    CudaGraphExec& graph = graph_for_attention(segmented);
    if (graph.ready()) return;
    active_segmented_attention_ = segmented;
    graph.capture_begin(stream_.get());
    try {
        enqueue_decode_step();
        graph.capture_end(stream_.get());
    } catch (...) {
        graph.abort_capture(stream_.get());
        throw;
    }
}

int32_t LfmModel::Impl::decode() {
    decode_async_begin();
    return decode_async_finish();
}

void LfmModel::Impl::decode_async_begin() {
    if (!local_kv_cache_available_) {
        throw std::runtime_error(
            "lane decode is unavailable after transferring KV to the shared paged cache");
    }
    if (phase_ != SessionPhase::Ready) {
        throw std::runtime_error("decode requires a successful prefill");
    }
    if (phase_ == SessionPhase::DecodePending) {
        throw std::runtime_error("decode_async_begin called twice");
    }
    if (position_ >= max_context_) {
        throw std::runtime_error("context limit reached");
    }
    decode_async_begin_time_ = std::chrono::steady_clock::now();
    const bool segmented = use_segmented_attention(position_);
    active_segmented_attention_ = segmented;
    if (options_.cuda_graph) {
        CudaGraphExec& graph = graph_for_attention(segmented);
        if (!graph.ready()) capture_decode_graph(segmented);
        graph.launch(stream_.get());
    } else {
        enqueue_decode_step();
    }
    LFM_CUDA(cudaMemcpyAsync(sampled_host_.data(), sampled_device_.data(),
                             sizeof(int32_t), cudaMemcpyDeviceToHost,
                             stream_.get()));
    phase_ = SessionPhase::DecodePending;
}

int32_t LfmModel::Impl::decode_async_finish() {
    if (phase_ != SessionPhase::DecodePending) {
        throw std::runtime_error("decode_async_finish without begin");
    }
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    phase_ = SessionPhase::Ready;
    ++position_;
    const auto ended = std::chrono::steady_clock::now();
    metrics_.cumulative_decode_ms +=
        std::chrono::duration<double, std::milli>(
            ended - decode_async_begin_time_).count();
    ++metrics_.decoded_tokens;
    return sampled_host_.data()[0];
}

DecodeBenchmark LfmModel::Impl::benchmark_decode(int warmup_steps,
                                                int measured_steps) {
    if (phase_ != SessionPhase::Ready) {
        throw std::runtime_error("benchmark_decode requires a successful prefill");
    }
    if (warmup_steps < 0 || measured_steps <= 0) {
        throw std::invalid_argument(
            "benchmark steps require warmup >= 0 and measured > 0");
    }
    const int total_steps = warmup_steps + measured_steps;
    if (position_ + total_steps > max_context_) {
        throw std::runtime_error("decode benchmark exceeds context limit");
    }
    if (options_.cuda_graph) {
        const int final_position = position_ + total_steps - 1;
        const bool starts_segmented = use_segmented_attention(position_);
        const bool ends_segmented = use_segmented_attention(final_position);
        if (!starts_segmented || !ends_segmented) capture_decode_graph(false);
        if (starts_segmented || ends_segmented) capture_decode_graph(true);
    }

    int simulated_position = position_;
    auto launch_step = [&]() {
        const bool segmented = use_segmented_attention(simulated_position);
        active_segmented_attention_ = segmented;
        if (options_.cuda_graph) {
            CudaGraphExec& graph = graph_for_attention(segmented);
            if (!graph.ready()) capture_decode_graph(segmented);
            graph.launch(stream_.get());
        } else {
            enqueue_decode_step();
        }
        ++simulated_position;
    };
    for (int i = 0; i < warmup_steps; ++i) launch_step();
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));

    CudaEvent begin;
    CudaEvent end;
    begin.record(stream_.get());
    for (int i = 0; i < measured_steps; ++i) launch_step();
    end.record(stream_.get());
    end.synchronize();

    position_ += total_steps;
    DecodeBenchmark result;
    result.warmup_steps = warmup_steps;
    result.measured_steps = measured_steps;
    result.elapsed_ms = CudaEvent::elapsed_ms(begin, end);
    return result;
}

ModelMemoryStats LfmModel::Impl::memory_stats() const {
    ModelMemoryStats stats;
    stats.weights = weights_ ? weights_->memory_bytes() : 0;
    for (const Layer& layer : layers_) {
        if (const AttentionLayer* attention = as_attention(layer)) {
            stats.kv_cache += attention->key_cache.bytes() + attention->value_cache.bytes() +
                attention->key_cache_int8.bytes() + attention->value_cache_int8.bytes() +
                attention->key_cache_scales.bytes() + attention->value_cache_scales.bytes();
        } else {
            stats.conv_state += as_convolution(layer)->conv_state.bytes();
        }
    }
    stats.rope_tables = rope_cos_.bytes() + rope_sin_.bytes();
    stats.activations =
        hidden_.bytes() + residual_.bytes() + normed_.bytes() +
        op_output_.bytes() + qkv_output_.bytes() + conv_projected_.bytes() +
        gate_up_.bytes() + activated_.bytes() + mlp_output_.bytes() +
        logits_.bytes() + paged_page_table_.bytes() +
        paged_prefill_tokens_.bytes() + prefill_tokens_.bytes() +
        prefill_hidden_.bytes() +
        prefill_residual_.bytes() + prefill_normed_.bytes() +
        prefill_op_output_.bytes() + prefill_q_.bytes() + prefill_k_.bytes() +
        prefill_v_.bytes() + prefill_conv_projected_.bytes() +
        prefill_gate_up_.bytes() + prefill_activated_.bytes() +
        prefill_mlp_output_.bytes();
    stats.sampling =
        position_device_.bytes() + sampled_device_.bytes() +
        seen_tokens_.bytes() + sampling_scores_.bytes() +
        topk_values_.bytes() + topk_indices_.bytes() + rng_state_.bytes();
    stats.matmul_workspace = lt_workspace_.bytes();
    stats.attention_workspace = attention_partial_max_.bytes() +
        attention_partial_denom_.bytes() + attention_partial_accum_.bytes();
    return stats;
}

void LfmModel::Impl::save_session(const std::string& path) {
    if (!local_kv_cache_available_) {
        throw std::runtime_error(
            "save_session requires a model with a local contiguous KV cache");
    }
    if (phase_ != SessionPhase::Ready) {
        throw std::runtime_error("cannot save a session before prefill or load_session");
    }
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    SessionHeader header;
    header.kv_cache_mode =
        options_.kv_cache_mode == KvCacheMode::Int8 ? 1U : 0U;
    header.position = position_;
    header.max_context = max_context_;
    LFM_CUDA(cudaMemcpy(&header.rng_state, rng_state_.data(),
                        sizeof(header.rng_state), cudaMemcpyDeviceToHost));

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create session file: " + path);
    write_scalar(out, header);

    auto write_device = [&](const void* device, size_t bytes) {
        std::vector<std::byte> host(bytes);
        if (bytes) {
            LFM_CUDA(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost));
            out.write(reinterpret_cast<const char*>(host.data()),
                      static_cast<std::streamsize>(bytes));
            if (!out) throw std::runtime_error("failed writing session payload");
        }
    };

    write_device(seen_tokens_.data(), seen_tokens_.bytes());
    write_device(logits_.data(), logits_.bytes());
    const size_t cache_elements =
        static_cast<size_t>(position_) * LfmConfig::kv_width;
    const size_t scale_elements =
        static_cast<size_t>(position_) * LfmConfig::kv_heads;
    for (const Layer& layer : layers_) {
        if (const AttentionLayer* attention = as_attention(layer)) {
            if (options_.kv_cache_mode == KvCacheMode::Int8) {
                write_device(attention->key_cache_int8.data(),
                             cache_elements * sizeof(int8_t));
                write_device(attention->value_cache_int8.data(),
                             cache_elements * sizeof(int8_t));
                write_device(attention->key_cache_scales.data(),
                             scale_elements * sizeof(float));
                write_device(attention->value_cache_scales.data(),
                             scale_elements * sizeof(float));
            } else {
                write_device(attention->key_cache.data(),
                             cache_elements * sizeof(__nv_bfloat16));
                write_device(attention->value_cache.data(),
                             cache_elements * sizeof(__nv_bfloat16));
            }
        } else {
            const ConvolutionLayer& convolution = *as_convolution(layer);
            write_device(convolution.conv_state.data(), convolution.conv_state.bytes());
        }
    }
}

void LfmModel::Impl::load_session(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open session file: " + path);
    SessionHeader header;
    read_scalar(in, header);
    const std::array<char, 8> expected{{'L', 'F', 'M', 'S', 'E', 'S', 'S', '1'}};
    if (header.magic != expected || header.version != 1) {
        throw std::runtime_error("unsupported session format");
    }
    const uint32_t expected_kv =
        options_.kv_cache_mode == KvCacheMode::Int8 ? 1U : 0U;
    if (header.kv_cache_mode != expected_kv) {
        throw std::runtime_error("session KV cache mode differs from model options");
    }
    if (header.position <= 0 || header.position > max_context_ ||
        header.layers != LfmConfig::layers ||
        header.kv_width != LfmConfig::kv_width ||
        header.kv_heads != LfmConfig::kv_heads ||
        header.vocab != LfmConfig::vocab) {
        throw std::runtime_error("session dimensions are incompatible");
    }

    reset();
    auto read_device = [&](void* device, size_t bytes) {
        std::vector<std::byte> host(bytes);
        if (bytes) {
            in.read(reinterpret_cast<char*>(host.data()),
                    static_cast<std::streamsize>(bytes));
            if (!in) throw std::runtime_error("truncated session payload");
            LFM_CUDA(cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice));
        }
    };

    read_device(seen_tokens_.data(), seen_tokens_.bytes());
    read_device(logits_.data(), logits_.bytes());
    const size_t cache_elements =
        static_cast<size_t>(header.position) * LfmConfig::kv_width;
    const size_t scale_elements =
        static_cast<size_t>(header.position) * LfmConfig::kv_heads;
    for (Layer& layer : layers_) {
        if (AttentionLayer* attention = as_attention(layer)) {
            if (options_.kv_cache_mode == KvCacheMode::Int8) {
                read_device(attention->key_cache_int8.data(),
                            cache_elements * sizeof(int8_t));
                read_device(attention->value_cache_int8.data(),
                            cache_elements * sizeof(int8_t));
                read_device(attention->key_cache_scales.data(),
                            scale_elements * sizeof(float));
                read_device(attention->value_cache_scales.data(),
                            scale_elements * sizeof(float));
            } else {
                read_device(attention->key_cache.data(),
                            cache_elements * sizeof(__nv_bfloat16));
                read_device(attention->value_cache.data(),
                            cache_elements * sizeof(__nv_bfloat16));
            }
        } else {
            ConvolutionLayer& convolution = *as_convolution(layer);
            read_device(convolution.conv_state.data(), convolution.conv_state.bytes());
        }
    }
    char trailing = 0;
    if (in.read(&trailing, 1)) {
        throw std::runtime_error("session file has trailing data");
    }
    if (!in.eof()) throw std::runtime_error("failed reading session file");

    position_ = header.position;
    LFM_CUDA(cudaMemcpy(position_device_.data(), &position_, sizeof(position_),
                        cudaMemcpyHostToDevice));
    LFM_CUDA(cudaMemcpy(rng_state_.data(), &header.rng_state,
                        sizeof(header.rng_state), cudaMemcpyHostToDevice));
    phase_ = SessionPhase::Ready;
    active_segmented_attention_ = use_segmented_attention(position_);
    metrics_ = {};
}


PrefixState LfmModel::Impl::export_prefix_state() const {
    if (phase_ != SessionPhase::Ready) {
        throw std::runtime_error("cannot export prefix state before prefill");
    }
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    PrefixState state;
    state.position = position_;
    state.seen_tokens.resize(seen_tokens_.size());
    state.logits_bf16.resize(logits_.size());
    size_t conv_elements = 0;
    for (const Layer& layer : layers_) {
        if (const ConvolutionLayer* convolution = as_convolution(layer)) {
            conv_elements += convolution->conv_state.size();
        }
    }
    state.conv_state_bf16.resize(conv_elements);
    LFM_CUDA(cudaMemcpy(state.seen_tokens.data(), seen_tokens_.data(),
                        seen_tokens_.bytes(), cudaMemcpyDeviceToHost));
    LFM_CUDA(cudaMemcpy(state.logits_bf16.data(), logits_.data(),
                        logits_.bytes(), cudaMemcpyDeviceToHost));
    size_t offset = 0;
    for (const Layer& layer : layers_) {
        const ConvolutionLayer* convolution = as_convolution(layer);
        if (!convolution) continue;
        const size_t count = convolution->conv_state.size();
        LFM_CUDA(cudaMemcpy(state.conv_state_bf16.data() + offset,
                            convolution->conv_state.data(), convolution->conv_state.bytes(),
                            cudaMemcpyDeviceToHost));
        offset += count;
    }
    return state;
}

void LfmModel::Impl::restore_prefix_state(const PrefixState& state) {
    if (state.position <= 0 || state.position > max_context_) {
        throw std::invalid_argument("prefix state position is invalid");
    }
    if (state.seen_tokens.size() != seen_tokens_.size() ||
        state.logits_bf16.size() != logits_.size()) {
        throw std::invalid_argument("prefix state sampling dimensions differ");
    }
    size_t expected_conv = 0;
    for (const Layer& layer : layers_) {
        if (const ConvolutionLayer* convolution = as_convolution(layer)) {
            expected_conv += convolution->conv_state.size();
        }
    }
    if (state.conv_state_bf16.size() != expected_conv) {
        throw std::invalid_argument("prefix state convolution dimensions differ");
    }
    phase_ = SessionPhase::Prefilling;
    position_ = state.position;
    LFM_CUDA(cudaMemcpyAsync(seen_tokens_.data(), state.seen_tokens.data(),
                             seen_tokens_.bytes(), cudaMemcpyHostToDevice,
                             stream_.get()));
    LFM_CUDA(cudaMemcpyAsync(logits_.data(), state.logits_bf16.data(),
                             logits_.bytes(), cudaMemcpyHostToDevice,
                             stream_.get()));
    // Prefix computation is deterministic, but generation randomness belongs to
    // the receiving request. Never inherit the seed/RNG stream of the request
    // that populated the shared prefix cache.
    const uint64_t request_seed = generation_.seed;
    LFM_CUDA(cudaMemcpyAsync(rng_state_.data(), &request_seed,
                             sizeof(request_seed), cudaMemcpyHostToDevice,
                             stream_.get()));
    LFM_CUDA(cudaMemcpyAsync(position_device_.data(), &position_,
                             sizeof(position_), cudaMemcpyHostToDevice,
                             stream_.get()));
    size_t offset = 0;
    for (Layer& layer : layers_) {
        ConvolutionLayer* convolution = as_convolution(layer);
        if (!convolution) continue;
        const size_t count = convolution->conv_state.size();
        LFM_CUDA(cudaMemcpyAsync(convolution->conv_state.data(),
                                 state.conv_state_bf16.data() + offset,
                                 convolution->conv_state.bytes(), cudaMemcpyHostToDevice,
                                 stream_.get()));
        offset += count;
    }
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    phase_ = SessionPhase::Ready;
    active_segmented_attention_ = use_segmented_attention(position_);
    metrics_ = {};
}

void LfmModel::Impl::release_local_kv_cache() {
    if (!local_kv_cache_available_) return;
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));
    for (Layer& layer : layers_) {
        AttentionLayer* attention = as_attention(layer);
        if (!attention) continue;
        attention->key_cache.reset(0);
        attention->value_cache.reset(0);
        attention->key_cache_int8.reset(0);
        attention->value_cache_int8.reset(0);
        attention->key_cache_scales.reset(0);
        attention->value_cache_scales.reset(0);
    }
    local_kv_cache_available_ = false;
    decode_graph_.reset();
    segmented_decode_graph_.reset();
}

std::vector<float> LfmModel::Impl::copy_logits() {
    if (phase_ != SessionPhase::Ready) {
        throw std::runtime_error("logits are unavailable before prefill");
    }
    std::vector<__nv_bfloat16> bf16_logits(LfmConfig::vocab);
    LFM_CUDA(cudaMemcpyAsync(
        bf16_logits.data(), logits_.data(), logits_.bytes(),
        cudaMemcpyDeviceToHost, stream_.get()));
    LFM_CUDA(cudaStreamSynchronize(stream_.get()));

    std::vector<float> result(LfmConfig::vocab);
    for (int i = 0; i < LfmConfig::vocab; ++i) {
        result[static_cast<size_t>(i)] =
            __bfloat162float(bf16_logits[static_cast<size_t>(i)]);
    }
    return result;
}


LfmModel::LfmModel(const std::string& safetensors_path,
                   int max_context,
                   ModelOptions options,
                   GenerationConfig generation)
    : impl_(std::make_unique<Impl>(safetensors_path, max_context,
                                   std::move(options), std::move(generation))),
      session_view_(*this),
      diagnostics_view_(*this),
      persistence_view_(*this) {}

LfmModel::~LfmModel() = default;

void LfmModel::reset(bool allocate_local_kv) { impl_->reset(allocate_local_kv); }
void LfmModel::prefill(const std::vector<int32_t>& tokens) { impl_->prefill(tokens); }
void LfmModel::prefill_chunk(const std::vector<int32_t>& tokens,
                             bool begin, bool finalize) {
    impl_->prefill_chunk(tokens, begin, finalize);
}
void LfmModel::prefill_chunk_paged(const std::vector<int32_t>& tokens,
                                   bool begin, bool finalize,
                                   PhysicalPagedKvCache& paged_kv,
                                   const std::vector<uint32_t>& page_table) {
    impl_->prefill_chunk_paged(tokens, begin, finalize, paged_kv, page_table);
}
int32_t LfmModel::decode() { return impl_->decode(); }
void LfmModel::decode_async_begin() { impl_->decode_async_begin(); }
int32_t LfmModel::decode_async_finish() { return impl_->decode_async_finish(); }
void LfmModel::set_generation_config(GenerationConfig generation) {
    impl_->set_generation_config(std::move(generation));
}
std::vector<float> LfmModel::copy_logits() { return impl_->copy_logits(); }
DecodeBenchmark LfmModel::benchmark_decode(int warmup_steps, int measured_steps) {
    return impl_->benchmark_decode(warmup_steps, measured_steps);
}
ModelMemoryStats LfmModel::memory_stats() const { return impl_->memory_stats(); }
RuntimeMetrics LfmModel::runtime_metrics() const { return impl_->runtime_metrics(); }
void LfmModel::clear_runtime_metrics() { impl_->clear_runtime_metrics(); }
void LfmModel::save_session(const std::string& path) { impl_->save_session(path); }
void LfmModel::load_session(const std::string& path) { impl_->load_session(path); }
PrefixState LfmModel::export_prefix_state() const { return impl_->export_prefix_state(); }
void LfmModel::restore_prefix_state(const PrefixState& state) {
    impl_->restore_prefix_state(state);
}
void LfmModel::release_local_kv_cache() { impl_->release_local_kv_cache(); }
bool LfmModel::local_kv_cache_available() const {
    return impl_->local_kv_cache_available();
}
SessionPhase LfmModel::phase() const { return impl_->phase(); }
bool LfmModel::ready_for_decode() const { return impl_->ready_for_decode(); }
bool LfmModel::decode_pending() const { return impl_->decode_pending(); }
int LfmModel::position() const { return impl_->position(); }
bool LfmModel::cuda_graph_ready() const { return impl_->cuda_graph_ready(); }

void LfmInferenceSession::reset(bool allocate_local_kv) {
    owner_->impl_->reset(allocate_local_kv);
}
void LfmInferenceSession::prefill(const std::vector<int32_t>& tokens) {
    owner_->impl_->prefill(tokens);
}
void LfmInferenceSession::prefill_chunk(const std::vector<int32_t>& tokens,
                                        bool begin, bool finalize) {
    owner_->impl_->prefill_chunk(tokens, begin, finalize);
}
void LfmInferenceSession::prefill_chunk_paged(
    const std::vector<int32_t>& tokens, bool begin, bool finalize,
    PhysicalPagedKvCache& paged_kv,
    const std::vector<uint32_t>& page_table) {
    owner_->impl_->prefill_chunk_paged(tokens, begin, finalize,
                                      paged_kv, page_table);
}
int32_t LfmInferenceSession::decode() { return owner_->impl_->decode(); }
void LfmInferenceSession::decode_async_begin() {
    owner_->impl_->decode_async_begin();
}
int32_t LfmInferenceSession::decode_async_finish() {
    return owner_->impl_->decode_async_finish();
}
void LfmInferenceSession::set_generation_config(GenerationConfig generation) {
    owner_->impl_->set_generation_config(std::move(generation));
}
void LfmInferenceSession::release_local_kv_cache() {
    owner_->impl_->release_local_kv_cache();
}
bool LfmInferenceSession::local_kv_cache_available() const {
    return owner_->impl_->local_kv_cache_available();
}
SessionPhase LfmInferenceSession::phase() const { return owner_->impl_->phase(); }
bool LfmInferenceSession::ready_for_decode() const {
    return owner_->impl_->ready_for_decode();
}
bool LfmInferenceSession::decode_pending() const {
    return owner_->impl_->decode_pending();
}
int LfmInferenceSession::position() const { return owner_->impl_->position(); }

std::vector<float> LfmDiagnostics::copy_logits() const {
    return owner_->impl_->copy_logits();
}
DecodeBenchmark LfmDiagnostics::benchmark_decode(int warmup_steps,
                                                  int measured_steps) {
    return owner_->impl_->benchmark_decode(warmup_steps, measured_steps);
}
ModelMemoryStats LfmDiagnostics::memory_stats() const {
    return owner_->impl_->memory_stats();
}
RuntimeMetrics LfmDiagnostics::runtime_metrics() const {
    return owner_->impl_->runtime_metrics();
}
void LfmDiagnostics::clear_runtime_metrics() {
    owner_->impl_->clear_runtime_metrics();
}
bool LfmDiagnostics::cuda_graph_ready() const {
    return owner_->impl_->cuda_graph_ready();
}

void LfmPersistence::save_session(const std::string& path) const {
    owner_->impl_->save_session(path);
}
void LfmPersistence::load_session(const std::string& path) {
    owner_->impl_->load_session(path);
}
PrefixState LfmPersistence::export_prefix_state() const {
    return owner_->impl_->export_prefix_state();
}
void LfmPersistence::restore_prefix_state(const PrefixState& state) {
    owner_->impl_->restore_prefix_state(state);
}

} // namespace lfm
