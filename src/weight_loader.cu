#include "lfm/weight_loader.hpp"
#include "lfm/quantization.hpp"
#include "lfm/expert_residency.hpp"

#include <cstring>
#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace lfm {

namespace {

size_t checked_element_count(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        throw std::runtime_error("weight shape is empty");
    }
    size_t count = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("weight shape has non-positive dimension");
        }
        count *= static_cast<size_t>(dim);
    }
    return count;
}

std::string layer_name(int index, const std::string& suffix) {
    return "model.layers." + std::to_string(index) + "." + suffix;
}

// Host-side dequantization of a GGUF Q4_K / Q6_K tensor to row-major BF16. Used
// as a fallback when a concatenation mixes quant formats (e.g. attention q/k/v
// where value projection is Q6_K while query/key are Q4_K) and cannot stay
// packed. The block decode mirrors the device kernel in gguf_kernels.cu.
namespace {
struct Q4KHost { __half d; __half dmin; uint8_t scales[12]; uint8_t qs[128]; };
struct Q6KHost { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; __half d; };

void q4k_decode(const Q4KHost* blk, int col, float& out) {
    const float d = __half2float(blk->d);
    const float dmin = __half2float(blk->dmin);
    const int sub = col >> 5, within = col & 31;
    uint8_t sc, m;
    if (sub < 4) { sc = blk->scales[sub] & 63; m = blk->scales[sub + 4] & 63; }
    else {
        sc = (blk->scales[sub + 4] & 0xF) | ((blk->scales[sub - 4] >> 6) << 4);
        m = (blk->scales[sub + 4] >> 4) | ((blk->scales[sub] >> 6) << 4);
    }
    const uint8_t* qs = blk->qs + (sub >> 1) * 32;
    const int q = (sub & 1) ? (qs[within] >> 4) : (qs[within] & 0xF);
    out = d * sc * static_cast<float>(q) - dmin * m;
}
void q6k_decode(const Q6KHost* blk, int col, float& out) {
    const float d = __half2float(blk->d);
    const int half = col >> 7, idx = col & 127;
    const int n = idx & 31, grp = idx >> 5;
    const uint8_t* ql = blk->ql + half * 64;
    const uint8_t* qh = blk->qh + half * 32;
    int q;
    if (grp == 0) q = (ql[n] & 0xF) | (((qh[n]) & 3) << 4);
    else if (grp == 1) q = (ql[n + 32] & 0xF) | (((qh[n] >> 2) & 3) << 4);
    else if (grp == 2) q = (ql[n] >> 4) | (((qh[n] >> 4) & 3) << 4);
    else q = (ql[n + 32] >> 4) | (((qh[n] >> 6) & 3) << 4);
    const int is = half * 8 + grp * 2 + (n >> 4);
    out = d * static_cast<float>(blk->scales[is]) * static_cast<float>(q - 32);
}
} // namespace

void dequantize_gguf_to_bf16(const HostTensorView& tensor,
                             std::vector<__nv_bfloat16>& out) {
    const int rows = static_cast<int>(tensor.shape[0]);
    const int cols = static_cast<int>(tensor.shape[1]);
    out.resize(static_cast<size_t>(rows) * cols);
    const GgmlTypeTrait trait = ggml_type_trait(tensor.ggml_type);
    const int blocks_per_row = cols / trait.block_size;
    const size_t row_bytes = static_cast<size_t>(blocks_per_row) * trait.type_size;
    for (int r = 0; r < rows; ++r) {
        const uint8_t* row_blocks =
            reinterpret_cast<const uint8_t*>(tensor.data) + static_cast<size_t>(r) * row_bytes;
        for (int c = 0; c < cols; ++c) {
            const int b = c / trait.block_size;
            const int within = c % trait.block_size;
            float v = 0.0f;
            if (tensor.ggml_type == GgmlType::Q4_K) {
                q4k_decode(reinterpret_cast<const Q4KHost*>(row_blocks) + b, within, v);
            } else {
                q6k_decode(reinterpret_cast<const Q6KHost*>(row_blocks) + b, within, v);
            }
            out[static_cast<size_t>(r) * cols + c] = __float2bfloat16(v);
        }
    }
}

const __nv_bfloat16* upload_bf16(SharedModelWeights& weights,
                                 const IWeightRepository& repo,
                                 const std::string& name,
                                 const std::vector<int64_t>& expected,
                                 const std::string& cache_key) {
    if (const auto cached = weights.tensors.find(cache_key);
        cached != weights.tensors.end()) {
        if (!expected.empty() && cached->second.shape != expected) {
            throw std::runtime_error("cached weight shape mismatch for " + cache_key);
        }
        return cached->second.bf16_storage.data();
    }
    const HostTensorView tensor = repo.tensor(name);
    if (!expected.empty()) {
        // GGUF convolutional depthwise weights are stored as [hidden, cache]
        // (2D) while the model builder expects [hidden, 1, cache] (3D). The
        // extra unit dimension is implicit, so accept the 2D form.
        bool shape_ok = (tensor.shape == expected);
        if (!shape_ok && tensor.shape.size() == 2 && expected.size() == 3 &&
            tensor.shape[0] == expected[0] &&
            tensor.shape[1] == expected[1] * expected[2] &&
            expected[1] == 1) {
            shape_ok = true;
        }
        if (!shape_ok) {
            throw std::runtime_error("unexpected shape for " + name);
        }
    }
    const size_t count = checked_element_count(tensor.shape);

    DeviceWeight weight;
    weight.shape = tensor.shape;
    weight.bf16_storage.reset(count);

    if (tensor.dtype == TensorDType::BF16) {
        if (tensor.bytes != count * sizeof(__nv_bfloat16)) {
            throw std::runtime_error("invalid BF16 byte count for " + name);
        }
        LFM_CUDA(cudaMemcpy(weight.bf16_storage.data(), tensor.data, tensor.bytes,
                            cudaMemcpyHostToDevice));
    } else if (tensor.dtype == TensorDType::F32) {
        if (tensor.bytes != count * sizeof(float)) {
            throw std::runtime_error("invalid F32 byte count for " + name);
        }
        const float* src = reinterpret_cast<const float*>(tensor.data);
        std::vector<__nv_bfloat16> converted(count);
        for (size_t i = 0; i < count; ++i) {
            converted[i] = __float2bfloat16(src[i]);
        }
        LFM_CUDA(cudaMemcpy(weight.bf16_storage.data(), converted.data(),
                            count * sizeof(__nv_bfloat16),
                            cudaMemcpyHostToDevice));
    } else {
        throw std::runtime_error(
            "only BF16/F32 source weights are supported; incompatible tensor: " + name);
    }
    auto [it, inserted] = weights.tensors.emplace(cache_key, std::move(weight));
    if (!inserted) throw std::runtime_error("duplicate weight: " + cache_key);
    return it->second.bf16_storage.data();
}

} // namespace

std::shared_ptr<SharedModelWeights> WeightLoader::acquire(
    const std::string& safetensors_path,
    WeightMode weight_mode) {
    // Process-wide cache so multiple sessions on the same device + checkpoint
    // + weight_mode share one immutable device allocation.
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::weak_ptr<SharedModelWeights>> cache;

    int device_id = 0;
    LFM_CUDA(cudaGetDevice(&device_id));
    std::ostringstream key_builder;
    key_builder << device_id << ':' << static_cast<int>(weight_mode)
                << ':' << safetensors_path;
    const std::string key = key_builder.str();

    std::shared_ptr<SharedModelWeights> weights;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto found = cache.find(key);
        if (found != cache.end()) weights = found->second.lock();
        if (!weights) {
            weights = std::make_shared<SharedModelWeights>();
            cache[key] = weights;
        }
    }
    return weights;
}

WeightLoader::WeightLoader(std::shared_ptr<SharedModelWeights> weights,
                           WeightMode weight_mode)
    : weights_(std::move(weights)), weight_mode_(weight_mode) {
    if (!weights_) {
        throw std::invalid_argument("WeightLoader requires non-null weights");
    }
}

const __nv_bfloat16* WeightLoader::load_weight(
    const IWeightRepository& repo,
    const std::string& name,
    std::vector<int64_t> expected) {
    return upload_bf16(*weights_, repo, name, expected, name);
}

const LinearWeight* WeightLoader::load_linear_weight(
    const IWeightRepository& repo,
    const std::string& name,
    std::vector<int64_t> expected) {
    if (const auto cached = weights_->tensors.find(name);
        cached != weights_->tensors.end()) {
        if (cached->second.shape != expected) {
            throw std::runtime_error("cached linear shape mismatch for " + name);
        }
        return &cached->second.linear;
    }
    const HostTensorView tensor = repo.tensor(name);
    if (tensor.shape != expected || tensor.shape.size() != 2) {
        throw std::runtime_error("unexpected linear tensor: " + name);
    }
    const int rows = static_cast<int>(tensor.shape[0]);
    const int cols = static_cast<int>(tensor.shape[1]);

    DeviceWeight weight;
    weight.shape = tensor.shape;

    // Native GGUF block-quantized weights stay packed; the matmul kernel
    // dequantizes on the fly. Upload the raw super-block bytes verbatim.
    if (tensor.dtype == TensorDType::Quantized) {
        const GgmlTypeTrait trait = ggml_type_trait(tensor.ggml_type);
        const size_t expected_bytes =
            static_cast<size_t>(rows) * static_cast<size_t>(cols) /
            trait.block_size * trait.type_size;
        if (tensor.bytes != expected_bytes) {
            throw std::runtime_error(
                "GGUF quantized byte count mismatch for " + name);
        }
        weight.gguf_blocks_storage.reset(tensor.bytes);
        LFM_CUDA(cudaMemcpy(weight.gguf_blocks_storage.data(), tensor.data,
                            tensor.bytes, cudaMemcpyHostToDevice));
        weight.linear.kind = (tensor.ggml_type == GgmlType::Q4_K)
                                ? LinearStorageKind::Q4_K
                                : LinearStorageKind::Q6_K;
        weight.linear.gguf_type = tensor.ggml_type;
        weight.linear.gguf_blocks = weight.gguf_blocks_storage.data();
        weight.linear.rows = rows;
        weight.linear.cols = cols;
        weight.linear.validate_storage();
        auto [it, inserted] =
            weights_->tensors.emplace(name, std::move(weight));
        if (!inserted) throw std::runtime_error("duplicate linear weight: " + name);
        return &it->second.linear;
    }

    if (tensor.dtype != TensorDType::BF16) {
        throw std::runtime_error("unexpected linear tensor dtype: " + name);
    }
    const size_t count = checked_element_count(tensor.shape);
    if (tensor.bytes != count * sizeof(__nv_bfloat16)) {
        throw std::runtime_error("invalid linear tensor byte count: " + name);
    }

    if (weight_mode_ == WeightMode::Int8) {
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
    } else if (weight_mode_ == WeightMode::Int4) {
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

const LinearWeight* WeightLoader::load_concat_linear_weight(
    const IWeightRepository& repo,
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
        const HostTensorView tensor = repo.tensor(name);
        if (tensor.shape != expected || tensor.shape.size() != 2) {
            throw std::runtime_error("unexpected concatenated tensor: " + name);
        }
        if (common_width < 0) common_width = tensor.shape[1];
        if (tensor.shape[1] != common_width) {
            throw std::runtime_error("concatenated tensors have different widths");
        }
        if (tensor.dtype == TensorDType::Quantized) {
            views.push_back(tensor);
            total_rows += tensor.shape[0];
            continue;
        }
        if (tensor.dtype != TensorDType::BF16) {
            throw std::runtime_error("unexpected concat tensor dtype: " + name);
        }
        const size_t count = checked_element_count(tensor.shape);
        if (tensor.bytes != count * sizeof(__nv_bfloat16)) {
            throw std::runtime_error("invalid concatenated tensor bytes: " + name);
        }
        total_count += count;
        total_rows += tensor.shape[0];
        views.push_back(tensor);
    }

    // GGUF block-quantized concat: stack the raw super-block payloads of the
    // parts, which are already row-contiguous in packed form. The result keeps
    // the same block format; the matmul kernel dequantizes on the fly. When the
    // parts mix quant formats (e.g. attention q/k/v where value is Q6_K but
    // query/key are Q4_K) a single packed matrix is impossible, so each part is
    // dequantized to BF16 and concatenated as BF16 instead.
    if (views.front().dtype == TensorDType::Quantized) {
        const GgmlType concat_type = views.front().ggml_type;
        bool same_type = true;
        for (const auto& v : views) {
            if (v.dtype != TensorDType::Quantized || v.ggml_type != concat_type) {
                same_type = false;
                break;
            }
        }
        if (same_type) {
            const GgmlTypeTrait trait = ggml_type_trait(concat_type);
            size_t total_bytes = 0;
            for (const auto& v : views) total_bytes += v.bytes;
            DeviceWeight weight;
            weight.shape = {total_rows, common_width};
            weight.gguf_blocks_storage.reset(total_bytes);
            uint8_t* dst = weight.gguf_blocks_storage.data();
            for (const auto& v : views) {
                LFM_CUDA(cudaMemcpy(dst, v.data, v.bytes, cudaMemcpyHostToDevice));
                dst += v.bytes;
            }
            weight.linear.kind = (concat_type == GgmlType::Q4_K)
                                    ? LinearStorageKind::Q4_K
                                    : LinearStorageKind::Q6_K;
            weight.linear.gguf_type = concat_type;
            weight.linear.gguf_blocks = weight.gguf_blocks_storage.data();
            weight.linear.rows = static_cast<int>(total_rows);
            weight.linear.cols = static_cast<int>(common_width);
            weight.linear.validate_storage();
            auto [it, inserted] =
                weights_->tensors.emplace(synthetic_name, std::move(weight));
            if (!inserted) throw std::runtime_error("duplicate linear weight: " + synthetic_name);
            return &it->second.linear;
        }
        // Mixed quant formats: dequantize each part to BF16 and concat.
        DeviceWeight weight;
        weight.shape = {total_rows, common_width};
        size_t total_count = 0;
        for (const auto& v : views) {
            total_count += static_cast<size_t>(v.shape[0]) * static_cast<size_t>(v.shape[1]);
        }
        weight.bf16_storage.reset(total_count);
        __nv_bfloat16* dst = weight.bf16_storage.data();
        for (const auto& v : views) {
            std::vector<__nv_bfloat16> part;
            dequantize_gguf_to_bf16(v, part);
            const size_t count = part.size();
            LFM_CUDA(cudaMemcpy(dst, part.data(), count * sizeof(__nv_bfloat16),
                                cudaMemcpyHostToDevice));
            dst += count;
        }
        weight.linear.kind = LinearStorageKind::Bf16;
        weight.linear.bf16 = weight.bf16_storage.data();
        weight.linear.rows = static_cast<int>(total_rows);
        weight.linear.cols = static_cast<int>(common_width);
        weight.linear.validate_storage();
        auto [it, inserted] =
            weights_->tensors.emplace(synthetic_name, std::move(weight));
        if (!inserted) throw std::runtime_error("duplicate linear weight: " + synthetic_name);
        return &it->second.linear;
    }

    DeviceWeight weight;
    weight.shape = {total_rows, common_width};
    if (weight_mode_ == WeightMode::Int8) {
        std::vector<int8_t> quantized(total_count);
        std::vector<float> scales(static_cast<size_t>(total_rows));
        size_t row_offset = 0;
        size_t quantized_offset = 0;
        for (const auto& view : views) {
            const int rows = static_cast<int>(view.shape[0]);
            const int cols = static_cast<int>(view.shape[1]);
            Int8RowwisePack pack = quantize_bf16_rows(
                view.data, static_cast<size_t>(rows), static_cast<size_t>(cols));
            for (size_t i = 0; i < pack.scales.size(); ++i) {
                scales[row_offset + i] = pack.scales[i];
            }
            for (size_t i = 0; i < pack.values.size(); ++i) {
                quantized[quantized_offset + i] = pack.values[i];
            }
            row_offset += pack.scales.size();
            quantized_offset += pack.values.size();
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
    } else if (weight_mode_ == WeightMode::Int4) {
        std::vector<uint8_t> quantized;
        std::vector<float> scales;
        quantized.reserve(total_count / 2 + 16);
        scales.reserve(static_cast<size_t>(total_rows));
        for (const auto& view : views) {
            const int rows = static_cast<int>(view.shape[0]);
            const int cols = static_cast<int>(view.shape[1]);
            Int4RowwisePack pack = quantize_bf16_rows_int4(
                view.data, static_cast<size_t>(rows), static_cast<size_t>(cols));
            quantized.insert(quantized.end(), pack.values.begin(), pack.values.end());
            scales.insert(scales.end(), pack.scales.begin(), pack.scales.end());
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
        __nv_bfloat16* dest = weight.bf16_storage.data();
        size_t offset = 0;
        for (const auto& view : views) {
            const size_t count = checked_element_count(view.shape);
            LFM_CUDA(cudaMemcpy(dest + offset, view.data,
                                count * sizeof(__nv_bfloat16),
                                cudaMemcpyHostToDevice));
            offset += count;
        }
        weight.linear.kind = LinearStorageKind::Bf16;
        weight.linear.bf16 = weight.bf16_storage.data();
    }
    weight.linear.rows = static_cast<int>(total_rows);
    weight.linear.cols = static_cast<int>(common_width);
    weight.linear.validate_storage();

    auto [it, inserted] = weights_->tensors.emplace(synthetic_name, std::move(weight));
    if (!inserted) throw std::runtime_error("duplicate linear weight: " + synthetic_name);
    return &it->second.linear;
}

const ExpertLinearWeight* WeightLoader::load_expert_linear_weight(
    const IWeightRepository& repo,
    const std::string& name,
    int experts, int rows_per_expert, int cols) {
    const std::string cache_key = name;
    if (const auto cached = expert_cache_.find(cache_key);
        cached != expert_cache_.end()) {
        return &cached->second;
    }
    if (experts <= 0 || rows_per_expert <= 0 || cols <= 0) {
        throw std::runtime_error("invalid expert weight dimensions for " + name);
    }
    const std::vector<int64_t> expected = {
        static_cast<int64_t>(experts) * rows_per_expert, cols};
    const HostTensorView tensor = repo.tensor(name);
    if (tensor.dtype != TensorDType::BF16) {
        throw std::runtime_error("expert weights must be BF16: " + name);
    }
    if (tensor.shape != expected) {
        throw std::runtime_error("unexpected packed expert shape for " + name);
    }
    const size_t count = checked_element_count(tensor.shape);
    if (tensor.bytes != count * sizeof(__nv_bfloat16)) {
        throw std::runtime_error("invalid expert byte count for " + name);
    }

    DeviceWeight weight;
    weight.shape = {experts, rows_per_expert, cols};
    weight.bf16_storage.reset(count);
    LFM_CUDA(cudaMemcpy(weight.bf16_storage.data(), tensor.data, tensor.bytes,
                        cudaMemcpyHostToDevice));

    ExpertLinearWeight ew;
    ew.kind = LinearStorageKind::Bf16;
    ew.bf16 = weight.bf16_storage.data();
    ew.experts = experts;
    ew.rows_per_expert = rows_per_expert;
    ew.cols = cols;

    auto [it, inserted] = weights_->tensors.emplace(cache_key, std::move(weight));
    if (!inserted) throw std::runtime_error("duplicate expert weight: " + cache_key);
    const ExpertLinearWeight& stored = expert_cache_.emplace(cache_key, ew).first->second;
    return &stored;
}

// Packs per-expert gate (w1) and up (w3) tensors into one [experts,
// 2*moe_inter, hidden] buffer. w1 occupies rows [0, moe_inter); w3 occupies
// [moe_inter, 2*moe_inter).
const ExpertLinearWeight* WeightLoader::load_moe_gate_up(
    const IWeightRepository& repo, int layer,
    int num_experts, int moe_intermediate, int hidden) {
    const std::string cache_key = layer_name(layer, "moe.gate_up");
    if (const auto cached = expert_cache_.find(cache_key);
        cached != expert_cache_.end()) {
        return &cached->second;
    }
    if (num_experts <= 0 || moe_intermediate <= 0 || hidden <= 0) {
        throw std::runtime_error("invalid MoE gate_up dimensions for layer " +
                                 std::to_string(layer));
    }
    const size_t rows_per_expert = static_cast<size_t>(2 * moe_intermediate);
    const size_t per_expert = rows_per_expert * static_cast<size_t>(hidden);
    const size_t total = static_cast<size_t>(num_experts) * per_expert;

    DeviceWeight weight;
    weight.shape = {num_experts, static_cast<int>(rows_per_expert), hidden};
    weight.bf16_storage.reset(total);
    __nv_bfloat16* base = weight.bf16_storage.data();

    const size_t moe_inter = static_cast<size_t>(moe_intermediate);
    const size_t hidden_c = static_cast<size_t>(hidden);
    const size_t w_bytes = moe_inter * hidden_c * sizeof(__nv_bfloat16);
    for (int e = 0; e < num_experts; ++e) {
        const std::string w1_name = layer_name(
            layer, "feed_forward.experts." + std::to_string(e) + ".w1.weight");
        const std::string w3_name = layer_name(
            layer, "feed_forward.experts." + std::to_string(e) + ".w3.weight");
        const HostTensorView w1 = repo.tensor(w1_name);
        const HostTensorView w3 = repo.tensor(w3_name);
        if (w1.shape != std::vector<int64_t>{moe_intermediate, hidden} ||
            w3.shape != std::vector<int64_t>{moe_intermediate, hidden}) {
            throw std::runtime_error("unexpected MoE expert tensor shape for " + w1_name);
        }
        const size_t e_off = static_cast<size_t>(e) * per_expert;
        LFM_CUDA(cudaMemcpy(base + e_off, w1.data, w_bytes, cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(base + e_off + moe_inter * hidden_c, w3.data,
                            w_bytes, cudaMemcpyHostToDevice));
    }

    ExpertLinearWeight ew;
    ew.kind = LinearStorageKind::Bf16;
    ew.bf16 = weight.bf16_storage.data();
    ew.experts = num_experts;
    ew.rows_per_expert = static_cast<int>(rows_per_expert);
    ew.cols = hidden;

    auto [it, inserted] = weights_->tensors.emplace(cache_key, std::move(weight));
    if (!inserted) throw std::runtime_error("duplicate expert weight: " + cache_key);
    const ExpertLinearWeight& stored = expert_cache_.emplace(cache_key, ew).first->second;
    return &stored;
}

const ExpertLinearWeight* WeightLoader::load_moe_down(
    const IWeightRepository& repo, int layer,
    int num_experts, int moe_intermediate, int hidden) {
    const std::string cache_key = layer_name(layer, "moe.down");
    if (const auto cached = expert_cache_.find(cache_key);
        cached != expert_cache_.end()) {
        return &cached->second;
    }
    if (num_experts <= 0 || moe_intermediate <= 0 || hidden <= 0) {
        throw std::runtime_error("invalid MoE down dimensions for layer " +
                                 std::to_string(layer));
    }
    const size_t rows_per_expert = static_cast<size_t>(hidden);
    const size_t per_expert = rows_per_expert * static_cast<size_t>(moe_intermediate);
    const size_t total = static_cast<size_t>(num_experts) * per_expert;

    DeviceWeight weight;
    weight.shape = {num_experts, hidden, moe_intermediate};
    weight.bf16_storage.reset(total);
    __nv_bfloat16* base = weight.bf16_storage.data();

    const size_t hidden_c = static_cast<size_t>(hidden);
    const size_t moe_inter = static_cast<size_t>(moe_intermediate);
    const size_t w_bytes = hidden_c * moe_inter * sizeof(__nv_bfloat16);
    for (int e = 0; e < num_experts; ++e) {
        const std::string w2_name = layer_name(
            layer, "feed_forward.experts." + std::to_string(e) + ".w2.weight");
        const HostTensorView w2 = repo.tensor(w2_name);
        if (w2.shape != std::vector<int64_t>{hidden, moe_intermediate}) {
            throw std::runtime_error("unexpected MoE expert tensor shape for " + w2_name);
        }
        const size_t e_off = static_cast<size_t>(e) * per_expert;
        LFM_CUDA(cudaMemcpy(base + e_off, w2.data, w_bytes, cudaMemcpyHostToDevice));
    }

    ExpertLinearWeight ew;
    ew.kind = LinearStorageKind::Bf16;
    ew.bf16 = weight.bf16_storage.data();
    ew.experts = num_experts;
    ew.rows_per_expert = hidden;
    ew.cols = moe_intermediate;

    auto [it, inserted] = weights_->tensors.emplace(cache_key, std::move(weight));
    if (!inserted) throw std::runtime_error("duplicate expert weight: " + cache_key);
    const ExpertLinearWeight& stored = expert_cache_.emplace(cache_key, ew).first->second;
    return &stored;
}

const float* WeightLoader::load_f32_weight(
    const IWeightRepository& repo,
    const std::string& name,
    std::vector<int64_t> expected) {
    if (const auto cached = weights_->tensors.find(name);
        cached != weights_->tensors.end()) {
        if (cached->second.shape != expected) {
            throw std::runtime_error("cached f32 shape mismatch for " + name);
        }
        return cached->second.scales_storage.data();
    }
    const HostTensorView tensor = repo.tensor(name);
    if (tensor.dtype != TensorDType::F32) {
        throw std::runtime_error("expected F32 tensor: " + name);
    }
    if (tensor.shape != expected) {
        throw std::runtime_error("unexpected f32 shape for " + name);
    }
    const size_t count = checked_element_count(tensor.shape);
    if (tensor.bytes != count * sizeof(float)) {
        throw std::runtime_error("invalid f32 byte count for " + name);
    }
    DeviceWeight weight;
    weight.shape = tensor.shape;
    weight.scales_storage.reset(count);
    LFM_CUDA(cudaMemcpy(weight.scales_storage.data(), tensor.data, tensor.bytes,
                        cudaMemcpyHostToDevice));
    auto [it, inserted] = weights_->tensors.emplace(name, std::move(weight));
    if (!inserted) throw std::runtime_error("duplicate f32 weight: " + name);
    return it->second.scales_storage.data();
}

const LinearWeight* WeightLoader::load_router(
    const IWeightRepository& repo, int layer,
    int num_experts, int hidden) {
    return load_linear_weight(
        repo, layer_name(layer, "feed_forward.gate.weight"),
        {num_experts, hidden});
}

WeightLoader::HostExpertLayer WeightLoader::load_moe_experts_host(
    const IWeightRepository& repo, int layer,
    int num_experts, int moe_intermediate, int hidden,
    HostExpertStore& store, ExpertHostMode host_mode) {
    if (num_experts <= 0 || moe_intermediate <= 0 || hidden <= 0) {
        throw std::runtime_error("invalid MoE expert dimensions for layer " +
                                 std::to_string(layer));
    }
    const size_t moe_inter = static_cast<size_t>(moe_intermediate);
    const size_t hidden_c = static_cast<size_t>(hidden);
    const size_t gate_up_elems = 2 * moe_inter * hidden_c;
    const size_t down_elems = hidden_c * moe_inter;
    const size_t gate_up_bytes = gate_up_elems * sizeof(__nv_bfloat16);
    const size_t down_bytes = down_elems * sizeof(__nv_bfloat16);
    const size_t w_bytes = moe_inter * hidden_c * sizeof(__nv_bfloat16);

    HostExpertLayer result;
    result.gate_up_bytes = gate_up_bytes;
    result.down_bytes = down_bytes;
    result.gate_up_host_dev.resize(static_cast<size_t>(num_experts));
    result.down_host_dev.resize(static_cast<size_t>(num_experts));

    const size_t layer_gate_up_bytes = gate_up_bytes * static_cast<size_t>(num_experts);
    const size_t layer_down_bytes = down_bytes * static_cast<size_t>(num_experts);

    // Mapped mode: allocate one persistent arena per layer (a single pinned +
    // registered region) and point each expert into it. This avoids 704 separate
    // cudaHostAlloc calls that can exhaust the driver's pinned-memory pool.
    // Pinned mode: copy each expert into its own pinned allocation.
    const __nv_bfloat16* gate_up_base = nullptr;
    const __nv_bfloat16* down_base = nullptr;
    if (host_mode == ExpertHostMode::Mapped) {
        gate_up_base = static_cast<const __nv_bfloat16*>(
            store.alloc_mapped(layer_gate_up_bytes));
        down_base = static_cast<const __nv_bfloat16*>(
            store.alloc_mapped(layer_down_bytes));
        for (int e = 0; e < num_experts; ++e) {
            result.gate_up_host_dev[static_cast<size_t>(e)] =
                gate_up_base + static_cast<size_t>(e) * (gate_up_bytes / sizeof(__nv_bfloat16));
            result.down_host_dev[static_cast<size_t>(e)] =
                down_base + static_cast<size_t>(e) * (down_bytes / sizeof(__nv_bfloat16));
        }
    }

    // Reusable host staging buffer: assemble each expert's packed layout
    // (gate_up = [w1; w3], down = [w2]).
    std::vector<__nv_bfloat16> gate_up_stage(gate_up_elems);
    std::vector<__nv_bfloat16> down_stage(down_elems);

    for (int e = 0; e < num_experts; ++e) {
        const std::string w1_name = layer_name(
            layer, "feed_forward.experts." + std::to_string(e) + ".w1.weight");
        const std::string w3_name = layer_name(
            layer, "feed_forward.experts." + std::to_string(e) + ".w3.weight");
        const std::string w2_name = layer_name(
            layer, "feed_forward.experts." + std::to_string(e) + ".w2.weight");
        const HostTensorView w1 = repo.tensor(w1_name);
        const HostTensorView w3 = repo.tensor(w3_name);
        const HostTensorView w2 = repo.tensor(w2_name);
        if (w1.dtype != TensorDType::BF16 || w3.dtype != TensorDType::BF16 ||
            w2.dtype != TensorDType::BF16) {
            throw std::runtime_error("MoE expert weights must be BF16: " + w1_name);
        }
        if (w1.shape != std::vector<int64_t>{moe_intermediate, hidden} ||
            w3.shape != std::vector<int64_t>{moe_intermediate, hidden} ||
            w2.shape != std::vector<int64_t>{hidden, moe_intermediate}) {
            throw std::runtime_error("unexpected MoE expert tensor shape for " + w1_name);
        }
        std::memcpy(gate_up_stage.data(), w1.data, w_bytes);
        std::memcpy(gate_up_stage.data() + moe_inter * hidden_c, w3.data, w_bytes);
        std::memcpy(down_stage.data(), w2.data, down_bytes);

        if (host_mode == ExpertHostMode::Mapped) {
            // Copy the packed expert into its slot in the persistent arena.
            const size_t gu_off = static_cast<size_t>(e) * gate_up_elems;
            const size_t dw_off = static_cast<size_t>(e) * down_elems;
            std::memcpy(const_cast<__nv_bfloat16*>(gate_up_base) + gu_off,
                        gate_up_stage.data(), gate_up_bytes);
            std::memcpy(const_cast<__nv_bfloat16*>(down_base) + dw_off,
                        down_stage.data(), down_bytes);
        } else {
            result.gate_up_host_dev[static_cast<size_t>(e)] =
                static_cast<const __nv_bfloat16*>(
                    store.store_pinned_copy(gate_up_stage.data(), gate_up_bytes));
            result.down_host_dev[static_cast<size_t>(e)] =
                static_cast<const __nv_bfloat16*>(
                    store.store_pinned_copy(down_stage.data(), down_bytes));
        }
    }
    return result;
}

std::vector<ExpertLocation> WeightLoader::build_expert_catalog(
    const IWeightRepository& repo, int layer,
    int num_experts, int moe_intermediate, int hidden) {
    if (num_experts <= 0 || moe_intermediate <= 0 || hidden <= 0) {
        throw std::runtime_error("invalid MoE expert dimensions for layer " +
                                 std::to_string(layer));
    }
    std::vector<ExpertLocation> catalog(static_cast<size_t>(num_experts));
    for (int e = 0; e < num_experts; ++e) {
        const std::string w1_name = layer_name(
            layer, "feed_forward.experts." + std::to_string(e) + ".w1.weight");
        const std::string w3_name = layer_name(
            layer, "feed_forward.experts." + std::to_string(e) + ".w3.weight");
        const std::string w2_name = layer_name(
            layer, "feed_forward.experts." + std::to_string(e) + ".w2.weight");

        TensorLocator w1_loc = repo.locate(w1_name);
        TensorLocator w3_loc = repo.locate(w3_name);
        TensorLocator w2_loc = repo.locate(w2_name);

        if (w1_loc.shape != std::vector<int64_t>{moe_intermediate, hidden} ||
            w3_loc.shape != std::vector<int64_t>{moe_intermediate, hidden} ||
            w2_loc.shape != std::vector<int64_t>{hidden, moe_intermediate}) {
            throw std::runtime_error("unexpected MoE expert tensor shape for " + w1_name);
        }

        catalog[static_cast<size_t>(e)] = ExpertLocation{w1_loc, w2_loc, w3_loc};
    }
    return catalog;
}

} // namespace lfm
