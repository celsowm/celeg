#include "lfm/weight_loader.hpp"
#include "lfm/quantization.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>

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

const LinearWeight* WeightLoader::load_linear_weight(
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

} // namespace lfm
