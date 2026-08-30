#include "backend/cuda/weights_loader.hpp"
#include "celeg/model/weights/quantization.hpp"
#include "backend/cuda/moe/expert_residency.hpp"
#include "kernels/gguf.cuh"
#include "celeg/checkpoint/gguf_blocks.hpp"
#include "celeg/checkpoint/tensor_codec.hpp"
#include "celeg/checkpoint/tensor_names.hpp"

#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace celeg {

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
    const std::vector<int64_t> target_shape = expected.empty() ? tensor.shape : expected;
    if (!expected.empty()) {
        if (!tensor_shape_is_compatible(tensor.shape, expected)) {
            throw std::runtime_error("unexpected shape for " + name);
        }
    }
    const size_t count = tensor_element_count(target_shape, name);

    DeviceWeight weight(weights.memory_kind);
    weight.shape = expected.empty() ? tensor.shape : expected;
    weight.bf16_storage.reset(count);

    if (tensor.dtype == TensorDType::BF16) {
        if (tensor.bytes != count * sizeof(__nv_bfloat16)) {
            throw std::runtime_error("invalid BF16 byte count for " + name);
        }
        CELEG_CUDA(cudaMemcpy(weight.bf16_storage.data(), tensor.data, tensor.bytes,
                            cudaMemcpyHostToDevice));
    } else if (tensor.dtype == TensorDType::F32 ||
               tensor.dtype == TensorDType::F16 ||
               tensor.dtype == TensorDType::Quantized) {
        const std::vector<float> decoded = decode_tensor_f32(
            tensor, target_shape, name);
        std::vector<__nv_bfloat16> converted(count);
        for (size_t i = 0; i < count; ++i) {
            converted[i] = __float2bfloat16(decoded[i]);
        }
        CELEG_CUDA(cudaMemcpy(weight.bf16_storage.data(), converted.data(),
                            count * sizeof(__nv_bfloat16),
                            cudaMemcpyHostToDevice));
    } else {
        throw std::runtime_error(
            "only BF16/F16/F32 source weights are supported; incompatible tensor: " + name);
    }
    auto [it, inserted] = weights.tensors.emplace(cache_key, std::move(weight));
    if (!inserted) throw std::runtime_error("duplicate weight: " + cache_key);
    return it->second.bf16_storage.data();
}

const __nv_bfloat16* WeightLoader::load_weight(
    const IWeightRepository& repo,
    const std::string& name,
    std::vector<int64_t> expected) {
    return upload_bf16(*weights_, repo, name, expected, name);
}

const __nv_bfloat16* WeightLoader::load_rms_norm_weight(
    const IWeightRepository& repo, const std::string& name,
    std::vector<int64_t> expected, NormWeightKind weight_kind) {
    const std::string cache_key = name + "#rms_norm_" +
        std::to_string(static_cast<int>(weight_kind));
    if (const auto cached = weights_->tensors.find(cache_key);
        cached != weights_->tensors.end()) {
        if (cached->second.shape != expected) {
            throw std::runtime_error("cached RMSNorm shape mismatch for " + name);
        }
        return cached->second.bf16_storage.data();
    }
    const size_t count = tensor_element_count(expected, name);
    std::vector<__nv_bfloat16> host(count);
    if (weight_kind == NormWeightKind::None) {
        for (__nv_bfloat16& value : host) value = __float2bfloat16(1.0f);
    } else {
        const __nv_bfloat16* source = load_weight(repo, name, expected);
        CELEG_CUDA(cudaMemcpy(host.data(), source, count * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToHost));
        if (weight_kind == NormWeightKind::OnePlusScale) {
            for (__nv_bfloat16& value : host) {
                value = __float2bfloat16(__bfloat162float(value) + 1.0f);
            }
        }
    }
    DeviceWeight adjusted(weights_->memory_kind);
    adjusted.shape = expected;
    adjusted.bf16_storage.reset(count);
    CELEG_CUDA(cudaMemcpy(adjusted.bf16_storage.data(), host.data(),
                          count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    auto [it, inserted] = weights_->tensors.emplace(cache_key, std::move(adjusted));
    if (!inserted) throw std::runtime_error("duplicate RMSNorm weight: " + name);
    return it->second.bf16_storage.data();
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
    const size_t count = tensor_element_count(tensor.shape, name);
    if (tensor.bytes != count * sizeof(float)) {
        throw std::runtime_error("invalid f32 byte count for " + name);
    }
    DeviceWeight weight(weights_->memory_kind);
    weight.shape = tensor.shape;
    weight.scales_storage.reset(count);
    CELEG_CUDA(cudaMemcpy(weight.scales_storage.data(), tensor.data, tensor.bytes,
                        cudaMemcpyHostToDevice));
    auto [it, inserted] = weights_->tensors.emplace(name, std::move(weight));
    if (!inserted) throw std::runtime_error("duplicate f32 weight: " + name);
    return it->second.scales_storage.data();
}

const LinearWeight* WeightLoader::load_router_weight_named(
    const IWeightRepository& repo, const std::string& name,
    int num_experts, int hidden) {
    if (const auto cached = weights_->tensors.find(name);
        cached != weights_->tensors.end()) {
        if (cached->second.shape != std::vector<int64_t>{
                static_cast<int64_t>(num_experts),
                static_cast<int64_t>(hidden)}) {
            throw std::runtime_error("cached router shape mismatch for " + name);
        }
        return &cached->second.linear;
    }
    const __nv_bfloat16* bf16 = load_weight(repo, name,
                                            {num_experts, hidden});

    DeviceWeight& slot = weights_->tensors.at(name);
    LinearWeight view;
    view.storage = Bf16LinearStorage{bf16};
    view.rows = num_experts;
    view.cols = hidden;
    slot.linear = view;
    return &slot.linear;
}

}
