#include "celeg/backend/cuda/weights_loader.hpp"
#include "celeg/model/weights/quantization.hpp"
#include "celeg/backend/cuda/moe/expert_residency.hpp"
#include "celeg/backend/cuda/kernels/gguf.cuh"
#include "celeg/checkpoint/gguf_blocks.hpp"
#include "celeg/checkpoint/tensor_names.hpp"
#include "weight_loader_internal.hpp"

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
    const size_t count = cuda_loader_detail::checked_element_count(tensor.shape);

    DeviceWeight weight;
    weight.shape = tensor.shape;
    weight.bf16_storage.reset(count);

    if (tensor.dtype == TensorDType::BF16) {
        if (tensor.bytes != count * sizeof(__nv_bfloat16)) {
            throw std::runtime_error("invalid BF16 byte count for " + name);
        }
        CELEG_CUDA(cudaMemcpy(weight.bf16_storage.data(), tensor.data, tensor.bytes,
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
        CELEG_CUDA(cudaMemcpy(weight.bf16_storage.data(), converted.data(),
                            count * sizeof(__nv_bfloat16),
                            cudaMemcpyHostToDevice));
    } else if (tensor.dtype == TensorDType::F16) {
        if (tensor.bytes != count * sizeof(__half)) {
            throw std::runtime_error("invalid F16 byte count for " + name);
        }
        const __half* src = reinterpret_cast<const __half*>(tensor.data);
        std::vector<__nv_bfloat16> converted(count);
        for (size_t i = 0; i < count; ++i) {
            converted[i] = __float2bfloat16(__half2float(src[i]));
        }
        CELEG_CUDA(cudaMemcpy(weight.bf16_storage.data(), converted.data(),
                            count * sizeof(__nv_bfloat16),
                            cudaMemcpyHostToDevice));
    } else if (tensor.dtype == TensorDType::Quantized) {
        if (tensor.shape.size() != 2 ||
            (tensor.ggml_type != GgmlType::Q4_K &&
             tensor.ggml_type != GgmlType::Q6_K)) {
            throw std::runtime_error(
                "router BF16 materialization supports only 2D GGUF Q4_K/Q6_K: " + name);
        }
        std::vector<__nv_bfloat16> dequantized;
        dequantize_gguf_to_bf16(tensor, dequantized);
        if (dequantized.size() != count) {
            throw std::runtime_error("invalid GGUF router element count for " + name);
        }
        CELEG_CUDA(cudaMemcpy(weight.bf16_storage.data(), dequantized.data(),
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
    const size_t count = cuda_loader_detail::checked_element_count(tensor.shape);
    if (tensor.bytes != count * sizeof(float)) {
        throw std::runtime_error("invalid f32 byte count for " + name);
    }
    DeviceWeight weight;
    weight.shape = tensor.shape;
    weight.scales_storage.reset(count);
    CELEG_CUDA(cudaMemcpy(weight.scales_storage.data(), tensor.data, tensor.bytes,
                        cudaMemcpyHostToDevice));
    auto [it, inserted] = weights_->tensors.emplace(name, std::move(weight));
    if (!inserted) throw std::runtime_error("duplicate f32 weight: " + name);
    return it->second.scales_storage.data();
}

const LinearWeight* WeightLoader::load_router_weight(
    const IWeightRepository& repo, int layer,
    int num_experts, int hidden) {
    // The MoE router must always materialize as BF16 regardless of the global
    // --weight-mode; the CUDA router kernel consumes a float copy of this
    // weight and would otherwise dereference a null BF16 pointer (Phase 1.1
    // hazard). We deliberately bypass load_linear_weight and go through the
    // BF16-only load_weight path.
    const std::string name = layer_name(layer, "feed_forward.gate.weight");
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

    // Construct a LinearWeight view pointing into the shared arena; the slot
    // was just emplaced by load_weight -> upload_bf16, so this is stable.
    DeviceWeight& slot = weights_->tensors[name];
    LinearWeight view;
    view.kind = LinearStorageKind::Bf16;
    view.bf16 = bf16;
    view.rows = num_experts;
    view.cols = hidden;
    slot.linear = view;
    return &slot.linear;
}

} // namespace celeg
