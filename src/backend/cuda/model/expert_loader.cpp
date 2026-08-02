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
    const size_t count = cuda_loader_detail::checked_element_count(tensor.shape);
    if (tensor.bytes != count * sizeof(__nv_bfloat16)) {
        throw std::runtime_error("invalid expert byte count for " + name);
    }

    DeviceWeight weight;
    weight.shape = {experts, rows_per_expert, cols};
    weight.bf16_storage.reset(count);
    CELEG_CUDA(cudaMemcpy(weight.bf16_storage.data(), tensor.data, tensor.bytes,
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

} // namespace celeg
