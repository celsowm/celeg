#include "backend/cuda/weight_setup_support.hpp"

#include "celeg/model/weights/roles.hpp"

#include <stdexcept>

namespace celeg {

std::string cuda_layer_name(int layer, const std::string& suffix) {
    return "model.layers." + std::to_string(layer) + "." + suffix;
}

std::string cuda_tensor_name(std::span<const TensorRequest> requests,
                             TensorRole role, int layer) {
    return resolved_tensor_name(requests, role, layer);
}

std::unique_ptr<IWeightLayout> make_cuda_embedding_layout(
    WeightMode mode, const LinearWeight& weight, const char* label) {
    if (const auto* gguf = std::get_if<GgufLinearStorage>(&weight.storage)) {
        if (gguf->segments.size() != 1) {
            throw std::runtime_error(std::string(label) +
                                     " must use one native GGUF segment");
        }
        return make_gguf_weight_layout(gguf->segments.front());
    }
    switch (mode) {
    case WeightMode::Int8: {
        const auto* storage = std::get_if<Int8LinearStorage>(&weight.storage);
        if (!storage || !storage->data) {
            throw std::runtime_error(std::string(label) + " has no INT8 storage");
        }
        return make_weight_layout(mode, storage->data, storage->scales);
    }
    case WeightMode::Int4: {
        const auto* storage = std::get_if<Int4LinearStorage>(&weight.storage);
        if (!storage || !storage->data) {
            throw std::runtime_error(std::string(label) + " has no INT4 storage");
        }
        return make_weight_layout(mode, storage->data, storage->scales);
    }
    default: {
        const auto* storage = std::get_if<Bf16LinearStorage>(&weight.storage);
        if (!storage || !storage->data) {
            throw std::runtime_error(std::string(label) + " has no BF16 storage");
        }
        return make_weight_layout(mode, storage->data, nullptr);
    }
    }
}

}
