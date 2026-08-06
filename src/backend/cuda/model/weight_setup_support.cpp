#include "celeg/backend/cuda/weight_setup_support.hpp"

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
    if (weight.gguf_quantized()) {
        if (weight.gguf_segments.size() != 1) {
            throw std::runtime_error(std::string(label) +
                                     " must use one native GGUF segment");
        }
        return make_gguf_weight_layout(weight.gguf_segments.front());
    }
    switch (mode) {
    case WeightMode::Int8:
        if (!weight.int8) throw std::runtime_error(std::string(label) + " has no INT8 storage");
        return make_weight_layout(mode, weight.int8, weight.scales);
    case WeightMode::Int4:
        if (!weight.int4) throw std::runtime_error(std::string(label) + " has no INT4 storage");
        return make_weight_layout(mode, weight.int4, weight.scales);
    default:
        if (!weight.bf16) throw std::runtime_error(std::string(label) + " has no BF16 storage");
        return make_weight_layout(mode, weight.bf16, weight.scales);
    }
}

} // namespace celeg
