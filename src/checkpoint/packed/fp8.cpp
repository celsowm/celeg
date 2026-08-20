#include "celeg/checkpoint/packed/fp8.hpp"

#include <bit>
#include <cstring>
#include <stdexcept>

namespace celeg {
namespace {

float bf16_to_float(const std::byte* source) {
    uint16_t bits = 0;
    std::memcpy(&bits, source, sizeof(bits));
    return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
}

float f32_from(const std::byte* source) {
    float value = 0.0f;
    std::memcpy(&value, source, sizeof(value));
    return value;
}

// Per-channel scale sidecars show up as F32 or BF16 depending on export
// tooling; both are accepted.
float read_scale(const HostTensorView& scale, size_t row) {
    if (scale.dtype == TensorDType::F32) {
        return f32_from(scale.data + row * sizeof(float));
    }
    if (scale.dtype == TensorDType::BF16) {
        return bf16_to_float(scale.data + row * sizeof(uint16_t));
    }
    throw std::runtime_error("unsupported FP8 weight_scale dtype");
}

}

void PackedFp8Matrix::validate() const {
    if (rows <= 0 || cols <= 0 || values.size() != static_cast<size_t>(rows) * cols ||
        scales.size() != static_cast<size_t>(rows)) {
        throw std::runtime_error("invalid packed FP8 matrix");
    }
}

bool has_packed_fp8_matrix(const IWeightRepository& repository,
                           std::string_view name) {
    const std::string prefix(name);
    if (!repository.contains(prefix) || !repository.contains(prefix + "_scale")) {
        return false;
    }
    const HostTensorView weight = repository.tensor(prefix);
    if (weight.dtype != TensorDType::F8_E4M3 || weight.shape.size() != 2) {
        return false;
    }
    const HostTensorView scale = repository.tensor(prefix + "_scale");
    return (scale.dtype == TensorDType::F32 || scale.dtype == TensorDType::BF16) &&
        scale.shape == std::vector<int64_t>{weight.shape[0], 1};
}

PackedFp8Matrix load_packed_fp8_matrix(
    const IWeightRepository& repository, std::string_view name,
    const std::vector<int64_t>& expected_shape) {
    if (expected_shape.size() != 2 || expected_shape[0] <= 0 || expected_shape[1] <= 0) {
        throw std::invalid_argument("packed FP8 weight requires a positive matrix shape");
    }
    const int rows = static_cast<int>(expected_shape[0]);
    const int cols = static_cast<int>(expected_shape[1]);
    const std::string prefix(name);
    const std::string scale_name = prefix + "_scale";
    if (!repository.contains(prefix) || !repository.contains(scale_name)) {
        throw std::out_of_range("incomplete compressed-tensors FP8 weight: " + prefix);
    }

    const HostTensorView weight = repository.tensor(prefix);
    const HostTensorView scale = repository.tensor(scale_name);
    if (weight.dtype != TensorDType::F8_E4M3 ||
        weight.shape != std::vector<int64_t>{rows, cols} ||
        weight.bytes != static_cast<size_t>(rows) * static_cast<size_t>(cols)) {
        throw std::runtime_error("invalid compressed-tensors FP8 weight payload: " + prefix);
    }
    if (scale.shape != std::vector<int64_t>{rows, 1}) {
        throw std::runtime_error("compressed-tensors FP8 scale shape disagrees: " + prefix);
    }

    PackedFp8Matrix result;
    result.rows = rows;
    result.cols = cols;
    result.values.resize(static_cast<size_t>(rows) * cols);
    std::memcpy(result.values.data(), weight.data, result.values.size());
    result.scales.resize(static_cast<size_t>(rows));
    for (int row = 0; row < rows; ++row) {
        result.scales[static_cast<size_t>(row)] = read_scale(scale, static_cast<size_t>(row));
    }
    result.validate();
    return result;
}

}
