#include "celeg/checkpoint/packed/nvfp4.hpp"

#include <cstring>
#include <stdexcept>

namespace celeg {
namespace {

float f32_from(const std::byte* source) {
    float value = 0.0f;
    std::memcpy(&value, source, sizeof(value));
    return value;
}

bool is_scalar_f32(const HostTensorView& view) {
    if (view.dtype != TensorDType::F32 || view.bytes != sizeof(float)) return false;
    size_t count = 1;
    for (const int64_t dimension : view.shape) count *= static_cast<size_t>(dimension);
    return count == 1;
}

std::string module_name(const std::string& weight_name) {
    constexpr std::string_view suffix = ".weight";
    if (weight_name.size() > suffix.size() &&
        weight_name.ends_with(suffix)) {
        return weight_name.substr(0, weight_name.size() - suffix.size());
    }
    return weight_name;
}

}

void PackedNvfp4Matrix::validate() const {
    if (rows <= 0 || cols <= 0 || cols % kNvfp4PackedBlockSize != 0 ||
        packed.size() != static_cast<size_t>(rows) * cols / 2 ||
        block_scales.size() != static_cast<size_t>(rows) * (cols / kNvfp4PackedBlockSize)) {
        throw std::runtime_error("invalid packed NVFP4 matrix");
    }
}

bool has_packed_nvfp4_matrix(const IWeightRepository& repository,
                             std::string_view name) {
    const std::string prefix(name);
    const std::string packed_name = prefix + "_packed";
    const std::string scale_name = prefix + "_scale";
    const std::string global_scale_name = prefix + "_global_scale";
    if (!repository.contains(packed_name) || !repository.contains(scale_name) ||
        !repository.contains(global_scale_name)) {
        return false;
    }
    const HostTensorView packed = repository.tensor(packed_name);
    if (packed.dtype != TensorDType::U8 || packed.shape.size() != 2) return false;
    const int64_t rows = packed.shape[0];
    const int64_t packed_cols = packed.shape[1];
    if (rows <= 0 || packed_cols <= 0) return false;
    const HostTensorView scale = repository.tensor(scale_name);
    if (scale.dtype != TensorDType::F8_E4M3 ||
        scale.shape != std::vector<int64_t>{rows, packed_cols * 2 / kNvfp4PackedBlockSize}) {
        return false;
    }
    return is_scalar_f32(repository.tensor(global_scale_name));
}

PackedNvfp4Matrix load_packed_nvfp4_matrix(
    const IWeightRepository& repository, std::string_view name,
    const std::vector<int64_t>& expected_shape) {
    if (expected_shape.size() != 2 || expected_shape[0] <= 0 || expected_shape[1] <= 0) {
        throw std::invalid_argument("packed NVFP4 weight requires a positive matrix shape");
    }
    const int rows = static_cast<int>(expected_shape[0]);
    const int cols = static_cast<int>(expected_shape[1]);
    if (cols % kNvfp4PackedBlockSize != 0) {
        throw std::runtime_error("NVFP4 weight width is not block-aligned: " + std::string(name));
    }
    const std::string prefix(name);
    const std::string packed_name = prefix + "_packed";
    const std::string scale_name = prefix + "_scale";
    const std::string global_scale_name = prefix + "_global_scale";
    if (!repository.contains(packed_name) || !repository.contains(scale_name) ||
        !repository.contains(global_scale_name)) {
        throw std::out_of_range("incomplete compressed-tensors NVFP4 weight: " + prefix);
    }

    const HostTensorView packed = repository.tensor(packed_name);
    const HostTensorView scale = repository.tensor(scale_name);
    const HostTensorView global_scale = repository.tensor(global_scale_name);
    const int64_t packed_cols = cols / 2;
    const int64_t scale_cols = cols / kNvfp4PackedBlockSize;
    if (packed.dtype != TensorDType::U8 ||
        packed.shape != std::vector<int64_t>{rows, packed_cols} ||
        packed.bytes != static_cast<size_t>(rows) * static_cast<size_t>(packed_cols)) {
        throw std::runtime_error("invalid compressed-tensors NVFP4 packed payload: " + prefix);
    }
    if (scale.dtype != TensorDType::F8_E4M3 ||
        scale.shape != std::vector<int64_t>{rows, scale_cols} ||
        scale.bytes != static_cast<size_t>(rows) * static_cast<size_t>(scale_cols)) {
        throw std::runtime_error("invalid compressed-tensors NVFP4 scale payload: " + prefix);
    }
    if (!is_scalar_f32(global_scale)) {
        throw std::runtime_error("invalid compressed-tensors NVFP4 global scale: " + prefix);
    }

    PackedNvfp4Matrix result;
    result.rows = rows;
    result.cols = cols;
    result.packed.resize(packed.bytes);
    std::memcpy(result.packed.data(), packed.data, packed.bytes);
    result.block_scales.resize(scale.bytes);
    std::memcpy(result.block_scales.data(), scale.data, scale.bytes);
    result.global_scale = f32_from(global_scale.data);

    const std::string input_scale_name = module_name(prefix) + ".input_global_scale";
    if (repository.contains(input_scale_name)) {
        const HostTensorView input_scale = repository.tensor(input_scale_name);
        if (!is_scalar_f32(input_scale)) {
            throw std::runtime_error("invalid compressed-tensors NVFP4 input global scale: " + prefix);
        }
        result.input_global_scale = f32_from(input_scale.data);
    }

    result.validate();
    return result;
}

}
