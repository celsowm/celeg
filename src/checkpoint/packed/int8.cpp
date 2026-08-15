#include "celeg/checkpoint/packed/int8.hpp"

#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace celeg {
namespace {

size_t element_count(const std::vector<int64_t>& shape, const char* label) {
    size_t result = 1;
    for (const int64_t dimension : shape) {
        if (dimension <= 0 || result > std::numeric_limits<size_t>::max() /
                                   static_cast<size_t>(dimension)) {
            throw std::runtime_error(std::string("invalid ") + label + " shape");
        }
        result *= static_cast<size_t>(dimension);
    }
    return result;
}

float bf16_to_float(const std::byte* source) {
    uint16_t bits = 0;
    std::memcpy(&bits, source, sizeof(bits));
    return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
}

uint32_t read_u32(const std::byte* source) {
    uint32_t value = 0;
    std::memcpy(&value, source, sizeof(value));
    return value;
}

int64_t read_i64(const std::byte* source) {
    int64_t value = 0;
    std::memcpy(&value, source, sizeof(value));
    return value;
}

}

void PackedInt8Matrix::validate() const {
    if (rows <= 0 || cols <= 0 || values.size() != static_cast<size_t>(rows) * cols ||
        scales.size() != static_cast<size_t>(rows)) {
        throw std::runtime_error("invalid packed INT8 matrix");
    }
}

bool has_packed_int8_matrix(const IWeightRepository& repository,
                            std::string_view name) {
    const std::string prefix(name);
    if (repository.contains(prefix) || !repository.contains(prefix + "_packed") ||
        !repository.contains(prefix + "_scale") || !repository.contains(prefix + "_shape")) {
        return false;
    }
    const HostTensorView packed = repository.tensor(prefix + "_packed");
    const HostTensorView scale = repository.tensor(prefix + "_scale");
    const HostTensorView shape = repository.tensor(prefix + "_shape");
    if (packed.shape.size() != 2 || scale.shape != std::vector<int64_t>{packed.shape[0], 1} ||
        shape.dtype != TensorDType::I64 || shape.bytes != 2 * sizeof(int64_t)) return false;
    int64_t dimensions[2] = {};
    std::memcpy(dimensions, shape.data, sizeof(dimensions));
    return dimensions[0] > 0 && dimensions[1] > 0 &&
        packed.shape[1] == (dimensions[1] + 3) / 4;
}

PackedInt8Matrix load_packed_int8_matrix(
    const IWeightRepository& repository, std::string_view name,
    const std::vector<int64_t>& expected_shape) {
    if (expected_shape.size() != 2 || expected_shape[0] <= 0 || expected_shape[1] <= 0) {
        throw std::invalid_argument("packed INT8 weight requires a positive matrix shape");
    }
    const int rows = static_cast<int>(expected_shape[0]);
    const int cols = static_cast<int>(expected_shape[1]);
    const std::string prefix(name);
    const std::string packed_name = prefix + "_packed";
    const std::string scale_name = prefix + "_scale";
    const std::string shape_name = prefix + "_shape";
    if (!repository.contains(packed_name) || !repository.contains(scale_name) ||
        !repository.contains(shape_name)) {
        throw std::out_of_range("incomplete compressed-tensors INT8 weight: " + prefix);
    }

    const HostTensorView packed = repository.tensor(packed_name);
    const HostTensorView scale = repository.tensor(scale_name);
    const HostTensorView shape = repository.tensor(shape_name);
    const size_t packed_cols = (static_cast<size_t>(cols) + 3) / 4;
    if (packed.dtype != TensorDType::I32 || packed.shape !=
            std::vector<int64_t>{rows, static_cast<int64_t>(packed_cols)} ||
        packed.bytes != static_cast<size_t>(rows) * packed_cols * sizeof(uint32_t)) {
        throw std::runtime_error("invalid compressed-tensors INT8 packed payload: " + prefix);
    }
    if (scale.dtype != TensorDType::BF16 || scale.shape != std::vector<int64_t>{rows, 1} ||
        scale.bytes != static_cast<size_t>(rows) * sizeof(uint16_t)) {
        throw std::runtime_error("invalid compressed-tensors INT8 scale payload: " + prefix);
    }
    if (shape.dtype != TensorDType::I64 || shape.shape != std::vector<int64_t>{2} ||
        shape.bytes != 2 * sizeof(int64_t) || read_i64(shape.data) != rows ||
        read_i64(shape.data + sizeof(int64_t)) != cols) {
        throw std::runtime_error("compressed-tensors INT8 shape sidecar disagrees: " + prefix);
    }

    PackedInt8Matrix result;
    result.rows = rows;
    result.cols = cols;
    result.values.resize(static_cast<size_t>(rows) * cols);
    result.scales.resize(static_cast<size_t>(rows));
    for (int row = 0; row < rows; ++row) {
        result.scales[static_cast<size_t>(row)] =
            bf16_to_float(scale.data + static_cast<size_t>(row) * sizeof(uint16_t));
        for (size_t word = 0; word < packed_cols; ++word) {
            const uint32_t packed_word = read_u32(
                packed.data + (static_cast<size_t>(row) * packed_cols + word) * 4);
            for (size_t byte = 0; byte < 4; ++byte) {
                const size_t column = word * 4 + byte;
                if (column >= static_cast<size_t>(cols)) break;
                const uint8_t unsigned_value = static_cast<uint8_t>(
                    (packed_word >> (byte * 8)) & 0xffu);
                result.values[static_cast<size_t>(row) * cols + column] =
                    static_cast<int8_t>(static_cast<int>(unsigned_value) - 128);
            }
        }
    }
    result.validate();
    return result;
}

}
