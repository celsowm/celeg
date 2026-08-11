#include "celeg/checkpoint/packed_int4.hpp"

#include "celeg/model/weights/quantization.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace celeg {
namespace {

size_t elements(const std::vector<int64_t>& shape, const char* label) {
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

int64_t read_i64(const std::byte* source) {
    int64_t value = 0;
    std::memcpy(&value, source, sizeof(value));
    return value;
}

int signed_nibble(uint8_t packed, int half) {
    const int value = (packed >> (half * 4)) & 0x0f;
    return value >= 8 ? value - 16 : value;
}

} // namespace

void PackedInt4Matrix::validate() const {
    if (rows <= 0 || cols <= 0 || group_size <= 0 || cols % group_size != 0 ||
        values.size() != static_cast<size_t>(rows) * packed_cols() ||
        scales.size() != static_cast<size_t>(rows) * groups_per_row()) {
        throw std::runtime_error("invalid packed INT4 matrix");
    }
}

bool has_packed_int4_matrix(const IWeightRepository& repository,
                            std::string_view name) {
    const std::string prefix(name);
    if (repository.contains(prefix) || !repository.contains(prefix + "_packed") ||
        !repository.contains(prefix + "_scale") || !repository.contains(prefix + "_shape")) {
        return false;
    }
    const HostTensorView packed = repository.tensor(prefix + "_packed");
    const HostTensorView scale = repository.tensor(prefix + "_scale");
    const HostTensorView shape = repository.tensor(prefix + "_shape");
    if (packed.shape.size() != 2 || scale.shape.size() != 2 ||
        scale.shape[0] != packed.shape[0] || shape.dtype != TensorDType::I64 ||
        shape.bytes != 2 * sizeof(int64_t)) return false;
    int64_t dimensions[2] = {};
    std::memcpy(dimensions, shape.data, sizeof(dimensions));
    return dimensions[0] > 0 && dimensions[1] > 0 &&
        packed.shape[1] == (dimensions[1] + 7) / 8 &&
        scale.shape[1] == (dimensions[1] + 31) / 32;
}

PackedInt4Matrix load_packed_int4_matrix(
    const IWeightRepository& repository, std::string_view name,
    const std::vector<int64_t>& expected_shape) {
    if (expected_shape.size() != 2 || expected_shape[0] <= 0 || expected_shape[1] <= 0) {
        throw std::invalid_argument("packed INT4 weight requires a positive matrix shape");
    }
    const int rows = static_cast<int>(expected_shape[0]);
    const int cols = static_cast<int>(expected_shape[1]);
    constexpr int group_size = 32;
    if (cols % group_size != 0) {
        throw std::invalid_argument("packed INT4 weight width must be group aligned");
    }

    const std::string prefix(name);
    const HostTensorView packed = repository.tensor(prefix + "_packed");
    const HostTensorView scale = repository.tensor(prefix + "_scale");
    const HostTensorView shape = repository.tensor(prefix + "_shape");
    const size_t packed_columns = (static_cast<size_t>(cols) + 7) / 8;
    const size_t source_packed_bytes = static_cast<size_t>(rows) * packed_columns *
        sizeof(uint32_t);
    if (packed.dtype != TensorDType::I32 ||
        packed.shape != std::vector<int64_t>{rows, static_cast<int64_t>(packed_columns)} ||
        packed.bytes != source_packed_bytes) {
        throw std::runtime_error("invalid compressed-tensors INT4 packed payload: " + prefix);
    }
    const size_t groups = static_cast<size_t>(cols) / group_size;
    if (scale.dtype != TensorDType::BF16 ||
        scale.shape != std::vector<int64_t>{rows, static_cast<int64_t>(groups)} ||
        scale.bytes != static_cast<size_t>(rows) * groups * sizeof(uint16_t)) {
        throw std::runtime_error("invalid compressed-tensors INT4 scale payload: " + prefix);
    }
    if (shape.dtype != TensorDType::I64 || shape.shape != std::vector<int64_t>{2} ||
        shape.bytes != 2 * sizeof(int64_t) || read_i64(shape.data) != rows ||
        read_i64(shape.data + sizeof(int64_t)) != cols) {
        throw std::runtime_error("compressed-tensors INT4 shape sidecar disagrees: " + prefix);
    }

    PackedInt4Matrix result;
    result.rows = rows;
    result.cols = cols;
    result.group_size = group_size;
    result.values.resize(static_cast<size_t>(rows) * result.packed_cols());
    // Each I32 packs eight nibbles.  The host is little-endian for the
    // safetensors payload, so copying its bytes preserves low-nibble order.
    std::memcpy(result.values.data(), packed.data, result.values.size());
    result.scales.resize(static_cast<size_t>(rows) * groups);
    for (size_t index = 0; index < result.scales.size(); ++index) {
        uint16_t bits = 0;
        std::memcpy(&bits, scale.data + index * sizeof(bits), sizeof(bits));
        result.scales[index] = bf16_bits_to_float(bits);
    }
    result.validate();
    return result;
}

std::vector<float> dequantize_packed_int4(const PackedInt4Matrix& matrix) {
    matrix.validate();
    std::vector<float> result(static_cast<size_t>(matrix.rows) * matrix.cols);
    for (int row = 0; row < matrix.rows; ++row) {
        const uint8_t* source = matrix.values.data() +
            static_cast<size_t>(row) * matrix.packed_cols();
        for (int column = 0; column < matrix.cols; ++column) {
            const int q = signed_nibble(source[column / 2], column & 1);
            const float scale = matrix.scales[static_cast<size_t>(row) *
                matrix.groups_per_row() + static_cast<size_t>(column / matrix.group_size)];
            result[static_cast<size_t>(row) * matrix.cols + column] =
                static_cast<float>(q) * scale;
        }
    }
    return result;
}

} // namespace celeg
