#include "celeg/checkpoint/tensor_codec.hpp"

#include "celeg/checkpoint/formats/gguf.hpp"
#include "celeg/quantization/scalars.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace celeg {
namespace {

std::string error_name(std::string_view name) {
    return name.empty() ? std::string("tensor") : std::string(name);
}

void require_bytes(const HostTensorView& tensor, std::size_t expected,
                   std::string_view name) {
    if (tensor.bytes != expected) {
        throw std::runtime_error("invalid tensor byte count for " + error_name(name));
    }
    if (expected != 0 && tensor.data == nullptr) {
        throw std::runtime_error("tensor data is null for " + error_name(name));
    }
}

std::size_t checked_byte_count(std::size_t count, std::size_t element_size,
                               std::string_view name) {
    if (count > std::numeric_limits<std::size_t>::max() / element_size) {
        throw std::runtime_error("tensor byte count overflows for " + error_name(name));
    }
    return count * element_size;
}

}

std::size_t tensor_element_count(std::span<const std::int64_t> shape,
                                 std::string_view context) {
    if (shape.empty()) {
        throw std::runtime_error("invalid tensor shape for " + error_name(context));
    }
    std::size_t count = 1;
    for (const std::int64_t dimension : shape) {
        if (dimension <= 0 || count > std::numeric_limits<std::size_t>::max() /
                                  static_cast<std::size_t>(dimension)) {
            throw std::runtime_error("invalid tensor shape for " + error_name(context));
        }
        count *= static_cast<std::size_t>(dimension);
    }
    return count;
}

bool tensor_shape_matches(std::span<const std::int64_t> actual,
                          std::span<const std::int64_t> expected) {
    return actual.size() == expected.size() &&
           std::equal(actual.begin(), actual.end(), expected.begin());
}

bool tensor_shape_is_compatible(std::span<const std::int64_t> actual,
                                std::span<const std::int64_t> expected) {
    if (tensor_shape_matches(actual, expected)) return true;
    if (expected.size() == 1) {
        return tensor_element_count(actual, "tensor shape") ==
               static_cast<std::size_t>(expected.front());
    }
    return actual.size() == 2 && expected.size() == 3 && expected[1] == 1 &&
           actual[0] == expected[0] && actual[1] == expected[2];
}

std::vector<float> decode_tensor_f32(const HostTensorView& tensor,
                                     std::span<const std::int64_t> expected,
                                     std::string_view name) {
    if (!tensor_shape_is_compatible(tensor.shape, expected)) {
        throw std::runtime_error("unexpected tensor shape for " + error_name(name));
    }
    const std::size_t count = tensor_element_count(expected, name);
    std::vector<float> result(count);
    if (tensor.dtype == TensorDType::F32) {
        require_bytes(tensor, checked_byte_count(count, sizeof(float), name), name);
        std::memcpy(result.data(), tensor.data, tensor.bytes);
        return result;
    }
    if (tensor.dtype == TensorDType::BF16 || tensor.dtype == TensorDType::F16) {
        require_bytes(tensor, checked_byte_count(count, sizeof(std::uint16_t), name), name);
        for (std::size_t index = 0; index < count; ++index) {
            std::uint16_t bits = 0;
            std::memcpy(&bits, tensor.data + index * sizeof(bits), sizeof(bits));
            result[index] = tensor.dtype == TensorDType::BF16
                ? bf16_bits_to_float(bits) : fp16_bits_to_float(bits);
        }
        return result;
    }
    if (tensor.dtype == TensorDType::Quantized && expected.size() == 2) {
        GgmlMatrixView matrix;
        matrix.type = ggml_type_from_block_encoding(tensor.block_encoding);
        matrix.rows = static_cast<std::uint32_t>(expected[0]);
        matrix.cols = static_cast<std::uint32_t>(expected[1]);
        matrix.data = tensor.data;
        matrix.bytes = tensor.bytes;
        matrix.validate();
        for (std::size_t row = 0; row < matrix.rows; ++row) {
            ggml_decode_row(matrix, row, result.data() + row * matrix.cols);
        }
        return result;
    }
    throw std::runtime_error("unsupported tensor dtype for " + error_name(name));
}

}
