#include "celeg/backend/cpu/weight_codec.hpp"

#include "celeg/checkpoint/weight_repository.hpp"
#include "celeg/model/weights/quantization.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace celeg {
namespace {

size_t checked_elements(const std::vector<int64_t>& shape) {
    size_t result = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) throw std::runtime_error("invalid tensor dimension");
        if (result > std::numeric_limits<size_t>::max() /
                    static_cast<size_t>(dim)) {
            throw std::overflow_error("tensor dimensions overflow");
        }
        result *= static_cast<size_t>(dim);
    }
    return result;
}

bool shape_matches(const std::vector<int64_t>& actual,
                   const std::vector<int64_t>& expected) {
    if (actual == expected) return true;
    if (expected.size() == 1) {
        size_t elements = 1;
        for (const int64_t dim : actual) elements *= static_cast<size_t>(dim);
        return elements == static_cast<size_t>(expected.front());
    }
    return actual.size() == 2 && expected.size() == 3 && expected[1] == 1 &&
           actual[0] == expected[0] && actual[1] == expected[2];
}

float fp16_to_float(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fu;
    uint32_t mantissa = bits & 0x03ffu;
    uint32_t result = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x03ffu;
            result = sign | static_cast<uint32_t>(127 - 15 - shift) << 23 |
                     mantissa << 13;
        }
    } else if (exponent == 31) {
        result = sign | 0x7f800000u | mantissa << 13;
    } else {
        result = sign | (exponent + (127 - 15)) << 23 | mantissa << 13;
    }
    return std::bit_cast<float>(result);
}

std::vector<float> read_vector(const HostTensorView& tensor,
                               const std::vector<int64_t>& expected,
                               const std::string& name) {
    if (!shape_matches(tensor.shape, expected)) {
        throw std::runtime_error("unexpected CPU tensor: " + name);
    }
    const size_t count = checked_elements(expected);
    std::vector<float> result(count);
    if (tensor.dtype == TensorDType::BF16 || tensor.dtype == TensorDType::F16) {
        if (tensor.bytes != count * sizeof(uint16_t)) {
            throw std::runtime_error("invalid 16-bit tensor bytes for " + name);
        }
        for (size_t i = 0; i < count; ++i) {
            uint16_t bits = 0;
            std::memcpy(&bits, tensor.data + i * sizeof(uint16_t), sizeof(bits));
            result[i] = tensor.dtype == TensorDType::BF16
                ? bf16_bits_to_float(bits) : fp16_to_float(bits);
        }
        return result;
    }
    if (tensor.dtype == TensorDType::F32) {
        if (tensor.bytes != count * sizeof(float)) {
            throw std::runtime_error("invalid F32 tensor bytes for " + name);
        }
        std::memcpy(result.data(), tensor.data, tensor.bytes);
        return result;
    }
    throw std::runtime_error("CPU vector requires F32, F16 or BF16 source: " + name);
}

} // namespace

CpuWeightCodec::CpuWeightCodec(IWeightRepository* source, CpuPackReader* reader,
                               CpuPackWriter* writer, size_t group_size)
    : source_(source), reader_(reader), writer_(writer), group_size_(group_size) {}

CpuLinearWeight CpuWeightCodec::matrix(
    const std::string& name, const std::vector<int64_t>& expected) const {
    if (reader_) return CpuLinearWeight::from_q4(reader_->read_q4_matrix(name));
    if (!source_) throw std::logic_error("CPU weight source is missing");
    const HostTensorView tensor = source_->tensor(name);
    if (tensor.shape != expected || expected.size() != 2) {
        throw std::runtime_error("unexpected CPU linear tensor: " + name);
    }
    if (tensor.dtype == TensorDType::Quantized) {
        const GgmlType type = ggml_type_from_block_encoding(tensor.block_encoding);
        if (type != GgmlType::Q4_K && type != GgmlType::Q6_K) {
            throw std::runtime_error("CPU GGUF supports only Q4_K/Q6_K linear tensor: " + name);
        }
        CpuGgufMatrix matrix;
        matrix.type = type;
        matrix.rows = static_cast<uint32_t>(expected[0]);
        matrix.cols = static_cast<uint32_t>(expected[1]);
        matrix.data = tensor.data;
        matrix.bytes = tensor.bytes;
        return CpuLinearWeight::from_gguf(matrix);
    }
    if (tensor.dtype != TensorDType::BF16) {
        throw std::runtime_error("Safetensors CPU linear tensor must be BF16: " + name);
    }
    Q4GroupMatrix packed = quantize_bf16_groupwise_q4(
        tensor.data, static_cast<size_t>(expected[0]),
        static_cast<size_t>(expected[1]), group_size_);
    try {
        if (writer_) writer_->add_q4_matrix(name, packed);
        return CpuLinearWeight::from_q4(std::move(packed));
    } catch (const std::exception& error) {
        throw std::runtime_error("invalid CPU linear weight '" + name + "': " + error.what());
    }
}

CpuLinearWeight CpuWeightCodec::concat(
    const std::string& synthetic,
    const std::vector<std::pair<std::string, std::vector<int64_t>>>& parts) const {
    if (reader_) return CpuLinearWeight::from_q4(reader_->read_q4_matrix(synthetic));
    if (!source_ || parts.empty()) throw std::logic_error("invalid CPU concat source");
    const int64_t cols = parts.front().second[1];
    size_t total_rows = 0;
    std::vector<HostTensorView> tensors;
    tensors.reserve(parts.size());
    bool quantized = true;
    for (const auto& [name, expected] : parts) {
        const HostTensorView tensor = source_->tensor(name);
        if (tensor.shape != expected || expected.size() != 2 || expected[1] != cols) {
            throw std::runtime_error("unexpected CPU concat tensor: " + name);
        }
        total_rows += static_cast<size_t>(expected[0]);
        quantized = quantized && tensor.dtype == TensorDType::Quantized;
        tensors.push_back(tensor);
    }
    if (quantized) {
        CpuLinearWeight result;
        result.rows = static_cast<uint32_t>(total_rows);
        result.cols = static_cast<uint32_t>(cols);
        for (size_t i = 0; i < tensors.size(); ++i) {
            const HostTensorView& tensor = tensors[i];
            const GgmlType type = ggml_type_from_block_encoding(tensor.block_encoding);
            if (type != GgmlType::Q4_K && type != GgmlType::Q6_K) {
                throw std::runtime_error("CPU GGUF concat supports only Q4_K/Q6_K: " + parts[i].first);
            }
            CpuGgufMatrix matrix;
            matrix.type = type;
            matrix.rows = static_cast<uint32_t>(tensor.shape[0]);
            matrix.cols = static_cast<uint32_t>(tensor.shape[1]);
            matrix.data = tensor.data;
            matrix.bytes = tensor.bytes;
            matrix.validate();
            result.segments.emplace_back(matrix);
        }
        result.validate();
        return result;
    }
    if (std::any_of(tensors.begin(), tensors.end(), [](const HostTensorView& tensor) {
            return tensor.dtype == TensorDType::Quantized;
        })) {
        throw std::runtime_error("CPU GGUF concat cannot mix quantized and dense tensors: " + synthetic);
    }
    std::vector<float> joined(total_rows * static_cast<size_t>(cols));
    size_t row_offset = 0;
    for (size_t i = 0; i < parts.size(); ++i) {
        const auto& [name, expected] = parts[i];
        if (tensors[i].dtype != TensorDType::BF16) {
            throw std::runtime_error("Safetensors CPU concat tensor must be BF16: " + name);
        }
        const std::vector<float> values = read_vector(tensors[i], expected, name);
        std::copy(values.begin(), values.end(), joined.begin() +
            static_cast<ptrdiff_t>(row_offset * static_cast<size_t>(cols)));
        row_offset += static_cast<size_t>(expected[0]);
    }
    Q4GroupMatrix packed = quantize_float_groupwise_q4(
        joined.data(), total_rows, static_cast<size_t>(cols), group_size_);
    if (writer_) writer_->add_q4_matrix(synthetic, packed);
    return CpuLinearWeight::from_q4(std::move(packed));
}

std::vector<float> CpuWeightCodec::vector(
    const std::string& name, const std::vector<int64_t>& expected) const {
    if (reader_) return reader_->read_bf16_vector(name);
    if (!source_) throw std::logic_error("CPU weight source is missing");
    const HostTensorView tensor = source_->tensor(name);
    std::vector<float> result = read_vector(tensor, expected, name);
    if (writer_) {
        if (tensor.dtype != TensorDType::BF16) {
            throw std::runtime_error("CPU pack vectors require BF16 source: " + name);
        }
        writer_->add_bf16_vector(name, tensor.data, result.size());
    }
    return result;
}

} // namespace celeg
