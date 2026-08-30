#include "celeg/backend/cpu/weight_codec.hpp"
#include "celeg/checkpoint/formats/gguf.hpp"

#include "celeg/checkpoint/weight_repository.hpp"
#include "celeg/checkpoint/packed/int8.hpp"
#include "celeg/checkpoint/packed/int4.hpp"
#include "celeg/checkpoint/tensor_codec.hpp"
#include "celeg/model/weights/quantization.hpp"

#include <algorithm>
#include <stdexcept>

namespace celeg {
namespace {

GgmlMatrixView gguf_matrix(const HostTensorView& tensor,
                          const std::string& name) {
    const GgmlType type = ggml_type_from_block_encoding(tensor.block_encoding);
    if (!gguf_type_dequantizable(type)) {
        throw std::runtime_error("unsupported CPU GGUF linear quantization: " +
            name + " (" + std::string(ggml_type_name(type)) + ")");
    }
    GgmlMatrixView matrix;
    matrix.type = type;
    matrix.rows = static_cast<uint32_t>(tensor.shape[0]);
    matrix.cols = static_cast<uint32_t>(tensor.shape[1]);
    matrix.data = tensor.data;
    matrix.bytes = tensor.bytes;
    matrix.validate();
    return matrix;
}

}

CpuWeightCodec::CpuWeightCodec(IWeightRepository* source, CpuPackReader* reader,
                               CpuPackWriter* writer, size_t group_size)
    : source_(source), reader_(reader), writer_(writer), group_size_(group_size) {}

CpuLinearWeight CpuWeightCodec::matrix(
    const std::string& name, const std::vector<int64_t>& expected) const {
    if (reader_) return CpuLinearWeight::from_q4(reader_->read_q4_matrix(name));
    if (!source_) throw std::logic_error("CPU weight source is missing");
    if (has_packed_int8_matrix(*source_, name)) {
        const PackedInt8Matrix packed = load_packed_int8_matrix(*source_, name, expected);
        CpuInt8Matrix matrix;
        matrix.rows = static_cast<uint32_t>(packed.rows);
        matrix.cols = static_cast<uint32_t>(packed.cols);
        *matrix.values = packed.values;
        *matrix.scales = packed.scales;
        return CpuLinearWeight::from_int8(std::move(matrix));
    }
    if (has_packed_int4_matrix(*source_, name)) {
        const PackedInt4Matrix packed = load_packed_int4_matrix(*source_, name, expected);
        const std::vector<float> values = dequantize_packed_int4(packed);
        Q4GroupMatrix repacked = quantize_float_groupwise_q4(
            values.data(), static_cast<size_t>(packed.rows),
            static_cast<size_t>(packed.cols), group_size_);
        if (writer_) writer_->add_q4_matrix(name, repacked);
        return CpuLinearWeight::from_q4(std::move(repacked));
    }
    const HostTensorView tensor = source_->tensor(name);
    if (tensor.shape != expected || expected.size() != 2) {
        throw std::runtime_error("unexpected CPU linear tensor: " + name +
            " expected=" + std::to_string(expected.size() > 0 ? expected[0] : 0) +
            "x" + std::to_string(expected.size() > 1 ? expected[1] : 0) +
            " actual=" + std::to_string(tensor.shape.size() > 0 ? tensor.shape[0] : 0) +
            "x" + std::to_string(tensor.shape.size() > 1 ? tensor.shape[1] : 0));
    }
    if (tensor.dtype == TensorDType::Quantized) {
        const GgmlMatrixView matrix = gguf_matrix(tensor, name);
        /// Every GGUF quant type now has a native dot kernel, so route it
        /// natively whenever the column count is a multiple of 256. Anything
        /// else (non-256 width, or a type without a native kernel) is
        /// dequantized and repacked into groupwise Q4, which keeps mixed-quant
        /// GGUF files (e.g. a Q4_0 file that sprinkles in Q4_1 tensors)
        /// loadable without special casing.
        if (!gguf_type_is_native_dot(matrix.type) || (matrix.cols % 256) != 0) {
            const std::vector<float> values = decode_tensor_f32(
                tensor, expected, name);
            return CpuLinearWeight::from_q4(quantize_float_groupwise_q4(
                values.data(), matrix.rows, matrix.cols, group_size_));
        }
        return CpuLinearWeight::from_ggml(matrix);
    }
    /// F16 and F32 both widen to float first; the neutral tensor codec decodes
    /// either representation before quantization.
    if (tensor.dtype == TensorDType::F32 || tensor.dtype == TensorDType::F16) {
        const std::vector<float> values = decode_tensor_f32(tensor, expected, name);
        Q4GroupMatrix packed = quantize_float_groupwise_q4(
            values.data(), static_cast<size_t>(expected[0]),
            static_cast<size_t>(expected[1]), group_size_);
        try {
            if (writer_) writer_->add_q4_matrix(name, packed);
            return CpuLinearWeight::from_q4(std::move(packed));
        } catch (const std::exception& error) {
            throw std::runtime_error("invalid CPU linear weight '" + name + "': " + error.what());
        }
    }
    if (tensor.dtype != TensorDType::BF16) {
        throw std::runtime_error("CPU linear tensor must be BF16, F16, F32 or a "
                                 "supported GGUF quantization: " + name);
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
    if (std::all_of(parts.begin(), parts.end(), [&](const auto& part) {
            return has_packed_int8_matrix(*source_, part.first);
        })) {
        const int64_t cols = parts.front().second.at(1);
        CpuLinearWeight result;
        result.cols = static_cast<uint32_t>(cols);
        for (const auto& [name, expected] : parts) {
            const PackedInt8Matrix packed = load_packed_int8_matrix(*source_, name, expected);
            if (packed.cols != cols) throw std::runtime_error("packed CPU concat width mismatch");
            CpuInt8Matrix matrix;
            matrix.rows = static_cast<uint32_t>(packed.rows);
            matrix.cols = static_cast<uint32_t>(packed.cols);
            *matrix.values = packed.values;
            *matrix.scales = packed.scales;
            result.rows += matrix.rows;
            result.segments.emplace_back(std::move(matrix));
        }
        result.validate();
        return result;
    }
    if (std::all_of(parts.begin(), parts.end(), [&](const auto& part) {
            return has_packed_int4_matrix(*source_, part.first);
        })) {
        const int64_t cols = parts.front().second.at(1);
        size_t total_rows = 0;
        std::vector<float> joined;
        for (const auto& [name, expected] : parts) {
            const PackedInt4Matrix packed = load_packed_int4_matrix(*source_, name, expected);
            if (packed.cols != cols) throw std::runtime_error("packed INT4 concat width mismatch");
            const std::vector<float> values = dequantize_packed_int4(packed);
            joined.insert(joined.end(), values.begin(), values.end());
            total_rows += static_cast<size_t>(packed.rows);
        }
        Q4GroupMatrix repacked = quantize_float_groupwise_q4(
            joined.data(), total_rows, static_cast<size_t>(cols), group_size_);
        if (writer_) writer_->add_q4_matrix(synthetic, repacked);
        return CpuLinearWeight::from_q4(std::move(repacked));
    }
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
        bool needs_repack = false;
        std::vector<GgmlMatrixView> matrices;
        matrices.reserve(tensors.size());
        for (size_t i = 0; i < tensors.size(); ++i) {
            matrices.push_back(gguf_matrix(tensors[i], parts[i].first));
            needs_repack = needs_repack ||
                !gguf_type_is_native_dot(matrices.back().type) ||
                (matrices.back().cols % 256) != 0;
        }
        if (needs_repack) {
            std::vector<float> joined(total_rows * static_cast<size_t>(cols));
            size_t row_offset = 0;
            for (size_t i = 0; i < matrices.size(); ++i) {
                const GgmlMatrixView& matrix = matrices[i];
                const std::vector<float> values = decode_tensor_f32(
                    tensors[i], parts[i].second, parts[i].first);
                std::copy(values.begin(), values.end(), joined.begin() +
                    static_cast<ptrdiff_t>(row_offset * static_cast<size_t>(cols)));
                row_offset += matrix.rows;
            }
            return CpuLinearWeight::from_q4(quantize_float_groupwise_q4(
                joined.data(), total_rows, static_cast<size_t>(cols), group_size_));
        }
        CpuLinearWeight result;
        result.rows = static_cast<uint32_t>(total_rows);
        result.cols = static_cast<uint32_t>(cols);
        for (const GgmlMatrixView& matrix : matrices) {
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
        if (tensors[i].dtype != TensorDType::BF16 && tensors[i].dtype != TensorDType::F32 &&
            tensors[i].dtype != TensorDType::F16) {
            throw std::runtime_error("CPU concat tensor must be BF16, F16 or F32: " + name);
        }
        const std::vector<float> values = decode_tensor_f32(tensors[i], expected, name);
        std::copy(values.begin(), values.end(), joined.begin() +
            static_cast<ptrdiff_t>(row_offset * static_cast<size_t>(cols)));
        row_offset += static_cast<size_t>(expected[0]);
    }
    Q4GroupMatrix packed = quantize_float_groupwise_q4(
        joined.data(), total_rows, static_cast<size_t>(cols), group_size_);
    if (writer_) writer_->add_q4_matrix(synthetic, packed);
    return CpuLinearWeight::from_q4(std::move(packed));
}

std::vector<CpuLinearWeight> CpuWeightCodec::packed_matrices(
    const std::string& name, const std::vector<int64_t>& expected) const {
    if (!source_ || expected.size() != 3 || expected[0] <= 0 ||
        expected[1] <= 0 || expected[2] <= 0) {
        throw std::logic_error("packed CPU matrix source is invalid");
    }
    const HostTensorView tensor = source_->tensor(name);
    if (tensor.shape != expected || tensor.dtype != TensorDType::BF16) {
        throw std::runtime_error("unexpected packed CPU tensor: " + name);
    }
    const int64_t entries = expected[0];
    const size_t rows = static_cast<size_t>(expected[1]);
    const size_t cols = static_cast<size_t>(expected[2]);
    const std::vector<float> values = decode_tensor_f32(tensor, expected, name);
    std::vector<CpuLinearWeight> result;
    result.reserve(static_cast<size_t>(entries));
    for (int64_t entry = 0; entry < entries; ++entry) {
        const float* matrix = values.data() + static_cast<size_t>(entry) * rows * cols;
        Q4GroupMatrix packed = quantize_float_groupwise_q4(matrix, rows, cols, group_size_);
        result.push_back(CpuLinearWeight::from_q4(std::move(packed)));
    }
    return result;
}

std::vector<float> CpuWeightCodec::vector(
    const std::string& name, const std::vector<int64_t>& expected) const {
    if (reader_) return reader_->read_bf16_vector(name);
    if (!source_) throw std::logic_error("CPU weight source is missing");
    const HostTensorView tensor = source_->tensor(name);
    std::vector<float> result = decode_tensor_f32(tensor, expected, name);
    if (writer_) {
        if (tensor.dtype == TensorDType::BF16) {
            writer_->add_bf16_vector(name, tensor.data, result.size());
        } else {
            std::vector<uint16_t> bf16(result.size());
            for (size_t i = 0; i < result.size(); ++i) {
                bf16[i] = float_to_bf16_bits(result[i]);
            }
            writer_->add_bf16_vector(
                name, reinterpret_cast<const std::byte*>(bf16.data()), bf16.size());
        }
    }
    return result;
}

}
