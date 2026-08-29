#include "detail.hpp"

#include "celeg/backend/cpu/gguf.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace celeg::metal_model_detail {

std::string ns_string(NSString* value) {
    return value ? std::string(value.UTF8String) : std::string{};
}

float bf16_to_float(uint16_t bits) {
    return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
}

float fp16_to_float(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1fu;
    const uint32_t fraction = bits & 0x3ffu;
    uint32_t value = 0;
    if (exponent == 0) {
        if (fraction == 0) return std::bit_cast<float>(sign);
        uint32_t normalized = fraction;
        int shift = 0;
        while ((normalized & 0x400u) == 0) {
            normalized <<= 1;
            ++shift;
        }
        normalized &= 0x3ffu;
        value = sign | static_cast<uint32_t>(127 - 15 - shift) << 23 |
                normalized << 13;
    } else if (exponent == 0x1fu) {
        value = sign | 0x7f800000u | fraction << 13;
    } else {
        value = sign | (exponent + 112u) << 23 | fraction << 13;
    }
    return std::bit_cast<float>(value);
}

size_t tensor_elements(const HostTensorView& view) {
    size_t count = 1;
    for (const int64_t dimension : view.shape) {
        if (dimension <= 0 || count > std::numeric_limits<size_t>::max() /
                                  static_cast<size_t>(dimension)) {
            throw std::runtime_error("invalid Metal tensor shape");
        }
        count *= static_cast<size_t>(dimension);
    }
    return count;
}

std::vector<float> tensor_values(const HostTensorView& view,
                                 std::span<const int64_t> expected,
                                 const std::string& name) {
    if (view.shape.size() != expected.size() ||
        !std::equal(view.shape.begin(), view.shape.end(), expected.begin())) {
        throw std::runtime_error("unexpected Metal tensor shape: " + name);
    }
    const size_t count = tensor_elements(view);
    std::vector<float> values(count);
    if (view.dtype == TensorDType::F32) {
        if (view.bytes != count * sizeof(float)) {
            throw std::runtime_error("invalid Metal F32 tensor: " + name);
        }
        std::memcpy(values.data(), view.data, view.bytes);
        return values;
    }
    if (view.dtype == TensorDType::BF16 || view.dtype == TensorDType::F16) {
        if (view.bytes != count * sizeof(uint16_t)) {
            throw std::runtime_error("invalid Metal 16-bit tensor: " + name);
        }
        for (size_t index = 0; index < count; ++index) {
            uint16_t bits = 0;
            std::memcpy(&bits, view.data + index * sizeof(bits), sizeof(bits));
            values[index] = view.dtype == TensorDType::BF16
                ? bf16_to_float(bits) : fp16_to_float(bits);
        }
        return values;
    }
    if (view.dtype == TensorDType::Quantized && expected.size() == 2) {
        CpuGgufMatrix matrix;
        matrix.type = ggml_type_from_block_encoding(view.block_encoding);
        matrix.rows = static_cast<uint32_t>(expected[0]);
        matrix.cols = static_cast<uint32_t>(expected[1]);
        matrix.data = view.data;
        matrix.bytes = view.bytes;
        matrix.validate();
        for (size_t row = 0; row < matrix.rows; ++row) {
            cpu_gguf_dequantize_row(matrix, row,
                                    values.data() + row * matrix.cols);
        }
        return values;
    }
    throw std::runtime_error("unsupported Metal tensor type: " + name);
}

const TensorRequest& request_for(std::span<const TensorRequest> requests,
                                 TensorRole role, int layer, int expert) {
    for (const TensorRequest& request : requests) {
        if (request.role == role && request.layer == layer && request.expert == expert) {
            return request;
        }
    }
    throw std::runtime_error("Metal weight request is missing: " +
                             std::string(tensor_role_name(role)));
}

std::string request_name(std::span<const TensorRequest> requests,
                         TensorRole role, int layer, int expert) {
    const TensorRequest& request = request_for(requests, role, layer, expert);
    if (!request.source_name) {
        throw std::runtime_error("Metal weight request was not resolved: " +
                                 std::string(tensor_role_name(role)));
    }
    return *request.source_name;
}

}

namespace celeg {

using metal_model_detail::tensor_values;
using metal_model_detail::request_for;

id<MTLBuffer> MetalModel::Impl::buffer(const std::vector<float>& values) const {
        id<MTLBuffer> result = [device newBufferWithBytes:values.data()
                                                    length:values.size() * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        if (!result) throw std::runtime_error("Metal buffer allocation failed");
        const_cast<MetalModel::Impl*>(this)->execution_metrics.resident_weight_bytes +=
            values.size() * sizeof(float);
        return result;
    }


id<MTLBuffer> MetalModel::Impl::zero_buffer(size_t bytes) const {
        id<MTLBuffer> result = [device newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
        if (!result) throw std::runtime_error("Metal state allocation failed");
        const_cast<MetalModel::Impl*>(this)->execution_metrics.resident_state_bytes += bytes;
        std::memset(result.contents, 0, bytes);
        return result;
    }


id<MTLBuffer> MetalModel::Impl::raw_buffer(const std::byte* data, size_t bytes) const {
        id<MTLBuffer> result = [device newBufferWithBytes:data
                                                    length:bytes
                                                   options:MTLResourceStorageModeShared];
        if (!result) throw std::runtime_error("Metal raw buffer allocation failed");
        const_cast<MetalModel::Impl*>(this)->execution_metrics.resident_weight_bytes += bytes;
        return result;
    }


std::vector<float> MetalModel::Impl::load_vector_values(TensorRole role, int layer, int width,
                                          bool allow_missing) const {
        const TensorRequest* selected = nullptr;
        for (const TensorRequest& request : model.weight_plan.requests) {
            if (request.role == role && request.layer == layer && request.expert < 0) {
                selected = &request;
                break;
            }
        }
        if (!selected) {
            if (!allow_missing) throw std::runtime_error("Metal vector request is missing");
            return std::vector<float>(static_cast<size_t>(width), 1.0f);
        }
        if (!selected->source_name) {
            throw std::runtime_error("Metal vector request was not resolved");
        }
        const std::string& name = *selected->source_name;
        std::vector<float> values = tensor_values(repository->tensor(name),
            std::span<const int64_t>(selected->expected_shape.data(),
                                     selected->expected_shape.size()), name);
        if (values.size() != static_cast<size_t>(width)) {
            throw std::runtime_error("Metal vector width mismatch: " + name);
        }
        if (selected->norm_weight_kind == NormWeightKind::OnePlusScale) {
            for (float& value : values) value += 1.0f;
        }
        return values;
    }


id<MTLBuffer> MetalModel::Impl::load_vector(TensorRole role, int layer, int width,
                              bool allow_missing) const {
        return buffer(load_vector_values(role, layer, width, allow_missing));
    }


MetalModel::Impl::Linear MetalModel::Impl::load_linear(TensorRole role, int layer, int rows, int cols) const {
        const TensorRequest& request = request_for(model.weight_plan.requests, role, layer);
        if (!request.source_name) throw std::runtime_error("Metal matrix request was not resolved");
        const std::string& name = *request.source_name;
        return load_linear_source(name, rows, cols);
    }


MetalModel::Impl::Linear MetalModel::Impl::load_linear_source(const std::string& name, int rows, int cols) const {
        const HostTensorView view = repository->tensor(name);
        const std::vector<int64_t> expected_values{rows, cols};
        const std::span<const int64_t> expected(expected_values.data(),
                                                 expected_values.size());
        if (view.shape.size() != expected.size() ||
            !std::equal(view.shape.begin(), view.shape.end(), expected.begin())) {
            throw std::runtime_error("unexpected Metal tensor shape: " + name);
        }
        if (view.dtype == TensorDType::Quantized) {
            const GgmlType type = ggml_type_from_block_encoding(view.block_encoding);
            if ((type == GgmlType::Q4_0 || type == GgmlType::Q4_K ||
                 type == GgmlType::Q5_K || type == GgmlType::Q6_K ||
                 type == GgmlType::Q8_0) &&
                cols % static_cast<int>(ggml_type_trait(type).block_size) == 0) {
                const GgmlTypeTrait trait = ggml_type_trait(type);
                const size_t row_bytes = static_cast<size_t>(cols) /
                    static_cast<size_t>(trait.block_size) *
                    static_cast<size_t>(trait.type_size);
                if (view.bytes != static_cast<size_t>(rows) * row_bytes) {
                    throw std::runtime_error("invalid Metal quantized tensor bytes: " + name);
                }
                return {raw_buffer(view.data, view.bytes),
                        static_cast<uint32_t>(rows), static_cast<uint32_t>(cols),
                        static_cast<uint32_t>(row_bytes),
                        type == GgmlType::Q4_0 ? LinearStorage::Q4_0
                        : type == GgmlType::Q4_K ? LinearStorage::Q4K
                        : type == GgmlType::Q5_K ? LinearStorage::Q5K
                        : type == GgmlType::Q6_K ? LinearStorage::Q6K
                                               : LinearStorage::Q8_0};
            }
        }
        if (view.dtype == TensorDType::F16 || view.dtype == TensorDType::BF16) {
            const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) *
                sizeof(uint16_t);
            if (view.bytes != bytes) {
                throw std::runtime_error("invalid Metal 16-bit tensor bytes: " + name);
            }
            return {raw_buffer(view.data, bytes), static_cast<uint32_t>(rows),
                    static_cast<uint32_t>(cols), 0,
                    view.dtype == TensorDType::F16 ? LinearStorage::Float16
                                                   : LinearStorage::BFloat16};
        }
        std::vector<float> values = tensor_values(view, expected, name);
        if (values.size() != static_cast<size_t>(rows) * cols) {
            throw std::runtime_error("Metal matrix dimensions mismatch: " + name);
        }
        return {buffer(values), static_cast<uint32_t>(rows), static_cast<uint32_t>(cols),
                0, LinearStorage::Float32};
    }


id<MTLBuffer> MetalModel::Impl::load_conv_kernel(int layer, int width, int cache_length) const {
        const TensorRequest& request = request_for(model.weight_plan.requests,
                                                   TensorRole::ShortConvKernel, layer);
        if (!request.source_name) throw std::runtime_error("Metal convolution request was not resolved");
        const std::string& name = *request.source_name;
        std::vector<float> channel_major = tensor_values(repository->tensor(name),
            std::span<const int64_t>(request.expected_shape.data(), request.expected_shape.size()), name);
        std::vector<float> tap_major(channel_major.size());
        for (int tap = 0; tap < cache_length; ++tap) {
            for (int channel = 0; channel < width; ++channel) {
                tap_major[static_cast<size_t>(tap) * width + channel] =
                    channel_major[static_cast<size_t>(channel) * cache_length + tap];
            }
        }
        return buffer(tap_major);
    }


}
