#include "detail.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace celeg::metal_model_detail {

std::string ns_string(NSString* value) {
    return value ? std::string(value.UTF8String) : std::string{};
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

using metal_model_detail::request_for;

id<MTLBuffer> MetalModel::Impl::immutable_buffer(const void* data, size_t bytes) const {
    if (options.storage_mode != MetalStorageMode::Private) {
        id<MTLBuffer> result = [device newBufferWithBytes:data
                                                    length:bytes
                                                   options:MTLResourceStorageModeShared];
        if (!result) throw std::runtime_error("Metal buffer allocation failed");
        return result;
    }
    id<MTLBuffer> staging = [device newBufferWithBytes:data length:bytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> result = [device newBufferWithLength:bytes
                                               options:MTLResourceStorageModePrivate];
    if (!staging || !result) throw std::runtime_error("Metal private buffer allocation failed");
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
    [encoder copyFromBuffer:staging sourceOffset:0 toBuffer:result destinationOffset:0 size:bytes];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal private buffer upload failed");
    }
    return result;
}

id<MTLBuffer> MetalModel::Impl::buffer(const std::vector<float>& values) {
    id<MTLBuffer> result = immutable_buffer(values.data(), values.size() * sizeof(float));
    if (!result) throw std::runtime_error("Metal buffer allocation failed");
    execution_metrics.resident_weight_bytes += values.size() * sizeof(float);
    return result;
}

id<MTLBuffer> MetalModel::Impl::zero_buffer(size_t bytes) {
    id<MTLBuffer> result = [device newBufferWithLength:bytes
                                               options:MTLResourceStorageModeShared];
    if (!result) throw std::runtime_error("Metal state allocation failed");
    execution_metrics.resident_state_bytes += bytes;
    std::memset(result.contents, 0, bytes);
    return result;
}

id<MTLBuffer> MetalModel::Impl::raw_buffer(const std::byte* data, size_t bytes) {
    id<MTLBuffer> result = immutable_buffer(data, bytes);
    if (!result) throw std::runtime_error("Metal raw buffer allocation failed");
    execution_metrics.resident_weight_bytes += bytes;
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
    std::vector<float> values = decode_tensor_f32(repository->tensor(name),
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
                                            bool allow_missing) {
    return buffer(load_vector_values(role, layer, width, allow_missing));
}

MetalModel::Impl::Linear MetalModel::Impl::load_linear(TensorRole role, int layer,
                                                        int rows, int cols) {
    const TensorRequest& request = request_for(model.weight_plan.requests, role, layer);
    if (!request.source_name) throw std::runtime_error("Metal matrix request was not resolved");
    const std::string& name = *request.source_name;
    return load_linear_source(name, rows, cols);
}

MetalModel::Impl::Linear MetalModel::Impl::load_linear_source(const std::string& name,
                                                               int rows, int cols) {
    const HostTensorView view = repository->tensor(name);
    const std::vector<int64_t> expected_values{rows, cols};
    const std::span<const int64_t> expected(expected_values.data(), expected_values.size());
    if (!tensor_shape_matches(view.shape, expected)) {
        throw std::runtime_error("unexpected Metal tensor shape: " + name);
    }
    if (view.dtype == TensorDType::Quantized) {
        const GgmlType type = ggml_type_from_block_encoding(view.block_encoding);
        if (const auto binding = metal_model_detail::metal_linear_binding(type);
            binding && binding->supports_width(cols)) {
            const GgmlTypeTrait trait = ggml_type_trait(type);
            const size_t row_bytes = static_cast<size_t>(cols) /
                static_cast<size_t>(trait.block_size) * static_cast<size_t>(trait.type_size);
            if (view.bytes != static_cast<size_t>(rows) * row_bytes) {
                throw std::runtime_error("invalid Metal quantized tensor bytes: " + name);
            }
            return {raw_buffer(view.data, view.bytes),
                    static_cast<uint32_t>(rows), static_cast<uint32_t>(cols),
                    static_cast<uint32_t>(row_bytes), binding->storage};
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
    std::vector<float> values = decode_tensor_f32(view, expected, name);
    if (values.size() != static_cast<size_t>(rows) * cols) {
        throw std::runtime_error("Metal matrix dimensions mismatch: " + name);
    }
    return {buffer(values), static_cast<uint32_t>(rows), static_cast<uint32_t>(cols),
            0, LinearStorage::Float32};
}

id<MTLBuffer> MetalModel::Impl::load_conv_kernel(int layer, int width, int cache_length) {
    const TensorRequest& request = request_for(model.weight_plan.requests,
                                               TensorRole::ShortConvKernel, layer);
    if (!request.source_name) throw std::runtime_error("Metal convolution request was not resolved");
    const std::string& name = *request.source_name;
    std::vector<float> channel_major = decode_tensor_f32(repository->tensor(name),
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
