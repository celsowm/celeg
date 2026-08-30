#include "celeg/checkpoint/tensor_codec.hpp"
#include "celeg/quantization/scalars.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

celeg::HostTensorView view(celeg::TensorDType dtype,
                           std::vector<std::int64_t> shape,
                           const void* data, std::size_t bytes) {
    return {dtype, std::move(shape), static_cast<const std::byte*>(data), bytes, {}};
}

}

int main() {
    const std::vector<std::int64_t> matrix_shape{2, 2};
    const float f32[] = {0.0f, 1.5f, -2.0f, 4.0f};
    const auto decoded_f32 = celeg::decode_tensor_f32(
        view(celeg::TensorDType::F32, matrix_shape, f32, sizeof(f32)),
        matrix_shape, "f32");
    if (decoded_f32 != std::vector<float>({0.0f, 1.5f, -2.0f, 4.0f})) return 1;

    const std::uint16_t bf16[] = {0x0000u, 0x3f80u, 0xc000u, 0x7f80u};
    const auto decoded_bf16 = celeg::decode_tensor_f32(
        view(celeg::TensorDType::BF16, matrix_shape, bf16, sizeof(bf16)),
        matrix_shape, "bf16");
    if (decoded_bf16[0] != 0.0f || decoded_bf16[1] != 1.0f ||
        decoded_bf16[2] != -2.0f || !std::isinf(decoded_bf16[3])) return 2;

    const std::uint16_t f16[] = {0x0000u, 0x3c00u, 0x0001u, 0x7e00u};
    const auto decoded_f16 = celeg::decode_tensor_f32(
        view(celeg::TensorDType::F16, matrix_shape, f16, sizeof(f16)),
        matrix_shape, "f16");
    if (decoded_f16[0] != 0.0f || decoded_f16[1] != 1.0f ||
        decoded_f16[2] <= 0.0f || !std::isnan(decoded_f16[3])) return 3;

    const std::vector<std::int64_t> flat_shape{4};
    if (!celeg::tensor_shape_is_compatible(matrix_shape, flat_shape)) return 4;
    const std::vector<std::int64_t> singleton_shape{2, 1, 2};
    if (!celeg::tensor_shape_is_compatible(matrix_shape, singleton_shape) ||
        celeg::tensor_shape_matches(matrix_shape, flat_shape)) return 5;

    bool overflow_rejected = false;
    try {
        const std::vector<std::int64_t> overflow_shape(80, 2);
        celeg::tensor_element_count(overflow_shape, "overflow");
    } catch (const std::runtime_error&) {
        overflow_rejected = true;
    }
    if (!overflow_rejected) return 6;

    bool bytes_rejected = false;
    try {
        celeg::decode_tensor_f32(
            view(celeg::TensorDType::F32, matrix_shape, f32, sizeof(float)),
            matrix_shape, "truncated");
    } catch (const std::runtime_error&) {
        bytes_rejected = true;
    }
    if (!bytes_rejected) return 7;
    return 0;
}
