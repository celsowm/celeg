#include "backend/cuda/weights_loader.hpp"
#include "kernels/gguf.cuh"
#include "celeg/checkpoint/formats/gguf.hpp"
#include "celeg/quantization/ggml.hpp"

#include <cuda_bf16.h>

#include <stdexcept>
#include <vector>

namespace celeg {
namespace {

void dequantize_gguf_to_bf16_impl(const HostTensorView& tensor,
                                  std::vector<__nv_bfloat16>& out) {
    if (tensor.shape.size() != 2) {
        throw std::runtime_error("GGUF dequantization requires a matrix tensor");
    }
    GgmlMatrixView matrix;
    matrix.type = ggml_type_from_block_encoding(tensor.block_encoding);
    matrix.rows = static_cast<std::uint32_t>(tensor.shape[0]);
    matrix.cols = static_cast<std::uint32_t>(tensor.shape[1]);
    matrix.data = tensor.data;
    matrix.bytes = tensor.bytes;
    matrix.validate();
    out.resize(static_cast<std::size_t>(matrix.rows) * matrix.cols);
    std::vector<float> decoded(matrix.cols);
    for (std::uint32_t row = 0; row < matrix.rows; ++row) {
        ggml_decode_row(matrix, row, decoded.data());
        __nv_bfloat16* destination = out.data() +
            static_cast<std::size_t>(row) * matrix.cols;
        for (std::uint32_t column = 0; column < matrix.cols; ++column) {
            destination[column] = __float2bfloat16(decoded[column]);
        }
    }
}

}

void dequantize_gguf_to_bf16(const HostTensorView& tensor,
                             std::vector<__nv_bfloat16>& out) {
    dequantize_gguf_to_bf16_impl(tensor, out);
}

}
