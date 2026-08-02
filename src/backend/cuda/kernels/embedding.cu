#include "kernel_common.cuh"

namespace celeg {
namespace {

__global__ void embedding_value_kernel(int32_t token,
                                       const __nv_bfloat16* table,
                                       __nv_bfloat16* out,
                                       int hidden) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) {
        out[i] = table[static_cast<size_t>(token) * hidden + i];
    }
}

__global__ void embedding_pointer_kernel(const int32_t* token,
                                         const __nv_bfloat16* table,
                                         __nv_bfloat16* out,
                                         int hidden) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) {
        out[i] = table[static_cast<size_t>(*token) * hidden + i];
    }
}

__global__ void embedding_batch_kernel(const int32_t* tokens,
                                       int rows,
                                       const __nv_bfloat16* table,
                                       __nv_bfloat16* out,
                                       int hidden) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * hidden;
    if (index >= total) return;
    const int row = static_cast<int>(index / hidden);
    const int column = static_cast<int>(index % hidden);
    out[index] = table[static_cast<size_t>(tokens[row]) * hidden + column];
}

__global__ void embedding_slice_kernel(const int32_t* token, const __nv_bfloat16* table,
                                       int table_width, int offset,
                                       __nv_bfloat16* out, int width, bool device_token) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const int row = device_token ? *token : *token;
    out[i] = table[static_cast<size_t>(row) * table_width + offset + i];
}

__global__ void embedding_slice_value_kernel(int32_t token, const __nv_bfloat16* table,
                                             int table_width, int offset,
                                             __nv_bfloat16* out, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    out[i] = table[static_cast<size_t>(token) * table_width + offset + i];
}

__global__ void embedding_slice_batch_kernel(const int32_t* tokens, int rows,
                                             const __nv_bfloat16* table, int table_width,
                                             int offset, __nv_bfloat16* out, int width) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * width;
    if (index >= total) return;
    const int row = static_cast<int>(index / width);
    const int column = static_cast<int>(index % width);
    out[index] = table[static_cast<size_t>(tokens[row]) * table_width + offset + column];
}

__global__ void embedding_int8_value_kernel(int32_t token,
                                             const int8_t* table,
                                             const float* scales,
                                             __nv_bfloat16* out,
                                             int hidden) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= hidden) return;
    const float value = static_cast<float>(
        table[static_cast<size_t>(token) * hidden + column]) * scales[token];
    out[column] = __float2bfloat16(value);
}

__global__ void embedding_int8_pointer_kernel(const int32_t* token,
                                               const int8_t* table,
                                               const float* scales,
                                               __nv_bfloat16* out,
                                               int hidden) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= hidden) return;
    const int row = *token;
    const float value = static_cast<float>(
        table[static_cast<size_t>(row) * hidden + column]) * scales[row];
    out[column] = __float2bfloat16(value);
}

__global__ void embedding_int8_batch_kernel(const int32_t* tokens,
                                             int rows,
                                             const int8_t* table,
                                             const float* scales,
                                             __nv_bfloat16* out,
                                             int hidden) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * hidden;
    if (index >= total) return;
    const int batch_row = static_cast<int>(index / hidden);
    const int column = static_cast<int>(index % hidden);
    const int token = tokens[batch_row];
    const float value = static_cast<float>(
        table[static_cast<size_t>(token) * hidden + column]) * scales[token];
    out[index] = __float2bfloat16(value);
}

// Weight-only symmetric per-output-channel INT8. Eight warps compute eight
// output rows for one activation row. The decode path (m=1) is therefore a
// true GEMV; prefill reuses the same kernel across activation rows.

__global__ void embedding_int4_value_kernel(int32_t token,
                                             const uint8_t* table,
                                             const float* scales,
                                             __nv_bfloat16* out,
                                             int hidden) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= hidden) return;
    const size_t packed_cols = static_cast<size_t>((hidden + 1) / 2);
    const uint8_t* row = table + static_cast<size_t>(token) * packed_cols;
    out[column] = __float2bfloat16(
        static_cast<float>(unpack_int4(row, column)) * scales[token]);
}

__global__ void embedding_int4_pointer_kernel(const int32_t* token,
                                               const uint8_t* table,
                                               const float* scales,
                                               __nv_bfloat16* out,
                                               int hidden) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= hidden) return;
    const int row_index = *token;
    const size_t packed_cols = static_cast<size_t>((hidden + 1) / 2);
    const uint8_t* row = table + static_cast<size_t>(row_index) * packed_cols;
    out[column] = __float2bfloat16(
        static_cast<float>(unpack_int4(row, column)) * scales[row_index]);
}

__global__ void embedding_int4_batch_kernel(const int32_t* tokens,
                                             int rows,
                                             const uint8_t* table,
                                             const float* scales,
                                             __nv_bfloat16* out,
                                             int hidden) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * hidden;
    if (index >= total) return;
    const int activation_row = static_cast<int>(index / hidden);
    const int column = static_cast<int>(index % hidden);
    const int token = tokens[activation_row];
    const size_t packed_cols = static_cast<size_t>((hidden + 1) / 2);
    const uint8_t* row = table + static_cast<size_t>(token) * packed_cols;
    out[index] = __float2bfloat16(
        static_cast<float>(unpack_int4(row, column)) * scales[token]);
}

// Weight-only symmetric per-output-channel INT4. The packed matrix keeps two
// signed values per byte. Eight warps compute eight output rows per block.
// Shared-memory activation staging: the activation tile is loaded once per
// block and shared across all 8 warps, halving global-memory traffic versus
// the naive per-warp load when m > 1 (multi-row prefill).
} // namespace

void launch_embedding(int32_t token, const __nv_bfloat16* table,
                      __nv_bfloat16* out, int hidden, cudaStream_t stream) {
    embedding_value_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        token, table, out, hidden);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_embedding_device(const int32_t* token, const __nv_bfloat16* table,
                             __nv_bfloat16* out, int hidden, cudaStream_t stream) {
    embedding_pointer_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        token, table, out, hidden);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_embedding_batch(const int32_t* tokens, int rows,
                            const __nv_bfloat16* table, __nv_bfloat16* out,
                            int hidden, cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * hidden;
    embedding_batch_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        tokens, rows, table, out, hidden);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_embedding_slice(int32_t token, const __nv_bfloat16* table,
                            int table_width, int offset, __nv_bfloat16* out,
                            int width, cudaStream_t stream) {
    embedding_slice_value_kernel<<<(width + 255) / 256, 256, 0, stream>>>(
        token, table, table_width, offset, out, width);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_embedding_slice_device(const int32_t* token, const __nv_bfloat16* table,
                                   int table_width, int offset, __nv_bfloat16* out,
                                   int width, cudaStream_t stream) {
    embedding_slice_kernel<<<(width + 255) / 256, 256, 0, stream>>>(
        token, table, table_width, offset, out, width, true);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_embedding_slice_batch(const int32_t* tokens, int rows,
                                  const __nv_bfloat16* table, int table_width,
                                  int offset, __nv_bfloat16* out, int width,
                                  cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * width;
    embedding_slice_batch_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        tokens, rows, table, table_width, offset, out, width);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_embedding_int8(int32_t token, const int8_t* table,
                           const float* scales, __nv_bfloat16* out,
                           int hidden, cudaStream_t stream) {
    embedding_int8_value_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        token, table, scales, out, hidden);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_embedding_int8_device(const int32_t* token, const int8_t* table,
                                  const float* scales, __nv_bfloat16* out,
                                  int hidden, cudaStream_t stream) {
    embedding_int8_pointer_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        token, table, scales, out, hidden);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_embedding_int8_batch(const int32_t* tokens, int rows,
                                 const int8_t* table, const float* scales,
                                 __nv_bfloat16* out, int hidden,
                                 cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * hidden;
    embedding_int8_batch_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        tokens, rows, table, scales, out, hidden);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_embedding_int4(int32_t token, const uint8_t* table,
                           const float* scales, __nv_bfloat16* out,
                           int hidden, cudaStream_t stream) {
    embedding_int4_value_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        token, table, scales, out, hidden);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_embedding_int4_device(const int32_t* token, const uint8_t* table,
                                  const float* scales, __nv_bfloat16* out,
                                  int hidden, cudaStream_t stream) {
    embedding_int4_pointer_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        token, table, scales, out, hidden);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_embedding_int4_batch(const int32_t* tokens, int rows,
                                 const uint8_t* table, const float* scales,
                                 __nv_bfloat16* out, int hidden,
                                 cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * hidden;
    embedding_int4_batch_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        tokens, rows, table, scales, out, hidden);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

} // namespace celeg
