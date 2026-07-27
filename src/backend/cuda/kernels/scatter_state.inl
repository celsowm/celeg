__global__ void scatter_bf16_rows_kernel(
    const __nv_bfloat16* source,
    __nv_bfloat16* const* destinations,
    int rows,
    int width) {
    const size_t total = static_cast<size_t>(rows) * width;
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int row = static_cast<int>(index / width);
    const int column = static_cast<int>(index % width);
    destinations[row][column] = source[index];
}

__global__ void scatter_decode_state_kernel(
    const int32_t* sampled,
    const int32_t* positions,
    int32_t* const* sampled_destinations,
    int32_t* const* position_destinations,
    int rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    *sampled_destinations[row] = sampled[row];
    *position_destinations[row] = positions[row] + 1;
}

__global__ void scatter_bf16_selected_rows_kernel(
    const __nv_bfloat16* source, const int32_t* source_rows,
    __nv_bfloat16* const* destinations, int rows, int width) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * width;
    if (index >= total) return;
    const int request = static_cast<int>(index / width);
    destinations[request][index % width] =
        source[static_cast<size_t>(source_rows[request]) * width + index % width];
}

__global__ void scatter_selected_decode_state_kernel(
    const int32_t* sampled, const int32_t* positions,
    const int32_t* source_rows, int32_t* const* sampled_destinations,
    int32_t* const* position_destinations, int rows) {
    const int request = blockIdx.x * blockDim.x + threadIdx.x;
    if (request >= rows) return;
    const int row = source_rows[request];
    *sampled_destinations[request] = sampled[row];
    *position_destinations[request] = positions[row] + 1;
}

void launch_scatter_bf16_rows(
    const __nv_bfloat16* source,
    __nv_bfloat16* const* destinations,
    int rows,
    int width,
    cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * width;
    scatter_bf16_rows_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        source, destinations, rows, width);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_scatter_bf16_selected_rows(
    const __nv_bfloat16* source, const int32_t* source_rows,
    __nv_bfloat16* const* destinations, int rows, int width,
    cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * width;
    scatter_bf16_selected_rows_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        source, source_rows, destinations, rows, width);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_scatter_decode_state(
    const int32_t* sampled,
    const int32_t* positions,
    int32_t* const* sampled_destinations,
    int32_t* const* position_destinations,
    int rows,
    cudaStream_t stream) {
    scatter_decode_state_kernel<<<(rows + 255) / 256, 256, 0, stream>>>(
        sampled, positions, sampled_destinations,
        position_destinations, rows);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_scatter_selected_decode_state(
    const int32_t* sampled, const int32_t* positions,
    const int32_t* source_rows, int32_t* const* sampled_destinations,
    int32_t* const* position_destinations, int rows, cudaStream_t stream) {
    scatter_selected_decode_state_kernel<<<(rows + 255) / 256, 256, 0, stream>>>(
        sampled, positions, source_rows, sampled_destinations,
        position_destinations, rows);
    LFM_KERNEL_DEBUG_SYNC(stream);
}


