__global__ void w4a16_linear_kernel(const __nv_bfloat16* x,
                                    const uint8_t* weight,
                                    const float* scales,
                                    __nv_bfloat16* y,
                                    int m, int n, int k, float beta,
                                    int tile_k) {
    constexpr int warps_per_block = 8;
    extern __shared__ char smem_raw[];
    __nv_bfloat16* s_act = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int output_row = blockIdx.x * warps_per_block + warp;
    if (warp >= warps_per_block) return;

    const size_t packed_cols = static_cast<size_t>((k + 1) / 2);
    const uint8_t* row_weight = (output_row < n)
        ? (weight + static_cast<size_t>(output_row) * packed_cols)
        : nullptr;
    const float row_scale = (output_row < n) ? scales[output_row] : 0.0f;

    for (int activation_row = blockIdx.y;
         activation_row < m;
         activation_row += gridDim.y) {
        const __nv_bfloat16* input =
            x + static_cast<size_t>(activation_row) * k;

        float accum = 0.0f;
        for (int base_k = 0; base_k < k; base_k += tile_k) {
            const int chunk = (k - base_k) < tile_k ? (k - base_k) : tile_k;

            {
                const int tid = threadIdx.x;
                const int stride = blockDim.x;
                for (int i = tid; i < chunk; i += stride) {
                    s_act[i] = input[base_k + i];
                }
            }
            __syncthreads();

            if (output_row < n) {
                float sum = 0.0f;
                for (int column = lane; column < chunk; column += 32) {
                    sum += bf16_float(s_act[column]) *
                           static_cast<float>(unpack_int4(row_weight, base_k + column));
                }
                accum += warp_sum(sum);
            }

            __syncthreads();
        }
        if (output_row < n && lane == 0) {
            float value = accum * row_scale;
            const size_t output_index =
                static_cast<size_t>(activation_row) * n + output_row;
            if (beta != 0.0f) value += beta * bf16_float(y[output_index]);
            y[output_index] = __float2bfloat16(value);
        }
    }
}

__global__ void w8a16_linear_kernel(const __nv_bfloat16* x,
                                    const int8_t* weight,
                                    const float* scales,
                                    __nv_bfloat16* y,
                                    int m, int n, int k, float beta) {
    constexpr int warps_per_block = 8;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int output_row = blockIdx.x * warps_per_block + warp;
    if (warp >= warps_per_block || output_row >= n) return;

    const int8_t* row_weight = weight + static_cast<size_t>(output_row) * k;
    // gridDim.y is capped by the CUDA y-dimension limit. Striding keeps the
    // batched path valid for contexts larger than 65,535 tokens.
    for (int activation_row = blockIdx.y;
         activation_row < m;
         activation_row += gridDim.y) {
        const __nv_bfloat16* input =
            x + static_cast<size_t>(activation_row) * k;
        float sum = 0.0f;
        for (int column = lane; column < k; column += 32) {
            sum += bf16_float(input[column]) *
                   static_cast<float>(row_weight[column]);
        }
        sum = warp_sum(sum);
        if (lane == 0) {
            float value = sum * scales[output_row];
            const size_t output_index =
                static_cast<size_t>(activation_row) * n + output_row;
            if (beta != 0.0f) value += beta * bf16_float(y[output_index]);
            y[output_index] = __float2bfloat16(value);
        }
    }
}

void launch_w8a16_linear(const __nv_bfloat16* x, const int8_t* weight,
                         const float* scales, __nv_bfloat16* y,
                         int m, int n, int k, float beta,
                         cudaStream_t stream) {
    constexpr int warps_per_block = 8;
    const unsigned grid_y = static_cast<unsigned>(m < 65535 ? m : 65535);
    const dim3 grid((n + warps_per_block - 1) / warps_per_block, grid_y);
    w8a16_linear_kernel<<<grid, warps_per_block * 32, 0, stream>>>(
        x, weight, scales, y, m, n, k, beta);
    LFM_KERNEL_DEBUG_SYNC(stream);
}


void launch_w4a16_linear(const __nv_bfloat16* x, const uint8_t* weight,
                         const float* scales, __nv_bfloat16* y,
                         int m, int n, int k, float beta,
                         cudaStream_t stream) {
    constexpr int warps_per_block = 8;
    // Use 512-element tiles (1 KB smem). For k <= 512, one chunk; for larger k,
    // multiple chunks sharing the same smem buffer per activation row.
    constexpr int tile_k = 512;
    const unsigned grid_x = static_cast<unsigned>((n + warps_per_block - 1) / warps_per_block);
    const unsigned grid_y = static_cast<unsigned>(m < 65535 ? m : 65535);
    const dim3 grid(grid_x, grid_y);
    const size_t smem_size = static_cast<size_t>(tile_k) * sizeof(__nv_bfloat16);
    w4a16_linear_kernel<<<grid, warps_per_block * 32, smem_size, stream>>>(
        x, weight, scales, y, m, n, k, beta, tile_k);
    LFM_KERNEL_DEBUG_SYNC(stream);
}


