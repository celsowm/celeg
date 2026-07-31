__global__ void rmsnorm_kernel(const __nv_bfloat16* x,
                               const __nv_bfloat16* weight,
                               __nv_bfloat16* out,
                               int width,
                               float eps) {
    const int row = blockIdx.x;
    const __nv_bfloat16* in = x + static_cast<size_t>(row) * width;
    __nv_bfloat16* dst = out + static_cast<size_t>(row) * width;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float v = bf16_float(in[i]);
        sum += v * v;
    }

    __shared__ float warp_sums[32];
    __shared__ float total;
    sum = block_sum(sum, warp_sums, &total);

    __shared__ float inv;
    if (threadIdx.x == 0) {
        inv = rsqrtf(sum / static_cast<float>(width) + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float normalized = rounded_bf16_float(bf16_float(in[i]) * inv);
        dst[i] = __float2bfloat16(normalized * bf16_float(weight[i]));
    }
}

__global__ void residual_kernel(__nv_bfloat16* x,
                                const __nv_bfloat16* residual,
                                int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        x[i] = __float2bfloat16(bf16_float(x[i]) + bf16_float(residual[i]));
    }
}

__global__ void scale_kernel(__nv_bfloat16* x, int count, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) x[i] = __float2bfloat16(bf16_float(x[i]) * scale);
}

__global__ void swiglu_fused_kernel(const __nv_bfloat16* gate_up,
                                    __nv_bfloat16* out,
                                    int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float gate = bf16_float(gate_up[i]);
        const float up = bf16_float(gate_up[count + i]);
        const float silu = rounded_bf16_float(gate / (1.0f + expf(-gate)));
        out[i] = __float2bfloat16(silu * up);
    }
}

void launch_rmsnorm(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                    __nv_bfloat16* out, int rows, int width, float eps,
                    cudaStream_t stream) {
    rmsnorm_kernel<<<rows, 256, 0, stream>>>(x, weight, out, width, eps);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_residual_add(__nv_bfloat16* x, const __nv_bfloat16* residual,
                         int count, cudaStream_t stream) {
    residual_kernel<<<(count + 255) / 256, 256, 0, stream>>>(x, residual, count);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_scale(__nv_bfloat16* x, int count, float scale, cudaStream_t stream) {
    if (scale == 1.0f) return;
    scale_kernel<<<(count + 255) / 256, 256, 0, stream>>>(x, count, scale);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_swiglu_fused(const __nv_bfloat16* gate_up, __nv_bfloat16* out,
                         int count, cudaStream_t stream) {
    swiglu_fused_kernel<<<(count + 255) / 256, 256, 0, stream>>>(gate_up, out, count);
    LFM_KERNEL_DEBUG_SYNC(stream);
}
