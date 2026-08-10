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
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_residual_add(__nv_bfloat16* x, const __nv_bfloat16* residual,
                         int count, cudaStream_t stream) {
    residual_kernel<<<(count + 255) / 256, 256, 0, stream>>>(x, residual, count);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_scale(__nv_bfloat16* x, int count, float scale, cudaStream_t stream) {
    if (scale == 1.0f) return;
    scale_kernel<<<(count + 255) / 256, 256, 0, stream>>>(x, count, scale);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

__global__ void relu2_kernel(const __nv_bfloat16* input, __nv_bfloat16* out,
                             int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float x = fmaxf(0.0f, bf16_float(input[i]));
        out[i] = __float2bfloat16(x * x);
    }
}

void launch_relu2(const __nv_bfloat16* input, __nv_bfloat16* out,
                  int count, cudaStream_t stream) {
    relu2_kernel<<<(count + 255) / 256, 256, 0, stream>>>(input, out, count);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

__global__ void tanh_softcap_kernel(__nv_bfloat16* x, int count, float cap) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    x[index] = __float2bfloat16(tanhf(bf16_float(x[index]) / cap) * cap);
}

void launch_tanh_softcap(__nv_bfloat16* x, int count, float cap, cudaStream_t stream) {
    tanh_softcap_kernel<<<(count + 255) / 256, 256, 0, stream>>>(x, count, cap);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_swiglu_fused(const __nv_bfloat16* gate_up, __nv_bfloat16* out,
                         int count, cudaStream_t stream) {
    swiglu_fused_kernel<<<(count + 255) / 256, 256, 0, stream>>>(gate_up, out, count);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

__global__ void gelu_tanh_kernel(const __nv_bfloat16* input,
                                 __nv_bfloat16* out, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    const float x = bf16_float(input[index]);
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    out[index] = __float2bfloat16(0.5f * x *
        (1.0f + tanhf(kSqrt2OverPi * (x + 0.044715f * x * x * x))));
}

void launch_gelu_tanh(const __nv_bfloat16* input, __nv_bfloat16* out,
                      int count, cudaStream_t stream) {
    gelu_tanh_kernel<<<(count + 255) / 256, 256, 0, stream>>>(input, out, count);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

__global__ void gated_gelu_tanh_kernel(const __nv_bfloat16* gate_up,
                                       __nv_bfloat16* out, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    const float x = bf16_float(gate_up[index]);
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    const float gelu = 0.5f * x *
        (1.0f + tanhf(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
    out[index] = __float2bfloat16(gelu * bf16_float(gate_up[count + index]));
}

void launch_gated_gelu_tanh(const __nv_bfloat16* gate_up, __nv_bfloat16* out,
                            int count, cudaStream_t stream) {
    gated_gelu_tanh_kernel<<<(count + 255) / 256, 256, 0, stream>>>(gate_up, out, count);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

__global__ void multiply_kernel(__nv_bfloat16* x, const __nv_bfloat16* y, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) x[index] = __float2bfloat16(bf16_float(x[index]) * bf16_float(y[index]));
}

void launch_multiply(__nv_bfloat16* x, const __nv_bfloat16* y, int count,
                     cudaStream_t stream) {
    multiply_kernel<<<(count + 255) / 256, 256, 0, stream>>>(x, y, count);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

__global__ void sigmoid_multiply_kernel(__nv_bfloat16* x,
                                        const __nv_bfloat16* gate, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) {
        const float value = bf16_float(gate[index]);
        x[index] = __float2bfloat16(bf16_float(x[index]) /
            (1.0f + expf(-value)));
    }
}

void launch_sigmoid_multiply(__nv_bfloat16* x, const __nv_bfloat16* gate,
                             int count, cudaStream_t stream) {
    sigmoid_multiply_kernel<<<(count + 255) / 256, 256, 0, stream>>>(x, gate, count);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

__global__ void extract_attention_output_gate_kernel(__nv_bfloat16* projected,
                                          __nv_bfloat16* gate,
                                          int rows, int width) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * width;
    if (index >= total) return;
    const int row = static_cast<int>(index / width);
    const int column = static_cast<int>(index % width);
    const size_t source = static_cast<size_t>(row) * width * 2 + column;
    const size_t source_gate = source + width;
    const __nv_bfloat16 q = projected[source];
    const __nv_bfloat16 g = projected[source_gate];
    projected[index] = q;
    gate[index] = g;
}

void launch_extract_attention_output_gate(__nv_bfloat16* projected, __nv_bfloat16* gate,
                               int rows, int width, cudaStream_t stream) {
    const size_t total = static_cast<size_t>(rows) * width;
    extract_attention_output_gate_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
        projected, gate, rows, width);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

__global__ void multiply_strided_kernel(__nv_bfloat16* x, const __nv_bfloat16* y,
                                        int rows, int width, int y_stride,
                                        int y_offset) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * width;
    if (index >= total) return;
    const int row = static_cast<int>(index / width);
    const int column = static_cast<int>(index % width);
    x[index] = __float2bfloat16(
        bf16_float(x[index]) * bf16_float(y[static_cast<size_t>(row) * y_stride +
                                            y_offset + column]));
}

void launch_multiply_strided(__nv_bfloat16* x, const __nv_bfloat16* y,
                             int rows, int width, int y_stride, int y_offset,
                             cudaStream_t stream) {
    const size_t total = static_cast<size_t>(rows) * width;
    multiply_strided_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
        x, y, rows, width, y_stride, y_offset);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

__global__ void scale_by_scalar_kernel(__nv_bfloat16* x,
                                       const __nv_bfloat16* scalar, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) {
        x[index] = __float2bfloat16(bf16_float(x[index]) * bf16_float(*scalar));
    }
}

void launch_scale_by_scalar(__nv_bfloat16* x, const __nv_bfloat16* scalar,
                            int count, cudaStream_t stream) {
    scale_by_scalar_kernel<<<(count + 255) / 256, 256, 0, stream>>>(x, scalar, count);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

__global__ void sigmoid_scale_by_scalar_kernel(__nv_bfloat16* x,
                                               const __nv_bfloat16* scalar,
                                               int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) {
        const float gate = 1.0f / (1.0f + expf(-bf16_float(*scalar)));
        x[index] = __float2bfloat16(bf16_float(x[index]) * gate);
    }
}

void launch_sigmoid_scale_by_scalar(__nv_bfloat16* x,
                                    const __nv_bfloat16* scalar,
                                    int count, cudaStream_t stream) {
    sigmoid_scale_by_scalar_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
        x, scalar, count);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

__global__ void sigmoid_multiply_strided_kernel(__nv_bfloat16* x,
                                                const __nv_bfloat16* gate,
                                                int rows, int width) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * width;
    if (index >= total) return;
    const int row = static_cast<int>(index / width);
    const float value = bf16_float(x[index]);
    const float g = bf16_float(gate[row]);
    x[index] = __float2bfloat16(value / (1.0f + expf(-g)));
}

void launch_sigmoid_multiply_strided(__nv_bfloat16* x, const __nv_bfloat16* gate,
                                     int rows, int width, cudaStream_t stream) {
    const size_t total = static_cast<size_t>(rows) * width;
    sigmoid_multiply_strided_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
        x, gate, rows, width);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}
