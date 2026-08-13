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

    // The reduction above stays scalar (each thread's fixed strided slice
    // determines the exact floating-point summation order); this loop has no
    // such constraint -- every output element depends only on its own input,
    // so packing two bf16 lanes per 32-bit transaction is bit-identical to
    // the scalar loop, just half the traffic on `in`, `weight`, and `dst`.
    if ((width & 1) == 0 && bf16x2_aligned(in, weight, dst)) {
        const int half = width >> 1;
        const __nv_bfloat162* in2 = reinterpret_cast<const __nv_bfloat162*>(in);
        const __nv_bfloat162* weight2 = reinterpret_cast<const __nv_bfloat162*>(weight);
        __nv_bfloat162* dst2 = reinterpret_cast<__nv_bfloat162*>(dst);
        for (int i = threadIdx.x; i < half; i += blockDim.x) {
            const __nv_bfloat162 iv = in2[i];
            const __nv_bfloat162 wv = weight2[i];
            const float n0 = rounded_bf16_float(__low2float(iv) * inv);
            const float n1 = rounded_bf16_float(__high2float(iv) * inv);
            dst2[i] = __floats2bfloat162_rn(n0 * __low2float(wv), n1 * __high2float(wv));
        }
    } else {
        for (int i = threadIdx.x; i < width; i += blockDim.x) {
            const float normalized = rounded_bf16_float(bf16_float(in[i]) * inv);
            dst[i] = __float2bfloat16(normalized * bf16_float(weight[i]));
        }
    }
}

__global__ void residual_kernel(__nv_bfloat16* x,
                                const __nv_bfloat16* residual,
                                int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((count & 1) == 0 && bf16x2_aligned(x, residual)) {
        if (i < count >> 1) {
            __nv_bfloat162* x2 = reinterpret_cast<__nv_bfloat162*>(x);
            const __nv_bfloat162* r2 = reinterpret_cast<const __nv_bfloat162*>(residual);
            const __nv_bfloat162 xv = x2[i];
            const __nv_bfloat162 rv = r2[i];
            x2[i] = __floats2bfloat162_rn(__low2float(xv) + __low2float(rv),
                                          __high2float(xv) + __high2float(rv));
        }
    } else if (i < count) {
        x[i] = __float2bfloat16(bf16_float(x[i]) + bf16_float(residual[i]));
    }
}

__global__ void scale_kernel(__nv_bfloat16* x, int count, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((count & 1) == 0 && bf16x2_aligned(x)) {
        if (i < count >> 1) {
            __nv_bfloat162* x2 = reinterpret_cast<__nv_bfloat162*>(x);
            const __nv_bfloat162 xv = x2[i];
            x2[i] = __floats2bfloat162_rn(__low2float(xv) * scale, __high2float(xv) * scale);
        }
    } else if (i < count) {
        x[i] = __float2bfloat16(bf16_float(x[i]) * scale);
    }
}

__global__ void swiglu_fused_kernel(const __nv_bfloat16* gate_up,
                                    __nv_bfloat16* out,
                                    int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((count & 1) == 0 && bf16x2_aligned(gate_up, out)) {
        if (i < count >> 1) {
            const __nv_bfloat162* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate_up);
            const __nv_bfloat162* up2 =
                reinterpret_cast<const __nv_bfloat162*>(gate_up + count);
            __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);
            const __nv_bfloat162 gv = gate2[i];
            const __nv_bfloat162 uv = up2[i];
            const float g0 = __low2float(gv);
            const float g1 = __high2float(gv);
            const float silu0 = rounded_bf16_float(g0 / (1.0f + expf(-g0)));
            const float silu1 = rounded_bf16_float(g1 / (1.0f + expf(-g1)));
            out2[i] = __floats2bfloat162_rn(silu0 * __low2float(uv), silu1 * __high2float(uv));
        }
    } else if (i < count) {
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

__device__ __forceinline__ float relu2_scalar(float x) {
    const float r = fmaxf(0.0f, x);
    return r * r;
}

__global__ void relu2_kernel(const __nv_bfloat16* input, __nv_bfloat16* out,
                             int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((count & 1) == 0 && bf16x2_aligned(input, out)) {
        if (i < count >> 1) {
            const __nv_bfloat162* in2 = reinterpret_cast<const __nv_bfloat162*>(input);
            __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);
            const __nv_bfloat162 iv = in2[i];
            out2[i] = __floats2bfloat162_rn(relu2_scalar(__low2float(iv)),
                                            relu2_scalar(__high2float(iv)));
        }
    } else if (i < count) {
        out[i] = __float2bfloat16(relu2_scalar(bf16_float(input[i])));
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

__device__ __forceinline__ float gelu_tanh_scalar(float x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
}

__global__ void gelu_tanh_kernel(const __nv_bfloat16* input,
                                 __nv_bfloat16* out, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if ((count & 1) == 0 && bf16x2_aligned(input, out)) {
        if (index < count >> 1) {
            const __nv_bfloat162* in2 = reinterpret_cast<const __nv_bfloat162*>(input);
            __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);
            const __nv_bfloat162 iv = in2[index];
            out2[index] = __floats2bfloat162_rn(gelu_tanh_scalar(__low2float(iv)),
                                                gelu_tanh_scalar(__high2float(iv)));
        }
        return;
    }
    if (index >= count) return;
    out[index] = __float2bfloat16(gelu_tanh_scalar(bf16_float(input[index])));
}

void launch_gelu_tanh(const __nv_bfloat16* input, __nv_bfloat16* out,
                      int count, cudaStream_t stream) {
    gelu_tanh_kernel<<<(count + 255) / 256, 256, 0, stream>>>(input, out, count);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

__global__ void gated_gelu_tanh_kernel(const __nv_bfloat16* gate_up,
                                       __nv_bfloat16* out, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if ((count & 1) == 0 && bf16x2_aligned(gate_up, out)) {
        if (index < count >> 1) {
            const __nv_bfloat162* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate_up);
            const __nv_bfloat162* up2 =
                reinterpret_cast<const __nv_bfloat162*>(gate_up + count);
            __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);
            const __nv_bfloat162 gv = gate2[index];
            const __nv_bfloat162 uv = up2[index];
            const float gelu0 = gelu_tanh_scalar(__low2float(gv));
            const float gelu1 = gelu_tanh_scalar(__high2float(gv));
            out2[index] = __floats2bfloat162_rn(gelu0 * __low2float(uv), gelu1 * __high2float(uv));
        }
        return;
    }
    if (index >= count) return;
    const float gelu = gelu_tanh_scalar(bf16_float(gate_up[index]));
    out[index] = __float2bfloat16(gelu * bf16_float(gate_up[count + index]));
}

void launch_gated_gelu_tanh(const __nv_bfloat16* gate_up, __nv_bfloat16* out,
                            int count, cudaStream_t stream) {
    gated_gelu_tanh_kernel<<<(count + 255) / 256, 256, 0, stream>>>(gate_up, out, count);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

__global__ void multiply_kernel(__nv_bfloat16* x, const __nv_bfloat16* y, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if ((count & 1) == 0 && bf16x2_aligned(x, y)) {
        if (index < count >> 1) {
            __nv_bfloat162* x2 = reinterpret_cast<__nv_bfloat162*>(x);
            const __nv_bfloat162* y2 = reinterpret_cast<const __nv_bfloat162*>(y);
            const __nv_bfloat162 xv = x2[index];
            const __nv_bfloat162 yv = y2[index];
            x2[index] = __floats2bfloat162_rn(__low2float(xv) * __low2float(yv),
                                              __high2float(xv) * __high2float(yv));
        }
        return;
    }
    if (index < count) x[index] = __float2bfloat16(bf16_float(x[index]) * bf16_float(y[index]));
}

void launch_multiply(__nv_bfloat16* x, const __nv_bfloat16* y, int count,
                     cudaStream_t stream) {
    multiply_kernel<<<(count + 255) / 256, 256, 0, stream>>>(x, y, count);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

__device__ __forceinline__ float sigmoid_multiply_scalar(float x, float gate) {
    return x / (1.0f + expf(-gate));
}

__global__ void sigmoid_multiply_kernel(__nv_bfloat16* x,
                                        const __nv_bfloat16* gate, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if ((count & 1) == 0 && bf16x2_aligned(x, gate)) {
        if (index < count >> 1) {
            __nv_bfloat162* x2 = reinterpret_cast<__nv_bfloat162*>(x);
            const __nv_bfloat162* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate);
            const __nv_bfloat162 xv = x2[index];
            const __nv_bfloat162 gv = gate2[index];
            x2[index] = __floats2bfloat162_rn(
                sigmoid_multiply_scalar(__low2float(xv), __low2float(gv)),
                sigmoid_multiply_scalar(__high2float(xv), __high2float(gv)));
        }
        return;
    }
    if (index < count) {
        x[index] = __float2bfloat16(
            sigmoid_multiply_scalar(bf16_float(x[index]), bf16_float(gate[index])));
    }
}

void launch_sigmoid_multiply(__nv_bfloat16* x, const __nv_bfloat16* gate,
                             int count, cudaStream_t stream) {
    sigmoid_multiply_kernel<<<(count + 255) / 256, 256, 0, stream>>>(x, gate, count);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

// Splits a packed [query|gate] projection into disjoint destination buffers.
// Must not write into `packed` in place: a compacting in-place variant has
// thread (row=r) reading packed[r*2*width + column] while thread
// (row=2r) writes that same address (its own index r*2*width + column lands
// exactly on row r's source), racing across independently scheduled blocks.
__global__ void extract_attention_output_gate_kernel(const __nv_bfloat16* packed,
                                          __nv_bfloat16* query_out,
                                          __nv_bfloat16* gate_out,
                                          int rows, int width) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * width;
    if (index >= total) return;
    const int row = static_cast<int>(index / width);
    const int column = static_cast<int>(index % width);
    const size_t source = static_cast<size_t>(row) * width * 2 + column;
    query_out[index] = packed[source];
    gate_out[index] = packed[source + width];
}

void launch_extract_attention_output_gate(const __nv_bfloat16* packed,
                               __nv_bfloat16* query_out, __nv_bfloat16* gate_out,
                               int rows, int width, cudaStream_t stream) {
    const size_t total = static_cast<size_t>(rows) * width;
    extract_attention_output_gate_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
        packed, query_out, gate_out, rows, width);
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

__global__ void sigmoid_multiply_headwise_kernel(__nv_bfloat16* x,
                                                 const __nv_bfloat16* gate,
                                                 int rows, int heads, int head_dim) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t width = static_cast<size_t>(heads) * head_dim;
    const size_t total = static_cast<size_t>(rows) * width;
    if (index >= total) return;
    const int row = static_cast<int>(index / width);
    const int head = static_cast<int>((index / static_cast<size_t>(head_dim)) % heads);
    const float value = bf16_float(x[index]);
    const float g = bf16_float(gate[static_cast<size_t>(row) * heads + head]);
    x[index] = __float2bfloat16(value / (1.0f + expf(-g)));
}

void launch_sigmoid_multiply_headwise(__nv_bfloat16* x, const __nv_bfloat16* gate,
                                      int rows, int heads, int head_dim,
                                      cudaStream_t stream) {
    const size_t total = static_cast<size_t>(rows) * heads * head_dim;
    sigmoid_multiply_headwise_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
        x, gate, rows, heads, head_dim);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}
