#include "kernel_common.cuh"
#include "celeg/backend/cuda/kernels/mamba2.hpp"

namespace celeg {

__global__ void mamba2_step_kernel(
    const __nv_bfloat16* projected, const __nv_bfloat16* conv_weight,
    const __nv_bfloat16* conv_bias, const __nv_bfloat16* dt_bias,
    const __nv_bfloat16* a_log, const __nv_bfloat16* d,
    __nv_bfloat16* conv_state, __nv_bfloat16* ssm_state,
    __nv_bfloat16* inner, int intermediate, int state_size, int num_heads,
    int head_dim, int group_count, int conv_kernel) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    const int conv_dim = intermediate + 2 * group_count * state_size;
    if (channel >= conv_dim) return;

    const __nv_bfloat16* xbc = projected + intermediate;
    __nv_bfloat16* history = conv_state + static_cast<size_t>(channel) * conv_kernel;
    for (int tap = 0; tap + 1 < conv_kernel; ++tap) history[tap] = history[tap + 1];
    history[conv_kernel - 1] = xbc[channel];
    float value = bf16_float(conv_bias[channel]);
    for (int tap = 0; tap < conv_kernel; ++tap) {
        value += bf16_float(history[tap]) *
            bf16_float(conv_weight[static_cast<size_t>(channel) * conv_kernel + tap]);
    }
    const float conv = value / (1.0f + expf(-value));
    if (channel >= intermediate) return;

    const int head = channel / head_dim;
    const int group = head / (num_heads / group_count);
    const float dt_raw = bf16_float(projected[conv_dim + intermediate + head]);
    const float dt = log1pf(expf(dt_raw + bf16_float(dt_bias[head])));
    const float decay = expf(-dt * expf(bf16_float(a_log[head])));
    __nv_bfloat16* state = ssm_state + static_cast<size_t>(channel) * state_size;
    const __nv_bfloat16* b = nullptr;
    const __nv_bfloat16* c = nullptr;
    // B and C are stored in the convolution output after the x channels.
    b = xbc + intermediate + group * state_size;
    c = xbc + intermediate + group_count * state_size + group * state_size;
    float output = 0.0f;
    for (int n = 0; n < state_size; ++n) {
        float s = decay * bf16_float(state[n]) + dt * bf16_float(b[n]) * conv;
        state[n] = __float2bfloat16(s);
        output += s * bf16_float(c[n]);
    }
    inner[channel] = __float2bfloat16(output + bf16_float(d[head]) * conv);
}

void launch_mamba2_step(const __nv_bfloat16* projected,
                        const __nv_bfloat16* conv_weight,
                        const __nv_bfloat16* conv_bias,
                        const __nv_bfloat16* dt_bias,
                        const __nv_bfloat16* a_log,
                        const __nv_bfloat16* d,
                        __nv_bfloat16* conv_state,
                        __nv_bfloat16* ssm_state,
                        __nv_bfloat16* inner,
                        int intermediate, int state_size, int num_heads,
                        int head_dim, int group_count, int conv_kernel,
                        cudaStream_t stream) {
    const int conv_dim = intermediate + 2 * group_count * state_size;
    mamba2_step_kernel<<<(conv_dim + 255) / 256, 256, 0, stream>>>(
        projected, conv_weight, conv_bias, dt_bias, a_log, d, conv_state,
        ssm_state, inner, intermediate, state_size, num_heads, head_dim,
        group_count, conv_kernel);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

} // namespace celeg
