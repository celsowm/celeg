#include "mamba_tests.hpp"

#include "utils.cuh"
#include "../support/assertions.hpp"
#include "../support/cuda_kernel_assertions.cuh"
#include "kernels/kernels.cuh"

#include <vector>

namespace celeg::cuda_test {

void run_mamba_tests(celeg::CudaStream& stream) {
{
    constexpr int rows = 8, intermediate = 4, state_size = 3, heads = 2,
        head_dim = 2, groups = 1, kernel = 4;
    constexpr int conv_dim = intermediate + 2 * groups * state_size;
    constexpr int projection = 2 * intermediate + 2 * groups * state_size + heads;
    std::vector<float> projected(rows * projection), conv_weight(conv_dim * kernel),
        conv_bias(conv_dim), dt_bias(heads), a_log(heads), d(heads);
    for (size_t i = 0; i < projected.size(); ++i) projected[i] = 0.01f * float(i + 1);
    for (size_t i = 0; i < conv_weight.size(); ++i) conv_weight[i] = 0.02f * float(i + 1);
    for (int i = 0; i < conv_dim; ++i) conv_bias[i] = -0.1f + 0.01f * float(i);
    for (int i = 0; i < heads; ++i) { dt_bias[i] = 0.2f; a_log[i] = -0.3f; d[i] = 0.4f; }
    auto convert = [](const std::vector<float>& values) {
        std::vector<__nv_bfloat16> result(values.size());
        for (size_t i = 0; i < values.size(); ++i) result[i] = to_bf16(values[i]);
        return result;
    };
    const auto hp = convert(projected), hw = convert(conv_weight), hb = convert(conv_bias),
        hdt = convert(dt_bias), ha = convert(a_log), hd = convert(d);
    celeg::DeviceBuffer<__nv_bfloat16> dp(hp.size()), dw(hw.size()), db(hb.size()),
        ddt(hdt.size()), da(ha.size()), dd(hd.size()), batch_conv(conv_dim * kernel),
        batch_state(intermediate * state_size), batch_inner(rows * intermediate),
        step_conv(conv_dim * kernel), step_state(intermediate * state_size),
        step_inner(rows * intermediate);
    CELEG_CUDA(cudaMemcpy(dp.data(), hp.data(), dp.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dw.data(), hw.data(), dw.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(db.data(), hb.data(), db.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(ddt.data(), hdt.data(), ddt.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(da.data(), ha.data(), da.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dd.data(), hd.data(), dd.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemset(batch_conv.data(), 0, batch_conv.bytes()));
    CELEG_CUDA(cudaMemset(batch_state.data(), 0, batch_state.bytes()));
    CELEG_CUDA(cudaMemset(step_conv.data(), 0, step_conv.bytes()));
    CELEG_CUDA(cudaMemset(step_state.data(), 0, step_state.bytes()));
    celeg::launch_mamba2_prefill(dp.data(), dw.data(), db.data(), ddt.data(), da.data(), dd.data(),
        batch_conv.data(), batch_state.data(), batch_inner.data(), rows, intermediate,
        state_size, heads, head_dim, groups, kernel, stream.get());
    for (int row = 0; row < rows; ++row) {
        celeg::launch_mamba2_step(dp.data() + row * projection, dw.data(), db.data(), ddt.data(),
            da.data(), dd.data(), step_conv.data(), step_state.data(),
            step_inner.data() + row * intermediate, intermediate, state_size, heads,
            head_dim, groups, kernel, stream.get());
    }
    std::vector<__nv_bfloat16> batch_out(rows * intermediate), step_out(rows * intermediate),
        batch_cs(conv_dim * kernel), step_cs(conv_dim * kernel), batch_ss(intermediate * state_size),
        step_ss(intermediate * state_size);
    CELEG_CUDA(cudaMemcpyAsync(batch_out.data(), batch_inner.data(), batch_inner.bytes(), cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(step_out.data(), step_inner.data(), step_inner.bytes(), cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(batch_cs.data(), batch_conv.data(), batch_conv.bytes(), cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(step_cs.data(), step_conv.data(), step_conv.bytes(), cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(batch_ss.data(), batch_state.data(), batch_state.bytes(), cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(step_ss.data(), step_state.data(), step_state.bytes(), cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    for (size_t i = 0; i < batch_out.size(); ++i) expect_near(to_float(batch_out[i]), to_float(step_out[i]), 0.20f);
    for (size_t i = 0; i < batch_cs.size(); ++i) expect_near(to_float(batch_cs[i]), to_float(step_cs[i]), 0.20f);
    for (size_t i = 0; i < batch_ss.size(); ++i) expect_near(to_float(batch_ss[i]), to_float(step_ss[i]), 0.20f);
}

{
    constexpr int rows = 32, intermediate = 7680, state_size = 128,
        heads = 96, head_dim = 80, groups = 8, kernel = 4;
    constexpr int conv_dim = intermediate + 2 * groups * state_size;
    constexpr int projection = 2 * intermediate + 2 * groups * state_size + heads;
    std::vector<__nv_bfloat16> projected(static_cast<size_t>(rows) * projection, to_bf16(0.01f)),
        conv_weight(static_cast<size_t>(conv_dim) * kernel, to_bf16(0.02f)),
        conv_bias(conv_dim, to_bf16(0.01f)), dt_bias(heads, to_bf16(0.1f)),
        a_log(heads, to_bf16(-0.2f)), d(heads, to_bf16(0.3f));
    celeg::DeviceBuffer<__nv_bfloat16> dp(projected.size()), dw(conv_weight.size()),
        db(conv_bias.size()), ddt(dt_bias.size()), da(a_log.size()), dd(d.size()),
        conv_state(static_cast<size_t>(conv_dim) * kernel),
        ssm_state(static_cast<size_t>(intermediate) * state_size),
        inner(static_cast<size_t>(rows) * intermediate);
    CELEG_CUDA(cudaMemcpy(dp.data(), projected.data(), dp.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dw.data(), conv_weight.data(), dw.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(db.data(), conv_bias.data(), db.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(ddt.data(), dt_bias.data(), ddt.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(da.data(), a_log.data(), da.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dd.data(), d.data(), dd.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemset(conv_state.data(), 0, conv_state.bytes()));
    CELEG_CUDA(cudaMemset(ssm_state.data(), 0, ssm_state.bytes()));
    celeg::launch_mamba2_prefill(dp.data(), dw.data(), db.data(), ddt.data(), da.data(), dd.data(),
        conv_state.data(), ssm_state.data(), inner.data(), rows, intermediate, state_size,
        heads, head_dim, groups, kernel, stream.get());
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
}
}

}
