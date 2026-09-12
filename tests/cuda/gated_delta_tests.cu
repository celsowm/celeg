#include "gated_delta_tests.hpp"

#include "utils.cuh"
#include "../support/assertions.hpp"
#include "../support/cuda_kernel_assertions.cuh"
#include "kernels/kernels.cuh"
#include "celeg/backend/cpu/gated_delta.hpp"

#include <algorithm>
#include <vector>

namespace celeg::cuda_test {

void run_gated_delta_tests(celeg::CudaStream& stream) {
{
    constexpr int rows = 64, kernel = 4, dim = 2, key_heads = 1, value_heads = 2;
    constexpr int qkv_width = 2 * key_heads * dim + value_heads * dim;
    constexpr int value_width = value_heads * dim;
    std::vector<float> qkv(rows * qkv_width), z(rows * value_width), b(rows * value_heads),
        a(rows * value_heads), conv(qkv_width * kernel), dt(value_heads), alog(value_heads), norm(dim, 1.0f);
    for (size_t i = 0; i < qkv.size(); ++i) qkv[i] = 0.003f * static_cast<float>(i + 1);
    for (size_t i = 0; i < z.size(); ++i) z[i] = -0.02f * static_cast<float>(i + 1);
    for (size_t i = 0; i < b.size(); ++i) { b[i] = -0.2f; a[i] = 0.1f; }
    for (size_t i = 0; i < conv.size(); ++i) conv[i] = 0.02f;
    std::fill(dt.begin(), dt.end(), 0.5f); std::fill(alog.begin(), alog.end(), -0.3f);
    std::vector<float> cpu_conv(qkv_width * kernel), cpu_state(value_heads * dim * dim), cpu_out(rows * value_width);
    celeg::cpu_gated_delta_net_prefill(qkv.data(), z.data(), b.data(), a.data(), conv.data(), dt.data(), alog.data(), norm.data(), cpu_conv.data(), cpu_state.data(), cpu_out.data(), rows, kernel, dim, dim, key_heads, value_heads, 1e-6f);
    std::vector<__nv_bfloat16> hq(qkv.size()), hz(z.size()), hb(b.size()), ha(a.size()), hc(conv.size()), hdt(dt.size()), hal(alog.size()), hn(norm.size());
    for (size_t i=0;i<hq.size();++i) hq[i]=to_bf16(qkv[i]); for(size_t i=0;i<hz.size();++i) hz[i]=to_bf16(z[i]); for(size_t i=0;i<hb.size();++i){hb[i]=to_bf16(b[i]);ha[i]=to_bf16(a[i]);} for(size_t i=0;i<hc.size();++i)hc[i]=to_bf16(conv[i]); for(int i=0;i<value_heads;++i){hdt[i]=to_bf16(dt[i]);hal[i]=to_bf16(alog[i]);} for(int i=0;i<dim;++i)hn[i]=to_bf16(norm[i]);
    celeg::DeviceBuffer<__nv_bfloat16> dq(qkv.size()), dz(z.size()), db(b.size()), da(a.size()), dc(conv.size()), ddt(dt.size()), dal(alog.size()), dn(norm.size()), dcs(qkv_width*kernel), drs(value_heads*dim*dim), dout(rows*value_width);
    CELEG_CUDA(cudaMemcpy(dq.data(),hq.data(),dq.bytes(),cudaMemcpyHostToDevice)); CELEG_CUDA(cudaMemcpy(dz.data(),hz.data(),dz.bytes(),cudaMemcpyHostToDevice)); CELEG_CUDA(cudaMemcpy(db.data(),hb.data(),db.bytes(),cudaMemcpyHostToDevice)); CELEG_CUDA(cudaMemcpy(da.data(),ha.data(),da.bytes(),cudaMemcpyHostToDevice)); CELEG_CUDA(cudaMemcpy(dc.data(),hc.data(),dc.bytes(),cudaMemcpyHostToDevice)); CELEG_CUDA(cudaMemcpy(ddt.data(),hdt.data(),ddt.bytes(),cudaMemcpyHostToDevice)); CELEG_CUDA(cudaMemcpy(dal.data(),hal.data(),dal.bytes(),cudaMemcpyHostToDevice)); CELEG_CUDA(cudaMemcpy(dn.data(),hn.data(),dn.bytes(),cudaMemcpyHostToDevice));
    celeg::launch_gated_delta_net(dq.data(), dz.data(), db.data(), da.data(),
        dc.data(), ddt.data(), dal.data(), dn.data(), dcs.data(), drs.data(),
        dout.data(), rows, kernel, dim, dim, key_heads, value_heads, 1e-6f, false,
        false, -5.0f, false, stream.get());
    std::vector<__nv_bfloat16> got(cpu_out.size()), got_conv(cpu_conv.size()), got_state(cpu_state.size()); CELEG_CUDA(cudaMemcpyAsync(got.data(),dout.data(),dout.bytes(),cudaMemcpyDeviceToHost,stream.get())); CELEG_CUDA(cudaMemcpyAsync(got_conv.data(),dcs.data(),dcs.bytes(),cudaMemcpyDeviceToHost,stream.get())); CELEG_CUDA(cudaMemcpyAsync(got_state.data(),drs.data(),drs.bytes(),cudaMemcpyDeviceToHost,stream.get())); CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    for(size_t i=0;i<got.size();++i) expect_near(to_float(got[i]),cpu_out[i],0.03f);
    for(size_t i=0;i<got_conv.size();++i) expect_near(to_float(got_conv[i]),cpu_conv[i],0.03f);
    for(size_t i=0;i<got_state.size();++i) expect_near(to_float(got_state[i]),cpu_state[i],0.03f);
}

/// rows < 64 with key_heads != value_heads (GQA-style repeat), matching a
/// short prefill of a Qwen3.5-shaped gated-deltanet layer. This exact
/// combination previously fell into a single-row-only kernel path that
/// silently left every row past row 0 uncomputed.
{
    constexpr int rows = 5, kernel = 4, dim = 8, key_heads = 2, value_heads = 6;
    constexpr int qkv_width = 2 * key_heads * dim + value_heads * dim;
    constexpr int value_width = value_heads * dim;
    std::vector<float> qkv(rows * qkv_width), z(rows * value_width), b(rows * value_heads),
        a(rows * value_heads), conv(qkv_width * kernel), dt(value_heads), alog(value_heads), norm(dim, 1.0f);
    for (size_t i = 0; i < qkv.size(); ++i) qkv[i] = 0.003f * static_cast<float>(i + 1);
    for (size_t i = 0; i < z.size(); ++i) z[i] = -0.02f * static_cast<float>(i + 1);
    for (size_t i = 0; i < b.size(); ++i) { b[i] = -0.2f; a[i] = 0.1f; }
    for (size_t i = 0; i < conv.size(); ++i) conv[i] = 0.02f;
    std::fill(dt.begin(), dt.end(), 0.5f); std::fill(alog.begin(), alog.end(), -0.3f);
    std::vector<float> cpu_conv(qkv_width * kernel), cpu_state(value_heads * dim * dim), cpu_out(rows * value_width);
    celeg::cpu_gated_delta_net_prefill(qkv.data(), z.data(), b.data(), a.data(), conv.data(), dt.data(), alog.data(), norm.data(), cpu_conv.data(), cpu_state.data(), cpu_out.data(), rows, kernel, dim, dim, key_heads, value_heads, 1e-6f);
    std::vector<__nv_bfloat16> hq(qkv.size()), hz(z.size()), hb(b.size()), ha(a.size()), hc(conv.size()), hdt(dt.size()), hal(alog.size()), hn(norm.size());
    for (size_t i=0;i<hq.size();++i) hq[i]=to_bf16(qkv[i]); for(size_t i=0;i<hz.size();++i) hz[i]=to_bf16(z[i]); for(size_t i=0;i<hb.size();++i){hb[i]=to_bf16(b[i]);ha[i]=to_bf16(a[i]);} for(size_t i=0;i<hc.size();++i)hc[i]=to_bf16(conv[i]); for(int i=0;i<value_heads;++i){hdt[i]=to_bf16(dt[i]);hal[i]=to_bf16(alog[i]);} for(int i=0;i<dim;++i)hn[i]=to_bf16(norm[i]);
    celeg::DeviceBuffer<__nv_bfloat16> dq(qkv.size()), dz(z.size()), db(b.size()), da(a.size()), dc(conv.size()), ddt(dt.size()), dal(alog.size()), dn(norm.size()), dcs(qkv_width*kernel), drs(value_heads*dim*dim), dout(rows*value_width);
    CELEG_CUDA(cudaMemcpy(dq.data(),hq.data(),dq.bytes(),cudaMemcpyHostToDevice)); CELEG_CUDA(cudaMemcpy(dz.data(),hz.data(),dz.bytes(),cudaMemcpyHostToDevice)); CELEG_CUDA(cudaMemcpy(db.data(),hb.data(),db.bytes(),cudaMemcpyHostToDevice)); CELEG_CUDA(cudaMemcpy(da.data(),ha.data(),da.bytes(),cudaMemcpyHostToDevice)); CELEG_CUDA(cudaMemcpy(dc.data(),hc.data(),dc.bytes(),cudaMemcpyHostToDevice)); CELEG_CUDA(cudaMemcpy(ddt.data(),hdt.data(),ddt.bytes(),cudaMemcpyHostToDevice)); CELEG_CUDA(cudaMemcpy(dal.data(),hal.data(),dal.bytes(),cudaMemcpyHostToDevice)); CELEG_CUDA(cudaMemcpy(dn.data(),hn.data(),dn.bytes(),cudaMemcpyHostToDevice));
    celeg::launch_gated_delta_net(dq.data(), dz.data(), db.data(), da.data(),
        dc.data(), ddt.data(), dal.data(), dn.data(), dcs.data(), drs.data(),
        dout.data(), rows, kernel, dim, dim, key_heads, value_heads, 1e-6f, false,
        false, -5.0f, false, stream.get());
    std::vector<__nv_bfloat16> got(cpu_out.size()), got_conv(cpu_conv.size()), got_state(cpu_state.size()); CELEG_CUDA(cudaMemcpyAsync(got.data(),dout.data(),dout.bytes(),cudaMemcpyDeviceToHost,stream.get())); CELEG_CUDA(cudaMemcpyAsync(got_conv.data(),dcs.data(),dcs.bytes(),cudaMemcpyDeviceToHost,stream.get())); CELEG_CUDA(cudaMemcpyAsync(got_state.data(),drs.data(),drs.bytes(),cudaMemcpyDeviceToHost,stream.get())); CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    for(size_t i=0;i<got.size();++i) expect_near(to_float(got[i]),cpu_out[i],0.03f);
    for(size_t i=0;i<got_conv.size();++i) expect_near(to_float(got_conv[i]),cpu_conv[i],0.03f);
    for(size_t i=0;i<got_state.size();++i) expect_near(to_float(got_state[i]),cpu_state[i],0.03f);
}

/// Agnes-shaped fused gated-deltanet layer: 16 key heads against 48 value
/// heads (3x GQA-style repeat) at head dim 128, covering both the
/// multi-row generic prefill path and the single-row decode path with
/// key_heads != value_heads.
for (const int agnes_rows : {59, 1}) {
    constexpr int kernel = 4, key_dim = 128, value_dim = 128, key_heads = 16, value_heads = 48;
    constexpr int qkv_width = 2 * key_heads * key_dim + value_heads * value_dim;
    constexpr int value_width = value_heads * value_dim;
    const int rows = agnes_rows;
    std::vector<float> qkv(static_cast<size_t>(rows) * qkv_width), z(static_cast<size_t>(rows) * value_width), b(static_cast<size_t>(rows) * value_heads),
        a(static_cast<size_t>(rows) * value_heads), conv(qkv_width * kernel), dt(value_heads), alog(value_heads), norm(value_dim, 1.0f);
    for (size_t i = 0; i < qkv.size(); ++i) qkv[i] = 0.003f * static_cast<float>((i % 4096) + 1);
    for (size_t i = 0; i < z.size(); ++i) z[i] = -0.002f * static_cast<float>((i % 4096) + 1);
    for (size_t i = 0; i < b.size(); ++i) { b[i] = -0.2f; a[i] = 0.1f; }
    for (size_t i = 0; i < conv.size(); ++i) conv[i] = 0.02f;
    std::fill(dt.begin(), dt.end(), 0.5f); std::fill(alog.begin(), alog.end(), -0.3f);
    std::vector<float> cpu_conv(qkv_width * kernel), cpu_state(static_cast<size_t>(value_heads) * key_dim * value_dim), cpu_out(static_cast<size_t>(rows) * value_width);
    celeg::cpu_gated_delta_net_prefill(qkv.data(), z.data(), b.data(), a.data(), conv.data(), dt.data(), alog.data(), norm.data(), cpu_conv.data(), cpu_state.data(), cpu_out.data(), rows, kernel, key_dim, value_dim, key_heads, value_heads, 1e-6f);
    std::vector<__nv_bfloat16> hq(qkv.size()), hz(z.size()), hb(b.size()), ha(a.size()), hc(conv.size()), hdt(dt.size()), hal(alog.size()), hn(norm.size());
    for(size_t i=0;i<hq.size();++i) hq[i]=to_bf16(qkv[i]); for(size_t i=0;i<hz.size();++i) hz[i]=to_bf16(z[i]); for(size_t i=0;i<hb.size();++i){hb[i]=to_bf16(b[i]);ha[i]=to_bf16(a[i]);} for(size_t i=0;i<hc.size();++i)hc[i]=to_bf16(conv[i]); for(int i=0;i<value_heads;++i){hdt[i]=to_bf16(dt[i]);hal[i]=to_bf16(alog[i]);} for(int i=0;i<value_dim;++i)hn[i]=to_bf16(norm[i]);
    celeg::DeviceBuffer<__nv_bfloat16> dq(qkv.size()), dz(z.size()), db(b.size()), da(a.size()), dc(conv.size()), ddt(dt.size()), dal(alog.size()), dn(norm.size()), dcs(cpu_conv.size()), drs(cpu_state.size()), dout(cpu_out.size());
    CELEG_CUDA(cudaMemcpy(dq.data(),hq.data(),dq.bytes(),cudaMemcpyHostToDevice)); CELEG_CUDA(cudaMemcpy(dz.data(),hz.data(),dz.bytes(),cudaMemcpyHostToDevice)); CELEG_CUDA(cudaMemcpy(db.data(),hb.data(),db.bytes(),cudaMemcpyHostToDevice)); CELEG_CUDA(cudaMemcpy(da.data(),ha.data(),da.bytes(),cudaMemcpyHostToDevice)); CELEG_CUDA(cudaMemcpy(dc.data(),hc.data(),dc.bytes(),cudaMemcpyHostToDevice)); CELEG_CUDA(cudaMemcpy(ddt.data(),hdt.data(),ddt.bytes(),cudaMemcpyHostToDevice)); CELEG_CUDA(cudaMemcpy(dal.data(),hal.data(),dal.bytes(),cudaMemcpyHostToDevice)); CELEG_CUDA(cudaMemcpy(dn.data(),hn.data(),dn.bytes(),cudaMemcpyHostToDevice));
    celeg::launch_gated_delta_net(dq.data(), dz.data(), db.data(), da.data(),
        dc.data(), ddt.data(), dal.data(), dn.data(), dcs.data(), drs.data(),
        dout.data(), rows, kernel, key_dim, value_dim, key_heads, value_heads, 1e-6f, false,
        false, -5.0f, false, stream.get());
    std::vector<__nv_bfloat16> got(cpu_out.size()), got_conv(cpu_conv.size()), got_state(cpu_state.size()); CELEG_CUDA(cudaMemcpyAsync(got.data(),dout.data(),dout.bytes(),cudaMemcpyDeviceToHost,stream.get())); CELEG_CUDA(cudaMemcpyAsync(got_conv.data(),dcs.data(),dcs.bytes(),cudaMemcpyDeviceToHost,stream.get())); CELEG_CUDA(cudaMemcpyAsync(got_state.data(),drs.data(),drs.bytes(),cudaMemcpyDeviceToHost,stream.get())); CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    for(size_t i=0;i<got.size();++i) expect_near(to_float(got[i]),cpu_out[i],0.05f);
    for(size_t i=0;i<got_conv.size();++i) expect_near(to_float(got_conv[i]),cpu_conv[i],0.05f);
    for(size_t i=0;i<got_state.size();++i) expect_near(to_float(got_state[i]),cpu_state[i],0.05f);
}

{
    constexpr int rows = 1, kernel = 4, dim = 128, heads = 1;
    constexpr int qkv_width = 3 * dim, value_width = dim;
    std::vector<float> qkv(qkv_width), z(value_width), b(heads, -0.2f),
        a(heads, 0.1f), conv(qkv_width * kernel, 0.02f), dt(heads, 0.5f),
        alog(heads, -0.3f), norm(dim, 1.0f), cpu_conv(qkv_width * kernel),
        cpu_state(dim * dim), cpu_out(value_width);
    for (size_t i = 0; i < qkv.size(); ++i) qkv[i] = 0.001f * static_cast<float>(i + 1);
    for (size_t i = 0; i < z.size(); ++i) z[i] = -0.002f * static_cast<float>(i + 1);
    celeg::cpu_gated_delta_net_prefill(qkv.data(), z.data(), b.data(), a.data(), conv.data(),
        dt.data(), alog.data(), norm.data(), cpu_conv.data(), cpu_state.data(), cpu_out.data(),
        rows, kernel, dim, dim, heads, heads, 1e-6f);
    auto convert = [](const std::vector<float>& values) {
        std::vector<__nv_bfloat16> result(values.size());
        for (size_t i = 0; i < values.size(); ++i) result[i] = to_bf16(values[i]);
        return result;
    };
    const auto hq = convert(qkv), hz = convert(z), hb = convert(b), ha = convert(a),
        hc = convert(conv), hdt = convert(dt), hal = convert(alog), hn = convert(norm);
    celeg::DeviceBuffer<__nv_bfloat16> dq(qkv.size()), dz(z.size()), db(b.size()), da(a.size()),
        dc(conv.size()), ddt(dt.size()), dal(alog.size()), dn(norm.size()),
        dcs(cpu_conv.size()), drs(cpu_state.size()), dout(cpu_out.size());
    CELEG_CUDA(cudaMemcpy(dq.data(), hq.data(), dq.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dz.data(), hz.data(), dz.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(db.data(), hb.data(), db.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(da.data(), ha.data(), da.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dc.data(), hc.data(), dc.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(ddt.data(), hdt.data(), ddt.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dal.data(), hal.data(), dal.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dn.data(), hn.data(), dn.bytes(), cudaMemcpyHostToDevice));
    celeg::launch_gated_delta_net(dq.data(), dz.data(), db.data(), da.data(), dc.data(),
        ddt.data(), dal.data(), dn.data(), dcs.data(), drs.data(), dout.data(), rows, kernel,
        dim, dim, heads, heads, 1e-6f, false, false, -5.0f, false, stream.get());
    std::vector<__nv_bfloat16> got_out(cpu_out.size()), got_conv(cpu_conv.size()), got_state(cpu_state.size());
    CELEG_CUDA(cudaMemcpyAsync(got_out.data(), dout.data(), dout.bytes(), cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(got_conv.data(), dcs.data(), dcs.bytes(), cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(got_state.data(), drs.data(), drs.bytes(), cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    for (size_t i = 0; i < got_out.size(); ++i) expect_near(to_float(got_out[i]), cpu_out[i], 0.03f);
    for (size_t i = 0; i < got_conv.size(); ++i) expect_near(to_float(got_conv[i]), cpu_conv[i], 0.03f);
    for (size_t i = 0; i < got_state.size(); ++i) expect_near(to_float(got_state[i]), cpu_state[i], 0.03f);
}
}

}
