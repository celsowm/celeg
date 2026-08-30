#include "attention_tests.hpp"

#include "utils.cuh"
#include "../support/assertions.hpp"
#include "../support/cuda_kernel_assertions.cuh"
#include "kernels/kernels.cuh"
#include "backend/cuda/paged_kv.hpp"
#include "celeg/model/reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace celeg::cuda_test {

void run_attention_tests(celeg::CudaStream& stream) {
{
    std::vector<__nv_bfloat16> q = {
        to_bf16(1.0f), to_bf16(2.0f), to_bf16(3.0f), to_bf16(4.0f)};
    std::vector<__nv_bfloat16> k = {
        to_bf16(4.0f), to_bf16(3.0f), to_bf16(2.0f), to_bf16(1.0f)};
    std::vector<__nv_bfloat16> norm(4, to_bf16(1.0f));
    celeg::DeviceBuffer<__nv_bfloat16> dq(4), dk(4), dn(4);
    CELEG_CUDA(cudaMemcpy(dq.data(), q.data(), dq.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dk.data(), k.data(), dk.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dn.data(), norm.data(), dn.bytes(), cudaMemcpyHostToDevice));
    celeg::launch_dynamic_qk_norm_rope(
        dq.data(), dk.data(), dn.data(), dn.data(), 1, 1, 4, 0,
        10000.0f, 1.0f, 1e-5f, true, celeg::CudaRopeScaling{},
        celeg::RopePairingKind::SplitHalf, stream.get());
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    CELEG_CUDA(cudaMemcpy(q.data(), dq.data(), dq.bytes(), cudaMemcpyDeviceToHost));
    CELEG_CUDA(cudaMemcpy(k.data(), dk.data(), dk.bytes(), cudaMemcpyDeviceToHost));
    std::vector<float> expected_q = celeg::reference::rmsnorm_bf16(
        {1, 2, 3, 4}, {1, 1, 1, 1}, 1e-5f);
    std::vector<float> expected_k = celeg::reference::rmsnorm_bf16(
        {4, 3, 2, 1}, {1, 1, 1, 1}, 1e-5f);
    for (int i = 0; i < 4; ++i) {
        expect_near(to_float(q[i]), expected_q[i], 0.01f);
        expect_near(to_float(k[i]), expected_k[i], 0.01f);
    }
}

{
    std::vector<float> qf = {1, 0, 1, 0};
    std::vector<float> kf = {1, 0, 0, 1};
    std::vector<float> vf = {2, 4, 6, 8};
    std::vector<__nv_bfloat16> q(4), k(4), v(4);
    for (int i = 0; i < 4; ++i) {
        q[i] = to_bf16(qf[i]);
        k[i] = to_bf16(kf[i]);
        v[i] = to_bf16(vf[i]);
    }
    celeg::DeviceBuffer<__nv_bfloat16> dq(4), dk(4), dv(4), dout(4);
    CELEG_CUDA(cudaMemcpy(dq.data(), q.data(), dq.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dk.data(), k.data(), dk.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dv.data(), v.data(), dv.bytes(), cudaMemcpyHostToDevice));
    celeg::launch_gqa_decode_strict({
        .query = dq.data(),
        .kv = {.keys = dk.data(), .values = dv.data()},
        .out = dout.data(),
        .geometry = {.q_heads = 2, .kv_heads = 1, .head_dim = 2},
        .extent = {.seq_len = 2},
        .stream = stream.get()});
    std::vector<__nv_bfloat16> output(4);
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    CELEG_CUDA(cudaMemcpy(output.data(), dout.data(), dout.bytes(), cudaMemcpyDeviceToHost));
    const auto expected = celeg::reference::gqa_decode_strict_bf16(
        qf, kf, vf, 2, 2, 1, 2);
    for (int i = 0; i < 4; ++i) expect_near(to_float(output[i]), expected[i], 0.01f);
}

{
    const std::vector<__nv_bfloat16> q = {to_bf16(1), to_bf16(0)};
    const std::vector<__nv_bfloat16> k = {
        to_bf16(8), to_bf16(0),
        to_bf16(0), to_bf16(1),
        to_bf16(1), to_bf16(0)};
    const std::vector<__nv_bfloat16> v = {
        to_bf16(100), to_bf16(200),
        to_bf16(10), to_bf16(20),
        to_bf16(3), to_bf16(7)};
    celeg::DeviceBuffer<__nv_bfloat16> dq(q.size()), dk(k.size()), dv(v.size());
    celeg::DeviceBuffer<__nv_bfloat16> full_out(2), sliding_out(2);
    CELEG_CUDA(cudaMemcpy(dq.data(), q.data(), dq.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dk.data(), k.data(), dk.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dv.data(), v.data(), dv.bytes(), cudaMemcpyHostToDevice));
    celeg::launch_gqa_decode_strict({
        .query = dq.data(),
        .kv = {.keys = dk.data(), .values = dv.data()},
        .out = full_out.data(),
        .geometry = {.q_heads = 1, .kv_heads = 1, .head_dim = 2,
                     .sliding_window = 0},
        .extent = {.seq_len = 3},
        .stream = stream.get()});
    celeg::launch_gqa_decode_strict({
        .query = dq.data(),
        .kv = {.keys = dk.data(), .values = dv.data()},
        .out = sliding_out.data(),
        .geometry = {.q_heads = 1, .kv_heads = 1, .head_dim = 2,
                     .sliding_window = 1},
        .extent = {.seq_len = 3},
        .stream = stream.get()});
    std::array<__nv_bfloat16, 2> full_host{};
    std::array<__nv_bfloat16, 2> sliding_host{};
    CELEG_CUDA(cudaMemcpyAsync(full_host.data(), full_out.data(), full_out.bytes(),
                               cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(sliding_host.data(), sliding_out.data(), sliding_out.bytes(),
                               cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    expect_near(to_float(sliding_host[0]), 3.0f, 0.01f);
    expect_near(to_float(sliding_host[1]), 7.0f, 0.01f);
    CELEG_TEST_CHECK(std::abs(to_float(full_host[0]) - to_float(sliding_host[0])) > 1.0f);
}

{
    const std::vector<float> qf = {1, 0};
    const std::vector<float> kf = {1, 0, 0, 1};
    const std::vector<float> vf = {2, 4, 6, 8};
    const std::vector<__nv_bfloat16> q = {to_bf16(1), to_bf16(0)};
    const std::vector<__nv_bfloat16> k = {
        to_bf16(1), to_bf16(0), to_bf16(0), to_bf16(1)};
    const std::vector<__nv_bfloat16> v = {
        to_bf16(2), to_bf16(4), to_bf16(6), to_bf16(8)};
    const float slope = 0.5f;
    const int32_t position = 1;
    celeg::DeviceBuffer<__nv_bfloat16> dq(2), dk(4), dv(4), dout(2);
    celeg::DeviceBuffer<float> dslope(1);
    celeg::DeviceBuffer<int32_t> dposition(1);
    CELEG_CUDA(cudaMemcpy(dq.data(), q.data(), dq.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dk.data(), k.data(), dk.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dv.data(), v.data(), dv.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dslope.data(), &slope, sizeof(slope), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dposition.data(), &position, sizeof(position), cudaMemcpyHostToDevice));
    celeg::launch_gqa_decode_alibi_device({
        .query = dq.data(),
        .kv = {.keys = dk.data(), .values = dv.data()},
        .out = dout.data(),
        .geometry = {.q_heads = 1, .kv_heads = 1, .head_dim = 2},
        .extent = {.position = dposition.data()},
        .alibi_slopes = dslope.data(),
        .stream = stream.get()});
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    std::vector<__nv_bfloat16> output(2);
    CELEG_CUDA(cudaMemcpy(output.data(), dout.data(), dout.bytes(), cudaMemcpyDeviceToHost));
    const float s0 = std::exp(1.0f / std::sqrt(2.0f) - 0.5f);
    const float s1 = std::exp(0.0f);
    const float denominator = s0 + s1;
    expect_near(to_float(output[0]), (s0 * 2 + s1 * 6) / denominator, 0.02f);
    expect_near(to_float(output[1]), (s0 * 4 + s1 * 8) / denominator, 0.02f);
}

{
    constexpr int latent_rank = 2;
    constexpr int rotary_width = 2;
    constexpr int page_tokens = 2;
    constexpr int page_elements = page_tokens * (latent_rank + rotary_width);
    constexpr int page_count = 2;
    const std::vector<__nv_bfloat16> query_content = {
        to_bf16(1.0f), to_bf16(0.0f)};
    const std::vector<__nv_bfloat16> query_rope = {
        to_bf16(1.0f), to_bf16(0.0f)};
    const std::vector<__nv_bfloat16> keys = {
        to_bf16(1.0f), to_bf16(0.0f),
        to_bf16(0.0f), to_bf16(1.0f)};
    const std::vector<__nv_bfloat16> values = {
        to_bf16(2.0f), to_bf16(4.0f), to_bf16(6.0f), to_bf16(8.0f)};
    const std::vector<__nv_bfloat16> key_ropes = {
        to_bf16(1.0f), to_bf16(0.0f), to_bf16(0.0f), to_bf16(1.0f)};
    const std::vector<uint32_t> page_table = {1, 1};
    const std::vector<int32_t> positions = {0, 1};
    const int32_t query_position = 1;
    celeg::DeviceBuffer<__nv_bfloat16> dquery_content(query_content.size());
    celeg::DeviceBuffer<__nv_bfloat16> dquery_rope(query_rope.size());
    celeg::DeviceBuffer<__nv_bfloat16> dkeys(keys.size());
    celeg::DeviceBuffer<__nv_bfloat16> dvalues(values.size());
    celeg::DeviceBuffer<__nv_bfloat16> dkey_ropes(key_ropes.size());
    celeg::DeviceBuffer<__nv_bfloat16> key_pool(page_count * page_elements);
    celeg::DeviceBuffer<__nv_bfloat16> value_pool(page_count * page_elements);
    celeg::DeviceBuffer<__nv_bfloat16> output(latent_rank);
    celeg::DeviceBuffer<uint32_t> dpage_table(page_table.size());
    celeg::DeviceBuffer<int32_t> dpositions(positions.size());
    celeg::DeviceBuffer<int32_t> dquery_position(1);
    key_pool.zero_async(stream.get());
    value_pool.zero_async(stream.get());
    CELEG_CUDA(cudaMemcpy(dquery_content.data(), query_content.data(),
                          dquery_content.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dquery_rope.data(), query_rope.data(),
                          dquery_rope.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dkeys.data(), keys.data(), dkeys.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dvalues.data(), values.data(), dvalues.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dkey_ropes.data(), key_ropes.data(), dkey_ropes.bytes(),
                          cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dpage_table.data(), page_table.data(), dpage_table.bytes(),
                          cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dpositions.data(), positions.data(), dpositions.bytes(),
                          cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dquery_position.data(), &query_position,
                          sizeof(query_position), cudaMemcpyHostToDevice));
    celeg::launch_store_latent_paged_batch(
        dkeys.data(), dvalues.data(), dkey_ropes.data(), key_pool.data(),
        value_pool.data(), dpage_table.data(), 1, dpositions.data(), 2, 0,
        page_tokens, page_elements, 0, latent_rank, rotary_width, stream.get());
    celeg::launch_latent_attention_paged_batch({
        .query = {.content = dquery_content.data(), .rope = dquery_rope.data()},
        .kv = {.keys = key_pool.data(), .values = value_pool.data()},
        .index = {.page_tables = dpage_table.data(),
                  .page_table_stride = 1,
                  .attention_slot = 0,
                  .page_tokens = page_tokens,
                  .page_vector_elements = page_elements,
                  .layer_vector_offset = 0},
        .out = output.data(),
        .positions = dquery_position.data(),
        .rows = 1,
        .geometry = {.query_heads = 1,
                     .latent_rank = latent_rank,
                     .rotary_width = rotary_width,
                     .score_scale = 1.0f,
                     .sliding_window = 0},
        .stream = stream.get()});
    std::vector<__nv_bfloat16> host_output(latent_rank);
    CELEG_CUDA(cudaMemcpyAsync(host_output.data(), output.data(), output.bytes(),
                               cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    const float s0 = std::exp(2.0f);
    const float denominator = s0 + 1.0f;
    expect_near(to_float(host_output[0]), (s0 * 2.0f + 6.0f) / denominator, 0.03f);
    expect_near(to_float(host_output[1]), (s0 * 4.0f + 8.0f) / denominator, 0.03f);
}

{
    constexpr int hidden = 2;
    constexpr int cache = 3;
    std::vector<float> projected_f = {1, 1, 1, 1, 2, 4};
    std::vector<float> weight_f = {1, 2, 3, 1, 2, 3};
    std::vector<__nv_bfloat16> projected(projected_f.size());
    std::vector<__nv_bfloat16> weight(weight_f.size());
    for (size_t i = 0; i < projected.size(); ++i) projected[i] = to_bf16(projected_f[i]);
    for (size_t i = 0; i < weight.size(); ++i) weight[i] = to_bf16(weight_f[i]);
    celeg::DeviceBuffer<__nv_bfloat16> dp(projected.size()), dw(weight.size());
    celeg::DeviceBuffer<__nv_bfloat16> ds(hidden * cache), dy(hidden);
    CELEG_CUDA(cudaMemcpy(dp.data(), projected.data(), dp.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dw.data(), weight.data(), dw.bytes(), cudaMemcpyHostToDevice));
    ds.zero_async(stream.get());
    celeg::launch_conv_decode(dp.data(), dw.data(), ds.data(), dy.data(),
                            hidden, cache, 0, stream.get());
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    std::vector<__nv_bfloat16> output(hidden);
    CELEG_CUDA(cudaMemcpy(output.data(), dy.data(), dy.bytes(), cudaMemcpyDeviceToHost));
    std::vector<float> state(hidden * cache, 0.0f);
    const auto expected = celeg::reference::conv_decode_bf16(
        projected_f, weight_f, state, hidden, cache, 0);
    for (int i = 0; i < hidden; ++i) {
        expect_near(to_float(output[static_cast<size_t>(i)]),
                    expected[static_cast<size_t>(i)], 0.01f);
    }
}



{
    constexpr int rows = 4;
    constexpr int hidden = 2;
    constexpr int cache = 3;
    std::vector<float> projected_f = {
        1, 2,  1, 1,  1, 2,
        2, 1,  1, 2,  2, 1,
        1, 1,  2, 1,  3, 2,
        2, 2,  1, 1,  1, 3,
    };
    std::vector<float> weight_f = {1, 2, 3, 3, 2, 1};
    std::vector<__nv_bfloat16> projected(projected_f.size());
    std::vector<__nv_bfloat16> weight(weight_f.size());
    for (size_t i = 0; i < projected.size(); ++i) projected[i] = to_bf16(projected_f[i]);
    for (size_t i = 0; i < weight.size(); ++i) weight[i] = to_bf16(weight_f[i]);
    celeg::DeviceBuffer<__nv_bfloat16> dp(projected.size()), dw(weight.size());
    celeg::DeviceBuffer<__nv_bfloat16> ds(hidden * cache), dy(rows * hidden);
    CELEG_CUDA(cudaMemcpy(dp.data(), projected.data(), dp.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dw.data(), weight.data(), dw.bytes(), cudaMemcpyHostToDevice));
    ds.zero_async(stream.get());
    celeg::launch_conv_prefill(dp.data(), dw.data(), ds.data(), dy.data(),
                             rows, hidden, cache, stream.get());
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    std::vector<__nv_bfloat16> output(rows * hidden), state_gpu(hidden * cache);
    CELEG_CUDA(cudaMemcpy(output.data(), dy.data(), dy.bytes(), cudaMemcpyDeviceToHost));
    CELEG_CUDA(cudaMemcpy(state_gpu.data(), ds.data(), ds.bytes(), cudaMemcpyDeviceToHost));

    std::vector<float> state(hidden * cache, 0.0f);
    for (int row = 0; row < rows; ++row) {
        std::vector<float> one(projected_f.begin() + row * 3 * hidden,
                               projected_f.begin() + (row + 1) * 3 * hidden);
        const auto expected = celeg::reference::conv_decode_bf16(
            one, weight_f, state, hidden, cache, row);
        for (int c = 0; c < hidden; ++c) {
            expect_near(to_float(output[row * hidden + c]), expected[c], 0.02f);
        }
    }
    for (int i = 0; i < hidden * cache; ++i) {
        expect_near(to_float(state_gpu[i]), state[i], 0.01f);
    }
}

{
    constexpr int requests = 2;
    constexpr int hidden = 2;
    constexpr int cache = 3;
    const std::vector<int> offsets = {0, 3};
    const std::vector<int> counts = {3, 2};
    const std::vector<int> positions = {1, 2, 3, 4, 5};
    std::vector<float> projected_f = {
        1, 2, 1, 1, 1, 2,  2, 1, 1, 2, 2, 1,  1, 1, 2, 1, 3, 2,
        3, 1, 1, 1, 2, 2,  2, 2, 1, 3, 1, 2,
    };
    std::vector<float> weight_f = {1, 2, 3, 3, 2, 1};
    std::vector<__nv_bfloat16> projected(projected_f.size()), weight(weight_f.size());
    for (size_t i = 0; i < projected.size(); ++i) projected[i] = to_bf16(projected_f[i]);
    for (size_t i = 0; i < weight.size(); ++i) weight[i] = to_bf16(weight_f[i]);
    celeg::DeviceBuffer<__nv_bfloat16> dp(projected.size()), dw(weight.size()),
        ds0(hidden * cache), ds1(hidden * cache), dy(positions.size() * hidden);
    celeg::DeviceBuffer<__nv_bfloat16*> states(requests);
    celeg::DeviceBuffer<int32_t> dpositions(positions.size()), doffsets(requests), dcounts(requests);
    const std::vector<__nv_bfloat16*> state_ptrs = {ds0.data(), ds1.data()};
    CELEG_CUDA(cudaMemcpy(dp.data(), projected.data(), dp.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dw.data(), weight.data(), dw.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(states.data(), state_ptrs.data(), states.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dpositions.data(), positions.data(), dpositions.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(doffsets.data(), offsets.data(), doffsets.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dcounts.data(), counts.data(), dcounts.bytes(), cudaMemcpyHostToDevice));
    ds0.zero_async(stream.get());
    ds1.zero_async(stream.get());
    celeg::launch_conv_ragged_prefill(dp.data(), dw.data(), states.data(), dy.data(),
                                    dpositions.data(), doffsets.data(), dcounts.data(),
                                    requests, hidden, cache, stream.get());
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    std::vector<__nv_bfloat16> output(positions.size() * hidden), state0(hidden * cache), state1(hidden * cache);
    CELEG_CUDA(cudaMemcpy(output.data(), dy.data(), dy.bytes(), cudaMemcpyDeviceToHost));
    CELEG_CUDA(cudaMemcpy(state0.data(), ds0.data(), ds0.bytes(), cudaMemcpyDeviceToHost));
    CELEG_CUDA(cudaMemcpy(state1.data(), ds1.data(), ds1.bytes(), cudaMemcpyDeviceToHost));
    std::vector<float> expected0(hidden * cache, 0.0f), expected1(hidden * cache, 0.0f);
    for (int request = 0; request < requests; ++request) {
        std::vector<float>& state = request == 0 ? expected0 : expected1;
        for (int token = 0; token < counts[request]; ++token) {
            const int row = offsets[request] + token;
            const std::vector<float> one(projected_f.begin() + row * 3 * hidden,
                                         projected_f.begin() + (row + 1) * 3 * hidden);
            const auto expected = celeg::reference::conv_decode_bf16(
                one, weight_f, state, hidden, cache, positions[row]);
            for (int channel = 0; channel < hidden; ++channel) {
                expect_near(to_float(output[row * hidden + channel]), expected[channel], 0.02f);
            }
        }
    }
    for (int i = 0; i < hidden * cache; ++i) {
        expect_near(to_float(state0[i]), expected0[i], 0.01f);
        expect_near(to_float(state1[i]), expected1[i], 0.01f);
    }
}

{
    constexpr int rows = 2;
    constexpr int q_heads = 2;
    constexpr int kv_heads = 1;
    constexpr int head_dim = 2;
    std::vector<float> qf = {1, 0, 0, 1, 1, 1, 1, -1};
    std::vector<float> kf = {1, 0, 0, 1};
    std::vector<float> vf = {2, 4, 6, 8};
    std::vector<__nv_bfloat16> q(qf.size()), k(kf.size()), v(vf.size());
    for (size_t i = 0; i < q.size(); ++i) q[i] = to_bf16(qf[i]);
    for (size_t i = 0; i < k.size(); ++i) k[i] = to_bf16(kf[i]);
    for (size_t i = 0; i < v.size(); ++i) v[i] = to_bf16(vf[i]);
    celeg::DeviceBuffer<__nv_bfloat16> dq(q.size()), dk(k.size()), dv(v.size()), dout(q.size());
    CELEG_CUDA(cudaMemcpy(dq.data(), q.data(), dq.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dk.data(), k.data(), dk.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dv.data(), v.data(), dv.bytes(), cudaMemcpyHostToDevice));
    celeg::launch_gqa_prefill_strict({
        .query = dq.data(),
        .kv = {.keys = dk.data(), .values = dv.data()},
        .out = dout.data(),
        .geometry = {.q_heads = q_heads, .kv_heads = kv_heads, .head_dim = head_dim},
        .extent = {.rows = rows},
        .stream = stream.get()});
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    std::vector<__nv_bfloat16> output(q.size());
    CELEG_CUDA(cudaMemcpy(output.data(), dout.data(), dout.bytes(), cudaMemcpyDeviceToHost));
    for (int row = 0; row < rows; ++row) {
        std::vector<float> row_q(qf.begin() + row * q_heads * head_dim,
                                 qf.begin() + (row + 1) * q_heads * head_dim);
        std::vector<float> prefix_k(kf.begin(), kf.begin() + (row + 1) * kv_heads * head_dim);
        std::vector<float> prefix_v(vf.begin(), vf.begin() + (row + 1) * kv_heads * head_dim);
        const auto expected = celeg::reference::gqa_decode_strict_bf16(
            row_q, prefix_k, prefix_v, row + 1, q_heads, kv_heads, head_dim);
        for (int i = 0; i < q_heads * head_dim; ++i) {
            expect_near(to_float(output[row * q_heads * head_dim + i]), expected[i], 0.02f);
        }
    }
}

{
    constexpr int rows = 2;
    constexpr int q_heads = 2;
    constexpr int kv_heads = 1;
    constexpr int head_dim = 4;
    std::vector<float> qf = {
        1, 0, 0, 1,  0, 1, 1, 0,
        1, 1, 0, 0,  0, 0, 1, 1};
    std::vector<float> kf = {1, 0, -1, 0.5f,  0, 1, 0.5f, -1};
    std::vector<float> vf = {2, 4, 6, 8,  1, 3, 5, 7};
    std::vector<__nv_bfloat16> q(qf.size()), k(kf.size()), v(vf.size());
    for (size_t i = 0; i < q.size(); ++i) q[i] = to_bf16(qf[i]);
    for (size_t i = 0; i < k.size(); ++i) k[i] = to_bf16(kf[i]);
    for (size_t i = 0; i < v.size(); ++i) v[i] = to_bf16(vf[i]);
    celeg::DeviceBuffer<__nv_bfloat16> dq(q.size()), dk(k.size()),
        dv(v.size()), dout_int8(q.size()), dout_bf16(q.size());
    celeg::DeviceBuffer<int8_t> key_cache(k.size()), value_cache(v.size());
    celeg::DeviceBuffer<float> key_scales(rows * kv_heads),
        value_scales(rows * kv_heads);
    CELEG_CUDA(cudaMemcpy(dq.data(), q.data(), dq.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dk.data(), k.data(), dk.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dv.data(), v.data(), dv.bytes(), cudaMemcpyHostToDevice));
    celeg::launch_store_kv_int8_prefill(
        dk.data(), dv.data(), key_cache.data(), value_cache.data(),
        key_scales.data(), value_scales.data(), rows, kv_heads, head_dim,
        stream.get());
    celeg::launch_gqa_prefill_strict_int8({
        .query = dq.data(),
        .kv = {.keys = key_cache.data(),
               .values = value_cache.data(),
               .key_scales = key_scales.data(),
               .value_scales = value_scales.data()},
        .out = dout_int8.data(),
        .geometry = {.q_heads = q_heads, .kv_heads = kv_heads, .head_dim = head_dim},
        .extent = {.rows = rows},
        .stream = stream.get()});
    celeg::launch_gqa_prefill_strict({
        .query = dq.data(),
        .kv = {.keys = dk.data(), .values = dv.data()},
        .out = dout_bf16.data(),
        .geometry = {.q_heads = q_heads, .kv_heads = kv_heads, .head_dim = head_dim},
        .extent = {.rows = rows},
        .stream = stream.get()});
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    std::vector<__nv_bfloat16> int8_output(q.size()), bf16_output(q.size());
    CELEG_CUDA(cudaMemcpy(int8_output.data(), dout_int8.data(),
                        dout_int8.bytes(), cudaMemcpyDeviceToHost));
    CELEG_CUDA(cudaMemcpy(bf16_output.data(), dout_bf16.data(),
                        dout_bf16.bytes(), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < int8_output.size(); ++i) {
        expect_near(to_float(int8_output[i]), to_float(bf16_output[i]), 0.08f);
    }
    std::vector<float> host_scales(rows * kv_heads);
    CELEG_CUDA(cudaMemcpy(host_scales.data(), value_scales.data(),
                        value_scales.bytes(), cudaMemcpyDeviceToHost));
    expect_near(host_scales[0], 8.0f / 127.0f, 1e-4f);
    expect_near(host_scales[1], 7.0f / 127.0f, 1e-4f);
}

}

}
