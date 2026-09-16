#include "attention_tests.hpp"

#include "utils.cuh"
#include "../support/assertions.hpp"
#include "../support/cuda_kernel_assertions.cuh"
#include "../support/numerical_compare.hpp"
#include "kernels/kernels.cuh"
#include "backend/cuda/paged_kv.hpp"
#include "celeg/model/reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
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

/// Partial-rotary reference: full-width RMS norm, then the first
/// `rotary_pairs` pairs rotate. Proportional RoPE pads its table to the full
/// head width, so pairs are full-dimension pairs; other scalings keep
/// prefix-local pairs.
auto partial_rope_reference = [](const std::vector<float>& input,
                                 const std::vector<float>& weight,
                                 int head_dim, int rotary_dim,
                                 bool full_dim_pairs, double theta,
                                 double fraction, bool proportional,
                                 int position, float eps) {
    const int rotated_pairs = rotary_dim / 2;
    const int pair_count = full_dim_pairs ? head_dim / 2 : rotated_pairs;
    std::vector<float> output = input;
    double sum = 0.0;
    for (float v : output) sum += static_cast<double>(v) * v;
    const float inv =
        1.0f / std::sqrt(static_cast<float>(sum / head_dim) + eps);
    for (int d = 0; d < head_dim; ++d) output[d] *= inv * weight[d];
    for (int pair = 0; pair < rotated_pairs; ++pair) {
        const int first = pair;
        const int second = pair_count + pair;
        double frequency =
            std::pow(theta, -2.0 * static_cast<double>(pair) /
                                static_cast<double>(rotary_dim));
        if (proportional) frequency = std::pow(frequency, fraction);
        const float angle =
            static_cast<float>(position) * static_cast<float>(frequency);
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        const float a = output[first];
        const float b = output[second];
        output[first] = a * c - b * s;
        output[second] = b * c + a * s;
    }
    return output;
};

{
    /// Proportional partial rotary (gemma-4 full-attention convention):
    /// head_dim 8, rotary fraction 0.25, theta 1e6. The rotated pair must be
    /// (0, 4); the old kernel rotated (0, 1).
    const std::vector<float> weight = {0.5f, 0.625f, 0.75f, 0.875f,
                                       1.0f, 1.125f, 1.25f, 1.375f};
    /// Loud partner swap: dims 1 and 4 differ by ~500x, so rotating (0, 1)
    /// instead of (0, 4) misses by O(1), well above bf16 noise.
    std::vector<float> input = {10, 1, 2, 3, 500, 5, 6, 7};
    const std::vector<float> expected = partial_rope_reference(
        input, weight, 8, 2, true, 1000000.0, 0.25, true, 7, 1e-6f);
    std::vector<__nv_bfloat16> q(8), norm(8);
    for (int i = 0; i < 8; ++i) {
        q[i] = to_bf16(input[i]);
        norm[i] = to_bf16(weight[i]);
    }
    celeg::DeviceBuffer<__nv_bfloat16> dq(8), dn(8);
    CELEG_CUDA(cudaMemcpy(dq.data(), q.data(), dq.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dn.data(), norm.data(), dn.bytes(), cudaMemcpyHostToDevice));
    celeg::CudaRopeScaling scaling{};
    scaling.kind = 6;
    scaling.factor = 1.0f;
    scaling.rotary_fraction = 0.25f;
    celeg::launch_dynamic_qk_norm_rope(
        dq.data(), nullptr, dn.data(), nullptr, 1, 0, 8, 7,
        1000000.0f, 0.25f, 1e-6f, true, scaling,
        celeg::RopePairingKind::SplitHalf, stream.get());
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    CELEG_CUDA(cudaMemcpy(q.data(), dq.data(), dq.bytes(), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 8; ++i) expect_near(to_float(q[i]), expected[i], 0.05f);
}

{
    /// Legacy partial rotary (prefix pairs + normalized tail): head_dim 8,
    /// rotary fraction 0.5, default scaling. Guards the proportional scoping
    /// above -- legacy pairing must not move.
    const std::vector<float> weight = {1.5f, 1.375f, 1.25f, 1.125f,
                                       1.0f, 0.875f, 0.75f, 0.625f};
    std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 320};
    const std::vector<float> expected = partial_rope_reference(
        input, weight, 8, 4, false, 10000.0, 0.5, false, 3, 1e-6f);
    std::vector<__nv_bfloat16> q(8), norm(8);
    for (int i = 0; i < 8; ++i) {
        q[i] = to_bf16(input[i]);
        norm[i] = to_bf16(weight[i]);
    }
    celeg::DeviceBuffer<__nv_bfloat16> dq(8), dn(8);
    CELEG_CUDA(cudaMemcpy(dq.data(), q.data(), dq.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dn.data(), norm.data(), dn.bytes(), cudaMemcpyHostToDevice));
    celeg::CudaRopeScaling scaling{};
    celeg::launch_dynamic_qk_norm_rope(
        dq.data(), nullptr, dn.data(), nullptr, 1, 0, 8, 3,
        10000.0f, 0.5f, 1e-6f, true, scaling,
        celeg::RopePairingKind::SplitHalf, stream.get());
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    CELEG_CUDA(cudaMemcpy(q.data(), dq.data(), dq.bytes(), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 8; ++i) expect_near(to_float(q[i]), expected[i], 0.05f);
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

void run_attention_data_movement_tests(celeg::CudaStream& stream) {
/// Packed attention query+gate extraction must de-interleave *per head*
/// (query_head0, gate_head0, query_head1, gate_head1, ...) -- the
/// HF/checkpoint convention from `q_proj(x).view(..., heads, 2*head_dim)`
/// then `chunk(2, dim=-1)` -- not split coarsely into one contiguous
/// query block followed by one contiguous gate block.
{
    constexpr int rows = 2, heads = 3, head_dim = 4, width = heads * head_dim;
    std::vector<__nv_bfloat16> packed(rows * width * 2);
    std::vector<float> expected_query(rows * width), expected_gate(rows * width);
    for (int row = 0; row < rows; ++row) {
        for (int head = 0; head < heads; ++head) {
            for (int d = 0; d < head_dim; ++d) {
                const float qv = static_cast<float>(row * 100 + head * 10 + d);
                const float gv = static_cast<float>(row * 100 + head * 10 + d) + 0.5f;
                const size_t base = static_cast<size_t>(row) * width * 2 +
                    static_cast<size_t>(head) * 2 * head_dim;
                packed[base + d] = to_bf16(qv);
                packed[base + head_dim + d] = to_bf16(gv);
                expected_query[row * width + head * head_dim + d] = qv;
                expected_gate[row * width + head * head_dim + d] = gv;
            }
        }
    }
    celeg::DeviceBuffer<__nv_bfloat16> dpacked(packed.size()), dquery(rows * width), dgate(rows * width);
    CELEG_CUDA(cudaMemcpy(dpacked.data(), packed.data(), dpacked.bytes(), cudaMemcpyHostToDevice));
    celeg::launch_extract_attention_output_gate(
        dpacked.data(), dquery.data(), dgate.data(), rows, width, head_dim, stream.get());
    std::vector<__nv_bfloat16> got_query(rows * width), got_gate(rows * width);
    CELEG_CUDA(cudaMemcpyAsync(got_query.data(), dquery.data(), dquery.bytes(), cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(got_gate.data(), dgate.data(), dgate.bytes(), cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    for (size_t i = 0; i < got_query.size(); ++i) {
        expect_near(to_float(got_query[i]), expected_query[i], 0.01f);
        expect_near(to_float(got_gate[i]), expected_gate[i], 0.01f);
    }
}

{
    constexpr int rows = 2;
    constexpr int q_width = 2;
    constexpr int kv_width = 1;
    std::vector<__nv_bfloat16> qkv = {
        to_bf16(1), to_bf16(2), to_bf16(3), to_bf16(4),
        to_bf16(5), to_bf16(6), to_bf16(7), to_bf16(8)};
    celeg::DeviceBuffer<__nv_bfloat16> input(qkv.size());
    celeg::DeviceBuffer<__nv_bfloat16> q(rows * q_width), k(rows * kv_width),
        v(rows * kv_width);
    CELEG_CUDA(cudaMemcpy(input.data(), qkv.data(), input.bytes(),
                        cudaMemcpyHostToDevice));
    celeg::launch_split_qkv_rows(input.data(), q.data(), k.data(), v.data(),
                               rows, q_width, kv_width, stream.get());
    std::vector<__nv_bfloat16> hq(q.size()), hk(k.size()), hv(v.size());
    CELEG_CUDA(cudaMemcpyAsync(hq.data(), q.data(), q.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(hk.data(), k.data(), k.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(hv.data(), v.data(), v.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    CELEG_TEST_CHECK(to_float(hq[0]) == 1 && to_float(hq[3]) == 6);
    CELEG_TEST_CHECK(to_float(hk[0]) == 3 && to_float(hk[1]) == 7);
    CELEG_TEST_CHECK(to_float(hv[0]) == 4 && to_float(hv[1]) == 8);

    std::vector<__nv_bfloat16> gate_up = {
        to_bf16(0), to_bf16(1), to_bf16(2), to_bf16(3),
        to_bf16(1), to_bf16(-1), to_bf16(4), to_bf16(2)};
    celeg::DeviceBuffer<__nv_bfloat16> dgu(gate_up.size()), out(4);
    CELEG_CUDA(cudaMemcpy(dgu.data(), gate_up.data(), dgu.bytes(),
                        cudaMemcpyHostToDevice));
    celeg::launch_swiglu_interleaved(dgu.data(), out.data(), 2, 2,
                                   stream.get());
    std::vector<__nv_bfloat16> hout(4);
    CELEG_CUDA(cudaMemcpyAsync(hout.data(), out.data(), out.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    expect_near(to_float(hout[0]), 0.0f);
    expect_near(to_float(hout[1]),
                (1.0f / (1.0f + std::exp(-1.0f))) * 3.0f);
    expect_near(to_float(hout[2]),
                (1.0f / (1.0f + std::exp(-1.0f))) * 4.0f);
}
}

namespace {

/// Geometry for one packed-gate decode-sequence case: a tiny exact case plus
/// the Qwen3.5 full-attention scale (24 query heads, 4 KV heads, head_dim
/// 256, partial rotary) that caught the missing decode-side extraction.
struct PackedGateDecodeCase {
    int q_heads;
    int kv_heads;
    int head_dim;
    int seq_len;
    float theta;
    float rotary_fraction;
};

/// Drives the exact launcher sequence the CUDA decode path must honor for a
/// packed Q+Gate projection: per-head extraction, QK-norm/RoPE prepare at
/// position 0 (identity rotation, normalization applied), contiguous
/// strict + online decode, and sigmoid gate apply -- checking every stage
/// against a host reference. A final halves-split pass (the old
/// `[Q|G]`-halves misreading) must clearly diverge, proving the test is
/// sensitive to the bug class.
void check_packed_gate_decode_case(celeg::CudaStream& stream,
                                   const PackedGateDecodeCase& geometry) {
    const int q_heads = geometry.q_heads;
    const int kv_heads = geometry.kv_heads;
    const int head_dim = geometry.head_dim;
    const int seq_len = geometry.seq_len;
    const int q_width = q_heads * head_dim;
    const int kv_width = kv_heads * head_dim;
    constexpr float kEps = 1e-6f;

    /// Deterministic synthetic q_proj output, interleaved per head: head h
    /// owns [q_h, g_h]. Gate values span both sigmoid regimes.
    std::vector<float> packed_f(2 * static_cast<size_t>(q_width));
    std::vector<float> query_f(q_width), gate_f(q_width);
    for (int head = 0; head < q_heads; ++head) {
        for (int d = 0; d < head_dim; ++d) {
            const float qv = 0.05f * static_cast<float>((head * 131 + d * 17) % 19) - 0.4f;
            const float gv = 2.0f * qv +
                static_cast<float>((head * 7 + d * 3) % 5 - 2);
            query_f[static_cast<size_t>(head) * head_dim + d] = qv;
            gate_f[static_cast<size_t>(head) * head_dim + d] = gv;
            const size_t base = static_cast<size_t>(head) * 2 * head_dim;
            packed_f[base + d] = qv;
            packed_f[base + head_dim + d] = gv;
        }
    }
    std::vector<float> k_cache_f(seq_len * kv_width), v_cache_f(seq_len * kv_width);
    for (int token = 0; token < seq_len; ++token) {
        for (int i = 0; i < kv_width; ++i) {
            k_cache_f[static_cast<size_t>(token) * kv_width + i] =
                0.07f * static_cast<float>((token * 37 + i * 11) % 13) - 0.35f;
            v_cache_f[static_cast<size_t>(token) * kv_width + i] =
                0.11f * static_cast<float>((token * 53 + i * 29) % 17) - 0.8f;
        }
    }
    const std::vector<float> ones(static_cast<size_t>(q_width), 1.0f);
    const std::vector<float> ones_kv(static_cast<size_t>(kv_width), 1.0f);

    auto to_device = [&](const std::vector<float>& host) {
        std::vector<__nv_bfloat16> bf16(host.size());
        for (size_t i = 0; i < host.size(); ++i) bf16[i] = to_bf16(host[i]);
        celeg::DeviceBuffer<__nv_bfloat16> device(bf16.size());
        CELEG_CUDA(cudaMemcpy(device.data(), bf16.data(), device.bytes(),
                              cudaMemcpyHostToDevice));
        return device;
    };
    celeg::DeviceBuffer<__nv_bfloat16> dpacked = to_device(packed_f);
    celeg::DeviceBuffer<__nv_bfloat16> dk = to_device(k_cache_f);
    celeg::DeviceBuffer<__nv_bfloat16> dv = to_device(v_cache_f);
    celeg::DeviceBuffer<__nv_bfloat16> dquery_norm = to_device(ones);
    celeg::DeviceBuffer<__nv_bfloat16> dkey_norm = to_device(ones_kv);
    celeg::DeviceBuffer<__nv_bfloat16> dquery(q_width), dgate(q_width);
    celeg::DeviceBuffer<__nv_bfloat16> dattn(q_width);
    celeg::DeviceBuffer<int32_t> dposition(1);
    /// The current token's key (last cache row): prepared single-row exactly
    /// like decode does. The multi-row cache itself stays as uploaded --
    /// cache preparation belongs to prefill, and the decode kernel only
    /// ever reads it.
    std::vector<float> cur_k_f(k_cache_f.end() - kv_width, k_cache_f.end());
    celeg::DeviceBuffer<__nv_bfloat16> dcur_k = to_device(cur_k_f);
    auto download = [&](const celeg::DeviceBuffer<__nv_bfloat16>& device) {
        std::vector<__nv_bfloat16> bf16(device.size());
        CELEG_CUDA(cudaMemcpyAsync(bf16.data(), device.data(), device.bytes(),
                                   cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        std::vector<float> out(bf16.size());
        for (size_t i = 0; i < bf16.size(); ++i) out[i] = to_float(bf16[i]);
        return out;
    };

    /// Stage 1: per-head extraction must recover the interleaved query/gate.
    celeg::launch_extract_attention_output_gate(
        dpacked.data(), dquery.data(), dgate.data(), 1, q_width, head_dim,
        stream.get());
    for (const auto& [got, expected] :
         {std::pair{download(dquery), query_f}, {download(dgate), gate_f}}) {
        CELEG_TEST_CHECK(got.size() == expected.size());
        for (size_t i = 0; i < got.size(); ++i) expect_near(got[i], expected[i], 0.01f);
    }

    /// Stage 2: QK-norm/RoPE prepare of the current token at position 0
    /// rotates by the identity, so the outputs must equal per-head RMSNorm
    /// of the extracted query and current key.
    const int32_t zero = 0;
    CELEG_CUDA(cudaMemcpy(dposition.data(), &zero, sizeof(zero), cudaMemcpyHostToDevice));
    celeg::launch_dynamic_qk_norm_rope_device(
        dquery.data(), dcur_k.data(), dquery_norm.data(), dkey_norm.data(),
        q_heads, kv_heads, head_dim, dposition.data(), geometry.theta,
        geometry.rotary_fraction, kEps, true, celeg::CudaRopeScaling{},
        celeg::RopePairingKind::SplitHalf, stream.get());
    std::vector<float> query_normed(q_width), cur_k_normed(kv_width);
    for (int head = 0; head < q_heads; ++head) {
        const std::vector<float> slice(query_f.begin() + head * head_dim,
                                       query_f.begin() + (head + 1) * head_dim);
        const std::vector<float> ref = celeg::reference::rmsnorm_bf16(
            slice, std::vector<float>(head_dim, 1.0f), kEps);
        std::copy(ref.begin(), ref.end(),
                  query_normed.begin() + head * head_dim);
    }
    for (int head = 0; head < kv_heads; ++head) {
        const std::vector<float> slice(cur_k_f.begin() + head * head_dim,
                                       cur_k_f.begin() + (head + 1) * head_dim);
        const std::vector<float> ref = celeg::reference::rmsnorm_bf16(
            slice, std::vector<float>(head_dim, 1.0f), kEps);
        std::copy(ref.begin(), ref.end(), cur_k_normed.begin() + head * head_dim);
    }
    for (const auto& [got, expected] :
         {std::pair{download(dquery), query_normed}, {download(dcur_k), cur_k_normed}}) {
        CELEG_TEST_CHECK(got.size() == expected.size());
        for (size_t i = 0; i < got.size(); ++i) expect_near(got[i], expected[i], 0.02f);
    }

    /// Stage 3: strict decode must match the bf16 host reference; the online
    /// device kernel (the real decode kernel) must agree with it as well.
    celeg::launch_gqa_decode_strict({
        .query = dquery.data(),
        .kv = {.keys = dk.data(), .values = dv.data()},
        .out = dattn.data(),
        .geometry = {.q_heads = q_heads, .kv_heads = kv_heads, .head_dim = head_dim},
        .extent = {.seq_len = seq_len},
        .stream = stream.get()});
    const std::vector<float> attn_ref = celeg::reference::gqa_decode_strict_bf16(
        query_normed, k_cache_f, v_cache_f, seq_len, q_heads, kv_heads, head_dim);
    /// The softmax must actually spread mass: a one-hot peak would make the
    /// rest of the test insensitive to the query.
    float ref_max_prob = 0.0f;
    {
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        float maximum = -std::numeric_limits<float>::infinity();
        for (int token = 0; token < seq_len; ++token) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) dot += query_normed[d] * k_cache_f[token * kv_width + d];
            maximum = std::max(maximum, dot * scale);
        }
        float denominator = 0.0f;
        for (int token = 0; token < seq_len; ++token) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) dot += query_normed[d] * k_cache_f[token * kv_width + d];
            denominator += std::exp(dot * scale - maximum);
            ref_max_prob = std::max(ref_max_prob, std::exp(dot * scale - maximum));
        }
        ref_max_prob /= denominator;
    }
    CELEG_TEST_CHECK(ref_max_prob < 0.99f);
    const std::vector<float> attn_strict = download(dattn);
    CELEG_TEST_CHECK(attn_strict.size() == attn_ref.size());
    for (size_t i = 0; i < attn_strict.size(); ++i) {
        expect_near(attn_strict[i], attn_ref[i], 0.02f);
    }
    const int32_t last = seq_len - 1;
    CELEG_CUDA(cudaMemcpy(dposition.data(), &last, sizeof(last), cudaMemcpyHostToDevice));
    celeg::launch_gqa_decode_online_device({
        .query = dquery.data(),
        .kv = {.keys = dk.data(), .values = dv.data()},
        .out = dattn.data(),
        .geometry = {.q_heads = q_heads, .kv_heads = kv_heads, .head_dim = head_dim},
        .extent = {.position = dposition.data()},
        .stream = stream.get()});
    const std::vector<float> attn_online = download(dattn);
    CELEG_TEST_CHECK(attn_online.size() == attn_ref.size());
    for (size_t i = 0; i < attn_online.size(); ++i) {
        expect_near(attn_online[i], attn_ref[i], 0.05f);
    }
    CELEG_TEST_CHECK(
        celeg::test::numerical::cosine_similarity(attn_online, attn_ref) >= 0.999);

    /// Stage 4: sigmoid gate apply over the strict-decoded output.
    celeg::launch_gqa_decode_strict({
        .query = dquery.data(),
        .kv = {.keys = dk.data(), .values = dv.data()},
        .out = dattn.data(),
        .geometry = {.q_heads = q_heads, .kv_heads = kv_heads, .head_dim = head_dim},
        .extent = {.seq_len = seq_len},
        .stream = stream.get()});
    celeg::launch_sigmoid_multiply(dattn.data(), dgate.data(), q_width, stream.get());
    const std::vector<float> gated = download(dattn);
    CELEG_TEST_CHECK(gated.size() == attn_ref.size());
    for (size_t i = 0; i < gated.size(); ++i) {
        const float expected = attn_ref[i] / (1.0f + std::exp(-gate_f[i]));
        expect_near(gated[i], expected, 0.02f);
    }

    /// Sensitivity: the coarse halves reading ([Q|G] split, the old bug)
    /// must clearly diverge from the reference through the same stages. The
    /// key cache is restored first so its single normalization matches the
    /// extracted path exactly, leaving the halves misreading as the only
    /// variable under test.
    std::vector<float> halves_q(packed_f.begin(), packed_f.begin() + q_width);
    std::vector<float> halves_g(packed_f.begin() + q_width, packed_f.end());
    celeg::DeviceBuffer<__nv_bfloat16> dhalves_q = to_device(halves_q);
    celeg::DeviceBuffer<__nv_bfloat16> dhalves_g = to_device(halves_g);
    celeg::DeviceBuffer<__nv_bfloat16> dhalves_k = to_device(cur_k_f);
    CELEG_CUDA(cudaMemcpy(dposition.data(), &zero, sizeof(zero), cudaMemcpyHostToDevice));
    celeg::launch_dynamic_qk_norm_rope_device(
        dhalves_q.data(), dhalves_k.data(), dquery_norm.data(), dkey_norm.data(),
        q_heads, kv_heads, head_dim, dposition.data(), geometry.theta,
        geometry.rotary_fraction, kEps, true, celeg::CudaRopeScaling{},
        celeg::RopePairingKind::SplitHalf, stream.get());
    celeg::launch_gqa_decode_strict({
        .query = dhalves_q.data(),
        .kv = {.keys = dk.data(), .values = dv.data()},
        .out = dattn.data(),
        .geometry = {.q_heads = q_heads, .kv_heads = kv_heads, .head_dim = head_dim},
        .extent = {.seq_len = seq_len},
        .stream = stream.get()});
    celeg::launch_sigmoid_multiply(dattn.data(), dhalves_g.data(), q_width, stream.get());
    const std::vector<float> halves_out = download(dattn);
    CELEG_TEST_CHECK(halves_out.size() == gated.size());
    /// Sensitivity floor: the halves misreading must clearly diverge
    /// (measured 0.35 tiny / 0.68 Qwen-scale on the reference machine, vs
    /// ~0.01 numerical noise), proving this test catches the skipped
    /// extraction it guards against.
    CELEG_TEST_CHECK(
        celeg::test::numerical::max_absolute_error(halves_out, gated) > 0.1f);
}

}

void run_packed_gate_decode_tests(celeg::CudaStream& stream) {
/// Tiny exact-geometry case plus the Qwen3.5 full-attention scale whose
/// packed gate the CUDA decode path once skipped.
check_packed_gate_decode_case(
    stream, {.q_heads = 2, .kv_heads = 1, .head_dim = 8, .seq_len = 3,
             .theta = 10000.0f, .rotary_fraction = 1.0f});
check_packed_gate_decode_case(
    stream, {.q_heads = 24, .kv_heads = 4, .head_dim = 256, .seq_len = 2,
             .theta = 10000000.0f, .rotary_fraction = 0.25f});
}

}
