#include "celeg/backend/cuda/utils.cuh"
#include "support/assertions.hpp"
#include "support/cuda_kernel_assertions.cuh"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/weight_layout.hpp"
#include "celeg/model/reference.hpp"
#include "celeg/backend/cuda/paged_kv.hpp"
#include "cuda/sampling_tests.hpp"
#include "cuda/attention_tests.hpp"
#include "celeg/model/resolved.hpp"
#include "celeg/model/runtime_types.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

using celeg::cuda_test::expect_near;
using celeg::cuda_test::to_bf16;
using celeg::cuda_test::to_float;

struct TestShape {
    celeg::RuntimeTopology topology;
    celeg::CompiledModelProgram program;
};

std::vector<TestShape> registered_model_shapes() {
    std::vector<TestShape> shapes;
    for (int model = 0; model < 2; ++model) {
        TestShape result;
        auto& shape = result.topology;
        const int query_heads = model == 0 ? 16 : 32;
        const std::vector<celeg::CompiledMixer> mixers = model == 0
            ? std::vector<celeg::CompiledMixer>{
                celeg::CompiledMixer::ShortConvolution, celeg::CompiledMixer::ShortConvolution,
                celeg::CompiledMixer::Attention, celeg::CompiledMixer::ShortConvolution,
                celeg::CompiledMixer::Attention, celeg::CompiledMixer::ShortConvolution,
                celeg::CompiledMixer::Attention, celeg::CompiledMixer::ShortConvolution,
                celeg::CompiledMixer::Attention, celeg::CompiledMixer::ShortConvolution,
                celeg::CompiledMixer::Attention, celeg::CompiledMixer::ShortConvolution,
                celeg::CompiledMixer::Attention, celeg::CompiledMixer::ShortConvolution}
            : std::vector<celeg::CompiledMixer>{
                celeg::CompiledMixer::ShortConvolution, celeg::CompiledMixer::ShortConvolution,
                celeg::CompiledMixer::Attention, celeg::CompiledMixer::ShortConvolution,
                celeg::CompiledMixer::ShortConvolution, celeg::CompiledMixer::Attention,
                celeg::CompiledMixer::ShortConvolution, celeg::CompiledMixer::ShortConvolution,
                celeg::CompiledMixer::Attention, celeg::CompiledMixer::ShortConvolution,
                celeg::CompiledMixer::Attention, celeg::CompiledMixer::ShortConvolution,
                celeg::CompiledMixer::Attention, celeg::CompiledMixer::ShortConvolution,
                celeg::CompiledMixer::Attention, celeg::CompiledMixer::ShortConvolution};
        celeg::ModelGraph graph;
        graph.hidden = model == 0 ? 1024 : 2048;
        graph.layers.resize(mixers.size());
        for (size_t index = 0; index < mixers.size(); ++index) {
            auto& layer = graph.layers[index];
            if (mixers[index] == celeg::CompiledMixer::Attention) {
                celeg::AttentionSpec attention;
                attention.query_heads = query_heads;
                attention.key_value_heads = 8;
                attention.head_dim = 64;
                attention.pattern = celeg::FullCausalPattern{};
                attention.position = celeg::RopePositionSpec{1.0e6, 1.0, {}};
                layer.mixer = attention;
            } else {
                layer.mixer = celeg::ShortConvolutionSpec{3, graph.hidden};
            }
            layer.feed_forward = celeg::DenseFeedForwardSpec{
                model == 0 ? 2560 : 12288, celeg::ActivationKind::SwiGLU};
        }
        shape = celeg::compose_runtime_topology(std::move(shape.dims), graph);
        result.program.hidden = graph.hidden;
        result.program.layers.resize(graph.layers.size());
        for (size_t index = 0; index < graph.layers.size(); ++index) {
            auto& compiled = result.program.layers[index];
            compiled.mixer = mixers[index];
            compiled.feed_forward = celeg::CompiledFeedForward::Dense;
            compiled.feed_forward_intermediate =
                model == 0 ? 2560 : 12288;
            if (const auto* attention = std::get_if<celeg::AttentionSpec>(
                    &graph.layers[index].mixer)) {
                compiled.attention = *attention;
            }
        }
        shapes.push_back(std::move(result));
        continue;
    }
    if (shapes.empty()) {
        std::cerr << "registered_model_shapes: no model shapes registered\n";
        std::abort();
    }
    return shapes;
}

} // namespace

int main() {
    celeg::CudaStream stream;

    // Embedding by scalar value.
    {
        std::vector<__nv_bfloat16> table(12);
        for (int i = 0; i < 12; ++i) table[i] = to_bf16(static_cast<float>(i));
        celeg::DeviceBuffer<__nv_bfloat16> device_table(table.size());
        celeg::DeviceBuffer<__nv_bfloat16> output(4);
        CELEG_CUDA(cudaMemcpy(device_table.data(), table.data(),
                            table.size() * sizeof(table[0]),
                            cudaMemcpyHostToDevice));
        celeg::launch_embedding(2, device_table.data(), output.data(), 4, stream.get());
        std::vector<__nv_bfloat16> host(4);
        CELEG_CUDA(cudaMemcpyAsync(host.data(), output.data(), output.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        for (int i = 0; i < 4; ++i) expect_near(to_float(host[i]), 8.0f + i);
    }



    // Batched embedding preserves row-major [tokens, hidden] layout.
    {
        std::vector<__nv_bfloat16> table(12);
        for (int i = 0; i < 12; ++i) table[i] = to_bf16(static_cast<float>(i));
        std::vector<int32_t> tokens = {2, 0};
        celeg::DeviceBuffer<__nv_bfloat16> device_table(table.size());
        celeg::DeviceBuffer<int32_t> device_tokens(tokens.size());
        celeg::DeviceBuffer<__nv_bfloat16> output(8);
        CELEG_CUDA(cudaMemcpy(device_table.data(), table.data(),
                            table.size() * sizeof(table[0]), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(device_tokens.data(), tokens.data(),
                            tokens.size() * sizeof(tokens[0]), cudaMemcpyHostToDevice));
        celeg::launch_embedding_batch(device_tokens.data(), 2, device_table.data(),
                                    output.data(), 4, stream.get());
        std::vector<__nv_bfloat16> host(8);
        CELEG_CUDA(cudaMemcpyAsync(host.data(), output.data(), output.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        for (int i = 0; i < 4; ++i) {
            expect_near(to_float(host[i]), 8.0f + i);
            expect_near(to_float(host[4 + i]), static_cast<float>(i));
        }
    }

    // The embedding strategy is format-independent for every embedding table,
    // including the per-layer table bound during model construction.
    {
        constexpr int hidden = 4;
        std::vector<__nv_bfloat16> bf16_table = {
            to_bf16(1.0f), to_bf16(2.0f), to_bf16(3.0f), to_bf16(4.0f),
            to_bf16(5.0f), to_bf16(6.0f), to_bf16(7.0f), to_bf16(8.0f)};
        celeg::DeviceBuffer<__nv_bfloat16> dbf16(bf16_table.size()), out(hidden);
        CELEG_CUDA(cudaMemcpy(dbf16.data(), bf16_table.data(), dbf16.bytes(),
                              cudaMemcpyHostToDevice));
        auto bf16_layout = celeg::make_weight_layout(
            celeg::WeightMode::Bf16, dbf16.data(), nullptr);
        bf16_layout->embed_token(1, out.data(), hidden, stream.get());

        std::vector<int8_t> int8_table = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<float> int8_scales = {1.0f, 0.5f};
        celeg::DeviceBuffer<int8_t> dint8(int8_table.size());
        celeg::DeviceBuffer<float> dint8_scales(int8_scales.size());
        CELEG_CUDA(cudaMemcpy(dint8.data(), int8_table.data(), dint8.bytes(),
                              cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dint8_scales.data(), int8_scales.data(),
                              dint8_scales.bytes(), cudaMemcpyHostToDevice));
        auto int8_layout = celeg::make_weight_layout(
            celeg::WeightMode::Int8, dint8.data(), dint8_scales.data());
        int8_layout->embed_token(1, out.data(), hidden, stream.get());

        std::vector<uint8_t> int4_table = {0x21U, 0x43U, 0x65U, 0x87U};
        std::vector<float> int4_scales = {0.25f, 0.5f};
        celeg::DeviceBuffer<uint8_t> dint4(int4_table.size());
        celeg::DeviceBuffer<float> dint4_scales(int4_scales.size());
        CELEG_CUDA(cudaMemcpy(dint4.data(), int4_table.data(), dint4.bytes(),
                              cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dint4_scales.data(), int4_scales.data(),
                              dint4_scales.bytes(), cudaMemcpyHostToDevice));
        auto int4_layout = celeg::make_weight_layout(
            celeg::WeightMode::Int4, dint4.data(), dint4_scales.data());
        int4_layout->embed_token(1, out.data(), hidden, stream.get());
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    }


    // Packed W4A16 linear dequantizes signed nibbles per output row.
    {
        std::vector<__nv_bfloat16> x = {
            to_bf16(1.0f), to_bf16(1.0f),
            to_bf16(1.0f), to_bf16(1.0f)};
        // row 0 = [1,2,3,4], row 1 = [-1,-2,-3,-4]
        std::vector<uint8_t> packed = {0x21U, 0x43U, 0xefU, 0xcdU};
        std::vector<float> scales = {0.5f, 0.25f};
        celeg::DeviceBuffer<__nv_bfloat16> dx(x.size()), dy(2);
        celeg::DeviceBuffer<uint8_t> dw(packed.size());
        celeg::DeviceBuffer<float> ds(scales.size());
        CELEG_CUDA(cudaMemcpy(dx.data(), x.data(), dx.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dw.data(), packed.data(), dw.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(ds.data(), scales.data(), ds.bytes(), cudaMemcpyHostToDevice));
        celeg::launch_w4a16_linear(dx.data(), dw.data(), ds.data(), dy.data(),
                                 1, 2, 4, 0.0f, stream.get());
        std::vector<__nv_bfloat16> output(2);
        CELEG_CUDA(cudaMemcpyAsync(output.data(), dy.data(), dy.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        expect_near(to_float(output[0]), 5.0f, 0.02f);
        expect_near(to_float(output[1]), -2.5f, 0.02f);
    }

    // Packed INT4 embedding shares the same row format as the LM head.
    {
        std::vector<uint8_t> packed = {0x21U, 0x43U, 0xefU, 0xcdU};
        std::vector<float> scales = {0.5f, 0.25f};
        celeg::DeviceBuffer<uint8_t> table(packed.size());
        celeg::DeviceBuffer<float> ds(scales.size());
        celeg::DeviceBuffer<__nv_bfloat16> output(4);
        CELEG_CUDA(cudaMemcpy(table.data(), packed.data(), table.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(ds.data(), scales.data(), ds.bytes(), cudaMemcpyHostToDevice));
        celeg::launch_embedding_int4(1, table.data(), ds.data(), output.data(),
                                   4, stream.get());
        std::vector<__nv_bfloat16> host(4);
        CELEG_CUDA(cudaMemcpyAsync(host.data(), output.data(), output.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        expect_near(to_float(host[0]), -0.25f, 0.02f);
        expect_near(to_float(host[1]), -0.50f, 0.02f);
        expect_near(to_float(host[2]), -0.75f, 0.02f);
        expect_near(to_float(host[3]), -1.00f, 0.02f);
    }

    // RMSNorm.
    {
        std::vector<__nv_bfloat16> x = {
            to_bf16(1.0f), to_bf16(2.0f), to_bf16(3.0f), to_bf16(4.0f)};
        std::vector<__nv_bfloat16> weight(4, to_bf16(1.0f));
        celeg::DeviceBuffer<__nv_bfloat16> dx(4), dw(4), dy(4);
        CELEG_CUDA(cudaMemcpy(dx.data(), x.data(), dx.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dw.data(), weight.data(), dw.bytes(), cudaMemcpyHostToDevice));
        celeg::launch_rmsnorm(dx.data(), dw.data(), dy.data(), 1, 4, 1e-5f, stream.get());
        std::vector<__nv_bfloat16> y(4);
        CELEG_CUDA(cudaMemcpyAsync(y.data(), dy.data(), dy.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        const float inv = 1.0f / std::sqrt(7.5f + 1e-5f);
        for (int i = 0; i < 4; ++i) expect_near(to_float(y[i]), (i + 1) * inv);
    }

    // Fused SwiGLU layout: [gate..., up...].
    {
        std::vector<__nv_bfloat16> gate_up = {
            to_bf16(0.0f), to_bf16(1.0f),
            to_bf16(2.0f), to_bf16(3.0f)};
        celeg::DeviceBuffer<__nv_bfloat16> input(4), output(2);
        CELEG_CUDA(cudaMemcpy(input.data(), gate_up.data(), input.bytes(),
                            cudaMemcpyHostToDevice));
        celeg::launch_swiglu_fused(input.data(), output.data(), 2, stream.get());
        std::vector<__nv_bfloat16> result(2);
        CELEG_CUDA(cudaMemcpyAsync(result.data(), output.data(), output.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        expect_near(to_float(result[0]), 0.0f);
        expect_near(to_float(result[1]), (1.0f / (1.0f + std::exp(-1.0f))) * 3.0f);
    }


    celeg::cuda_test::run_attention_tests(stream);
    // Weight-only INT8 linear supports both GEMV and batched rows.
    {
        constexpr int m = 2;
        constexpr int n = 3;
        constexpr int k = 4;
        std::vector<__nv_bfloat16> x = {
            to_bf16(1), to_bf16(2), to_bf16(3), to_bf16(4),
            to_bf16(2), to_bf16(0), to_bf16(-1), to_bf16(1)};
        std::vector<int8_t> weight = {
            1, 2, 3, 4,
            -1, 1, -1, 1,
            1, 0, 0, 0};
        std::vector<float> scales = {0.5f, 2.0f, 3.0f};
        celeg::DeviceBuffer<__nv_bfloat16> dx(x.size()), dy(m * n);
        celeg::DeviceBuffer<int8_t> dw(weight.size());
        celeg::DeviceBuffer<float> ds(scales.size());
        CELEG_CUDA(cudaMemcpy(dx.data(), x.data(), dx.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dw.data(), weight.data(), dw.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(ds.data(), scales.data(), ds.bytes(), cudaMemcpyHostToDevice));
        celeg::launch_w8a16_linear(dx.data(), dw.data(), ds.data(), dy.data(),
                                 m, n, k, 0.0f, stream.get());
        std::vector<__nv_bfloat16> result(m * n);
        CELEG_CUDA(cudaMemcpyAsync(result.data(), dy.data(), dy.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        const std::vector<float> expected = {15.0f, 4.0f, 3.0f, 1.5f, 0.0f, 6.0f};
        for (size_t i = 0; i < expected.size(); ++i) {
            expect_near(to_float(result[i]), expected[i], 0.05f);
        }
    }

    // Quantized embedding dequantizes with one scale per vocabulary row.
    {
        std::vector<int8_t> table = {1, 2, 3, 4, -1, 2, -3, 4};
        std::vector<float> scales = {0.5f, 2.0f};
        int32_t token = 1;
        celeg::DeviceBuffer<int8_t> dt(table.size());
        celeg::DeviceBuffer<float> ds(scales.size());
        celeg::DeviceBuffer<int32_t> dtoken(1);
        celeg::DeviceBuffer<__nv_bfloat16> out(4);
        CELEG_CUDA(cudaMemcpy(dt.data(), table.data(), dt.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(ds.data(), scales.data(), ds.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dtoken.data(), &token, sizeof(token), cudaMemcpyHostToDevice));
        celeg::launch_embedding_int8_device(dtoken.data(), dt.data(), ds.data(),
                                          out.data(), 4, stream.get());
        std::vector<__nv_bfloat16> result(4);
        CELEG_CUDA(cudaMemcpyAsync(result.data(), out.data(), out.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        const std::vector<float> expected = {-2.0f, 4.0f, -6.0f, 8.0f};
        for (int i = 0; i < 4; ++i) expect_near(to_float(result[i]), expected[i]);
    }

    // Segmented online GQA matches the single-block online implementation.
    {
        constexpr int seq_len = 5;
        constexpr int q_heads = 2;
        constexpr int kv_heads = 1;
        constexpr int head_dim = 2;
        constexpr int chunk_tokens = 2;
        constexpr int chunks = 3;
        std::vector<__nv_bfloat16> q = {
            to_bf16(1.0f), to_bf16(0.5f),
            to_bf16(-0.5f), to_bf16(1.0f)};
        std::vector<__nv_bfloat16> k(seq_len * kv_heads * head_dim);
        std::vector<__nv_bfloat16> v(seq_len * kv_heads * head_dim);
        for (int token_index = 0; token_index < seq_len; ++token_index) {
            k[token_index * 2] = to_bf16(0.25f * (token_index + 1));
            k[token_index * 2 + 1] = to_bf16(1.0f - 0.1f * token_index);
            v[token_index * 2] = to_bf16(static_cast<float>(token_index));
            v[token_index * 2 + 1] = to_bf16(static_cast<float>(token_index + 2));
        }
        int32_t position = seq_len - 1;
        celeg::DeviceBuffer<__nv_bfloat16> dq(q.size()), dk(k.size()), dv(v.size());
        celeg::DeviceBuffer<__nv_bfloat16> reference(q.size()), segmented(q.size());
        celeg::DeviceBuffer<int32_t> dposition(1);
        celeg::DeviceBuffer<float> partial_max(q_heads * chunks);
        celeg::DeviceBuffer<float> partial_denom(q_heads * chunks);
        celeg::DeviceBuffer<float> partial_accum(q_heads * chunks * head_dim);
        CELEG_CUDA(cudaMemcpy(dq.data(), q.data(), dq.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dk.data(), k.data(), dk.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dv.data(), v.data(), dv.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dposition.data(), &position, sizeof(position), cudaMemcpyHostToDevice));
        celeg::launch_gqa_decode_online_device(
            dq.data(), dk.data(), dv.data(), reference.data(), dposition.data(),
            q_heads, kv_heads, head_dim, 0, stream.get());
        celeg::launch_gqa_decode_segmented_device(
            dq.data(), dk.data(), dv.data(), segmented.data(), dposition.data(),
            q_heads, kv_heads, head_dim, chunk_tokens, chunks, 0,
            partial_max.data(), partial_denom.data(), partial_accum.data(),
            stream.get());
        std::vector<__nv_bfloat16> a(q.size()), b(q.size());
        CELEG_CUDA(cudaMemcpyAsync(a.data(), reference.data(), reference.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaMemcpyAsync(b.data(), segmented.data(), segmented.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        for (size_t i = 0; i < a.size(); ++i) {
            expect_near(to_float(a[i]), to_float(b[i]), 0.02f);
        }
    }


    // Packed QKV splitting and row-interleaved SwiGLU preserve batch rows.
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

    // Packed sampling supports independent logits, penalties and RNG state.
    {
        constexpr int rows = 2;
        constexpr int vocab = 4;
        std::vector<__nv_bfloat16> a = {
            to_bf16(1), to_bf16(5), to_bf16(2), to_bf16(0)};
        std::vector<__nv_bfloat16> b = {
            to_bf16(4), to_bf16(3), to_bf16(6), to_bf16(0)};
        std::vector<uint8_t> seen_a(vocab, 0), seen_b(vocab, 0);
        seen_b[2] = 1; // penalty moves row B from token 2 to token 0
        uint64_t rng_a = 1, rng_b = 2;
        celeg::DeviceBuffer<__nv_bfloat16> da(vocab), db(vocab);
        celeg::DeviceBuffer<uint8_t> dsa(vocab), dsb(vocab);
        celeg::DeviceBuffer<uint64_t> dra(1), drb(1);
        CELEG_CUDA(cudaMemcpy(da.data(), a.data(), da.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(db.data(), b.data(), db.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dsa.data(), seen_a.data(), dsa.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dsb.data(), seen_b.data(), dsb.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dra.data(), &rng_a, sizeof(rng_a), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(drb.data(), &rng_b, sizeof(rng_b), cudaMemcpyHostToDevice));
        std::vector<__nv_bfloat16*> logits_ptrs = {da.data(), db.data()};
        std::vector<uint8_t*> seen_ptrs = {dsa.data(), dsb.data()};
        std::vector<uint64_t*> rng_ptrs = {dra.data(), drb.data()};
        celeg::DeviceBuffer<__nv_bfloat16*> dlogits(rows);
        celeg::DeviceBuffer<uint8_t*> dseen(rows);
        celeg::DeviceBuffer<uint64_t*> drng(rows);
        CELEG_CUDA(cudaMemcpy(dlogits.data(), logits_ptrs.data(), dlogits.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dseen.data(), seen_ptrs.data(), dseen.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(drng.data(), rng_ptrs.data(), drng.bytes(), cudaMemcpyHostToDevice));
        std::vector<float> temp = {0.0f, 0.0f};
        std::vector<float> penalty = {1.0f, 2.0f};
        std::vector<int32_t> topk = {1, 1};
        std::vector<float> topp = {1.0f, 1.0f};
        celeg::DeviceBuffer<float> dtemp(rows), dpenalty(rows), dtopp(rows);
        celeg::DeviceBuffer<int32_t> dtopk(rows), result(rows);
        celeg::DeviceBuffer<float> scores(rows * vocab), values(rows * celeg::kMaxTopK);
        celeg::DeviceBuffer<int32_t> indices(rows * celeg::kMaxTopK);
        CELEG_CUDA(cudaMemcpy(dtemp.data(), temp.data(), dtemp.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dpenalty.data(), penalty.data(), dpenalty.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dtopk.data(), topk.data(), dtopk.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dtopp.data(), topp.data(), dtopp.bytes(), cudaMemcpyHostToDevice));
        celeg::launch_packed_sample_topk(
            dlogits.data(), dseen.data(), drng.data(), dtemp.data(),
            dpenalty.data(), dtopk.data(), dtopp.data(), scores.data(),
            values.data(), indices.data(), rows, vocab, celeg::kMaxTopK,
            result.data(), stream.get());
        std::vector<int32_t> tokens(rows);
        CELEG_CUDA(cudaMemcpyAsync(tokens.data(), result.data(), result.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        CELEG_TEST_CHECK(tokens[0] == 1);
        CELEG_TEST_CHECK(tokens[1] == 0);
    }

    // Physical paged BF16 GQA follows a non-contiguous page table.
    {
        constexpr int page_tokens = 2;
        constexpr int attention_layers = 1;
        constexpr int q_heads = 1;
        constexpr int kv_heads = 1;
        constexpr int head_dim = 2;
        constexpr int page_count = 2;
        std::vector<__nv_bfloat16> keys(page_count * attention_layers *
                                        page_tokens * kv_heads * head_dim,
                                        to_bf16(0.0f));
        std::vector<__nv_bfloat16> values(keys.size(), to_bf16(0.0f));
        // Use physical page 1; page 0 deliberately remains empty.
        const size_t base = static_cast<size_t>(page_tokens) * head_dim;
        keys[base + 0] = to_bf16(1.0f);
        keys[base + 1] = to_bf16(0.0f);
        keys[base + 2] = to_bf16(0.0f);
        keys[base + 3] = to_bf16(1.0f);
        values[base + 0] = to_bf16(2.0f);
        values[base + 1] = to_bf16(4.0f);
        values[base + 2] = to_bf16(6.0f);
        values[base + 3] = to_bf16(8.0f);
        std::vector<__nv_bfloat16> query = {to_bf16(1.0f), to_bf16(0.0f)};
        std::vector<uint32_t> table = {1};
        std::vector<int32_t> positions = {1};
        celeg::DeviceBuffer<__nv_bfloat16> dq(query.size()), dk(keys.size()),
            dv(values.size()), dout(head_dim);
        celeg::DeviceBuffer<uint32_t> dtable(table.size());
        celeg::DeviceBuffer<int32_t> dpositions(positions.size());
        CELEG_CUDA(cudaMemcpy(dq.data(), query.data(), dq.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dk.data(), keys.data(), dk.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dv.data(), values.data(), dv.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dtable.data(), table.data(), dtable.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dpositions.data(), positions.data(), dpositions.bytes(), cudaMemcpyHostToDevice));
        celeg::launch_gqa_decode_paged_batch(
            dq.data(), dk.data(), dv.data(), dtable.data(), 1,
            dout.data(), dpositions.data(), 1, 0, page_tokens,
            page_tokens * kv_heads * head_dim, 0,
            q_heads, kv_heads, head_dim, 0, false,
            stream.get());
        std::vector<__nv_bfloat16> output(head_dim);
        CELEG_CUDA(cudaMemcpyAsync(output.data(), dout.data(), dout.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        const auto expected = celeg::reference::gqa_decode_strict_bf16(
            {1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f},
            {2.0f, 4.0f, 6.0f, 8.0f}, 2, q_heads, kv_heads, head_dim);
        for (int i = 0; i < head_dim; ++i) {
            expect_near(to_float(output[static_cast<size_t>(i)]),
                        expected[static_cast<size_t>(i)], 0.02f);
        }
    }


    // Segmented paged BF16 attention matches the single-block online path.
    {
        constexpr int page_tokens = 2;
        constexpr int attention_layers = 1;
        constexpr int q_heads = 1;
        constexpr int kv_heads = 1;
        constexpr int head_dim = 2;
        constexpr int page_count = 2;
        std::vector<__nv_bfloat16> keys(page_count * attention_layers *
                                        page_tokens * kv_heads * head_dim,
                                        to_bf16(0.0f));
        std::vector<__nv_bfloat16> values(keys.size(), to_bf16(0.0f));
        // Logical order is physical page 1 followed by physical page 0.
        const size_t p1 = static_cast<size_t>(page_tokens) * head_dim;
        keys[p1 + 0] = to_bf16(1.0f); keys[p1 + 1] = to_bf16(0.0f);
        keys[p1 + 2] = to_bf16(0.0f); keys[p1 + 3] = to_bf16(1.0f);
        values[p1 + 0] = to_bf16(2.0f); values[p1 + 1] = to_bf16(4.0f);
        values[p1 + 2] = to_bf16(6.0f); values[p1 + 3] = to_bf16(8.0f);
        keys[0] = to_bf16(0.5f); keys[1] = to_bf16(0.5f);
        values[0] = to_bf16(10.0f); values[1] = to_bf16(12.0f);
        std::vector<__nv_bfloat16> query = {to_bf16(1.0f), to_bf16(0.0f)};
        std::vector<uint32_t> table = {1, 0};
        std::vector<int32_t> positions = {2};
        celeg::DeviceBuffer<__nv_bfloat16> dq(query.size()), dk(keys.size()),
            dv(values.size()), normal(head_dim), segmented(head_dim);
        celeg::DeviceBuffer<uint32_t> dtable(table.size());
        celeg::DeviceBuffer<int32_t> dpositions(positions.size());
        celeg::DeviceBuffer<float> pmax(3), pdenom(3), paccum(3 * head_dim);
        CELEG_CUDA(cudaMemcpy(dq.data(), query.data(), dq.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dk.data(), keys.data(), dk.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dv.data(), values.data(), dv.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dtable.data(), table.data(), dtable.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dpositions.data(), positions.data(), dpositions.bytes(), cudaMemcpyHostToDevice));
        celeg::launch_gqa_decode_paged_batch(
            dq.data(), dk.data(), dv.data(), dtable.data(), 2, normal.data(),
            dpositions.data(), 1, 0, page_tokens,
            page_tokens * kv_heads * head_dim, 0,
            q_heads, kv_heads, head_dim, 0, true, stream.get());
        celeg::launch_gqa_decode_paged_segmented_batch(
            dq.data(), dk.data(), dv.data(), dtable.data(), 2,
            segmented.data(), dpositions.data(), 1, 0, page_tokens,
            page_tokens * kv_heads * head_dim, 0,
            q_heads, kv_heads, head_dim, 1, 3, 0,
            pmax.data(), pdenom.data(), paccum.data(), stream.get());
        std::vector<__nv_bfloat16> a(head_dim), b(head_dim);
        CELEG_CUDA(cudaMemcpyAsync(a.data(), normal.data(), normal.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaMemcpyAsync(b.data(), segmented.data(), segmented.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        for (int i = 0; i < head_dim; ++i) {
            expect_near(to_float(a[static_cast<size_t>(i)]),
                        to_float(b[static_cast<size_t>(i)]), 0.03f);
        }
    }

    // Segmented paged INT8 attention matches the single-block online path.
    {
        constexpr int page_tokens = 2;
        constexpr int attention_layers = 1;
        constexpr int q_heads = 1;
        constexpr int kv_heads = 1;
        constexpr int head_dim = 2;
        constexpr int page_count = 2;
        const size_t vector_count = page_count * attention_layers *
                                    page_tokens * kv_heads * head_dim;
        const size_t scale_count = page_count * attention_layers *
                                   page_tokens * kv_heads;
        std::vector<int8_t> keys(vector_count, 0), values(vector_count, 0);
        std::vector<float> key_scales(scale_count, 0.01f);
        std::vector<float> value_scales(scale_count, 0.02f);
        const size_t p1 = static_cast<size_t>(page_tokens) * head_dim;
        keys[p1 + 0] = 100; keys[p1 + 1] = 0;
        keys[p1 + 2] = 0;   keys[p1 + 3] = 100;
        values[p1 + 0] = 100; values[p1 + 1] = 50;
        values[p1 + 2] = 25;  values[p1 + 3] = 75;
        keys[0] = 50; keys[1] = 50;
        values[0] = 80; values[1] = 40;
        std::vector<__nv_bfloat16> query = {to_bf16(1.0f), to_bf16(0.0f)};
        std::vector<uint32_t> table = {1, 0};
        std::vector<int32_t> positions = {2};
        celeg::DeviceBuffer<__nv_bfloat16> dq(query.size()), normal(head_dim),
            segmented(head_dim);
        celeg::DeviceBuffer<int8_t> dk(keys.size()), dv(values.size());
        celeg::DeviceBuffer<float> dks(key_scales.size()), dvs(value_scales.size());
        celeg::DeviceBuffer<uint32_t> dtable(table.size());
        celeg::DeviceBuffer<int32_t> dpositions(positions.size());
        celeg::DeviceBuffer<float> pmax(3), pdenom(3), paccum(3 * head_dim);
        CELEG_CUDA(cudaMemcpy(dq.data(), query.data(), dq.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dk.data(), keys.data(), dk.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dv.data(), values.data(), dv.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dks.data(), key_scales.data(), dks.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dvs.data(), value_scales.data(), dvs.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dtable.data(), table.data(), dtable.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dpositions.data(), positions.data(), dpositions.bytes(), cudaMemcpyHostToDevice));
        celeg::launch_gqa_decode_int8_paged_batch(
            dq.data(), dk.data(), dv.data(), dks.data(), dvs.data(),
            dtable.data(), 2, normal.data(), dpositions.data(), 1, 0,
            page_tokens, page_tokens * kv_heads * head_dim, 0,
            page_tokens * kv_heads, 0,
            q_heads, kv_heads, head_dim, 0, true,
            stream.get());
        celeg::launch_gqa_decode_int8_paged_segmented_batch(
            dq.data(), dk.data(), dv.data(), dks.data(), dvs.data(),
            dtable.data(), 2, segmented.data(), dpositions.data(), 1, 0,
            page_tokens, page_tokens * kv_heads * head_dim, 0,
            page_tokens * kv_heads, 0,
            q_heads, kv_heads, head_dim, 1, 3, 0,
            pmax.data(), pdenom.data(), paccum.data(), stream.get());
        std::vector<__nv_bfloat16> a(head_dim), b(head_dim);
        CELEG_CUDA(cudaMemcpyAsync(a.data(), normal.data(), normal.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaMemcpyAsync(b.data(), segmented.data(), segmented.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        for (int i = 0; i < head_dim; ++i) {
            expect_near(to_float(a[static_cast<size_t>(i)]),
                        to_float(b[static_cast<size_t>(i)]), 0.03f);
        }
    }

    // Per-row seen-token marking updates separate request histories.
    {
        std::vector<int32_t> tokens = {3, 5};
        celeg::DeviceBuffer<int32_t> dtokens(tokens.size());
        celeg::DeviceBuffer<uint8_t> seen_a(8), seen_b(8);
        celeg::PinnedBuffer<uint8_t*> hseen(2);
        celeg::DeviceBuffer<uint8_t*> dseen(2);
        seen_a.zero_async(stream.get());
        seen_b.zero_async(stream.get());
        hseen.data()[0] = seen_a.data();
        hseen.data()[1] = seen_b.data();
        CELEG_CUDA(cudaMemcpy(dtokens.data(), tokens.data(), dtokens.bytes(),
                            cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dseen.data(), hseen.data(), dseen.bytes(),
                            cudaMemcpyHostToDevice));
        celeg::launch_mark_seen_batch_ptrs(dtokens.data(), dseen.data(), 2, 8,
                                         stream.get());
        std::vector<uint8_t> a(8), b(8);
        CELEG_CUDA(cudaMemcpyAsync(a.data(), seen_a.data(), seen_a.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaMemcpyAsync(b.data(), seen_b.data(), seen_b.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        CELEG_TEST_CHECK(a[3] == 1 && a[5] == 0);
        CELEG_TEST_CHECK(b[5] == 1 && b[3] == 0);
    }

    // Copy-on-write cloning duplicates all physical page storage while keeping
    // independent reference counts. Run against every registered model shape so a
    // regression that affects only one (e.g. different attention_layers) is
    // caught without editing this test.
    for (const TestShape& shape : registered_model_shapes()) {
        celeg::PhysicalPagedKvCache cache(3, 1, 4, celeg::KvCacheMode::Bf16,
                                          shape.topology.exec, shape.program);
        auto source = cache.allocate_tokens(1);
        CELEG_TEST_CHECK(source && source->size() == 1);
        const uint32_t source_page = source->front();
        const size_t page_elements = cache.page_vector_elements();
        std::vector<__nv_bfloat16> contents(page_elements, to_bf16(0.0f));
        contents[0] = to_bf16(3.5f);
        CELEG_CUDA(cudaMemcpy(cache.key_bf16() +
                            static_cast<size_t>(source_page) * page_elements,
                            contents.data(), contents.size() * sizeof(__nv_bfloat16),
                            cudaMemcpyHostToDevice));
        auto cloned = cache.clone_page(source_page);
        CELEG_TEST_CHECK(cloned && *cloned != source_page);
        CELEG_TEST_CHECK(cache.ref_count(source_page) == 1);
        CELEG_TEST_CHECK(cache.ref_count(*cloned) == 1);
        __nv_bfloat16 copied{};
        CELEG_CUDA(cudaMemcpy(&copied, cache.key_bf16() +
                            static_cast<size_t>(*cloned) * page_elements,
                            sizeof(copied), cudaMemcpyDeviceToHost));
        expect_near(to_float(copied), 3.5f, 0.01f);
        cache.release(*source);
        cache.release(std::vector<uint32_t>{*cloned});
        CELEG_TEST_CHECK(cache.free_pages() == cache.total_pages());
    }

    // Partial-page COW copies initialized token slots without transferring the unused suffix.
    for (const TestShape& shape : registered_model_shapes()) {
        constexpr int page_tokens = 4;
        celeg::PhysicalPagedKvCache cache(3, page_tokens, 8,
                                        celeg::KvCacheMode::Bf16,
                                        shape.topology.exec, shape.program);
        auto source = cache.allocate_tokens(page_tokens);
        CELEG_TEST_CHECK(source && source->size() == 1);
        const uint32_t source_page = source->front();
        const size_t page_elements = cache.page_vector_elements();
        std::vector<__nv_bfloat16> contents(page_elements, to_bf16(9.0f));
        CELEG_CUDA(cudaMemcpy(cache.key_bf16() +
                            static_cast<size_t>(source_page) * page_elements,
                            contents.data(), contents.size() * sizeof(__nv_bfloat16),
                            cudaMemcpyHostToDevice));
        auto cloned = cache.clone_page_prefix(source_page, 1);
        CELEG_TEST_CHECK(cloned);
        std::vector<__nv_bfloat16> copied(page_elements);
        CELEG_CUDA(cudaMemcpy(copied.data(), cache.key_bf16() +
                            static_cast<size_t>(*cloned) * page_elements,
                            copied.size() * sizeof(__nv_bfloat16),
                            cudaMemcpyDeviceToHost));
        for (int layer = 0; layer < cache.attention_layers(); ++layer) {
            const size_t layer_base = cache.layer_vector_offset(layer);
            expect_near(to_float(copied[layer_base]), 9.0f, 0.01f);
        }
        cache.release(*source);
        cache.release(std::vector<uint32_t>{*cloned});
    }

    // INT8 copy-on-write clones both quantized vectors and scale planes.
    for (const TestShape& shape : registered_model_shapes()) {
        celeg::PhysicalPagedKvCache cache(3, 1, 4, celeg::KvCacheMode::Int8,
                                          shape.topology.exec, shape.program);
        auto source = cache.allocate_tokens(1);
        CELEG_TEST_CHECK(source && source->size() == 1);
        const uint32_t source_page = source->front();
        const size_t page_elements = cache.page_vector_elements();
        const size_t scale_elements = cache.page_scale_elements();
        const int8_t quantized = -37;
        const float scale = 0.03125f;
        CELEG_CUDA(cudaMemcpy(cache.key_int8() +
                            static_cast<size_t>(source_page) * page_elements,
                            &quantized, sizeof(quantized), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(cache.key_scales() +
                            static_cast<size_t>(source_page) * scale_elements,
                            &scale, sizeof(scale), cudaMemcpyHostToDevice));
        auto cloned = cache.clone_page(source_page);
        CELEG_TEST_CHECK(cloned && *cloned != source_page);
        int8_t copied_quantized = 0;
        float copied_scale = 0.0f;
        CELEG_CUDA(cudaMemcpy(&copied_quantized, cache.key_int8() +
                            static_cast<size_t>(*cloned) * page_elements,
                            sizeof(copied_quantized), cudaMemcpyDeviceToHost));
        CELEG_CUDA(cudaMemcpy(&copied_scale, cache.key_scales() +
                            static_cast<size_t>(*cloned) * scale_elements,
                            sizeof(copied_scale), cudaMemcpyDeviceToHost));
        CELEG_TEST_CHECK(copied_quantized == quantized);
        expect_near(copied_scale, scale, 1e-7f);
        cache.release(*source);
        cache.release(std::vector<uint32_t>{*cloned});
    }

    // Paged INT8 store writes the selected physical page and scale slot.
    {
        constexpr int page_tokens = 2;
        constexpr int attention_layers = 1;
        constexpr int kv_heads = 1;
        constexpr int head_dim = 2;
        constexpr int page_count = 2;
        std::vector<__nv_bfloat16> k = {to_bf16(1.0f), to_bf16(-0.5f)};
        std::vector<__nv_bfloat16> v = {to_bf16(2.0f), to_bf16(-1.0f)};
        std::vector<uint32_t> table = {1};
        std::vector<int32_t> positions = {0};
        const size_t vector_count = page_count * attention_layers *
                                    page_tokens * kv_heads * head_dim;
        const size_t scale_count = page_count * attention_layers *
                                   page_tokens * kv_heads;
        celeg::DeviceBuffer<__nv_bfloat16> dk(k.size()), dv(v.size());
        celeg::DeviceBuffer<int8_t> key_pool(vector_count), value_pool(vector_count);
        celeg::DeviceBuffer<float> key_scales(scale_count), value_scales(scale_count);
        celeg::DeviceBuffer<uint32_t> dtable(table.size());
        celeg::DeviceBuffer<int32_t> dpositions(positions.size());
        key_pool.zero_async(stream.get());
        value_pool.zero_async(stream.get());
        key_scales.zero_async(stream.get());
        value_scales.zero_async(stream.get());
        CELEG_CUDA(cudaMemcpy(dk.data(), k.data(), dk.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dv.data(), v.data(), dv.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dtable.data(), table.data(), dtable.bytes(), cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dpositions.data(), positions.data(), dpositions.bytes(), cudaMemcpyHostToDevice));
        celeg::launch_store_kv_int8_paged_batch(
            dk.data(), dv.data(), key_pool.data(), value_pool.data(),
            key_scales.data(), value_scales.data(), dtable.data(), 1,
            dpositions.data(), 1, 0, page_tokens,
            page_tokens * kv_heads * head_dim, 0,
            page_tokens * kv_heads, 0, kv_heads, head_dim, stream.get());
        std::vector<int8_t> host_keys(vector_count), host_values(vector_count);
        std::vector<float> host_key_scales(scale_count), host_value_scales(scale_count);
        CELEG_CUDA(cudaMemcpyAsync(host_keys.data(), key_pool.data(), key_pool.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaMemcpyAsync(host_values.data(), value_pool.data(), value_pool.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaMemcpyAsync(host_key_scales.data(), key_scales.data(), key_scales.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaMemcpyAsync(host_value_scales.data(), value_scales.data(), value_scales.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        const size_t vector_base = static_cast<size_t>(page_tokens) * head_dim;
        const size_t scale_base = static_cast<size_t>(page_tokens);
        expect_near(host_key_scales[scale_base], 1.0f / 127.0f, 1e-4f);
        expect_near(host_value_scales[scale_base], 2.0f / 127.0f, 1e-4f);
        CELEG_TEST_CHECK(host_keys[vector_base] == 127);
        CELEG_TEST_CHECK(host_values[vector_base] == 127);
    }

    // CUDA Graph replay uses device-resident mutable state.
    {
        int32_t initial = 7;
        celeg::DeviceBuffer<int32_t> position(1);
        CELEG_CUDA(cudaMemcpy(position.data(), &initial, sizeof(initial), cudaMemcpyHostToDevice));
        celeg::CudaGraphExec graph;
        graph.capture_begin(stream.get());
        celeg::launch_increment_position(position.data(), stream.get());
        celeg::launch_increment_position(position.data(), stream.get());
        graph.capture_end(stream.get());
        graph.launch(stream.get());
        graph.launch(stream.get());
        int32_t result = 0;
        CELEG_CUDA(cudaMemcpyAsync(&result, position.data(), sizeof(result),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        CELEG_TEST_CHECK(result == 11);
    }

    std::cout << "cuda_kernels_test: ok\n";
    return 0;
}
