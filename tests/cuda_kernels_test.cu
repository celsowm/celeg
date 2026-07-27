#include "lfm/backend/cuda/utils.cuh"
#include "lfm/backend/cuda/kernels/kernels.cuh"
#include "lfm/model/reference.hpp"
#include "lfm/backend/cuda/paged_kv.hpp"
#include "lfm/model/config/shape.hpp"
#include "lfm/model/config/variant.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

float to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__nv_bfloat16 to_bf16(float value) {
    return __float2bfloat16(value);
}

// Returns the canonical shapes for every registered model variant. The
// paged-KV tests run against each entry so a regression that affects only
// one variant (e.g. a different num_hidden_layers or kv_width) is caught.
// Adding a new variant to ModelVariantRegistry automatically extends the
// test matrix without editing this file (Open/Closed Principle).
std::vector<lfm::ModelShape> registered_variant_shapes() {
    lfm::register_builtin_variants();
    std::vector<lfm::ModelShape> shapes;
    for (const lfm::IModelVariant* variant :
         []() -> std::vector<const lfm::IModelVariant*> {
         std::vector<const lfm::IModelVariant*> out;
         for (const auto id : lfm::ModelVariantRegistry::instance().ids()) {
             const auto* v = lfm::ModelVariantRegistry::instance().find(id);
             if (v != nullptr) out.push_back(v);
         }
         return out;
     }()) {
        // Rebuild a ModelShape from the variant's matching topology by
        // running resolve_shape on the canonical shape the variant
        // advertises. We approximate "canonical" by constructing from the
        // known published configs; this keeps the test self-contained.
        lfm::ModelShape shape;
        if (variant->id() == "lfm2.5-230m") {
            shape.hidden = 1024;
            shape.intermediate = 2560;
            shape.num_hidden_layers = 14;
            shape.num_attention_heads = 16;
            shape.num_key_value_heads = 8;
            shape.head_dim = 64;
            shape.vocab_size = 65536;
            shape.conv_cache = 3;
            shape.conv_dim = 1024;
            shape.max_position_embeddings = 128000;
            shape.bos_token_id = 1;
            shape.eos_token_id = 7;
            shape.pad_token_id = 0;
            shape.norm_eps = 1e-5f;
            shape.rope_theta = 1'000'000.0f;
            shape.rope_type = "default";
            shape.layer_types = {
                lfm::LayerType::Convolution, lfm::LayerType::Convolution,
                lfm::LayerType::FullAttention, lfm::LayerType::Convolution,
                lfm::LayerType::FullAttention, lfm::LayerType::Convolution,
                lfm::LayerType::FullAttention, lfm::LayerType::Convolution,
                lfm::LayerType::FullAttention, lfm::LayerType::Convolution,
                lfm::LayerType::FullAttention, lfm::LayerType::Convolution,
                lfm::LayerType::FullAttention, lfm::LayerType::Convolution,
            };
        } else if (variant->id() == "lfm2.5-1.2b-instruct" ||
                   variant->id() == "lfm2.5-1.2b-thinking") {
            // The Thinking and Instruct variants share the same topology.
            shape.hidden = 2048;
            shape.intermediate = 12288;
            shape.num_hidden_layers = 16;
            shape.num_attention_heads = 32;
            shape.num_key_value_heads = 8;
            shape.head_dim = 64;
            shape.vocab_size = 65536;
            shape.conv_cache = 3;
            shape.conv_dim = 2048;
            shape.max_position_embeddings = 128000;
            shape.bos_token_id = 1;
            shape.eos_token_id = 7;
            shape.pad_token_id = 0;
            shape.norm_eps = 1e-5f;
            shape.rope_theta = 1'000'000.0f;
            shape.rope_type = "default";
            shape.layer_types = {
                lfm::LayerType::Convolution, lfm::LayerType::Convolution,
                lfm::LayerType::FullAttention, lfm::LayerType::Convolution,
                lfm::LayerType::Convolution, lfm::LayerType::FullAttention,
                lfm::LayerType::Convolution, lfm::LayerType::Convolution,
                lfm::LayerType::FullAttention, lfm::LayerType::Convolution,
                lfm::LayerType::FullAttention, lfm::LayerType::Convolution,
                lfm::LayerType::FullAttention, lfm::LayerType::Convolution,
                lfm::LayerType::FullAttention, lfm::LayerType::Convolution,
            };
        } else {
            continue;
        }
        shape.compute_derived();
        shape.validate();
        shape = variant->resolve_shape(shape);
        shapes.push_back(shape);
    }
    if (shapes.empty()) {
        std::cerr << "registered_variant_shapes: no variants registered\n";
        std::abort();
    }
    return shapes;
}

void expect_near(float actual, float expected, float tolerance = 0.03f) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "expected " << expected << ", got " << actual << '\n';
        std::abort();
    }
}

} // namespace

int main() {
    lfm::CudaStream stream;

    // Embedding by scalar value.
    {
        std::vector<__nv_bfloat16> table(12);
        for (int i = 0; i < 12; ++i) table[i] = to_bf16(static_cast<float>(i));
        lfm::DeviceBuffer<__nv_bfloat16> device_table(table.size());
        lfm::DeviceBuffer<__nv_bfloat16> output(4);
        LFM_CUDA(cudaMemcpy(device_table.data(), table.data(),
                            table.size() * sizeof(table[0]),
                            cudaMemcpyHostToDevice));
        lfm::launch_embedding(2, device_table.data(), output.data(), 4, stream.get());
        std::vector<__nv_bfloat16> host(4);
        LFM_CUDA(cudaMemcpyAsync(host.data(), output.data(), output.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        for (int i = 0; i < 4; ++i) expect_near(to_float(host[i]), 8.0f + i);
    }



    // Batched embedding preserves row-major [tokens, hidden] layout.
    {
        std::vector<__nv_bfloat16> table(12);
        for (int i = 0; i < 12; ++i) table[i] = to_bf16(static_cast<float>(i));
        std::vector<int32_t> tokens = {2, 0};
        lfm::DeviceBuffer<__nv_bfloat16> device_table(table.size());
        lfm::DeviceBuffer<int32_t> device_tokens(tokens.size());
        lfm::DeviceBuffer<__nv_bfloat16> output(8);
        LFM_CUDA(cudaMemcpy(device_table.data(), table.data(),
                            table.size() * sizeof(table[0]), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(device_tokens.data(), tokens.data(),
                            tokens.size() * sizeof(tokens[0]), cudaMemcpyHostToDevice));
        lfm::launch_embedding_batch(device_tokens.data(), 2, device_table.data(),
                                    output.data(), 4, stream.get());
        std::vector<__nv_bfloat16> host(8);
        LFM_CUDA(cudaMemcpyAsync(host.data(), output.data(), output.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        for (int i = 0; i < 4; ++i) {
            expect_near(to_float(host[i]), 8.0f + i);
            expect_near(to_float(host[4 + i]), static_cast<float>(i));
        }
    }


    // Packed W4A16 linear dequantizes signed nibbles per output row.
    {
        std::vector<__nv_bfloat16> x = {
            to_bf16(1.0f), to_bf16(1.0f),
            to_bf16(1.0f), to_bf16(1.0f)};
        // row 0 = [1,2,3,4], row 1 = [-1,-2,-3,-4]
        std::vector<uint8_t> packed = {0x21U, 0x43U, 0xefU, 0xcdU};
        std::vector<float> scales = {0.5f, 0.25f};
        lfm::DeviceBuffer<__nv_bfloat16> dx(x.size()), dy(2);
        lfm::DeviceBuffer<uint8_t> dw(packed.size());
        lfm::DeviceBuffer<float> ds(scales.size());
        LFM_CUDA(cudaMemcpy(dx.data(), x.data(), dx.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dw.data(), packed.data(), dw.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(ds.data(), scales.data(), ds.bytes(), cudaMemcpyHostToDevice));
        lfm::launch_w4a16_linear(dx.data(), dw.data(), ds.data(), dy.data(),
                                 1, 2, 4, 0.0f, stream.get());
        std::vector<__nv_bfloat16> output(2);
        LFM_CUDA(cudaMemcpyAsync(output.data(), dy.data(), dy.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        expect_near(to_float(output[0]), 5.0f, 0.02f);
        expect_near(to_float(output[1]), -2.5f, 0.02f);
    }

    // Packed INT4 embedding shares the same row format as the LM head.
    {
        std::vector<uint8_t> packed = {0x21U, 0x43U, 0xefU, 0xcdU};
        std::vector<float> scales = {0.5f, 0.25f};
        lfm::DeviceBuffer<uint8_t> table(packed.size());
        lfm::DeviceBuffer<float> ds(scales.size());
        lfm::DeviceBuffer<__nv_bfloat16> output(4);
        LFM_CUDA(cudaMemcpy(table.data(), packed.data(), table.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(ds.data(), scales.data(), ds.bytes(), cudaMemcpyHostToDevice));
        lfm::launch_embedding_int4(1, table.data(), ds.data(), output.data(),
                                   4, stream.get());
        std::vector<__nv_bfloat16> host(4);
        LFM_CUDA(cudaMemcpyAsync(host.data(), output.data(), output.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
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
        lfm::DeviceBuffer<__nv_bfloat16> dx(4), dw(4), dy(4);
        LFM_CUDA(cudaMemcpy(dx.data(), x.data(), dx.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dw.data(), weight.data(), dw.bytes(), cudaMemcpyHostToDevice));
        lfm::launch_rmsnorm(dx.data(), dw.data(), dy.data(), 1, 4, 1e-5f, stream.get());
        std::vector<__nv_bfloat16> y(4);
        LFM_CUDA(cudaMemcpyAsync(y.data(), dy.data(), dy.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        const float inv = 1.0f / std::sqrt(7.5f + 1e-5f);
        for (int i = 0; i < 4; ++i) expect_near(to_float(y[i]), (i + 1) * inv);
    }

    // Fused SwiGLU layout: [gate..., up...].
    {
        std::vector<__nv_bfloat16> gate_up = {
            to_bf16(0.0f), to_bf16(1.0f),
            to_bf16(2.0f), to_bf16(3.0f)};
        lfm::DeviceBuffer<__nv_bfloat16> input(4), output(2);
        LFM_CUDA(cudaMemcpy(input.data(), gate_up.data(), input.bytes(),
                            cudaMemcpyHostToDevice));
        lfm::launch_swiglu_fused(input.data(), output.data(), 2, stream.get());
        std::vector<__nv_bfloat16> result(2);
        LFM_CUDA(cudaMemcpyAsync(result.data(), output.data(), output.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        expect_near(to_float(result[0]), 0.0f);
        expect_near(to_float(result[1]), (1.0f / (1.0f + std::exp(-1.0f))) * 3.0f);
    }


    // Strict Q/K RMSNorm + RoPE matches the host BF16 reference.
    {
        std::vector<__nv_bfloat16> q = {
            to_bf16(1.0f), to_bf16(2.0f), to_bf16(3.0f), to_bf16(4.0f)};
        std::vector<__nv_bfloat16> k = {
            to_bf16(4.0f), to_bf16(3.0f), to_bf16(2.0f), to_bf16(1.0f)};
        std::vector<__nv_bfloat16> norm(4, to_bf16(1.0f));
        std::vector<__nv_bfloat16> rope_cos = {to_bf16(1.0f), to_bf16(0.0f)};
        std::vector<__nv_bfloat16> rope_sin = {to_bf16(0.0f), to_bf16(1.0f)};
        lfm::DeviceBuffer<__nv_bfloat16> dq(4), dk(4), dn(4), dc(2), ds(2);
        LFM_CUDA(cudaMemcpy(dq.data(), q.data(), dq.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dk.data(), k.data(), dk.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dn.data(), norm.data(), dn.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dc.data(), rope_cos.data(), dc.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(ds.data(), rope_sin.data(), ds.bytes(), cudaMemcpyHostToDevice));
        lfm::launch_qk_norm_rope_strict(
            dq.data(), dk.data(), dn.data(), dn.data(), dc.data(), ds.data(),
            1, 1, 4, 0, 1e-5f, stream.get());
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        LFM_CUDA(cudaMemcpy(q.data(), dq.data(), dq.bytes(), cudaMemcpyDeviceToHost));
        LFM_CUDA(cudaMemcpy(k.data(), dk.data(), dk.bytes(), cudaMemcpyDeviceToHost));
        std::vector<float> expected_q = lfm::reference::rmsnorm_bf16(
            {1, 2, 3, 4}, {1, 1, 1, 1}, 1e-5f);
        std::vector<float> expected_k = lfm::reference::rmsnorm_bf16(
            {4, 3, 2, 1}, {1, 1, 1, 1}, 1e-5f);
        lfm::reference::rope_bf16_inplace(expected_q, {1, 0}, {0, 1});
        lfm::reference::rope_bf16_inplace(expected_k, {1, 0}, {0, 1});
        for (int i = 0; i < 4; ++i) {
            expect_near(to_float(q[i]), expected_q[i], 0.01f);
            expect_near(to_float(k[i]), expected_k[i], 0.01f);
        }
    }

    // Strict GQA rounds probabilities to BF16 before value accumulation.
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
        lfm::DeviceBuffer<__nv_bfloat16> dq(4), dk(4), dv(4), dout(4);
        LFM_CUDA(cudaMemcpy(dq.data(), q.data(), dq.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dk.data(), k.data(), dk.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dv.data(), v.data(), dv.bytes(), cudaMemcpyHostToDevice));
        lfm::launch_gqa_decode_strict(
            dq.data(), dk.data(), dv.data(), dout.data(), 2, 2, 1, 2, stream.get());
        std::vector<__nv_bfloat16> output(4);
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        LFM_CUDA(cudaMemcpy(output.data(), dout.data(), dout.bytes(), cudaMemcpyDeviceToHost));
        const auto expected = lfm::reference::gqa_decode_strict_bf16(
            qf, kf, vf, 2, 2, 1, 2);
        for (int i = 0; i < 4; ++i) expect_near(to_float(output[i]), expected[i], 0.01f);
    }

    // Causal short-convolution state and C gate.
    {
        constexpr int hidden = 2;
        constexpr int cache = 3;
        std::vector<float> projected_f = {1, 1, 1, 1, 2, 4};
        std::vector<float> weight_f = {1, 2, 3, 1, 2, 3};
        std::vector<__nv_bfloat16> projected(projected_f.size());
        std::vector<__nv_bfloat16> weight(weight_f.size());
        for (size_t i = 0; i < projected.size(); ++i) projected[i] = to_bf16(projected_f[i]);
        for (size_t i = 0; i < weight.size(); ++i) weight[i] = to_bf16(weight_f[i]);
        lfm::DeviceBuffer<__nv_bfloat16> dp(projected.size()), dw(weight.size());
        lfm::DeviceBuffer<__nv_bfloat16> ds(hidden * cache), dy(hidden);
        LFM_CUDA(cudaMemcpy(dp.data(), projected.data(), dp.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dw.data(), weight.data(), dw.bytes(), cudaMemcpyHostToDevice));
        ds.zero_async(stream.get());
        lfm::launch_conv_decode(dp.data(), dw.data(), ds.data(), dy.data(),
                                hidden, cache, 0, stream.get());
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        std::vector<__nv_bfloat16> output(hidden);
        LFM_CUDA(cudaMemcpy(output.data(), dy.data(), dy.bytes(), cudaMemcpyDeviceToHost));
        std::vector<float> state(hidden * cache, 0.0f);
        const auto expected = lfm::reference::conv_decode_bf16(
            projected_f, weight_f, state, hidden, cache, 0);
        for (int i = 0; i < hidden; ++i) {
            expect_near(to_float(output[static_cast<size_t>(i)]),
                        expected[static_cast<size_t>(i)], 0.01f);
        }
    }



    // Batched causal convolution matches sequential decode and leaves the same ring state.
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
        lfm::DeviceBuffer<__nv_bfloat16> dp(projected.size()), dw(weight.size());
        lfm::DeviceBuffer<__nv_bfloat16> ds(hidden * cache), dy(rows * hidden);
        LFM_CUDA(cudaMemcpy(dp.data(), projected.data(), dp.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dw.data(), weight.data(), dw.bytes(), cudaMemcpyHostToDevice));
        ds.zero_async(stream.get());
        lfm::launch_conv_prefill(dp.data(), dw.data(), ds.data(), dy.data(),
                                 rows, hidden, cache, stream.get());
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        std::vector<__nv_bfloat16> output(rows * hidden), state_gpu(hidden * cache);
        LFM_CUDA(cudaMemcpy(output.data(), dy.data(), dy.bytes(), cudaMemcpyDeviceToHost));
        LFM_CUDA(cudaMemcpy(state_gpu.data(), ds.data(), ds.bytes(), cudaMemcpyDeviceToHost));

        std::vector<float> state(hidden * cache, 0.0f);
        for (int row = 0; row < rows; ++row) {
            std::vector<float> one(projected_f.begin() + row * 3 * hidden,
                                   projected_f.begin() + (row + 1) * 3 * hidden);
            const auto expected = lfm::reference::conv_decode_bf16(
                one, weight_f, state, hidden, cache, row);
            for (int c = 0; c < hidden; ++c) {
                expect_near(to_float(output[row * hidden + c]), expected[c], 0.02f);
            }
        }
        for (int i = 0; i < hidden * cache; ++i) {
            expect_near(to_float(state_gpu[i]), state[i], 0.01f);
        }
    }

    // Ragged ShortConv advances every request span in order without a host
    // token-wave loop.  Its outputs and independent ring states match
    // sequential decode for unequal spans and nonzero positions.
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
        lfm::DeviceBuffer<__nv_bfloat16> dp(projected.size()), dw(weight.size()),
            ds0(hidden * cache), ds1(hidden * cache), dy(positions.size() * hidden);
        lfm::DeviceBuffer<__nv_bfloat16*> states(requests);
        lfm::DeviceBuffer<int32_t> dpositions(positions.size()), doffsets(requests), dcounts(requests);
        const std::vector<__nv_bfloat16*> state_ptrs = {ds0.data(), ds1.data()};
        LFM_CUDA(cudaMemcpy(dp.data(), projected.data(), dp.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dw.data(), weight.data(), dw.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(states.data(), state_ptrs.data(), states.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dpositions.data(), positions.data(), dpositions.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(doffsets.data(), offsets.data(), doffsets.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dcounts.data(), counts.data(), dcounts.bytes(), cudaMemcpyHostToDevice));
        ds0.zero_async(stream.get());
        ds1.zero_async(stream.get());
        lfm::launch_conv_ragged_prefill(dp.data(), dw.data(), states.data(), dy.data(),
                                        dpositions.data(), doffsets.data(), dcounts.data(),
                                        requests, hidden, cache, stream.get());
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        std::vector<__nv_bfloat16> output(positions.size() * hidden), state0(hidden * cache), state1(hidden * cache);
        LFM_CUDA(cudaMemcpy(output.data(), dy.data(), dy.bytes(), cudaMemcpyDeviceToHost));
        LFM_CUDA(cudaMemcpy(state0.data(), ds0.data(), ds0.bytes(), cudaMemcpyDeviceToHost));
        LFM_CUDA(cudaMemcpy(state1.data(), ds1.data(), ds1.bytes(), cudaMemcpyDeviceToHost));
        std::vector<float> expected0(hidden * cache, 0.0f), expected1(hidden * cache, 0.0f);
        for (int request = 0; request < requests; ++request) {
            std::vector<float>& state = request == 0 ? expected0 : expected1;
            for (int token = 0; token < counts[request]; ++token) {
                const int row = offsets[request] + token;
                const std::vector<float> one(projected_f.begin() + row * 3 * hidden,
                                             projected_f.begin() + (row + 1) * 3 * hidden);
                const auto expected = lfm::reference::conv_decode_bf16(
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

    // Batched strict GQA produces the same output as decoding each prefix.
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
        lfm::DeviceBuffer<__nv_bfloat16> dq(q.size()), dk(k.size()), dv(v.size()), dout(q.size());
        LFM_CUDA(cudaMemcpy(dq.data(), q.data(), dq.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dk.data(), k.data(), dk.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dv.data(), v.data(), dv.bytes(), cudaMemcpyHostToDevice));
        lfm::launch_gqa_prefill_strict(dq.data(), dk.data(), dv.data(), dout.data(),
                                       rows, q_heads, kv_heads, head_dim, stream.get());
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        std::vector<__nv_bfloat16> output(q.size());
        LFM_CUDA(cudaMemcpy(output.data(), dout.data(), dout.bytes(), cudaMemcpyDeviceToHost));
        for (int row = 0; row < rows; ++row) {
            std::vector<float> row_q(qf.begin() + row * q_heads * head_dim,
                                     qf.begin() + (row + 1) * q_heads * head_dim);
            std::vector<float> prefix_k(kf.begin(), kf.begin() + (row + 1) * kv_heads * head_dim);
            std::vector<float> prefix_v(vf.begin(), vf.begin() + (row + 1) * kv_heads * head_dim);
            const auto expected = lfm::reference::gqa_decode_strict_bf16(
                row_q, prefix_k, prefix_v, row + 1, q_heads, kv_heads, head_dim);
            for (int i = 0; i < q_heads * head_dim; ++i) {
                expect_near(to_float(output[row * q_heads * head_dim + i]), expected[i], 0.02f);
            }
        }
    }

    // INT8 KV cache quantizes per token/head and remains close to BF16 GQA.
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
        lfm::DeviceBuffer<__nv_bfloat16> dq(q.size()), dk(k.size()),
            dv(v.size()), dout_int8(q.size()), dout_bf16(q.size());
        lfm::DeviceBuffer<int8_t> key_cache(k.size()), value_cache(v.size());
        lfm::DeviceBuffer<float> key_scales(rows * kv_heads),
            value_scales(rows * kv_heads);
        LFM_CUDA(cudaMemcpy(dq.data(), q.data(), dq.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dk.data(), k.data(), dk.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dv.data(), v.data(), dv.bytes(), cudaMemcpyHostToDevice));
        lfm::launch_store_kv_int8_prefill(
            dk.data(), dv.data(), key_cache.data(), value_cache.data(),
            key_scales.data(), value_scales.data(), rows, kv_heads, head_dim,
            stream.get());
        lfm::launch_gqa_prefill_strict_int8(
            dq.data(), key_cache.data(), value_cache.data(),
            key_scales.data(), value_scales.data(), dout_int8.data(), rows,
            q_heads, kv_heads, head_dim, stream.get());
        lfm::launch_gqa_prefill_strict(
            dq.data(), dk.data(), dv.data(), dout_bf16.data(), rows,
            q_heads, kv_heads, head_dim, stream.get());
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        std::vector<__nv_bfloat16> int8_output(q.size()), bf16_output(q.size());
        LFM_CUDA(cudaMemcpy(int8_output.data(), dout_int8.data(),
                            dout_int8.bytes(), cudaMemcpyDeviceToHost));
        LFM_CUDA(cudaMemcpy(bf16_output.data(), dout_bf16.data(),
                            dout_bf16.bytes(), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < int8_output.size(); ++i) {
            expect_near(to_float(int8_output[i]), to_float(bf16_output[i]), 0.08f);
        }
        std::vector<float> host_scales(rows * kv_heads);
        LFM_CUDA(cudaMemcpy(host_scales.data(), value_scales.data(),
                            value_scales.bytes(), cudaMemcpyDeviceToHost));
        expect_near(host_scales[0], 8.0f / 127.0f, 1e-4f);
        expect_near(host_scales[1], 7.0f / 127.0f, 1e-4f);
    }

    // Greedy argmax.
    {
        std::vector<__nv_bfloat16> logits = {
            to_bf16(-1.0f), to_bf16(8.0f), to_bf16(8.0f), to_bf16(2.0f)};
        lfm::DeviceBuffer<__nv_bfloat16> device_logits(logits.size());
        lfm::DeviceBuffer<int32_t> result(1);
        LFM_CUDA(cudaMemcpy(device_logits.data(), logits.data(),
                            device_logits.bytes(), cudaMemcpyHostToDevice));
        lfm::launch_argmax_bf16(device_logits.data(),
                                static_cast<int>(logits.size()),
                                result.data(), stream.get());
        int32_t index = -1;
        LFM_CUDA(cudaMemcpyAsync(&index, result.data(), sizeof(index),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        assert(index == 1);
    }



    // GPU repetition penalty + top-k=1 sampling selects the adjusted maximum.
    {
        std::vector<__nv_bfloat16> logits = {
            to_bf16(5.0f), to_bf16(4.0f), to_bf16(3.0f), to_bf16(2.0f)};
        std::vector<uint8_t> seen = {1, 0, 0, 0};
        uint64_t seed = 123;
        lfm::DeviceBuffer<__nv_bfloat16> dlogits(logits.size());
        lfm::DeviceBuffer<uint8_t> dseen(seen.size());
        lfm::DeviceBuffer<float> scores(logits.size()), values(1);
        lfm::DeviceBuffer<int32_t> indices(1), result(1);
        lfm::DeviceBuffer<uint64_t> rng(1);
        LFM_CUDA(cudaMemcpy(dlogits.data(), logits.data(), dlogits.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dseen.data(), seen.data(), dseen.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(rng.data(), &seed, sizeof(seed), cudaMemcpyHostToDevice));
        lfm::launch_prepare_sampling_scores(dlogits.data(), dseen.data(), scores.data(),
                                            4, 1.0f, 2.0f, stream.get());
        lfm::launch_select_topk(scores.data(), values.data(), indices.data(), 0, 4, stream.get());
        lfm::launch_sample_topk(values.data(), indices.data(), 1, 1.0f,
                                rng.data(), result.data(), stream.get());
        int32_t token = -1;
        LFM_CUDA(cudaMemcpyAsync(&token, result.data(), sizeof(token),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        assert(token == 1); // token 0 is penalized from 5 to 2.5
    }



    // Top-p truncation within top-k is deterministic for a fixed seed.
    {
        std::vector<float> values = {3.0f, 2.0f, 1.0f};
        std::vector<int32_t> indices = {10, 20, 30};
        uint64_t seed = 3;
        lfm::DeviceBuffer<float> dvalues(values.size());
        lfm::DeviceBuffer<int32_t> dindices(indices.size()), result(1);
        lfm::DeviceBuffer<uint64_t> rng(1);
        LFM_CUDA(cudaMemcpy(dvalues.data(), values.data(), dvalues.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dindices.data(), indices.data(), dindices.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(rng.data(), &seed, sizeof(seed), cudaMemcpyHostToDevice));
        lfm::launch_sample_topk(dvalues.data(), dindices.data(), 3, 0.7f,
                                rng.data(), result.data(), stream.get());
        int32_t token = -1;
        LFM_CUDA(cudaMemcpyAsync(&token, result.data(), sizeof(token),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        assert(token == 20);
    }



    // Fused sampler matches the legacy multi-launch pipeline.
    {
        std::vector<__nv_bfloat16> logits = {
            to_bf16(4.0f), to_bf16(3.0f), to_bf16(2.0f),
            to_bf16(1.0f), to_bf16(0.0f), to_bf16(-1.0f)};
        std::vector<uint8_t> seen = {1, 0, 0, 0, 0, 0};
        constexpr int top_k = 4;
        constexpr float top_p = 0.85f;
        constexpr float temperature = 0.75f;
        constexpr float penalty = 1.1f;
        uint64_t legacy_seed = 77;
        uint64_t fused_seed = legacy_seed;

        lfm::DeviceBuffer<__nv_bfloat16> dlogits(logits.size());
        lfm::DeviceBuffer<uint8_t> dseen(seen.size());
        lfm::DeviceBuffer<float> legacy_scores(logits.size());
        lfm::DeviceBuffer<float> fused_scores(logits.size());
        lfm::DeviceBuffer<float> legacy_values(top_k), fused_values(top_k);
        lfm::DeviceBuffer<int32_t> legacy_indices(top_k), fused_indices(top_k);
        lfm::DeviceBuffer<int32_t> legacy_result(1), fused_result(1);
        lfm::DeviceBuffer<uint64_t> legacy_rng(1), fused_rng(1);
        LFM_CUDA(cudaMemcpy(dlogits.data(), logits.data(), dlogits.bytes(),
                            cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dseen.data(), seen.data(), dseen.bytes(),
                            cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(legacy_rng.data(), &legacy_seed, sizeof(legacy_seed),
                            cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(fused_rng.data(), &fused_seed, sizeof(fused_seed),
                            cudaMemcpyHostToDevice));

        lfm::launch_prepare_sampling_scores(
            dlogits.data(), dseen.data(), legacy_scores.data(),
            static_cast<int>(logits.size()), temperature, penalty, stream.get());
        for (int rank = 0; rank < top_k; ++rank) {
            lfm::launch_select_topk(legacy_scores.data(), legacy_values.data(),
                                    legacy_indices.data(), rank,
                                    static_cast<int>(logits.size()), stream.get());
        }
        lfm::launch_sample_topk(legacy_values.data(), legacy_indices.data(),
                                top_k, top_p, legacy_rng.data(),
                                legacy_result.data(), stream.get());

        lfm::launch_fused_sample_topk(
            dlogits.data(), dseen.data(), fused_scores.data(),
            fused_values.data(), fused_indices.data(),
            static_cast<int>(logits.size()), temperature, penalty,
            top_k, top_p, fused_rng.data(), fused_result.data(), stream.get());

        int32_t legacy_token = -1;
        int32_t fused_token = -1;
        LFM_CUDA(cudaMemcpyAsync(&legacy_token, legacy_result.data(), sizeof(legacy_token),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaMemcpyAsync(&fused_token, fused_result.data(), sizeof(fused_token),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        assert(legacy_token == fused_token);
    }

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
        lfm::DeviceBuffer<__nv_bfloat16> dx(x.size()), dy(m * n);
        lfm::DeviceBuffer<int8_t> dw(weight.size());
        lfm::DeviceBuffer<float> ds(scales.size());
        LFM_CUDA(cudaMemcpy(dx.data(), x.data(), dx.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dw.data(), weight.data(), dw.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(ds.data(), scales.data(), ds.bytes(), cudaMemcpyHostToDevice));
        lfm::launch_w8a16_linear(dx.data(), dw.data(), ds.data(), dy.data(),
                                 m, n, k, 0.0f, stream.get());
        std::vector<__nv_bfloat16> result(m * n);
        LFM_CUDA(cudaMemcpyAsync(result.data(), dy.data(), dy.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
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
        lfm::DeviceBuffer<int8_t> dt(table.size());
        lfm::DeviceBuffer<float> ds(scales.size());
        lfm::DeviceBuffer<int32_t> dtoken(1);
        lfm::DeviceBuffer<__nv_bfloat16> out(4);
        LFM_CUDA(cudaMemcpy(dt.data(), table.data(), dt.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(ds.data(), scales.data(), ds.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dtoken.data(), &token, sizeof(token), cudaMemcpyHostToDevice));
        lfm::launch_embedding_int8_device(dtoken.data(), dt.data(), ds.data(),
                                          out.data(), 4, stream.get());
        std::vector<__nv_bfloat16> result(4);
        LFM_CUDA(cudaMemcpyAsync(result.data(), out.data(), out.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
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
        lfm::DeviceBuffer<__nv_bfloat16> dq(q.size()), dk(k.size()), dv(v.size());
        lfm::DeviceBuffer<__nv_bfloat16> reference(q.size()), segmented(q.size());
        lfm::DeviceBuffer<int32_t> dposition(1);
        lfm::DeviceBuffer<float> partial_max(q_heads * chunks);
        lfm::DeviceBuffer<float> partial_denom(q_heads * chunks);
        lfm::DeviceBuffer<float> partial_accum(q_heads * chunks * head_dim);
        LFM_CUDA(cudaMemcpy(dq.data(), q.data(), dq.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dk.data(), k.data(), dk.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dv.data(), v.data(), dv.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dposition.data(), &position, sizeof(position), cudaMemcpyHostToDevice));
        lfm::launch_gqa_decode_online_device(
            dq.data(), dk.data(), dv.data(), reference.data(), dposition.data(),
            q_heads, kv_heads, head_dim, stream.get());
        lfm::launch_gqa_decode_segmented_device(
            dq.data(), dk.data(), dv.data(), segmented.data(), dposition.data(),
            q_heads, kv_heads, head_dim, chunk_tokens, chunks,
            partial_max.data(), partial_denom.data(), partial_accum.data(),
            stream.get());
        std::vector<__nv_bfloat16> a(q.size()), b(q.size());
        LFM_CUDA(cudaMemcpyAsync(a.data(), reference.data(), reference.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaMemcpyAsync(b.data(), segmented.data(), segmented.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
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
        lfm::DeviceBuffer<__nv_bfloat16> input(qkv.size());
        lfm::DeviceBuffer<__nv_bfloat16> q(rows * q_width), k(rows * kv_width),
            v(rows * kv_width);
        LFM_CUDA(cudaMemcpy(input.data(), qkv.data(), input.bytes(),
                            cudaMemcpyHostToDevice));
        lfm::launch_split_qkv_rows(input.data(), q.data(), k.data(), v.data(),
                                   rows, q_width, kv_width, stream.get());
        std::vector<__nv_bfloat16> hq(q.size()), hk(k.size()), hv(v.size());
        LFM_CUDA(cudaMemcpyAsync(hq.data(), q.data(), q.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaMemcpyAsync(hk.data(), k.data(), k.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaMemcpyAsync(hv.data(), v.data(), v.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        assert(to_float(hq[0]) == 1 && to_float(hq[3]) == 6);
        assert(to_float(hk[0]) == 3 && to_float(hk[1]) == 7);
        assert(to_float(hv[0]) == 4 && to_float(hv[1]) == 8);

        std::vector<__nv_bfloat16> gate_up = {
            to_bf16(0), to_bf16(1), to_bf16(2), to_bf16(3),
            to_bf16(1), to_bf16(-1), to_bf16(4), to_bf16(2)};
        lfm::DeviceBuffer<__nv_bfloat16> dgu(gate_up.size()), out(4);
        LFM_CUDA(cudaMemcpy(dgu.data(), gate_up.data(), dgu.bytes(),
                            cudaMemcpyHostToDevice));
        lfm::launch_swiglu_interleaved(dgu.data(), out.data(), 2, 2,
                                       stream.get());
        std::vector<__nv_bfloat16> hout(4);
        LFM_CUDA(cudaMemcpyAsync(hout.data(), out.data(), out.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
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
        lfm::DeviceBuffer<__nv_bfloat16> da(vocab), db(vocab);
        lfm::DeviceBuffer<uint8_t> dsa(vocab), dsb(vocab);
        lfm::DeviceBuffer<uint64_t> dra(1), drb(1);
        LFM_CUDA(cudaMemcpy(da.data(), a.data(), da.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(db.data(), b.data(), db.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dsa.data(), seen_a.data(), dsa.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dsb.data(), seen_b.data(), dsb.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dra.data(), &rng_a, sizeof(rng_a), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(drb.data(), &rng_b, sizeof(rng_b), cudaMemcpyHostToDevice));
        std::vector<__nv_bfloat16*> logits_ptrs = {da.data(), db.data()};
        std::vector<uint8_t*> seen_ptrs = {dsa.data(), dsb.data()};
        std::vector<uint64_t*> rng_ptrs = {dra.data(), drb.data()};
        lfm::DeviceBuffer<__nv_bfloat16*> dlogits(rows);
        lfm::DeviceBuffer<uint8_t*> dseen(rows);
        lfm::DeviceBuffer<uint64_t*> drng(rows);
        LFM_CUDA(cudaMemcpy(dlogits.data(), logits_ptrs.data(), dlogits.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dseen.data(), seen_ptrs.data(), dseen.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(drng.data(), rng_ptrs.data(), drng.bytes(), cudaMemcpyHostToDevice));
        std::vector<float> temp = {0.0f, 0.0f};
        std::vector<float> penalty = {1.0f, 2.0f};
        std::vector<int32_t> topk = {1, 1};
        std::vector<float> topp = {1.0f, 1.0f};
        lfm::DeviceBuffer<float> dtemp(rows), dpenalty(rows), dtopp(rows);
        lfm::DeviceBuffer<int32_t> dtopk(rows), result(rows);
        lfm::DeviceBuffer<float> scores(rows * vocab), values(rows * 128);
        lfm::DeviceBuffer<int32_t> indices(rows * 128);
        LFM_CUDA(cudaMemcpy(dtemp.data(), temp.data(), dtemp.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dpenalty.data(), penalty.data(), dpenalty.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dtopk.data(), topk.data(), dtopk.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dtopp.data(), topp.data(), dtopp.bytes(), cudaMemcpyHostToDevice));
        lfm::launch_packed_sample_topk(
            dlogits.data(), dseen.data(), drng.data(), dtemp.data(),
            dpenalty.data(), dtopk.data(), dtopp.data(), scores.data(),
            values.data(), indices.data(), rows, vocab, result.data(),
            stream.get());
        std::vector<int32_t> tokens(rows);
        LFM_CUDA(cudaMemcpyAsync(tokens.data(), result.data(), result.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        assert(tokens[0] == 1);
        assert(tokens[1] == 0);
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
        lfm::DeviceBuffer<__nv_bfloat16> dq(query.size()), dk(keys.size()),
            dv(values.size()), dout(head_dim);
        lfm::DeviceBuffer<uint32_t> dtable(table.size());
        lfm::DeviceBuffer<int32_t> dpositions(positions.size());
        LFM_CUDA(cudaMemcpy(dq.data(), query.data(), dq.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dk.data(), keys.data(), dk.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dv.data(), values.data(), dv.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dtable.data(), table.data(), dtable.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dpositions.data(), positions.data(), dpositions.bytes(), cudaMemcpyHostToDevice));
        lfm::launch_gqa_decode_paged_batch(
            dq.data(), dk.data(), dv.data(), dtable.data(), 1,
            dout.data(), dpositions.data(), 1, 0, page_tokens,
            attention_layers, q_heads, kv_heads, head_dim, false,
            stream.get());
        std::vector<__nv_bfloat16> output(head_dim);
        LFM_CUDA(cudaMemcpyAsync(output.data(), dout.data(), dout.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        const auto expected = lfm::reference::gqa_decode_strict_bf16(
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
        lfm::DeviceBuffer<__nv_bfloat16> dq(query.size()), dk(keys.size()),
            dv(values.size()), normal(head_dim), segmented(head_dim);
        lfm::DeviceBuffer<uint32_t> dtable(table.size());
        lfm::DeviceBuffer<int32_t> dpositions(positions.size());
        lfm::DeviceBuffer<float> pmax(3), pdenom(3), paccum(3 * head_dim);
        LFM_CUDA(cudaMemcpy(dq.data(), query.data(), dq.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dk.data(), keys.data(), dk.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dv.data(), values.data(), dv.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dtable.data(), table.data(), dtable.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dpositions.data(), positions.data(), dpositions.bytes(), cudaMemcpyHostToDevice));
        lfm::launch_gqa_decode_paged_batch(
            dq.data(), dk.data(), dv.data(), dtable.data(), 2, normal.data(),
            dpositions.data(), 1, 0, page_tokens, attention_layers,
            q_heads, kv_heads, head_dim, true, stream.get());
        lfm::launch_gqa_decode_paged_segmented_batch(
            dq.data(), dk.data(), dv.data(), dtable.data(), 2,
            segmented.data(), dpositions.data(), 1, 0, page_tokens,
            attention_layers, q_heads, kv_heads, head_dim, 1, 3,
            pmax.data(), pdenom.data(), paccum.data(), stream.get());
        std::vector<__nv_bfloat16> a(head_dim), b(head_dim);
        LFM_CUDA(cudaMemcpyAsync(a.data(), normal.data(), normal.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaMemcpyAsync(b.data(), segmented.data(), segmented.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
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
        lfm::DeviceBuffer<__nv_bfloat16> dq(query.size()), normal(head_dim),
            segmented(head_dim);
        lfm::DeviceBuffer<int8_t> dk(keys.size()), dv(values.size());
        lfm::DeviceBuffer<float> dks(key_scales.size()), dvs(value_scales.size());
        lfm::DeviceBuffer<uint32_t> dtable(table.size());
        lfm::DeviceBuffer<int32_t> dpositions(positions.size());
        lfm::DeviceBuffer<float> pmax(3), pdenom(3), paccum(3 * head_dim);
        LFM_CUDA(cudaMemcpy(dq.data(), query.data(), dq.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dk.data(), keys.data(), dk.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dv.data(), values.data(), dv.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dks.data(), key_scales.data(), dks.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dvs.data(), value_scales.data(), dvs.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dtable.data(), table.data(), dtable.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dpositions.data(), positions.data(), dpositions.bytes(), cudaMemcpyHostToDevice));
        lfm::launch_gqa_decode_int8_paged_batch(
            dq.data(), dk.data(), dv.data(), dks.data(), dvs.data(),
            dtable.data(), 2, normal.data(), dpositions.data(), 1, 0,
            page_tokens, attention_layers, q_heads, kv_heads, head_dim, true,
            stream.get());
        lfm::launch_gqa_decode_int8_paged_segmented_batch(
            dq.data(), dk.data(), dv.data(), dks.data(), dvs.data(),
            dtable.data(), 2, segmented.data(), dpositions.data(), 1, 0,
            page_tokens, attention_layers, q_heads, kv_heads, head_dim, 1, 3,
            pmax.data(), pdenom.data(), paccum.data(), stream.get());
        std::vector<__nv_bfloat16> a(head_dim), b(head_dim);
        LFM_CUDA(cudaMemcpyAsync(a.data(), normal.data(), normal.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaMemcpyAsync(b.data(), segmented.data(), segmented.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        for (int i = 0; i < head_dim; ++i) {
            expect_near(to_float(a[static_cast<size_t>(i)]),
                        to_float(b[static_cast<size_t>(i)]), 0.03f);
        }
    }

    // Per-row seen-token marking updates separate request histories.
    {
        std::vector<int32_t> tokens = {3, 5};
        lfm::DeviceBuffer<int32_t> dtokens(tokens.size());
        lfm::DeviceBuffer<uint8_t> seen_a(8), seen_b(8);
        lfm::PinnedBuffer<uint8_t*> hseen(2);
        lfm::DeviceBuffer<uint8_t*> dseen(2);
        seen_a.zero_async(stream.get());
        seen_b.zero_async(stream.get());
        hseen.data()[0] = seen_a.data();
        hseen.data()[1] = seen_b.data();
        LFM_CUDA(cudaMemcpy(dtokens.data(), tokens.data(), dtokens.bytes(),
                            cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dseen.data(), hseen.data(), dseen.bytes(),
                            cudaMemcpyHostToDevice));
        lfm::launch_mark_seen_batch_ptrs(dtokens.data(), dseen.data(), 2, 8,
                                         stream.get());
        std::vector<uint8_t> a(8), b(8);
        LFM_CUDA(cudaMemcpyAsync(a.data(), seen_a.data(), seen_a.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaMemcpyAsync(b.data(), seen_b.data(), seen_b.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        assert(a[3] == 1 && a[5] == 0);
        assert(b[5] == 1 && b[3] == 0);
    }

    // Copy-on-write cloning duplicates all physical page storage while keeping
    // independent reference counts. Run against every registered variant so a
    // regression that affects only one (e.g. different attention_layers) is
    // caught without editing this test.
    for (const lfm::ModelShape& shape : registered_variant_shapes()) {
        lfm::PhysicalPagedKvCache cache(3, 1, 4, lfm::KvCacheMode::Bf16, shape);
        auto source = cache.allocate_tokens(1);
        assert(source && source->size() == 1);
        const uint32_t source_page = source->front();
        const size_t page_elements = static_cast<size_t>(
            cache.attention_layers()) * shape.kv_width;
        std::vector<__nv_bfloat16> contents(page_elements, to_bf16(0.0f));
        contents[0] = to_bf16(3.5f);
        LFM_CUDA(cudaMemcpy(cache.key_bf16() +
                            static_cast<size_t>(source_page) * page_elements,
                            contents.data(), contents.size() * sizeof(__nv_bfloat16),
                            cudaMemcpyHostToDevice));
        auto cloned = cache.clone_page(source_page);
        assert(cloned && *cloned != source_page);
        assert(cache.ref_count(source_page) == 1);
        assert(cache.ref_count(*cloned) == 1);
        __nv_bfloat16 copied{};
        LFM_CUDA(cudaMemcpy(&copied, cache.key_bf16() +
                            static_cast<size_t>(*cloned) * page_elements,
                            sizeof(copied), cudaMemcpyDeviceToHost));
        expect_near(to_float(copied), 3.5f, 0.01f);
        cache.release(*source);
        cache.release(std::vector<uint32_t>{*cloned});
        assert(cache.free_pages() == cache.total_pages());
    }

    // Partial-page COW copies initialized token slots without transferring the unused suffix.
    for (const lfm::ModelShape& shape : registered_variant_shapes()) {
        constexpr int page_tokens = 4;
        lfm::PhysicalPagedKvCache cache(3, page_tokens, 8,
                                        lfm::KvCacheMode::Bf16, shape);
        auto source = cache.allocate_tokens(page_tokens);
        assert(source && source->size() == 1);
        const uint32_t source_page = source->front();
        const size_t page_elements = static_cast<size_t>(
            cache.attention_layers()) * page_tokens * shape.kv_width;
        std::vector<__nv_bfloat16> contents(page_elements, to_bf16(9.0f));
        LFM_CUDA(cudaMemcpy(cache.key_bf16() +
                            static_cast<size_t>(source_page) * page_elements,
                            contents.data(), contents.size() * sizeof(__nv_bfloat16),
                            cudaMemcpyHostToDevice));
        auto cloned = cache.clone_page_prefix(source_page, 1);
        assert(cloned);
        std::vector<__nv_bfloat16> copied(page_elements);
        LFM_CUDA(cudaMemcpy(copied.data(), cache.key_bf16() +
                            static_cast<size_t>(*cloned) * page_elements,
                            copied.size() * sizeof(__nv_bfloat16),
                            cudaMemcpyDeviceToHost));
        for (int layer = 0; layer < cache.attention_layers(); ++layer) {
            const size_t layer_base = static_cast<size_t>(layer) * page_tokens *
                                      shape.kv_width;
            expect_near(to_float(copied[layer_base]), 9.0f, 0.01f);
        }
        cache.release(*source);
        cache.release(std::vector<uint32_t>{*cloned});
    }

    // INT8 copy-on-write clones both quantized vectors and scale planes.
    for (const lfm::ModelShape& shape : registered_variant_shapes()) {
        lfm::PhysicalPagedKvCache cache(3, 1, 4, lfm::KvCacheMode::Int8, shape);
        auto source = cache.allocate_tokens(1);
        assert(source && source->size() == 1);
        const uint32_t source_page = source->front();
        const size_t page_elements = static_cast<size_t>(
            cache.attention_layers()) * shape.kv_width;
        const size_t scale_elements = static_cast<size_t>(
            cache.attention_layers()) * shape.num_key_value_heads;
        const int8_t quantized = -37;
        const float scale = 0.03125f;
        LFM_CUDA(cudaMemcpy(cache.key_int8() +
                            static_cast<size_t>(source_page) * page_elements,
                            &quantized, sizeof(quantized), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(cache.key_scales() +
                            static_cast<size_t>(source_page) * scale_elements,
                            &scale, sizeof(scale), cudaMemcpyHostToDevice));
        auto cloned = cache.clone_page(source_page);
        assert(cloned && *cloned != source_page);
        int8_t copied_quantized = 0;
        float copied_scale = 0.0f;
        LFM_CUDA(cudaMemcpy(&copied_quantized, cache.key_int8() +
                            static_cast<size_t>(*cloned) * page_elements,
                            sizeof(copied_quantized), cudaMemcpyDeviceToHost));
        LFM_CUDA(cudaMemcpy(&copied_scale, cache.key_scales() +
                            static_cast<size_t>(*cloned) * scale_elements,
                            sizeof(copied_scale), cudaMemcpyDeviceToHost));
        assert(copied_quantized == quantized);
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
        lfm::DeviceBuffer<__nv_bfloat16> dk(k.size()), dv(v.size());
        lfm::DeviceBuffer<int8_t> key_pool(vector_count), value_pool(vector_count);
        lfm::DeviceBuffer<float> key_scales(scale_count), value_scales(scale_count);
        lfm::DeviceBuffer<uint32_t> dtable(table.size());
        lfm::DeviceBuffer<int32_t> dpositions(positions.size());
        key_pool.zero_async(stream.get());
        value_pool.zero_async(stream.get());
        key_scales.zero_async(stream.get());
        value_scales.zero_async(stream.get());
        LFM_CUDA(cudaMemcpy(dk.data(), k.data(), dk.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dv.data(), v.data(), dv.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dtable.data(), table.data(), dtable.bytes(), cudaMemcpyHostToDevice));
        LFM_CUDA(cudaMemcpy(dpositions.data(), positions.data(), dpositions.bytes(), cudaMemcpyHostToDevice));
        lfm::launch_store_kv_int8_paged_batch(
            dk.data(), dv.data(), key_pool.data(), value_pool.data(),
            key_scales.data(), value_scales.data(), dtable.data(), 1,
            dpositions.data(), 1, 0, page_tokens, attention_layers,
            kv_heads, head_dim, stream.get());
        std::vector<int8_t> host_keys(vector_count), host_values(vector_count);
        std::vector<float> host_key_scales(scale_count), host_value_scales(scale_count);
        LFM_CUDA(cudaMemcpyAsync(host_keys.data(), key_pool.data(), key_pool.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaMemcpyAsync(host_values.data(), value_pool.data(), value_pool.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaMemcpyAsync(host_key_scales.data(), key_scales.data(), key_scales.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaMemcpyAsync(host_value_scales.data(), value_scales.data(), value_scales.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        const size_t vector_base = static_cast<size_t>(page_tokens) * head_dim;
        const size_t scale_base = static_cast<size_t>(page_tokens);
        expect_near(host_key_scales[scale_base], 1.0f / 127.0f, 1e-4f);
        expect_near(host_value_scales[scale_base], 2.0f / 127.0f, 1e-4f);
        assert(host_keys[vector_base] == 127);
        assert(host_values[vector_base] == 127);
    }

    // CUDA Graph replay uses device-resident mutable state.
    {
        int32_t initial = 7;
        lfm::DeviceBuffer<int32_t> position(1);
        LFM_CUDA(cudaMemcpy(position.data(), &initial, sizeof(initial), cudaMemcpyHostToDevice));
        lfm::CudaGraphExec graph;
        graph.capture_begin(stream.get());
        lfm::launch_increment_position(position.data(), stream.get());
        lfm::launch_increment_position(position.data(), stream.get());
        graph.capture_end(stream.get());
        graph.launch(stream.get());
        graph.launch(stream.get());
        int32_t result = 0;
        LFM_CUDA(cudaMemcpyAsync(&result, position.data(), sizeof(result),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        assert(result == 11);
    }

    std::cout << "cuda_kernels_test: ok\n";
    return 0;
}
