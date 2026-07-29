#include "lfm/backend/cuda/kernels/gguf.cuh"
#include "lfm/backend/cuda/kernels/mmq.hpp"
#include "lfm/backend/cuda/gemm_dispatcher.hpp"
#include "lfm/backend/cuda/utils.cuh"
#include "lfm/detail/checkpoint/bootstrap.hpp"
#include "lfm/model/model.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

namespace {

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(status) << "\n";
        std::exit(1);
    }
}

template <typename T>
float host_bf16(T value) { return __bfloat162float(value); }

bool close(float a, float b) {
    return std::abs(a - b) <= 0.05f * std::max(1.0f, std::abs(b));
}

} // namespace

int main() {
    constexpr int k = 256;
    constexpr int q4_bytes = 144;
    constexpr int q6_bytes = 210;
    std::vector<uint8_t> q4(q4_bytes, 0);
    // d=1, dmin=0, all sub-block scales=1, all nibbles=1 => every value is 1.
    q4[0] = 0x00; q4[1] = 0x3c; // half(1.0)
    q4[2] = 0x00; q4[3] = 0x00;
    for (int i = 4; i < 16; ++i) q4[i] = 1;
    for (int i = 16; i < q4_bytes; ++i) q4[i] = 0x11;
    std::vector<uint8_t> q6(q6_bytes, 0);
    // Zero scales make every Q6_K value exactly zero regardless of packed q.

    uint8_t* d_q4 = nullptr;
    uint8_t* d_q6 = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&d_q4), q4.size()), "cudaMalloc q4");
    check(cudaMalloc(reinterpret_cast<void**>(&d_q6), q6.size()), "cudaMalloc q6");
    check(cudaMemcpy(d_q4, q4.data(), q4.size(), cudaMemcpyHostToDevice), "copy q4");
    check(cudaMemcpy(d_q6, q6.data(), q6.size(), cudaMemcpyHostToDevice), "copy q6");

    std::vector<__nv_bfloat16> hx(k, __float2bfloat16(2.0f));
    __nv_bfloat16* d_x = nullptr;
    __nv_bfloat16* d_y = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&d_x), 2 * k * sizeof(*d_x)), "cudaMalloc x");
    check(cudaMalloc(reinterpret_cast<void**>(&d_y), 2 * sizeof(*d_y)), "cudaMalloc y");
    std::vector<__nv_bfloat16> hx2(2 * k, __float2bfloat16(2.0f));
    check(cudaMemcpy(d_x, hx2.data(), hx2.size() * sizeof(*d_x), cudaMemcpyHostToDevice), "copy x");

    lfm::launch_gguf_linear_segment(d_x, d_q4, lfm::GgmlType::Q4_K, d_y,
                                    1, 1, k, q4_bytes, 1, 0.0f, nullptr);
    __nv_bfloat16 y = 0;
    check(cudaMemcpy(&y, d_y, sizeof(y), cudaMemcpyDeviceToHost), "copy y q4");
    if (!close(host_bf16(y), 512.0f)) return 2;

    lfm::launch_gguf_linear_segment(d_x, d_q6, lfm::GgmlType::Q6_K, d_y,
                                    1, 1, k, q6_bytes, 1, 3.0f, nullptr);
    check(cudaMemcpy(&y, d_y, sizeof(y), cudaMemcpyDeviceToHost), "copy y q6");
    if (!close(host_bf16(y), 1536.0f)) return 3;

    // Exercise the m>1 path and row-local embedding gather.
    std::vector<__nv_bfloat16> hy(2, __float2bfloat16(0.0f));
    lfm::launch_gguf_linear_segment(d_x, d_q4, lfm::GgmlType::Q4_K, d_y,
                                    2, 1, k, q4_bytes, 1, 0.0f, nullptr);
    std::vector<__nv_bfloat16> gemm_y(2);
    check(cudaMemcpy(gemm_y.data(), d_y, 2 * sizeof(y), cudaMemcpyDeviceToHost), "copy y gemm");
    if (!close(host_bf16(gemm_y[0]), 512.0f) || !close(host_bf16(gemm_y[1]), 512.0f)) return 4;

    // Exercise the dispatcher with disjoint GGUF segments. Every segment
    // owns a separate output range, so each must receive the caller's beta.
    lfm::ModelOptions dispatcher_options;
    lfm::GemmDispatcher dispatcher(nullptr, dispatcher_options);
    lfm::LinearWeight segmented;
    segmented.kind = lfm::LinearStorageKind::Q4_K;
    segmented.rows = 2;
    segmented.cols = k;
    segmented.gguf_segments = {
        {d_q4, lfm::GgmlType::Q4_K, 0, 1, k, q4_bytes},
        {d_q6, lfm::GgmlType::Q6_K, 1, 1, k, q6_bytes},
    };
    const lfm::ExecutionPlan dispatcher_plan =
        lfm::ExecutionPlan::compile(dispatcher_options, 1024);
    std::vector<__nv_bfloat16> segmented_y(2, __float2bfloat16(9.0f));
    check(cudaMemcpy(d_y, segmented_y.data(), 2 * sizeof(*d_y),
                     cudaMemcpyHostToDevice), "copy segmented y");
    dispatcher.linear(d_x, segmented, d_y, 1, 2, k, 0.0f, dispatcher_plan);
    check(cudaMemcpy(segmented_y.data(), d_y, 2 * sizeof(*d_y),
                     cudaMemcpyDeviceToHost), "copy segmented result");
    if (!close(host_bf16(segmented_y[0]), 512.0f) ||
        !close(host_bf16(segmented_y[1]), 0.0f)) return 9;
    segmented_y.assign(2, __float2bfloat16(3.0f));
    check(cudaMemcpy(d_y, segmented_y.data(), 2 * sizeof(*d_y),
                     cudaMemcpyHostToDevice), "copy segmented beta y");
    dispatcher.linear(d_x, segmented, d_y, 1, 2, k, 2.0f, dispatcher_plan);
    check(cudaMemcpy(segmented_y.data(), d_y, 2 * sizeof(*d_y),
                     cudaMemcpyDeviceToHost), "copy segmented beta result");
    if (!close(host_bf16(segmented_y[0]), 518.0f) ||
        !close(host_bf16(segmented_y[1]), 6.0f)) return 10;

    lfm::GgufLinearSegment segment{d_q4, lfm::GgmlType::Q4_K, 0, 1, k, q4_bytes};
    __nv_bfloat16* d_embed = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&d_embed), k * sizeof(*d_embed)), "cudaMalloc embed");
    lfm::launch_gguf_embedding(0, segment, d_embed, nullptr);
    check(cudaMemcpy(hy.data(), d_embed, 2 * sizeof(*d_embed), cudaMemcpyDeviceToHost), "copy embed");
    if (!close(host_bf16(hy[0]), 1.0f)) return 5;

    // MMQ (Q8_1 x __dp4a) correctness: dequantize a randomized multi-super-
    // block Q4_K row via the already-validated launch_gguf_dequant (same
    // q4k_value math the whole model pipeline depends on) for a float
    // reference, then compare launch_q4k_mmq's int8-quantized-activation
    // result against a plain double-precision dot product of that
    // reference weight with the unquantized activation. int8 quantization
    // is lossy, so the tolerance here is a relative error bound, not
    // bit-exactness.
    {
        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> byte_dist(0, 255);
        constexpr int mmq_n = 3;
        constexpr int mmq_k = 512; // 2 super-blocks
        constexpr int mmq_super_blocks = mmq_k / 256;
        const size_t mmq_row_bytes = static_cast<size_t>(mmq_super_blocks) * q4_bytes;
        std::vector<uint8_t> mmq_weight(static_cast<size_t>(mmq_n) * mmq_row_bytes);
        for (auto& byte : mmq_weight) byte = static_cast<uint8_t>(byte_dist(rng));
        // Keep d/dmin (the first 4 bytes of every super-block) in a sane
        // magnitude range instead of arbitrary half bit patterns (which can
        // land on inf/nan and blow up the reference dot product).
        for (int row = 0; row < mmq_n; ++row) {
            for (int sb = 0; sb < mmq_super_blocks; ++sb) {
                uint8_t* blk = mmq_weight.data() +
                    (static_cast<size_t>(row) * mmq_super_blocks + sb) * q4_bytes;
                const __half d = __float2half(0.01f + 0.02f * (byte_dist(rng) % 100));
                const __half dmin = __float2half(0.01f + 0.01f * (byte_dist(rng) % 100));
                std::memcpy(blk, &d, sizeof(d));
                std::memcpy(blk + sizeof(d), &dmin, sizeof(dmin));
            }
        }

        uint8_t* d_mmq_weight = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&d_mmq_weight), mmq_weight.size()),
              "cudaMalloc mmq weight");
        check(cudaMemcpy(d_mmq_weight, mmq_weight.data(), mmq_weight.size(),
                         cudaMemcpyHostToDevice), "copy mmq weight");

        __nv_bfloat16* d_mmq_dequant = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&d_mmq_dequant),
                         static_cast<size_t>(mmq_n) * mmq_k * sizeof(__nv_bfloat16)),
              "cudaMalloc mmq dequant");
        lfm::launch_gguf_dequant(d_mmq_weight, lfm::GgmlType::Q4_K, d_mmq_dequant,
                                 mmq_n, mmq_k, nullptr);
        std::vector<__nv_bfloat16> mmq_weight_dequant(static_cast<size_t>(mmq_n) * mmq_k);
        check(cudaMemcpy(mmq_weight_dequant.data(), d_mmq_dequant,
                         mmq_weight_dequant.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost), "copy mmq dequant");

        std::uniform_real_distribution<float> act_dist(-3.0f, 3.0f);
        std::vector<__nv_bfloat16> mmq_activation(mmq_k);
        std::vector<float> mmq_activation_f(mmq_k);
        for (int i = 0; i < mmq_k; ++i) {
            const float value = act_dist(rng);
            mmq_activation_f[static_cast<size_t>(i)] = value;
            mmq_activation[static_cast<size_t>(i)] = __float2bfloat16(value);
        }
        __nv_bfloat16* d_mmq_activation = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&d_mmq_activation),
                         mmq_activation.size() * sizeof(__nv_bfloat16)),
              "cudaMalloc mmq activation");
        check(cudaMemcpy(d_mmq_activation, mmq_activation.data(),
                         mmq_activation.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice), "copy mmq activation");

        const int mmq_blocks_per_row = mmq_k / lfm::kMmqQ8_1BlockSize;
        int8_t* d_q8 = nullptr;
        float* d_q8_scale = nullptr;
        float* d_q8_sum = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&d_q8), mmq_activation.size()),
              "cudaMalloc q8");
        check(cudaMalloc(reinterpret_cast<void**>(&d_q8_scale),
                         static_cast<size_t>(mmq_blocks_per_row) * sizeof(float)),
              "cudaMalloc q8 scale");
        check(cudaMalloc(reinterpret_cast<void**>(&d_q8_sum),
                         static_cast<size_t>(mmq_blocks_per_row) * sizeof(float)),
              "cudaMalloc q8 sum");
        lfm::launch_quantize_q8_1(d_mmq_activation, d_q8, d_q8_scale, d_q8_sum,
                                  1, mmq_k, nullptr);

        __nv_bfloat16* d_mmq_y = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&d_mmq_y),
                         static_cast<size_t>(mmq_n) * sizeof(__nv_bfloat16)),
              "cudaMalloc mmq y");
        lfm::launch_q4k_mmq(d_q8, d_q8_scale, d_q8_sum, d_mmq_weight, d_mmq_y,
                            1, mmq_n, mmq_k, mmq_row_bytes, mmq_n, 0.0f, nullptr);
        std::vector<__nv_bfloat16> mmq_y(mmq_n);
        check(cudaMemcpy(mmq_y.data(), d_mmq_y, mmq_y.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost), "copy mmq y");

        for (int row = 0; row < mmq_n; ++row) {
            double reference = 0.0;
            for (int i = 0; i < mmq_k; ++i) {
                reference += static_cast<double>(
                                 host_bf16(mmq_weight_dequant[static_cast<size_t>(row) * mmq_k + i])) *
                             static_cast<double>(mmq_activation_f[static_cast<size_t>(i)]);
            }
            const float mmq_value = host_bf16(mmq_y[static_cast<size_t>(row)]);
            const double relative_error =
                std::abs(mmq_value - reference) / std::max(1.0, std::abs(reference));
            // Q8_1 quantizes each 32-element activation block to 127 levels;
            // a few percent relative error against the unquantized-activation
            // reference is expected quantization noise, not a bug.
            if (relative_error > 0.05) return 11;
        }

        cudaFree(d_mmq_y); cudaFree(d_q8_sum); cudaFree(d_q8_scale); cudaFree(d_q8);
        cudaFree(d_mmq_activation); cudaFree(d_mmq_dequant); cudaFree(d_mmq_weight);
    }

    if (const char* real_path = std::getenv("LFM_GGUF_TEST_FILE");
        real_path != nullptr && *real_path != '\0') {
        const auto bootstrap = lfm::detail::load_model_bootstrap(std::filesystem::path(real_path));
        lfm::LfmModel model(real_path, 1024);
        model.session().prefill({bootstrap.config.bos_token_id});
        const std::vector<float> first = model.diagnostics().copy_logits();
        for (float value : first) if (!std::isfinite(value)) return 6;
        model.session().reset();
        model.session().prefill({bootstrap.config.bos_token_id});
        const std::vector<float> second = model.diagnostics().copy_logits();
        if (first.size() != second.size()) return 7;
        for (size_t i = 0; i < std::min<size_t>(first.size(), 64); ++i) {
            if (first[i] != second[i]) return 8;
        }
    }

    cudaFree(d_embed); cudaFree(d_y); cudaFree(d_x); cudaFree(d_q6); cudaFree(d_q4);
    std::cout << "cuda_gguf_kernels_test: ok\n";
    return 0;
}
