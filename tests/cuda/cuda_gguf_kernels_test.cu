#include "celeg/backend/cuda/kernels/gguf.cuh"
#include "celeg/backend/cuda/kernels/mmq.hpp"
#include "celeg/backend/cuda/gemm_dispatcher.hpp"
#include "celeg/backend/cuda/utils.cuh"
#include "celeg/detail/checkpoint/bootstrap.hpp"
#include "celeg/backend/cuda/model.hpp"
#include "celeg/backend/cuda/weights_loader.hpp"
#include "celeg/checkpoint/formats/gguf.hpp"

#include <cuda_bf16.h>
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
#include <stdexcept>
#include <vector>

namespace {

#include "data/gguf_iq_reference.inc"

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

}

int main() {
    if (std::getenv("CELEG_MMQ_TENSOR_CORES") == nullptr) {
#if defined(_WIN32)
        _putenv_s("CELEG_MMQ_TENSOR_CORES", "1");
#else
        setenv("CELEG_MMQ_TENSOR_CORES", "1", 1);
#endif
    }
    constexpr int k = 256;
    constexpr int q4_bytes = 144;
    constexpr int q6_bytes = 210;
    std::vector<uint8_t> q4(q4_bytes, 0);
    q4[0] = 0x00; q4[1] = 0x3c;
    q4[2] = 0x00; q4[3] = 0x00;
    for (int i = 4; i < 16; ++i) q4[i] = 1;
    for (int i = 16; i < q4_bytes; ++i) q4[i] = 0x11;
    std::vector<uint8_t> q6(q6_bytes, 0);

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

    celeg::launch_gguf_linear_segment(d_x, d_q4, celeg::GgmlType::Q4_K, d_y,
                                    1, 1, k, q4_bytes, 1, 0.0f, nullptr);
    __nv_bfloat16 y = 0;
    check(cudaMemcpy(&y, d_y, sizeof(y), cudaMemcpyDeviceToHost), "copy y q4");
    if (!close(host_bf16(y), 512.0f)) return 2;

    celeg::launch_gguf_linear_segment(d_x, d_q6, celeg::GgmlType::Q6_K, d_y,
                                    1, 1, k, q6_bytes, 1, 3.0f, nullptr);
    check(cudaMemcpy(&y, d_y, sizeof(y), cudaMemcpyDeviceToHost), "copy y q6");
    if (!close(host_bf16(y), 1536.0f)) return 3;

    std::vector<__nv_bfloat16> hy(2, __float2bfloat16(0.0f));
    celeg::launch_gguf_linear_segment(d_x, d_q4, celeg::GgmlType::Q4_K, d_y,
                                    2, 1, k, q4_bytes, 1, 0.0f, nullptr);
    std::vector<__nv_bfloat16> gemm_y(2);
    check(cudaMemcpy(gemm_y.data(), d_y, 2 * sizeof(y), cudaMemcpyDeviceToHost), "copy y gemm");
    if (!close(host_bf16(gemm_y[0]), 512.0f) || !close(host_bf16(gemm_y[1]), 512.0f)) return 4;

    {
        constexpr int blocks_per_row = k / celeg::kMmqQ8_1BlockSize;
        int8_t* d_prefill_q8 = nullptr;
        float* d_prefill_scales = nullptr;
        float* d_prefill_sums = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&d_prefill_q8), 2 * k), "cudaMalloc prefill q8");
        check(cudaMalloc(reinterpret_cast<void**>(&d_prefill_scales),
                         2 * blocks_per_row * sizeof(float)), "cudaMalloc prefill scales");
        check(cudaMalloc(reinterpret_cast<void**>(&d_prefill_sums),
                         2 * blocks_per_row * sizeof(float)), "cudaMalloc prefill sums");
        celeg::launch_quantize_q8_1(d_x, d_prefill_q8, d_prefill_scales, d_prefill_sums,
                                  2, k, nullptr);
        celeg::launch_q4k_mmq(d_prefill_q8, d_prefill_scales, d_prefill_sums, d_q4, d_y,
                              2, 1, k, q4_bytes, 1, 0.0f, nullptr);
        check(cudaMemcpy(gemm_y.data(), d_y, 2 * sizeof(y), cudaMemcpyDeviceToHost),
              "copy tiled q4 prefill");
        if (!close(host_bf16(gemm_y[0]), 512.0f) || !close(host_bf16(gemm_y[1]), 512.0f)) return 12;
        gemm_y.assign(2, __float2bfloat16(4.0f));
        check(cudaMemcpy(d_y, gemm_y.data(), 2 * sizeof(y), cudaMemcpyHostToDevice),
              "copy tiled q6 beta");
        celeg::launch_q6k_mmq(d_prefill_q8, d_prefill_scales, d_prefill_sums, d_q6, d_y,
                              2, 1, k, q6_bytes, 1, 3.0f, nullptr);
        check(cudaMemcpy(gemm_y.data(), d_y, 2 * sizeof(y), cudaMemcpyDeviceToHost),
              "copy tiled q6 prefill");
        if (!close(host_bf16(gemm_y[0]), 12.0f) || !close(host_bf16(gemm_y[1]), 12.0f)) return 13;
        cudaFree(d_prefill_sums); cudaFree(d_prefill_scales); cudaFree(d_prefill_q8);
    }

    {
        constexpr int tensor_rows = 16;
        constexpr int tensor_blocks_per_row = k / celeg::kMmqQ8_1BlockSize;
        std::vector<uint8_t> q4_matrix(static_cast<size_t>(tensor_rows) * q4_bytes);
        std::vector<uint8_t> q6_matrix(static_cast<size_t>(tensor_rows) * q6_bytes);
        for (int row = 0; row < tensor_rows; ++row) {
            std::memcpy(q4_matrix.data() + static_cast<size_t>(row) * q4_bytes, q4.data(), q4_bytes);
            std::memcpy(q6_matrix.data() + static_cast<size_t>(row) * q6_bytes, q6.data(), q6_bytes);
        }
        std::vector<__nv_bfloat16> tensor_x(static_cast<size_t>(tensor_rows) * k, __float2bfloat16(2.0f));
        uint8_t* d_tensor_q4 = nullptr; uint8_t* d_tensor_q6 = nullptr;
        __nv_bfloat16* d_tensor_x = nullptr; __nv_bfloat16* d_tensor_y = nullptr;
        int8_t* d_tensor_q8 = nullptr; float* d_tensor_scales = nullptr; float* d_tensor_sums = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&d_tensor_q4), q4_matrix.size()), "cudaMalloc tensor q4");
        check(cudaMalloc(reinterpret_cast<void**>(&d_tensor_q6), q6_matrix.size()), "cudaMalloc tensor q6");
        check(cudaMalloc(reinterpret_cast<void**>(&d_tensor_x), tensor_x.size() * sizeof(*d_tensor_x)), "cudaMalloc tensor x");
        check(cudaMalloc(reinterpret_cast<void**>(&d_tensor_y), tensor_rows * tensor_rows * sizeof(*d_tensor_y)), "cudaMalloc tensor y");
        check(cudaMalloc(reinterpret_cast<void**>(&d_tensor_q8), tensor_x.size()), "cudaMalloc tensor q8");
        check(cudaMalloc(reinterpret_cast<void**>(&d_tensor_scales),
                         tensor_rows * tensor_blocks_per_row * sizeof(float)), "cudaMalloc tensor scales");
        check(cudaMalloc(reinterpret_cast<void**>(&d_tensor_sums),
                         tensor_rows * tensor_blocks_per_row * sizeof(float)), "cudaMalloc tensor sums");
        check(cudaMemcpy(d_tensor_q4, q4_matrix.data(), q4_matrix.size(), cudaMemcpyHostToDevice), "copy tensor q4");
        check(cudaMemcpy(d_tensor_q6, q6_matrix.data(), q6_matrix.size(), cudaMemcpyHostToDevice), "copy tensor q6");
        check(cudaMemcpy(d_tensor_x, tensor_x.data(), tensor_x.size() * sizeof(*d_tensor_x), cudaMemcpyHostToDevice), "copy tensor x");
        celeg::launch_quantize_q8_1(d_tensor_x, d_tensor_q8, d_tensor_scales, d_tensor_sums,
                                  tensor_rows, k, nullptr);
        celeg::launch_q4k_mmq(d_tensor_q8, d_tensor_scales, d_tensor_sums, d_tensor_q4, d_tensor_y,
                              tensor_rows, tensor_rows, k, q4_bytes, tensor_rows, 0.0f, nullptr);
        std::vector<__nv_bfloat16> tensor_y(static_cast<size_t>(tensor_rows) * tensor_rows);
        check(cudaMemcpy(tensor_y.data(), d_tensor_y, tensor_y.size() * sizeof(*d_tensor_y), cudaMemcpyDeviceToHost),
              "copy tensor q4 result");
        for (const __nv_bfloat16 value : tensor_y) if (!close(host_bf16(value), 512.0f)) return 14;
        __nv_bfloat16* d_tensor_y_dp4a = nullptr;
        __nv_bfloat16* d_tensor_y_mma = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&d_tensor_y_dp4a),
                         tensor_y.size() * sizeof(*d_tensor_y_dp4a)),
              "cudaMalloc tensor dp4a result");
        check(cudaMalloc(reinterpret_cast<void**>(&d_tensor_y_mma),
                         tensor_y.size() * sizeof(*d_tensor_y_mma)),
              "cudaMalloc tensor mma result");
        celeg::launch_q4k_mmq_with_policy(
            d_tensor_q8, d_tensor_scales, d_tensor_sums, d_tensor_q4,
            d_tensor_y_dp4a, tensor_rows, tensor_rows, k, q4_bytes,
            tensor_rows, 0.0f, false, nullptr);
        celeg::launch_q4k_mmq_with_policy(
            d_tensor_q8, d_tensor_scales, d_tensor_sums, d_tensor_q4,
            d_tensor_y_mma, tensor_rows, tensor_rows, k, q4_bytes,
            tensor_rows, 0.0f, true, nullptr);
        std::vector<__nv_bfloat16> tensor_y_dp4a(tensor_y.size());
        std::vector<__nv_bfloat16> tensor_y_mma(tensor_y.size());
        check(cudaMemcpy(tensor_y_dp4a.data(), d_tensor_y_dp4a,
                         tensor_y_dp4a.size() * sizeof(*d_tensor_y_dp4a),
                         cudaMemcpyDeviceToHost), "copy tensor dp4a result");
        check(cudaMemcpy(tensor_y_mma.data(), d_tensor_y_mma,
                         tensor_y_mma.size() * sizeof(*d_tensor_y_mma),
                         cudaMemcpyDeviceToHost), "copy tensor mma result");
        for (size_t i = 0; i < tensor_y.size(); ++i) {
            if (!close(host_bf16(tensor_y_dp4a[i]), host_bf16(tensor_y_mma[i]))) return 16;
        }
        cudaFree(d_tensor_y_mma);
        cudaFree(d_tensor_y_dp4a);
        std::fill(tensor_y.begin(), tensor_y.end(), __float2bfloat16(4.0f));
        check(cudaMemcpy(d_tensor_y, tensor_y.data(), tensor_y.size() * sizeof(*d_tensor_y), cudaMemcpyHostToDevice),
              "copy tensor q6 beta");
        celeg::launch_q6k_mmq(d_tensor_q8, d_tensor_scales, d_tensor_sums, d_tensor_q6, d_tensor_y,
                              tensor_rows, tensor_rows, k, q6_bytes, tensor_rows, 3.0f, nullptr);
        check(cudaMemcpy(tensor_y.data(), d_tensor_y, tensor_y.size() * sizeof(*d_tensor_y), cudaMemcpyDeviceToHost),
              "copy tensor q6 result");
        for (const __nv_bfloat16 value : tensor_y) if (!close(host_bf16(value), 12.0f)) return 15;
        cudaFree(d_tensor_sums); cudaFree(d_tensor_scales); cudaFree(d_tensor_q8); cudaFree(d_tensor_y);
        cudaFree(d_tensor_x); cudaFree(d_tensor_q6); cudaFree(d_tensor_q4);
    }

    celeg::CudaModelOptions dispatcher_options;
    celeg::GemmDispatcher dispatcher(nullptr, dispatcher_options);
    celeg::LinearWeight segmented;
    segmented.rows = 2;
    segmented.cols = k;
    segmented.storage = celeg::GgufLinearStorage{{
        {d_q4, celeg::GgmlType::Q4_K, 0, 1, k, q4_bytes},
        {d_q6, celeg::GgmlType::Q6_K, 1, 1, k, q6_bytes},
    }};
    const celeg::CudaExecutionPlan dispatcher_plan =
        celeg::CudaExecutionPlan::compile(dispatcher_options, 1024);
    const auto& binding = dispatcher.compile_linear_binding(segmented,
                                                              dispatcher_plan);
    if (binding.weight != &segmented || binding.rows != 2 ||
        binding.cols != k || binding.plan_fingerprint != dispatcher_plan.fingerprint() ||
        &dispatcher.compile_linear_binding(segmented, dispatcher_plan) != &binding) {
        return 14;
    }
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

    celeg::CudaDeviceCapabilities wrong_device =
        celeg::discover_cuda_device_capabilities();
    wrong_device.device_ordinal += 1;
    const auto wrong_device_plan = celeg::CudaExecutionPlan::compile(
        dispatcher_options, 1024, wrong_device);
    bool rejected_wrong_device = false;
    try {
        dispatcher.linear(d_x, segmented, d_y, 1, 2, k, 0.0f,
                          wrong_device_plan);
    } catch (const std::invalid_argument&) {
        rejected_wrong_device = true;
    }
    if (!rejected_wrong_device) return 11;

    bool rejected_bad_fanout = false;
    try {
        const celeg::GemmDispatcher::NativeFanoutScope invalid_scope(
            &dispatcher, d_x, 1, k - 1);
    } catch (const std::invalid_argument&) {
        rejected_bad_fanout = true;
    }
    if (!rejected_bad_fanout) return 12;
    bool rejected_nested_fanout = false;
    {
        celeg::GemmDispatcher::NativeFanoutScope outer(&dispatcher, d_x, 1, k);
        try {
            const celeg::GemmDispatcher::NativeFanoutScope inner(
                &dispatcher, d_x, 1, k);
        } catch (const std::logic_error&) {
            rejected_nested_fanout = true;
        }
    }
    if (!rejected_nested_fanout) return 13;

    celeg::GgufLinearSegment segment{d_q4, celeg::GgmlType::Q4_K, 0, 1, k, q4_bytes};
    __nv_bfloat16* d_embed = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&d_embed), k * sizeof(*d_embed)), "cudaMalloc embed");
    celeg::launch_gguf_embedding(0, segment, d_embed, nullptr);
    check(cudaMemcpy(hy.data(), d_embed, 2 * sizeof(*d_embed), cudaMemcpyDeviceToHost), "copy embed");
    if (!close(host_bf16(hy[0]), 1.0f)) return 5;

    {
        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> byte_dist(0, 255);
        constexpr int mmq_n = 16;
        constexpr int mmq_m = 16;
        constexpr int mmq_k = 512;
        constexpr int mmq_super_blocks = mmq_k / 256;
        const size_t mmq_row_bytes = static_cast<size_t>(mmq_super_blocks) * q4_bytes;
        std::vector<uint8_t> mmq_weight(static_cast<size_t>(mmq_n) * mmq_row_bytes);
        for (auto& byte : mmq_weight) byte = static_cast<uint8_t>(byte_dist(rng));
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
        celeg::launch_gguf_dequant(d_mmq_weight, celeg::GgmlType::Q4_K, d_mmq_dequant,
                                 mmq_n, mmq_k, nullptr);
        std::vector<__nv_bfloat16> mmq_weight_dequant(static_cast<size_t>(mmq_n) * mmq_k);
        check(cudaMemcpy(mmq_weight_dequant.data(), d_mmq_dequant,
                         mmq_weight_dequant.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost), "copy mmq dequant");

        std::uniform_real_distribution<float> act_dist(-3.0f, 3.0f);
        std::vector<__nv_bfloat16> mmq_activation(static_cast<size_t>(mmq_m) * mmq_k);
        std::vector<float> mmq_activation_f(static_cast<size_t>(mmq_m) * mmq_k);
        for (size_t i = 0; i < mmq_activation.size(); ++i) {
            const float value = act_dist(rng);
            mmq_activation_f[i] = value;
            mmq_activation[i] = __float2bfloat16(value);
        }
        __nv_bfloat16* d_mmq_activation = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&d_mmq_activation),
                         mmq_activation.size() * sizeof(__nv_bfloat16)),
              "cudaMalloc mmq activation");
        check(cudaMemcpy(d_mmq_activation, mmq_activation.data(),
                         mmq_activation.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice), "copy mmq activation");

        const int mmq_blocks_per_row = mmq_k / celeg::kMmqQ8_1BlockSize;
        int8_t* d_q8 = nullptr;
        float* d_q8_scale = nullptr;
        float* d_q8_sum = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&d_q8), mmq_activation.size()),
              "cudaMalloc q8");
        check(cudaMalloc(reinterpret_cast<void**>(&d_q8_scale),
                         static_cast<size_t>(mmq_m) * mmq_blocks_per_row * sizeof(float)),
              "cudaMalloc q8 scale");
        check(cudaMalloc(reinterpret_cast<void**>(&d_q8_sum),
                         static_cast<size_t>(mmq_m) * mmq_blocks_per_row * sizeof(float)),
              "cudaMalloc q8 sum");
        celeg::launch_quantize_q8_1(d_mmq_activation, d_q8, d_q8_scale, d_q8_sum,
                                  mmq_m, mmq_k, nullptr);

        __nv_bfloat16* d_mmq_y = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&d_mmq_y),
                         static_cast<size_t>(mmq_m) * mmq_n * sizeof(__nv_bfloat16)),
              "cudaMalloc mmq y");
        celeg::launch_q4k_mmq(d_q8, d_q8_scale, d_q8_sum, d_mmq_weight, d_mmq_y,
                            mmq_m, mmq_n, mmq_k, mmq_row_bytes, mmq_n, 0.0f, nullptr);
        std::vector<__nv_bfloat16> mmq_y(static_cast<size_t>(mmq_m) * mmq_n);
        check(cudaMemcpy(mmq_y.data(), d_mmq_y, mmq_y.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost), "copy mmq y");

        __nv_bfloat16* d_decode_y = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&d_decode_y),
                         static_cast<size_t>(mmq_m) * mmq_n * sizeof(__nv_bfloat16)),
              "cudaMalloc decode y");
        for (int activation_row = 0; activation_row < mmq_m; ++activation_row) {
            celeg::launch_q4k_mmq(d_q8 + static_cast<size_t>(activation_row) * mmq_k,
                                  d_q8_scale + static_cast<size_t>(activation_row) * mmq_blocks_per_row,
                                  d_q8_sum + static_cast<size_t>(activation_row) * mmq_blocks_per_row,
                                  d_mmq_weight,
                                  d_decode_y + static_cast<size_t>(activation_row) * mmq_n,
                                  1, mmq_n, mmq_k, mmq_row_bytes, mmq_n, 0.0f, nullptr);
        }
        std::vector<__nv_bfloat16> decode_y(static_cast<size_t>(mmq_m) * mmq_n);
        check(cudaMemcpy(decode_y.data(), d_decode_y, decode_y.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost), "copy decode y");

        for (int activation_row = 0; activation_row < mmq_m; ++activation_row) {
        std::vector<double> block_step(mmq_blocks_per_row);
        for (int block = 0; block < mmq_blocks_per_row; ++block) {
            float absmax = 0.0f;
            for (int i = 0; i < celeg::kMmqQ8_1BlockSize; ++i) {
                absmax = std::max(absmax, std::abs(mmq_activation_f[
                    static_cast<size_t>(activation_row) * mmq_k +
                    block * celeg::kMmqQ8_1BlockSize + i]));
            }
            block_step[block] = absmax > 0.0f ? absmax / 127.0 : 1.0;
        }

        for (int row = 0; row < mmq_n; ++row) {
            const float mmq_value = host_bf16(mmq_y[
                static_cast<size_t>(activation_row) * mmq_n + row]);

            const float decode_value = host_bf16(decode_y[
                static_cast<size_t>(activation_row) * mmq_n + row]);
            if (std::abs(mmq_value - decode_value) >
                std::max(1.0, 0.02 * std::abs(static_cast<double>(decode_value)))) {
                std::cerr << "prefill/decode mismatch row=" << row
                          << " activation_row=" << activation_row
                          << " prefill=" << mmq_value
                          << " decode=" << decode_value << "\n";
                return 11;
            }

            double reference = 0.0;
            double noise_variance = 0.0;
            for (int i = 0; i < mmq_k; ++i) {
                const double weight = static_cast<double>(
                    host_bf16(mmq_weight_dequant[static_cast<size_t>(row) * mmq_k + i]));
                reference += weight * static_cast<double>(mmq_activation_f[
                    static_cast<size_t>(activation_row) * mmq_k + i]);
                const double term = weight * block_step[i / celeg::kMmqQ8_1BlockSize] * 0.5;
                noise_variance += term * term;
            }
            const double tolerance = std::max(1.0, 3.0 * std::sqrt(noise_variance) +
                                                   0.01 * std::abs(reference));
            if (std::abs(mmq_value - reference) > tolerance) {
                std::cerr << "random mmq mismatch row=" << row
                          << " activation_row=" << activation_row
                          << " got=" << mmq_value
                          << " reference=" << reference
                          << " tolerance=" << tolerance << "\n";
                return 11;
            }
        }
        }

        cudaFree(d_decode_y);
        cudaFree(d_mmq_y); cudaFree(d_q8_sum); cudaFree(d_q8_scale); cudaFree(d_q8);
        cudaFree(d_mmq_activation); cudaFree(d_mmq_dequant); cudaFree(d_mmq_weight);
    }

    {
        std::mt19937 rng(6789);
        std::uniform_int_distribution<int> byte_dist(0, 255);
        constexpr int mmq_n = 16;
        constexpr int mmq_m = 16;
        constexpr int mmq_k = 512;
        constexpr int mmq_super_blocks = mmq_k / 256;
        constexpr int mmq_blocks_per_row = mmq_k / celeg::kMmqQ8_1BlockSize;
        const size_t mmq_row_bytes = static_cast<size_t>(mmq_super_blocks) * q6_bytes;

        std::vector<uint8_t> weight(static_cast<size_t>(mmq_n) * mmq_row_bytes);
        for (auto& byte : weight) byte = static_cast<uint8_t>(byte_dist(rng));
        for (int row = 0; row < mmq_n; ++row) {
            for (int sb = 0; sb < mmq_super_blocks; ++sb) {
                uint8_t* blk = weight.data() +
                    (static_cast<size_t>(row) * mmq_super_blocks + sb) * q6_bytes;
                const __half d = __float2half(0.005f + 0.01f * (byte_dist(rng) % 100));
                std::memcpy(blk + q6_bytes - sizeof(d), &d, sizeof(d));
            }
        }

        uint8_t* d_weight = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&d_weight), weight.size()), "cudaMalloc q6 rand weight");
        check(cudaMemcpy(d_weight, weight.data(), weight.size(), cudaMemcpyHostToDevice),
              "copy q6 rand weight");

        __nv_bfloat16* d_dequant = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&d_dequant),
                         static_cast<size_t>(mmq_n) * mmq_k * sizeof(__nv_bfloat16)),
              "cudaMalloc q6 rand dequant");
        celeg::launch_gguf_dequant(d_weight, celeg::GgmlType::Q6_K, d_dequant, mmq_n, mmq_k, nullptr);
        std::vector<__nv_bfloat16> dequant(static_cast<size_t>(mmq_n) * mmq_k);
        check(cudaMemcpy(dequant.data(), d_dequant, dequant.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost), "copy q6 rand dequant");

        std::uniform_real_distribution<float> act_dist(-3.0f, 3.0f);
        std::vector<__nv_bfloat16> activation(static_cast<size_t>(mmq_m) * mmq_k);
        std::vector<float> activation_f(static_cast<size_t>(mmq_m) * mmq_k);
        for (size_t i = 0; i < activation.size(); ++i) {
            const float value = act_dist(rng);
            activation_f[i] = value;
            activation[i] = __float2bfloat16(value);
        }
        __nv_bfloat16* d_activation = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&d_activation),
                         activation.size() * sizeof(__nv_bfloat16)), "cudaMalloc q6 rand activation");
        check(cudaMemcpy(d_activation, activation.data(), activation.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice), "copy q6 rand activation");

        int8_t* d_q8 = nullptr; float* d_q8_scale = nullptr; float* d_q8_sum = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&d_q8), activation.size()), "cudaMalloc q6 rand q8");
        check(cudaMalloc(reinterpret_cast<void**>(&d_q8_scale),
                         static_cast<size_t>(mmq_m) * mmq_blocks_per_row * sizeof(float)),
              "cudaMalloc q6 rand scale");
        check(cudaMalloc(reinterpret_cast<void**>(&d_q8_sum),
                         static_cast<size_t>(mmq_m) * mmq_blocks_per_row * sizeof(float)),
              "cudaMalloc q6 rand sum");
        celeg::launch_quantize_q8_1(d_activation, d_q8, d_q8_scale, d_q8_sum, mmq_m, mmq_k, nullptr);

        __nv_bfloat16* d_prefill = nullptr; __nv_bfloat16* d_decode = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&d_prefill),
                         static_cast<size_t>(mmq_m) * mmq_n * sizeof(__nv_bfloat16)),
              "cudaMalloc q6 rand prefill");
        check(cudaMalloc(reinterpret_cast<void**>(&d_decode),
                         static_cast<size_t>(mmq_m) * mmq_n * sizeof(__nv_bfloat16)),
              "cudaMalloc q6 rand decode");
        celeg::launch_q6k_mmq(d_q8, d_q8_scale, d_q8_sum, d_weight, d_prefill,
                              mmq_m, mmq_n, mmq_k, mmq_row_bytes, mmq_n, 0.0f, nullptr);
        for (int activation_row = 0; activation_row < mmq_m; ++activation_row) {
            celeg::launch_q6k_mmq(d_q8 + static_cast<size_t>(activation_row) * mmq_k,
                                  d_q8_scale + static_cast<size_t>(activation_row) * mmq_blocks_per_row,
                                  d_q8_sum + static_cast<size_t>(activation_row) * mmq_blocks_per_row,
                                  d_weight, d_decode + static_cast<size_t>(activation_row) * mmq_n,
                                  1, mmq_n, mmq_k, mmq_row_bytes, mmq_n, 0.0f, nullptr);
        }
        std::vector<__nv_bfloat16> prefill_y(static_cast<size_t>(mmq_m) * mmq_n);
        std::vector<__nv_bfloat16> decode_y(static_cast<size_t>(mmq_m) * mmq_n);
        check(cudaMemcpy(prefill_y.data(), d_prefill, prefill_y.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost), "copy q6 rand prefill");
        check(cudaMemcpy(decode_y.data(), d_decode, decode_y.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost), "copy q6 rand decode");

        for (int activation_row = 0; activation_row < mmq_m; ++activation_row) {
            std::vector<double> block_step(mmq_blocks_per_row);
            for (int block = 0; block < mmq_blocks_per_row; ++block) {
                float absmax = 0.0f;
                for (int i = 0; i < celeg::kMmqQ8_1BlockSize; ++i) {
                    absmax = std::max(absmax, std::abs(activation_f[
                        static_cast<size_t>(activation_row) * mmq_k +
                        block * celeg::kMmqQ8_1BlockSize + i]));
                }
                block_step[block] = absmax > 0.0f ? absmax / 127.0 : 1.0;
            }
            for (int row = 0; row < mmq_n; ++row) {
                const float prefill_value =
                    host_bf16(prefill_y[static_cast<size_t>(activation_row) * mmq_n + row]);
                const float decode_value =
                    host_bf16(decode_y[static_cast<size_t>(activation_row) * mmq_n + row]);
                if (std::abs(prefill_value - decode_value) >
                    std::max(1.0, 0.02 * std::abs(static_cast<double>(decode_value)))) {
                    std::cerr << "q6 prefill/decode mismatch row=" << row
                              << " activation_row=" << activation_row
                              << " prefill=" << prefill_value
                              << " decode=" << decode_value << "\n";
                    return 16;
                }
                double reference = 0.0;
                double noise_variance = 0.0;
                for (int i = 0; i < mmq_k; ++i) {
                    const double w = static_cast<double>(
                        host_bf16(dequant[static_cast<size_t>(row) * mmq_k + i]));
                    reference += w * static_cast<double>(activation_f[
                        static_cast<size_t>(activation_row) * mmq_k + i]);
                    const double term = w * block_step[i / celeg::kMmqQ8_1BlockSize] * 0.5;
                    noise_variance += term * term;
                }
                const double tolerance = std::max(1.0, 3.0 * std::sqrt(noise_variance) +
                                                       0.01 * std::abs(reference));
                if (std::abs(prefill_value - reference) > tolerance) {
                    std::cerr << "q6 random mmq mismatch row=" << row
                              << " activation_row=" << activation_row
                              << " got=" << prefill_value
                              << " reference=" << reference
                              << " tolerance=" << tolerance << "\n";
                    return 17;
                }
            }
        }

        cudaFree(d_decode); cudaFree(d_prefill); cudaFree(d_q8_sum); cudaFree(d_q8_scale);
        cudaFree(d_q8); cudaFree(d_activation); cudaFree(d_dequant); cudaFree(d_weight);
    }

    if (const char* real_path = std::getenv("CELEG_GGUF_TEST_FILE");
        real_path != nullptr && *real_path != '\0') {
        const auto bootstrap = celeg::detail::load_model_bootstrap(std::filesystem::path(real_path));
        celeg::CudaModel model(real_path, 1024);
        model.session().prefill({bootstrap.model.topology.dims.token_policy.bos_token_id});
        const std::vector<float> first = model.diagnostics().copy_logits();
        for (float value : first) if (!std::isfinite(value)) return 6;
        model.session().reset();
        model.session().prefill({bootstrap.model.topology.dims.token_policy.bos_token_id});
        const std::vector<float> second = model.diagnostics().copy_logits();
        if (first.size() != second.size()) return 7;
        for (size_t i = 0; i < std::min<size_t>(first.size(), 64); ++i) {
            if (first[i] != second[i]) return 8;
        }
    }

    // The CUDA loader decodes IQ blocks on the host before uploading BF16.
    // The shared decoders are checked element-exactly against ggml in
    // cpu_gguf_kernels_test; what is specific here is the loader's own
    // block/row striding and the BF16 narrowing, so the same ggml fixture is
    // pushed through dequantize_gguf_to_bf16 end to end.
    {
        struct IqCase {
            const char* name;
            celeg::GgmlType type;
            const unsigned char* blocks;
            size_t block_bytes;
            const float* reference;
        };
        const IqCase iq_cases[] = {
            {"IQ2_S", celeg::GgmlType::IQ2_S, kIQ2_SBlocks, sizeof(kIQ2_SBlocks),
             kIQ2_SReference},
            {"IQ3_XXS", celeg::GgmlType::IQ3_XXS, kIQ3_XXSBlocks, sizeof(kIQ3_XXSBlocks),
             kIQ3_XXSReference},
            {"IQ3_S", celeg::GgmlType::IQ3_S, kIQ3_SBlocks, sizeof(kIQ3_SBlocks),
             kIQ3_SReference},
            {"IQ4_NL", celeg::GgmlType::IQ4_NL, kIQ4_NLBlocks, sizeof(kIQ4_NLBlocks),
             kIQ4_NLReference},
            {"IQ4_XS", celeg::GgmlType::IQ4_XS, kIQ4_XSBlocks, sizeof(kIQ4_XSBlocks),
             kIQ4_XSReference},
        };
        for (const IqCase& iq_case : iq_cases) {
            if (!celeg::ggml_row_decoder(iq_case.type).has_value()) {
                std::cerr << iq_case.name << ": neutral decoder is unavailable\n";
                return 9;
            }
            celeg::HostTensorView view;
            view.dtype = celeg::TensorDType::Quantized;
            view.block_encoding = celeg::block_encoding_from_ggml_type(iq_case.type);
            view.shape = {4, 256};
            view.data = reinterpret_cast<const std::byte*>(iq_case.blocks);
            view.bytes = iq_case.block_bytes;
            std::vector<__nv_bfloat16> decoded;
            celeg::dequantize_gguf_to_bf16(view, decoded);
            if (decoded.size() != 4 * 256) return 10;
            for (size_t i = 0; i < decoded.size(); ++i) {
                const float got = __bfloat162float(decoded[i]);
                const float want = iq_case.reference[i];
                // BF16 keeps 8 mantissa bits, so ~0.4% relative is the
                // representation floor, not decoder slack.
                const float tolerance = 6e-3f * std::max(1.0f, std::abs(want));
                if (std::abs(got - want) >= tolerance) {
                    std::cerr << iq_case.name << " element " << i << ": got " << got
                              << " want " << want << "\n";
                    return 11;
                }
            }
        }
    }

    cudaFree(d_embed); cudaFree(d_y); cudaFree(d_x); cudaFree(d_q6); cudaFree(d_q4);
    std::cout << "cuda_gguf_kernels_test: ok\n";
    return 0;
}
