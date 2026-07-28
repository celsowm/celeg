#include "lfm/backend/cuda/kernels/gguf.cuh"
#include "lfm/backend/cuda/utils.cuh"
#include "lfm/detail/checkpoint/bootstrap.hpp"
#include "lfm/model/model.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iostream>
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

    lfm::GgufLinearSegment segment{d_q4, lfm::GgmlType::Q4_K, 0, 1, k, q4_bytes};
    __nv_bfloat16* d_embed = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&d_embed), k * sizeof(*d_embed)), "cudaMalloc embed");
    lfm::launch_gguf_embedding(0, segment, d_embed, nullptr);
    check(cudaMemcpy(hy.data(), d_embed, 2 * sizeof(*d_embed), cudaMemcpyDeviceToHost), "copy embed");
    if (!close(host_bf16(hy[0]), 1.0f)) return 5;

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
