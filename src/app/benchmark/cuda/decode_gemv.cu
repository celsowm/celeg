// Isolated microbenchmark for the m=1 (decode) GEMV path.
//
// Decode issues one skinny GEMV per projection per layer, and those are
// bandwidth-bound in principle, so it is tempting to assume they dominate. This
// benchmark checks that assumption directly: it replays the exact per-token
// sequence of GEMV shapes a model decodes and reports achieved GB/s against the
// GPU's peak.
//
// It exists because the assumption was wrong once already. A weight-traffic
// argument (BF16 weights move 3x the bytes of native Q4_K blocks) suggested the
// GEMV path was running at ~17% of peak and that a native-quantized kernel would
// give a 3x decode win. This benchmark showed the BF16 GEMV was already at 77%
// of peak and only ~22% of the decode step, so the ceiling on that work was a
// few percent, not 3x. The real cost was elsewhere -- see
// scripts/profile_decode.py.
//
// Pair the two: this answers "how fast is this kernel", profile_decode.py
// answers "does that kernel matter".
//
// Usage:
//   lfm25-decode-gemv-benchmark [iterations] [--shapes n,k,count ...]
// Defaults to the LFM2.5-230M shape set.

#include "lfm/backend/cuda/utils.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

__inline__ __device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

// Mirrors bf16_gemv_kernel in src/backend/cuda/runtime/gemm_dispatcher.cu.
// Kept as a copy rather than linked so this stays a kernel-shape study that can
// be edited freely without touching the production path.
__global__ void bf16_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                                 const __nv_bfloat16* __restrict__ weight,
                                 __nv_bfloat16* __restrict__ y,
                                 int n, int k, float beta) {
    constexpr int warps_per_block = 8;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * warps_per_block + warp;
    if (row >= n) return;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(
        weight + static_cast<size_t>(row) * k);
    const int k2 = k >> 1;
    float sum = 0.0f;
    for (int i = lane; i < k2; i += 32) {
        const __nv_bfloat162 xv = x2[i];
        const __nv_bfloat162 wv = w2[i];
        sum += __bfloat162float(xv.x) * __bfloat162float(wv.x) +
               __bfloat162float(xv.y) * __bfloat162float(wv.y);
    }
    sum = warp_sum(sum);
    if (lane == 0) {
        float value = sum;
        if (beta != 0.0f) value += beta * __bfloat162float(y[row]);
        y[row] = __float2bfloat16(value);
    }
}

struct Shape {
    int n;
    int k;
    int count;
    std::string name;
};

// Per-token GEMV sequence for LFM2.5-230M: hidden 1024, intermediate 2560,
// 6 attention layers + 8 convolution layers, 14 MLP blocks, tied lm_head.
std::vector<Shape> default_shapes() {
    return {
        {2048, 1024, 6, "qkv"},
        {1024, 1024, 6, "attn_out"},
        {3072, 1024, 8, "conv_in"},
        {1024, 1024, 8, "conv_out"},
        {5120, 1024, 14, "w13"},
        {1024, 2560, 14, "w2"},
        {65536, 1024, 1, "lm_head"},
    };
}

} // namespace

int main(int argc, char** argv) {
    int iterations = 200;
    std::vector<Shape> shapes;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--shapes" && i + 1 < argc) {
            int n = 0, k = 0, c = 0;
            if (std::sscanf(argv[++i], "%d,%d,%d", &n, &k, &c) == 3) {
                shapes.push_back({n, k, c, "custom"});
            }
        } else {
            iterations = std::atoi(argv[i]);
        }
    }
    if (shapes.empty()) shapes = default_shapes();
    if (iterations <= 0) iterations = 200;

    cudaDeviceProp prop{};
    LFM_CUDA(cudaGetDeviceProperties(&prop, 0));
    // cudaDeviceProp::memoryClockRate was removed in CUDA 13; the attribute
    // query is the portable spelling. Clock is kHz, bus width is bits, x2 for
    // DDR.
    int mem_clock_khz = 0;
    int bus_width_bits = 0;
    LFM_CUDA(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0));
    LFM_CUDA(cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
    const double peak_gbs =
        2.0 * mem_clock_khz * 1e3 * (bus_width_bits / 8.0) / 1e9;
    std::printf("%s: %d SMs, %.0f GB/s theoretical peak\n\n",
                prop.name, prop.multiProcessorCount, peak_gbs);

    lfm::CudaStream stream;
    int max_n = 0, max_k = 0;
    double token_bytes = 0.0;
    for (const Shape& s : shapes) {
        max_n = s.n > max_n ? s.n : max_n;
        max_k = s.k > max_k ? s.k : max_k;
        token_bytes += static_cast<double>(s.n) * s.k * 2 * s.count;
    }

    lfm::DeviceBuffer<__nv_bfloat16> x(static_cast<size_t>(max_k));
    lfm::DeviceBuffer<__nv_bfloat16> y(static_cast<size_t>(max_n));
    x.zero_async(stream.get());
    std::vector<lfm::DeviceBuffer<__nv_bfloat16>> weights(shapes.size());
    for (size_t i = 0; i < shapes.size(); ++i) {
        weights[i].reset(static_cast<size_t>(shapes[i].n) * shapes[i].k);
        weights[i].zero_async(stream.get());
    }
    LFM_CUDA(cudaStreamSynchronize(stream.get()));

    auto launch = [&](size_t i) {
        constexpr int wpb = 8;
        const Shape& s = shapes[i];
        bf16_gemv_kernel<<<(s.n + wpb - 1) / wpb, wpb * 32, 0, stream.get()>>>(
            x.data(), weights[i].data(), y.data(), s.n, s.k, 0.0f);
    };

    lfm::CudaEvent begin;
    lfm::CudaEvent end;

    std::printf("%-10s %7s %7s %5s %11s %10s %8s\n",
                "shape", "n", "k", "count", "ms/token", "GB/s", "of peak");
    for (size_t i = 0; i < shapes.size(); ++i) {
        for (int w = 0; w < 20; ++w) launch(i);
        LFM_CUDA(cudaStreamSynchronize(stream.get()));
        begin.record(stream.get());
        for (int it = 0; it < iterations; ++it) launch(i);
        end.record(stream.get());
        end.synchronize();
        const double per_call = lfm::CudaEvent::elapsed_ms(begin, end) / iterations;
        const double bytes = static_cast<double>(shapes[i].n) * shapes[i].k * 2;
        const double gbs = bytes / (per_call * 1e-3) / 1e9;
        std::printf("%-10s %7d %7d %5d %11.4f %10.1f %7.0f%%\n",
                    shapes[i].name.c_str(), shapes[i].n, shapes[i].k,
                    shapes[i].count, per_call * shapes[i].count, gbs,
                    100.0 * gbs / peak_gbs);
    }

    // Replaying the whole token sequence streams the entire weight set, so L2
    // cannot hold anything between calls. Per-shape loops above re-read the same
    // matrix and can flatter small shapes that fit in cache; this is the number
    // that corresponds to real decode.
    auto sequence = [&]() {
        for (size_t i = 0; i < shapes.size(); ++i)
            for (int c = 0; c < shapes[i].count; ++c) launch(i);
    };
    for (int w = 0; w < 5; ++w) sequence();
    LFM_CUDA(cudaStreamSynchronize(stream.get()));
    const int seq_iters = 50;
    begin.record(stream.get());
    for (int it = 0; it < seq_iters; ++it) sequence();
    end.record(stream.get());
    end.synchronize();
    const double per_token = lfm::CudaEvent::elapsed_ms(begin, end) / seq_iters;
    const double gbs = token_bytes / (per_token * 1e-3) / 1e9;
    std::printf("%-10s %7s %7s %5s %11.4f %10.1f %7.0f%%   <-- full token sequence\n",
                "TOKEN", "", "", "", per_token, gbs, 100.0 * gbs / peak_gbs);
    std::printf("\nweight traffic %.1f MB/token; GEMV alone implies %.0f tok/s.\n",
                token_bytes / 1e6, 1000.0 / per_token);
    std::printf("Compare against end-to-end decode (scripts/profile_decode.py) to see\n"
                "what share of a decode step this actually is.\n");
    return 0;
}
