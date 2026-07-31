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
//   celeg-decode-gemv-benchmark [iterations] [--shapes n,k,count ...]
// Defaults to the LFM2.5-230M shape set.

#include "celeg/backend/cuda/utils.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Pull in the shared kernel definitions (bf16_gemv_kernel, w8a16_gemv_kernel)
// from the same header the production backend uses -- no hand-mirrored copies.
#include "celeg/backend/cuda/kernels/gemv_kernels.cuh"

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
    CELEG_CUDA(cudaGetDeviceProperties(&prop, 0));
    // cudaDeviceProp::memoryClockRate was removed in CUDA 13; the attribute
    // query is the portable spelling. Clock is kHz, bus width is bits, x2 for
    // DDR.
    int mem_clock_khz = 0;
    int bus_width_bits = 0;
    CELEG_CUDA(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0));
    CELEG_CUDA(cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
    const double peak_gbs =
        2.0 * mem_clock_khz * 1e3 * (bus_width_bits / 8.0) / 1e9;
    std::printf("%s: %d SMs, %.0f GB/s theoretical peak\n\n",
                prop.name, prop.multiProcessorCount, peak_gbs);

    celeg::CudaStream stream;
    int max_n = 0, max_k = 0;
    double token_bytes = 0.0;
    for (const Shape& s : shapes) {
        max_n = s.n > max_n ? s.n : max_n;
        max_k = s.k > max_k ? s.k : max_k;
        token_bytes += static_cast<double>(s.n) * s.k * 2 * s.count;
    }

    celeg::DeviceBuffer<__nv_bfloat16> x(static_cast<size_t>(max_k));
    celeg::DeviceBuffer<__nv_bfloat16> y(static_cast<size_t>(max_n));
    x.zero_async(stream.get());
    std::vector<celeg::DeviceBuffer<__nv_bfloat16>> weights(shapes.size());
    for (size_t i = 0; i < shapes.size(); ++i) {
        weights[i].reset(static_cast<size_t>(shapes[i].n) * shapes[i].k);
        weights[i].zero_async(stream.get());
    }
    std::vector<celeg::DeviceBuffer<int8_t>> i8_weights(shapes.size());
    std::vector<celeg::DeviceBuffer<float>> i8_scales(shapes.size());
    for (size_t i = 0; i < shapes.size(); ++i) {
        i8_weights[i].reset(static_cast<size_t>(shapes[i].n) * shapes[i].k);
        i8_weights[i].zero_async(stream.get());
        i8_scales[i].reset(static_cast<size_t>(shapes[i].n));
        i8_scales[i].zero_async(stream.get());
    }
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));

    auto launch_bf16 = [&](size_t i) {
        constexpr int wpb = 8;
        const Shape& s = shapes[i];
        bf16_gemv_kernel<<<(s.n + wpb - 1) / wpb, wpb * 32, 0, stream.get()>>>(
            x.data(), weights[i].data(), y.data(), s.n, s.k, 0.0f);
    };
    auto launch_w8a16 = [&](size_t i) {
        constexpr int wpb = 8;
        const Shape& s = shapes[i];
        w8a16_gemv_kernel<<<(s.n + wpb - 1) / wpb, wpb * 32, 0, stream.get()>>>(
            x.data(), i8_weights[i].data(), i8_scales[i].data(), y.data(), 1, s.n, s.k, 0.0f);
    };

    celeg::CudaEvent begin;
    celeg::CudaEvent end;

    // INT8 weight traffic per token
    double i8_token_bytes = 0.0;
    for (const Shape& s : shapes)
        i8_token_bytes += static_cast<double>(s.n) * s.k * 1 * s.count;

    std::printf("\n=== BF16 GEMV ===\n");
    std::printf("%-10s %7s %7s %5s %11s %10s %8s\n",
                "shape", "n", "k", "count", "ms/token", "GB/s", "of peak");
    for (size_t i = 0; i < shapes.size(); ++i) {
        for (int w = 0; w < 20; ++w) launch_bf16(i);
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        begin.record(stream.get());
        for (int it = 0; it < iterations; ++it) launch_bf16(i);
        end.record(stream.get());
        end.synchronize();
        const double per_call = celeg::CudaEvent::elapsed_ms(begin, end) / iterations;
        const double bytes = static_cast<double>(shapes[i].n) * shapes[i].k * 2;
        const double gbs = bytes / (per_call * 1e-3) / 1e9;
        std::printf("%-10s %7d %7d %5d %11.4f %10.1f %7.0f%%\n",
                    shapes[i].name.c_str(), shapes[i].n, shapes[i].k,
                    shapes[i].count, per_call * shapes[i].count, gbs,
                    100.0 * gbs / peak_gbs);
    }

    std::printf("\n=== W8A16 GEMV (INT8 weights) ===\n");
    std::printf("%-10s %7s %7s %5s %11s %10s %8s\n",
                "shape", "n", "k", "count", "ms/token", "GB/s", "of peak");
    for (size_t i = 0; i < shapes.size(); ++i) {
        for (int w = 0; w < 20; ++w) launch_w8a16(i);
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        begin.record(stream.get());
        for (int it = 0; it < iterations; ++it) launch_w8a16(i);
        end.record(stream.get());
        end.synchronize();
        const double per_call = celeg::CudaEvent::elapsed_ms(begin, end) / iterations;
        const double bytes = static_cast<double>(shapes[i].n) * shapes[i].k * 1;
        const double gbs = bytes / (per_call * 1e-3) / 1e9;
        std::printf("%-10s %7d %7d %5d %11.4f %10.1f %7.0f%%\n",
                    shapes[i].name.c_str(), shapes[i].n, shapes[i].k,
                    shapes[i].count, per_call * shapes[i].count, gbs,
                    100.0 * gbs / peak_gbs);
    }

    // Full token sequence for both
    const int seq_iters = 50;
    auto seq_bf16 = [&]() {
        for (size_t i = 0; i < shapes.size(); ++i)
            for (int c = 0; c < shapes[i].count; ++c) launch_bf16(i);
    };
    auto seq_w8a16 = [&]() {
        for (size_t i = 0; i < shapes.size(); ++i)
            for (int c = 0; c < shapes[i].count; ++c) launch_w8a16(i);
    };

    for (int w = 0; w < 5; ++w) seq_bf16();
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    begin.record(stream.get());
    for (int it = 0; it < seq_iters; ++it) seq_bf16();
    end.record(stream.get());
    end.synchronize();
    double per_token = celeg::CudaEvent::elapsed_ms(begin, end) / seq_iters;
    double gbs = token_bytes / (per_token * 1e-3) / 1e9;
    std::printf("\n%-10s %7s %7s %5s %11.4f %10.1f %7.0f%%   <-- BF16 token\n",
                "TOKEN", "", "", "", per_token, gbs, 100.0 * gbs / peak_gbs);

    for (int w = 0; w < 5; ++w) seq_w8a16();
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    begin.record(stream.get());
    for (int it = 0; it < seq_iters; ++it) seq_w8a16();
    end.record(stream.get());
    end.synchronize();
    per_token = celeg::CudaEvent::elapsed_ms(begin, end) / seq_iters;
    gbs = i8_token_bytes / (per_token * 1e-3) / 1e9;
    std::printf("%-10s %7s %7s %5s %11.4f %10.1f %7.0f%%   <-- W8A16 token\n",
                "TOKEN", "", "", "", per_token, gbs, 100.0 * gbs / peak_gbs);

    std::printf("\nBF16 weight traffic: %.1f MB/token\n", token_bytes / 1e6);
    std::printf("INT8 weight traffic: %.1f MB/token\n", i8_token_bytes / 1e6);
    std::printf("Compare against end-to-end decode to see non-GEMV overhead.\n");
    return 0;
}
