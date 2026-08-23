
#include "celeg/backend/cuda/utils.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>

namespace {

#include "celeg/backend/cuda/kernels/gemv_kernels.cuh"

struct Shape {
    int n;
    int k;
    int count;
    std::string name;
};

// Each shape is benchmarked by re-running one matrix in a tight loop, so
// without care the whole matrix stays resident in L2 and the benchmark
// measures cache bandwidth rather than the DRAM streaming that real decode
// does. This card has 100.7 MB of L2 and every shape here except lm_head is
// under 11 MB, so an earlier revision of this benchmark reported 143% and
// 228% "of theoretical peak" -- not suspicious numbers so much as
// impossible ones. Each shape therefore allocates enough copies of its
// weight matrix to overflow L2 several times over and rotates through them,
// one copy per iteration. Real decode touches a given layer's weights once
// per token and evicts them long before coming back, which is what this
// rotation approximates.
constexpr double kRotationBytesTarget = 300e6;

size_t rotation_copies(size_t matrix_bytes) {
    if (matrix_bytes == 0) return 1;
    const size_t wanted =
        static_cast<size_t>(kRotationBytesTarget / matrix_bytes) + 1;
    return wanted < 2 ? 2 : wanted;
}
uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

template <typename T, typename MakeValue>
void fill_random(celeg::DeviceBuffer<T>& buffer, cudaStream_t stream,
                 uint32_t seed, MakeValue make_value) {
    std::vector<T> host(buffer.size());
    uint32_t state = seed | 1u;
    for (T& value : host) value = make_value(xorshift32(state));
    CELEG_CUDA(cudaMemcpyAsync(buffer.data(), host.data(), buffer.bytes(),
                               cudaMemcpyHostToDevice, stream));
    CELEG_CUDA(cudaStreamSynchronize(stream));
}

void fill_random_bf16(celeg::DeviceBuffer<__nv_bfloat16>& buffer,
                      cudaStream_t stream, uint32_t seed) {
    fill_random(buffer, stream, seed, [](uint32_t bits) {
        // Uniform in [-1, 1); the exact distribution is irrelevant, only
        // that adjacent values differ so nothing compresses.
        const float unit = static_cast<float>(bits >> 8) / 8388608.0f;
        return __float2bfloat16(unit * 2.0f - 1.0f);
    });
}

void fill_random_i8(celeg::DeviceBuffer<int8_t>& buffer, cudaStream_t stream,
                    uint32_t seed) {
    fill_random(buffer, stream, seed, [](uint32_t bits) {
        return static_cast<int8_t>(bits & 0xFFu);
    });
}

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

}

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
    fill_random_bf16(x, stream.get(), 0x1234u);

    // One buffer per shape holding `copies[i]` back-to-back matrices. Only
    // the first is filled from the host; the rest are device-to-device
    // copies of it. Identical contents are fine -- they live at different
    // addresses, which is all the cache cares about.
    std::vector<size_t> copies(shapes.size());
    std::vector<celeg::DeviceBuffer<__nv_bfloat16>> weights(shapes.size());
    std::vector<celeg::DeviceBuffer<int8_t>> i8_weights(shapes.size());
    std::vector<celeg::DeviceBuffer<float>> i8_scales(shapes.size());
    for (size_t i = 0; i < shapes.size(); ++i) {
        const size_t elements =
            static_cast<size_t>(shapes[i].n) * shapes[i].k;
        copies[i] = rotation_copies(elements * sizeof(int8_t));

        weights[i].reset(elements * copies[i]);
        celeg::DeviceBuffer<__nv_bfloat16> seed_bf16(elements);
        fill_random_bf16(seed_bf16, stream.get(), 0x9E37u + 7919u * i);
        for (size_t c = 0; c < copies[i]; ++c) {
            CELEG_CUDA(cudaMemcpyAsync(weights[i].data() + elements * c,
                                       seed_bf16.data(),
                                       elements * sizeof(__nv_bfloat16),
                                       cudaMemcpyDeviceToDevice, stream.get()));
        }

        i8_weights[i].reset(elements * copies[i]);
        celeg::DeviceBuffer<int8_t> seed_i8(elements);
        fill_random_i8(seed_i8, stream.get(), 0xBEEFu + 7919u * i);
        for (size_t c = 0; c < copies[i]; ++c) {
            CELEG_CUDA(cudaMemcpyAsync(i8_weights[i].data() + elements * c,
                                       seed_i8.data(), elements * sizeof(int8_t),
                                       cudaMemcpyDeviceToDevice, stream.get()));
        }

        i8_scales[i].reset(static_cast<size_t>(shapes[i].n));
        fill_random(i8_scales[i], stream.get(), 0x5EEDu + 7919u * i,
                    [](uint32_t bits) {
                        return static_cast<float>(bits >> 8) / 8388608.0f;
                    });
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    }
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));

    auto launch_bf16 = [&](size_t i, size_t iter) {
        constexpr int wpb = 8;
        const Shape& s = shapes[i];
        const size_t elements = static_cast<size_t>(s.n) * s.k;
        bf16_gemv_kernel<<<(s.n + wpb - 1) / wpb, wpb * 32, 0, stream.get()>>>(
            x.data(), weights[i].data() + elements * (iter % copies[i]),
            y.data(), s.n, s.k, 0.0f);
    };
    auto launch_w8a16 = [&](size_t i, size_t iter) {
        constexpr int wpb = W8A16_WARPS_PER_BLOCK;
        const Shape& s = shapes[i];
        const size_t elements = static_cast<size_t>(s.n) * s.k;
        w8a16_gemv_kernel<<<(s.n + wpb - 1) / wpb, wpb * 32, 0, stream.get()>>>(
            x.data(), i8_weights[i].data() + elements * (iter % copies[i]),
            i8_scales[i].data(), y.data(), 1, s.n, s.k, 0.0f);
    };

    auto launch_ksplit = [&](int nwarps, size_t i, size_t iter) {
        const Shape& s = shapes[i];
        const size_t elements = static_cast<size_t>(s.n) * s.k;
        const int8_t* w = i8_weights[i].data() + elements * (iter % copies[i]);
        const dim3 grid(static_cast<unsigned>(s.n), 1);
        switch (nwarps) {
            case 4:
                w8a16_gemv_ksplit_kernel<4><<<grid, 4 * 32, 0, stream.get()>>>(
                    x.data(), w, i8_scales[i].data(), y.data(), 1, s.n, s.k, 0.0f);
                break;
            case 8:
                w8a16_gemv_ksplit_kernel<8><<<grid, 8 * 32, 0, stream.get()>>>(
                    x.data(), w, i8_scales[i].data(), y.data(), 1, s.n, s.k, 0.0f);
                break;
            default:
                w8a16_gemv_ksplit_kernel<16><<<grid, 16 * 32, 0, stream.get()>>>(
                    x.data(), w, i8_scales[i].data(), y.data(), 1, s.n, s.k, 0.0f);
                break;
        }
    };

    // The two kernels must agree before any timing is worth reading.
    {
        std::vector<__nv_bfloat16> reference(static_cast<size_t>(max_n));
        std::vector<__nv_bfloat16> candidate(static_cast<size_t>(max_n));
        bool all_match = true;
        for (size_t i = 0; i < shapes.size(); ++i) {
            launch_w8a16(i, 0);
            CELEG_CUDA(cudaStreamSynchronize(stream.get()));
            CELEG_CUDA(cudaMemcpy(reference.data(), y.data(),
                                  shapes[i].n * sizeof(__nv_bfloat16),
                                  cudaMemcpyDeviceToHost));
            for (int nwarps : {4, 8, 16}) {
                launch_ksplit(nwarps, i, 0);
                CELEG_CUDA(cudaStreamSynchronize(stream.get()));
                CELEG_CUDA(cudaMemcpy(candidate.data(), y.data(),
                                      shapes[i].n * sizeof(__nv_bfloat16),
                                      cudaMemcpyDeviceToHost));
                for (int r = 0; r < shapes[i].n; ++r) {
                    const float a = __bfloat162float(reference[r]);
                    const float b = __bfloat162float(candidate[r]);
                    // Different summation order over k, so compare with a
                    // relative tolerance rather than bit-exactly.
                    if (std::fabs(a - b) > 2e-2f * std::fmax(1.0f, std::fabs(a))) {
                        std::printf("MISMATCH %s nwarps=%d row %d: %g vs %g\n",
                                    shapes[i].name.c_str(), nwarps, r, a, b);
                        all_match = false;
                        break;
                    }
                }
            }
        }
        std::printf("k-split correctness vs w8a16_gemv_kernel: %s\n",
                    all_match ? "OK" : "FAILED");
        if (!all_match) return 1;
    }

    celeg::CudaEvent begin;
    celeg::CudaEvent end;

    double i8_token_bytes = 0.0;
    for (const Shape& s : shapes)
        i8_token_bytes += static_cast<double>(s.n) * s.k * 1 * s.count;

    std::printf("\n=== BF16 GEMV ===\n");
    std::printf("%-10s %7s %7s %5s %11s %10s %8s\n",
                "shape", "n", "k", "count", "ms/token", "GB/s", "of peak");
    for (size_t i = 0; i < shapes.size(); ++i) {
        for (int w = 0; w < 20; ++w) launch_bf16(i, w);
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        begin.record(stream.get());
        for (int it = 0; it < iterations; ++it) launch_bf16(i, it);
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
        for (int w = 0; w < 20; ++w) launch_w8a16(i, w);
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        begin.record(stream.get());
        for (int it = 0; it < iterations; ++it) launch_w8a16(i, it);
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

    for (int nwarps : {4, 8, 16}) {
        std::printf("\n=== W8A16 GEMV K-SPLIT (nwarps=%d) ===\n", nwarps);
        std::printf("%-10s %7s %7s %5s %11s %10s %8s\n",
                    "shape", "n", "k", "count", "ms/token", "GB/s", "of peak");
        for (size_t i = 0; i < shapes.size(); ++i) {
            for (int w = 0; w < 20; ++w) launch_ksplit(nwarps, i, w);
            CELEG_CUDA(cudaStreamSynchronize(stream.get()));
            begin.record(stream.get());
            for (int it = 0; it < iterations; ++it) launch_ksplit(nwarps, i, it);
            end.record(stream.get());
            end.synchronize();
            const double per_call =
                celeg::CudaEvent::elapsed_ms(begin, end) / iterations;
            const double bytes =
                static_cast<double>(shapes[i].n) * shapes[i].k * 1;
            const double gbs = bytes / (per_call * 1e-3) / 1e9;
            std::printf("%-10s %7d %7d %5d %11.4f %10.1f %7.0f%%\n",
                        shapes[i].name.c_str(), shapes[i].n, shapes[i].k,
                        shapes[i].count, per_call * shapes[i].count, gbs,
                        100.0 * gbs / peak_gbs);
        }
    }

    const int seq_iters = 50;
    auto seq_bf16 = [&](size_t iter) {
        for (size_t i = 0; i < shapes.size(); ++i)
            for (int c = 0; c < shapes[i].count; ++c)
                launch_bf16(i, iter * shapes[i].count + c);
    };
    auto seq_w8a16 = [&](size_t iter) {
        for (size_t i = 0; i < shapes.size(); ++i)
            for (int c = 0; c < shapes[i].count; ++c)
                launch_w8a16(i, iter * shapes[i].count + c);
    };

    for (int w = 0; w < 5; ++w) seq_bf16(w);
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    begin.record(stream.get());
    for (int it = 0; it < seq_iters; ++it) seq_bf16(it);
    end.record(stream.get());
    end.synchronize();
    double per_token = celeg::CudaEvent::elapsed_ms(begin, end) / seq_iters;
    double gbs = token_bytes / (per_token * 1e-3) / 1e9;
    std::printf("\n%-10s %7s %7s %5s %11.4f %10.1f %7.0f%%   <-- BF16 token\n",
                "TOKEN", "", "", "", per_token, gbs, 100.0 * gbs / peak_gbs);

    for (int w = 0; w < 5; ++w) seq_w8a16(w);
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    begin.record(stream.get());
    for (int it = 0; it < seq_iters; ++it) seq_w8a16(it);
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
