#include "kernel_common.cuh"
#include "celeg/attention/merge_semantics.hpp"
#include "celeg/attention/pattern_semantics.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace celeg {
#include "attention_common.cuh"
}

namespace {

__device__ __forceinline__ float baseline_merge_segmented_attention_lane(
    size_t base, int count, int lane, int head_dim,
    const float* partial_max, const float* partial_denom,
    const float* partial_accum) {
    float global_max = -FLT_MAX;
    for (int chunk = 0; chunk < count; ++chunk) {
        global_max = fmaxf(global_max, partial_max[base + chunk]);
    }
    float denominator = 0.0f;
    float accumulator = 0.0f;
    for (int chunk = 0; chunk < count; ++chunk) {
        const float local_denom = partial_denom[base + chunk];
        if (local_denom == 0.0f) continue;
        const float factor = expf(partial_max[base + chunk] - global_max);
        denominator += local_denom * factor;
        if (lane < head_dim) {
            accumulator += partial_accum[
                (base + chunk) * static_cast<size_t>(head_dim) + lane] * factor;
        }
    }
    return accumulator / denominator;
}

__global__ void baseline_merge_kernel(
    float* out, int rows, int heads, int head_dim, int chunks,
    const float* partial_max, const float* partial_denom,
    const float* partial_accum) {
    const int flat = static_cast<int>(blockIdx.x);
    const int row = flat / heads;
    const int head = flat % heads;
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows) return;
    const size_t base =
        (static_cast<size_t>(row) * heads + head) * chunks;
    if (lane < head_dim) {
        out[(static_cast<size_t>(row) * heads + head) * head_dim + lane] =
            baseline_merge_segmented_attention_lane(
                base, chunks, lane, head_dim,
                partial_max, partial_denom, partial_accum);
    }
}

__global__ void shared_merge_kernel(
    float* out, int rows, int heads, int head_dim, int chunks,
    const float* partial_max, const float* partial_denom,
    const float* partial_accum) {
    const int flat = static_cast<int>(blockIdx.x);
    const int row = flat / heads;
    const int head = flat % heads;
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows) return;
    const size_t base =
        (static_cast<size_t>(row) * heads + head) * chunks;
    const float merged = celeg::merge_segmented_attention_lane(
        base, chunks, lane, head_dim,
        partial_max, partial_denom, partial_accum);
    if (lane < head_dim) {
        out[(static_cast<size_t>(row) * heads + head) * head_dim + lane] = merged;
    }
}

struct Shape {
    int rows;
    int heads;
    int head_dim;
    int chunks;
};

uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float random_float(uint32_t& state, float low, float high) {
    const float unit = static_cast<float>(xorshift32(state) >> 8) / 16777216.0f;
    return low + (high - low) * unit;
}

template <typename T>
void upload(celeg::DeviceBuffer<T>& dst, const std::vector<T>& src,
            cudaStream_t stream) {
    CELEG_CUDA(cudaMemcpyAsync(dst.data(), src.data(),
                              src.size() * sizeof(T),
                              cudaMemcpyHostToDevice, stream));
}

void print_attributes(const char* name, const void* kernel) {
    cudaFuncAttributes attributes{};
    CELEG_CUDA(cudaFuncGetAttributes(&attributes, kernel));
    std::printf("%-10s regs=%d local=%zu shared=%zu max_threads=%d\n",
                name, attributes.numRegs,
                static_cast<size_t>(attributes.localSizeBytes),
                static_cast<size_t>(attributes.sharedSizeBytes),
                attributes.maxThreadsPerBlock);
}

template <typename Launch>
double time_kernel(Launch&& launch, cudaStream_t stream, int iterations) {
    for (int warmup = 0; warmup < 50; ++warmup) launch();
    CELEG_CUDA(cudaStreamSynchronize(stream));
    celeg::CudaEvent begin;
    celeg::CudaEvent end;
    begin.record(stream);
    for (int iteration = 0; iteration < iterations; ++iteration) launch();
    end.record(stream);
    end.synchronize();
    return celeg::CudaEvent::elapsed_ms(begin, end) * 1000.0 /
           static_cast<double>(iterations);
}

bool run_shape(const Shape& shape, cudaStream_t stream, int iterations) {
    const size_t groups = static_cast<size_t>(shape.rows) * shape.heads;
    const size_t partial_count = groups * shape.chunks;
    const size_t output_count = groups * shape.head_dim;
    const size_t accum_count = partial_count * shape.head_dim;

    std::vector<float> host_max(partial_count);
    std::vector<float> host_denom(partial_count);
    std::vector<float> host_accum(accum_count);
    uint32_t state = 0xC0FFEEu ^ static_cast<uint32_t>(shape.chunks * 131 + shape.head_dim);
    for (size_t index = 0; index < partial_count; ++index) {
        host_max[index] = random_float(state, -8.0f, 8.0f);
        host_denom[index] = (index % 11 == 0)
            ? 0.0f
            : random_float(state, 0.01f, 4.0f);
    }
    for (float& value : host_accum) value = random_float(state, -3.0f, 3.0f);

    celeg::DeviceBuffer<float> partial_max(partial_count);
    celeg::DeviceBuffer<float> partial_denom(partial_count);
    celeg::DeviceBuffer<float> partial_accum(accum_count);
    celeg::DeviceBuffer<float> baseline(output_count);
    celeg::DeviceBuffer<float> shared(output_count);
    upload(partial_max, host_max, stream);
    upload(partial_denom, host_denom, stream);
    upload(partial_accum, host_accum, stream);
    CELEG_CUDA(cudaStreamSynchronize(stream));

    const int threads = shape.head_dim <= 32 ? 32 :
        shape.head_dim <= 64 ? 64 :
        shape.head_dim <= 128 ? 128 :
        shape.head_dim <= 256 ? 256 : 512;
    const dim3 grid(static_cast<unsigned>(groups));

    auto launch_baseline = [&] {
        baseline_merge_kernel<<<grid, threads, 0, stream>>>(
            baseline.data(), shape.rows, shape.heads, shape.head_dim, shape.chunks,
            partial_max.data(), partial_denom.data(), partial_accum.data());
    };
    auto launch_shared = [&] {
        shared_merge_kernel<<<grid, threads, 0, stream>>>(
            shared.data(), shape.rows, shape.heads, shape.head_dim, shape.chunks,
            partial_max.data(), partial_denom.data(), partial_accum.data());
    };

    launch_baseline();
    launch_shared();
    CELEG_CUDA(cudaStreamSynchronize(stream));

    std::vector<float> host_baseline(output_count);
    std::vector<float> host_shared(output_count);
    CELEG_CUDA(cudaMemcpy(host_baseline.data(), baseline.data(),
                          output_count * sizeof(float), cudaMemcpyDeviceToHost));
    CELEG_CUDA(cudaMemcpy(host_shared.data(), shared.data(),
                          output_count * sizeof(float), cudaMemcpyDeviceToHost));
    const bool bitwise = std::memcmp(host_baseline.data(), host_shared.data(),
                                     output_count * sizeof(float)) == 0;
    if (!bitwise) {
        for (size_t index = 0; index < output_count; ++index) {
            if (std::memcmp(&host_baseline[index], &host_shared[index], sizeof(float)) != 0) {
                std::printf("mismatch index=%zu baseline=%g shared=%g\n",
                            index, host_baseline[index], host_shared[index]);
                break;
            }
        }
    }

    const double baseline_us = time_kernel(launch_baseline, stream, iterations);
    const double shared_us = time_kernel(launch_shared, stream, iterations);
    std::printf("rows=%d heads=%d dim=%d chunks=%d bitwise=%s baseline_us=%.3f shared_us=%.3f ratio=%.4f\n",
                shape.rows, shape.heads, shape.head_dim, shape.chunks,
                bitwise ? "yes" : "NO", baseline_us, shared_us,
                baseline_us > 0.0 ? shared_us / baseline_us : 0.0);
    return bitwise;
}

}  // namespace

int main(int argc, char** argv) {
    int iterations = argc > 1 ? std::atoi(argv[1]) : 2000;
    if (iterations <= 0) iterations = 2000;

    cudaDeviceProp properties{};
    CELEG_CUDA(cudaGetDeviceProperties(&properties, 0));
    std::printf("device=%s iterations=%d\n", properties.name, iterations);
    print_attributes("baseline", reinterpret_cast<const void*>(baseline_merge_kernel));
    print_attributes("shared", reinterpret_cast<const void*>(shared_merge_kernel));

    celeg::CudaStream stream;
    const std::vector<Shape> shapes = {
        {1, 32, 128, 4},
        {1, 32, 128, 8},
        {1, 32, 128, 16},
        {4, 32, 128, 16},
        {8, 32, 128, 32},
        {8, 32, 256, 32},
        {16, 32, 256, 64},
    };

    bool all_bitwise = true;
    for (const Shape& shape : shapes) {
        all_bitwise = run_shape(shape, stream.get(), iterations) && all_bitwise;
    }
    return all_bitwise ? 0 : 1;
}
