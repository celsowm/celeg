#include "kernels/attention_output.hpp"
#include "utils.cuh"

#include <cmath>
#include <stdexcept>

namespace celeg {
namespace {

__global__ void orthogonalize_current_value_kernel(
    __nv_bfloat16* output,
    const __nv_bfloat16* current_value,
    int query_heads,
    int key_value_heads,
    int head_dim,
    float minimum_norm_squared) {
    const int row = static_cast<int>(blockIdx.y);
    const int query_head = static_cast<int>(blockIdx.x);
    if (query_head >= query_heads) return;

    const int queries_per_value = query_heads / key_value_heads;
    const int value_head = query_head / queries_per_value;
    __nv_bfloat16* head_output = output +
        (static_cast<size_t>(row) * query_heads + query_head) * head_dim;
    const __nv_bfloat16* value = current_value +
        (static_cast<size_t>(row) * key_value_heads + value_head) * head_dim;

    float projection = 0.0f;
    float norm_squared = 0.0f;
    for (int dimension = static_cast<int>(threadIdx.x);
         dimension < head_dim; dimension += static_cast<int>(blockDim.x)) {
        const float y = __bfloat162float(head_output[dimension]);
        const float v = __bfloat162float(value[dimension]);
        projection += y * v;
        norm_squared += v * v;
    }

    __shared__ float projection_sums[256];
    __shared__ float norm_sums[256];
    const int lane = static_cast<int>(threadIdx.x);
    projection_sums[lane] = projection;
    norm_sums[lane] = norm_squared;
    __syncthreads();

    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            projection_sums[lane] += projection_sums[lane + stride];
            norm_sums[lane] += norm_sums[lane + stride];
        }
        __syncthreads();
    }

    const float coefficient = projection_sums[0] /
        fmaxf(norm_sums[0], minimum_norm_squared);
    for (int dimension = lane; dimension < head_dim;
         dimension += static_cast<int>(blockDim.x)) {
        const float y = __bfloat162float(head_output[dimension]);
        const float v = __bfloat162float(value[dimension]);
        head_output[dimension] = __float2bfloat16(y - coefficient * v);
    }
}

}

void launch_orthogonalize_current_value(
    __nv_bfloat16* output,
    const __nv_bfloat16* current_value,
    int rows,
    int query_heads,
    int key_value_heads,
    int head_dim,
    float minimum_norm_squared,
    cudaStream_t stream) {
    if (!output || !current_value) {
        throw std::invalid_argument(
            "attention current-value orthogonalization needs output and value data");
    }
    if (rows <= 0 || query_heads <= 0 || key_value_heads <= 0 || head_dim <= 0 ||
        query_heads % key_value_heads != 0) {
        throw std::invalid_argument(
            "attention current-value orthogonalization has invalid dimensions");
    }
    if (!(minimum_norm_squared > 0.0f) || !std::isfinite(minimum_norm_squared)) {
        throw std::invalid_argument(
            "attention current-value orthogonalization has invalid norm floor");
    }
    constexpr int threads = 256;
    orthogonalize_current_value_kernel<<<
        dim3(static_cast<unsigned>(query_heads), static_cast<unsigned>(rows)),
        threads, 0, stream>>>(output, current_value, query_heads, key_value_heads,
                             head_dim, minimum_norm_squared);
    CELEG_CUDA(cudaGetLastError());
}

}
