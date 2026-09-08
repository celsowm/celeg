#include "kernels/rope_pairing.hpp"
#include "reductions.cuh"
#include "rope_scaling.cuh"
#include "utils.cuh"

#include <stdexcept>

namespace celeg {
namespace {

using cuda_reductions::warp_sum;

__global__ void paired_qk_norm_rope_kernel(
    __nv_bfloat16* data,
    const __nv_bfloat16* norm_weight,
    int rows,
    int heads,
    int head_dim,
    const int32_t* positions,
    float theta,
    int rotary_pairs,
    float epsilon,
    bool normalize,
    bool adjacent_pairs,
    CudaRopeScaling scaling,
    float attention_scale) {
    const int block = static_cast<int>(blockIdx.x);
    const int row = block / heads;
    const int head = block % heads;
    if (row >= rows) return;

    __nv_bfloat16* vector = data +
        (static_cast<size_t>(row) * heads + head) * head_dim;
    float square_sum = 0.0f;
    if (normalize) {
        for (int dimension = static_cast<int>(threadIdx.x);
             dimension < head_dim; dimension += static_cast<int>(blockDim.x)) {
            const float value = __bfloat162float(vector[dimension]);
            square_sum += value * value;
        }
        square_sum = warp_sum(square_sum);
    }

    __shared__ float warp_sums[32];
    __shared__ float inverse_norm;
    if (normalize) {
        const int lane = static_cast<int>(threadIdx.x) & 31;
        const int warp = static_cast<int>(threadIdx.x) >> 5;
        if (lane == 0) warp_sums[warp] = square_sum;
        __syncthreads();
        if (warp == 0) {
            float total = lane < (static_cast<int>(blockDim.x) + 31) / 32
                ? warp_sums[lane] : 0.0f;
            total = warp_sum(total);
            if (lane == 0) {
                inverse_norm = rsqrtf(total / static_cast<float>(head_dim) + epsilon);
            }
        }
        __syncthreads();
    } else if (threadIdx.x == 0) {
        inverse_norm = 1.0f;
    }
    __syncthreads();

    const int position = positions ? positions[row] : row;
    for (int pair = static_cast<int>(threadIdx.x);
         pair < rotary_pairs; pair += static_cast<int>(blockDim.x)) {
        const int first = adjacent_pairs ? 2 * pair : pair;
        const int second = adjacent_pairs ? first + 1 : rotary_pairs + pair;
        const float norm_first = normalize ? __bfloat162float(norm_weight[first]) : 1.0f;
        const float norm_second = normalize ? __bfloat162float(norm_weight[second]) : 1.0f;
        const float a = __bfloat162float(vector[first]) * inverse_norm * norm_first;
        const float b = __bfloat162float(vector[second]) * inverse_norm * norm_second;
        const float frequency = cuda_rope::scaled_frequency(
            theta, pair, 2 * rotary_pairs, position, scaling);
        const float angle = static_cast<float>(position) * frequency;
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        vector[first] = __float2bfloat16(a * cosine - b * sine);
        vector[second] = __float2bfloat16(b * cosine + a * sine);
    }

    if (normalize) {
        for (int dimension = static_cast<int>(threadIdx.x) + 2 * rotary_pairs;
             dimension < head_dim; dimension += static_cast<int>(blockDim.x)) {
            vector[dimension] = __float2bfloat16(
                __bfloat162float(vector[dimension]) * inverse_norm *
                __bfloat162float(norm_weight[dimension]));
        }
    }
    __syncthreads();
    if (attention_scale != 1.0f) {
        for (int dimension = static_cast<int>(threadIdx.x);
             dimension < head_dim; dimension += static_cast<int>(blockDim.x)) {
            vector[dimension] = __float2bfloat16(
                __bfloat162float(vector[dimension]) * attention_scale);
        }
    }
}

int rope_threads(int head_dim) {
    int threads = 32;
    while (threads < head_dim && threads < 256) threads <<= 1;
    return threads;
}

}

void launch_qk_norm_rope_positions(
    __nv_bfloat16* query,
    __nv_bfloat16* key,
    const __nv_bfloat16* query_norm,
    const __nv_bfloat16* key_norm,
    int rows,
    int query_heads,
    int key_value_heads,
    int head_dim,
    const int32_t* positions,
    float rope_theta,
    float rotary_fraction,
    float epsilon,
    bool normalize,
    RopePairingKind pairing,
    CudaRopeScaling scaling,
    cudaStream_t stream) {
    if (!query || rows <= 0 || query_heads <= 0 ||
        key_value_heads <= 0 || head_dim <= 0 || (head_dim % 2) != 0 ||
        !(rope_theta > 0.0f) || !(rotary_fraction > 0.0f) ||
        rotary_fraction > 1.0f || (normalize && !query_norm)) {
        throw std::invalid_argument("invalid position-vector RoPE arguments");
    }
    if (key && normalize && !key_norm) {
        throw std::invalid_argument("position-vector RoPE key norm is missing");
    }
    const int rotary_dimension = static_cast<int>(
        static_cast<float>(head_dim) * rotary_fraction);
    if (rotary_dimension <= 0 || (rotary_dimension % 2) != 0) {
        throw std::invalid_argument("position-vector RoPE rotary dimension must be even");
    }
    const int pairs = rotary_dimension / 2;
    const int threads = rope_threads(head_dim);
    const bool adjacent_pairs = pairing == RopePairingKind::AdjacentPairs;
    const float query_attention_scale = scaling.kind == 3
        ? scaling.attention_factor * scaling.attention_factor
        : 1.0f;
    paired_qk_norm_rope_kernel<<<rows * query_heads, threads, 0, stream>>>(
        query, query_norm, rows, query_heads, head_dim, positions, rope_theta,
        pairs, epsilon, normalize, adjacent_pairs, scaling, query_attention_scale);
    if (key) {
        paired_qk_norm_rope_kernel<<<rows * key_value_heads, threads, 0, stream>>>(
            key, key_norm, rows, key_value_heads, head_dim, positions, rope_theta,
            pairs, epsilon, normalize, adjacent_pairs, scaling, 1.0f);
    }
    CELEG_CUDA(cudaGetLastError());
}

}
