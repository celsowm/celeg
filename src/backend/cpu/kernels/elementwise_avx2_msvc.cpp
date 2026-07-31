#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))

#include "elementwise_avx2_msvc.hpp"

#include <cmath>
#include <immintrin.h>

namespace celeg::detail {

void cpu_rmsnorm_avx2_msvc(const float* input, const float* weight, float* output,
                           size_t width, float eps) {
    double sum = 0.0;
    size_t i = 0;
    __m256 sum_vec = _mm256_setzero_ps();
    for (; i + 7 < width; i += 8) {
        __m256 in = _mm256_loadu_ps(input + i);
        sum_vec = _mm256_fmadd_ps(in, in, sum_vec);
    }
    alignas(32) float temp[8];
    _mm256_storeu_ps(temp, sum_vec);
    for (int j = 0; j < 8; ++j) sum += temp[j];
    for (; i < width; ++i) {
        sum += static_cast<double>(input[i]) * input[i];
    }

    const float inv = 1.0f / std::sqrt(static_cast<float>(sum / width) + eps);
    __m256 inv_vec = _mm256_set1_ps(inv);

    i = 0;
    for (; i + 7 < width; i += 8) {
        __m256 in = _mm256_loadu_ps(input + i);
        __m256 w = _mm256_loadu_ps(weight + i);
        __m256 out = _mm256_mul_ps(_mm256_mul_ps(in, inv_vec), w);
        _mm256_storeu_ps(output + i, out);
    }
    for (; i < width; ++i) {
        output[i] = input[i] * inv * weight[i];
    }
}

void cpu_residual_add_avx2_msvc(float* data, const float* residual, size_t count) {
    size_t i = 0;
    for (; i + 7 < count; i += 8) {
        __m256 d = _mm256_loadu_ps(data + i);
        __m256 r = _mm256_loadu_ps(residual + i);
        _mm256_storeu_ps(data + i, _mm256_add_ps(d, r));
    }
    for (; i < count; ++i) {
        data[i] += residual[i];
    }
}

void cpu_swiglu_avx2_msvc(const float* gate_up, float* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const float gate = gate_up[i];
        const float up = gate_up[count + i];
        output[i] = (gate / (1.0f + std::exp(-gate))) * up;
    }
}

void cpu_qk_norm_rope_avx2_msvc(float* data, const float* norm_weight,
                                const float* cos_vals, const float* sin_vals,
                                int heads, int head_dim, float eps) {
    const int half = head_dim / 2;
    for (int head = 0; head < heads; ++head) {
        float* vector = data + static_cast<size_t>(head) * head_dim;
        double sum = 0.0;
        int d = 0;
        __m256 sum_vec = _mm256_setzero_ps();
        for (; d + 7 < head_dim; d += 8) {
            __m256 v = _mm256_loadu_ps(vector + d);
            sum_vec = _mm256_fmadd_ps(v, v, sum_vec);
        }
        alignas(32) float temp[8];
        _mm256_storeu_ps(temp, sum_vec);
        for (int j = 0; j < 8; ++j) sum += temp[j];
        for (; d < head_dim; ++d) {
            sum += static_cast<double>(vector[d]) * vector[d];
        }

        const float inv = 1.0f / std::sqrt(static_cast<float>(sum / head_dim) + eps);
        __m256 inv_vec = _mm256_set1_ps(inv);

        int i = 0;
        for (; i + 7 < half; i += 8) {
            __m256 v_d = _mm256_loadu_ps(vector + i);
            __m256 v_dh = _mm256_loadu_ps(vector + i + half);
            __m256 nw_d = _mm256_loadu_ps(norm_weight + i);
            __m256 nw_dh = _mm256_loadu_ps(norm_weight + i + half);
            __m256 cv = _mm256_loadu_ps(cos_vals + i);
            __m256 sv = _mm256_loadu_ps(sin_vals + i);

            __m256 a = _mm256_mul_ps(_mm256_mul_ps(v_d, inv_vec), nw_d);
            __m256 b = _mm256_mul_ps(_mm256_mul_ps(v_dh, inv_vec), nw_dh);

            __m256 res_d = _mm256_fmsub_ps(a, cv, _mm256_mul_ps(b, sv));
            __m256 res_dh = _mm256_fmadd_ps(b, cv, _mm256_mul_ps(a, sv));

            _mm256_storeu_ps(vector + i, res_d);
            _mm256_storeu_ps(vector + i + half, res_dh);
        }
        for (; i < half; ++i) {
            const float a = vector[i] * inv * norm_weight[i];
            const float b = vector[i + half] * inv * norm_weight[i + half];
            vector[i] = a * cos_vals[i] - b * sin_vals[i];
            vector[i + half] = b * cos_vals[i] + a * sin_vals[i];
        }
    }
}

} // namespace celeg::detail

#endif
