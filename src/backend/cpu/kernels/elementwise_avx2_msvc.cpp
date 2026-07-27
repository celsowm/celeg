#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))

#include "elementwise_avx2_msvc.hpp"

#include <cmath>

namespace lfm::detail {

void cpu_rmsnorm_avx2_msvc(const float* input, const float* weight, float* output,
                           size_t width, float eps) {
    double sum = 0.0;
    for (size_t i = 0; i < width; ++i) sum += static_cast<double>(input[i]) * input[i];
    const float inv = 1.0f / std::sqrt(static_cast<float>(sum / width) + eps);
    for (size_t i = 0; i < width; ++i) output[i] = input[i] * inv * weight[i];
}

void cpu_residual_add_avx2_msvc(float* data, const float* residual, size_t count) {
    for (size_t i = 0; i < count; ++i) data[i] += residual[i];
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
        for (int d = 0; d < head_dim; ++d) sum += static_cast<double>(vector[d]) * vector[d];
        const float inv = 1.0f / std::sqrt(static_cast<float>(sum / head_dim) + eps);
        for (int d = 0; d < half; ++d) {
            const float a = vector[d] * inv * norm_weight[d];
            const float b = vector[d + half] * inv * norm_weight[d + half];
            vector[d] = a * cos_vals[d] - b * sin_vals[d];
            vector[d + half] = b * cos_vals[d] + a * sin_vals[d];
        }
    }
}

} // namespace lfm::detail

#endif
