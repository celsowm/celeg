#include "lfm/backend/cpu/kernels.hpp"
#include "lfm/backend/cpu/isa.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define LFM_CPU_X86 1
#else
#define LFM_CPU_X86 0
#endif

#if defined(_MSC_VER) && LFM_CPU_X86
#include "elementwise_avx2_msvc.hpp"
#endif

namespace lfm {
namespace {

#if LFM_CPU_X86
static const bool g_has_avx2_fma = []() {
    auto caps = detect_cpu_capabilities();
    return caps.avx2 && caps.fma;
}();
#endif

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx2,fma")))
void cpu_qk_norm_rope_avx2(float* data, const float* norm_weight,
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
#endif

} // namespace

void cpu_qk_norm_rope(float* data, const float* norm_weight,
                      int heads, int head_dim, int position,
                      float rope_theta, float eps) {
    if (!data || !norm_weight || heads <= 0 || head_dim <= 0 ||
        (head_dim % 2) != 0 || position < 0) {
        throw std::invalid_argument("invalid QK norm/RoPE arguments");
    }
    const int half = head_dim / 2;
    thread_local std::vector<float> cos_vals;
    thread_local std::vector<float> sin_vals;
    cos_vals.resize(static_cast<size_t>(half));
    sin_vals.resize(static_cast<size_t>(half));
    for (int d = 0; d < half; ++d) {
        const float frequency = std::pow(rope_theta, -2.0f * static_cast<float>(d) / head_dim);
        const float angle = static_cast<float>(position) * frequency;
        cos_vals[d] = std::cos(angle);
        sin_vals[d] = std::sin(angle);
    }
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    if (g_has_avx2_fma) {
        cpu_qk_norm_rope_avx2(data, norm_weight, cos_vals.data(), sin_vals.data(), heads, head_dim, eps);
        return;
    }
#elif defined(_MSC_VER) && LFM_CPU_X86
    if (g_has_avx2_fma) {
        detail::cpu_qk_norm_rope_avx2_msvc(data, norm_weight, cos_vals.data(), sin_vals.data(), heads, head_dim, eps);
        return;
    }
#endif
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

void cpu_rope(float* data, int heads, int head_dim, int position,
              float rope_theta) {
    if (!data || heads <= 0 || head_dim <= 0 || (head_dim % 2) != 0 ||
        position < 0 || !(rope_theta > 0.0f)) {
        throw std::invalid_argument("invalid RoPE arguments");
    }
    const int half = head_dim / 2;
    std::vector<float> cos_vals(static_cast<size_t>(half));
    std::vector<float> sin_vals(static_cast<size_t>(half));
    for (int pair = 0; pair < half; ++pair) {
        const float frequency = std::pow(rope_theta,
            -2.0f * static_cast<float>(pair) / static_cast<float>(head_dim));
        const float angle = static_cast<float>(position) * frequency;
        cos_vals[static_cast<size_t>(pair)] = std::cos(angle);
        sin_vals[static_cast<size_t>(pair)] = std::sin(angle);
    }
    for (int head = 0; head < heads; ++head) {
        float* row = data + static_cast<size_t>(head) * head_dim;
        for (int pair = 0; pair < half; ++pair) {
            const float x0 = row[pair];
            const float x1 = row[half + pair];
            row[pair] = x0 * cos_vals[static_cast<size_t>(pair)] -
                        x1 * sin_vals[static_cast<size_t>(pair)];
            row[half + pair] = x1 * cos_vals[static_cast<size_t>(pair)] +
                               x0 * sin_vals[static_cast<size_t>(pair)];
        }
    }
}

} // namespace lfm
