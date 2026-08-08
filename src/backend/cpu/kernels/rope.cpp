#include "celeg/backend/cpu/kernels.hpp"
#include "celeg/backend/cpu/isa.hpp"
#include "celeg/model/position.hpp"

#include <cmath>
#include <array>
#include <stdexcept>
#include <vector>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define CELEG_CPU_X86 1
#else
#define CELEG_CPU_X86 0
#endif

#if defined(_MSC_VER) && CELEG_CPU_X86
#include "elementwise_avx2_msvc.hpp"
#endif

namespace celeg {
namespace {

#if CELEG_CPU_X86
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
                      const RopePositionSpec& rope, float eps) {
    if (!data || !norm_weight || heads <= 0 || head_dim <= 0 ||
        (head_dim % 2) != 0 || position < 0) {
        throw std::invalid_argument("invalid QK norm/RoPE arguments");
    }
    const int half = static_cast<int>(static_cast<float>(head_dim) * rope.rotary_fraction) / 2;
    thread_local std::vector<float> cos_vals;
    thread_local std::vector<float> sin_vals;
    cos_vals.resize(static_cast<size_t>(half));
    sin_vals.resize(static_cast<size_t>(half));
    for (int d = 0; d < half; ++d) {
        const float frequency = static_cast<float>(rope_frequency(
            rope, d, head_dim, position));
        const float angle = static_cast<float>(position) * frequency;
        cos_vals[d] = std::cos(angle);
        sin_vals[d] = std::sin(angle);
    }
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    if (g_has_avx2_fma) {
        if (rope.rotary_fraction == 1.0 && rope.scaling.kind == RopeScalingKind::None) {
            cpu_qk_norm_rope_avx2(data, norm_weight, cos_vals.data(), sin_vals.data(), heads, head_dim, eps);
            return;
        }
    }
#elif defined(_MSC_VER) && CELEG_CPU_X86
    if (g_has_avx2_fma) {
        if (rope.rotary_fraction == 1.0 && rope.scaling.kind == RopeScalingKind::None) {
            detail::cpu_qk_norm_rope_avx2_msvc(data, norm_weight, cos_vals.data(), sin_vals.data(), heads, head_dim, eps);
            return;
        }
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
              const RopePositionSpec& rope) {
    if (!data || heads <= 0 || head_dim <= 0 || (head_dim % 2) != 0 ||
        position < 0 || !(rope.theta > 0.0)) {
        throw std::invalid_argument("invalid RoPE arguments");
    }
    const int half = static_cast<int>(static_cast<float>(head_dim) * rope.rotary_fraction) / 2;
    std::vector<float> cos_vals(static_cast<size_t>(half));
    std::vector<float> sin_vals(static_cast<size_t>(half));
    for (int pair = 0; pair < half; ++pair) {
        const float frequency = static_cast<float>(rope_frequency(
            rope, pair, head_dim, position));
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

namespace {

int mrope_axis_for_pair(int pair, const std::array<int, 3>& sections,
                        bool interleaved) {
    if (interleaved) return pair % 3;
    if (pair < sections[0]) return 0;
    if (pair < sections[0] + sections[1]) return 1;
    return 2;
}

void validate_mrope(int head_dim, const std::array<int, 3>& sections,
                    float rope_theta, float rotary_fraction) {
    if (head_dim <= 0 || (head_dim % 2) != 0 || !(rope_theta > 0.0f) ||
        !(rotary_fraction > 0.0f) || rotary_fraction > 1.0f ||
        sections[0] <= 0 || sections[1] <= 0 || sections[2] <= 0) {
        throw std::invalid_argument("invalid MRoPE arguments");
    }
    const int rotary_dim = static_cast<int>(static_cast<float>(head_dim) * rotary_fraction);
    if ((rotary_dim % 2) != 0 || sections[0] + sections[1] + sections[2] != rotary_dim / 2) {
        throw std::invalid_argument("MRoPE sections do not match rotary dimension");
    }
}

} // namespace

void cpu_qk_norm_rope_mrope(float* data, const float* norm_weight,
                            int heads, int head_dim,
                            const std::array<int32_t, 3>& positions,
                            const std::array<int, 3>& sections,
                    bool interleaved, const RopePositionSpec& rope, float eps) {
    if (!data || !norm_weight || heads <= 0) {
        throw std::invalid_argument("invalid MRoPE QK arguments");
    }
    validate_mrope(head_dim, sections, static_cast<float>(rope.theta),
                   static_cast<float>(rope.rotary_fraction));
    const int rotary_dim = static_cast<int>(static_cast<float>(head_dim) * rope.rotary_fraction);
    const int pairs = rotary_dim / 2;
    std::vector<float> cos_values(static_cast<size_t>(pairs));
    std::vector<float> sin_values(static_cast<size_t>(pairs));
    for (int pair = 0; pair < pairs; ++pair) {
        const int axis = mrope_axis_for_pair(pair, sections, interleaved);
        const float frequency = static_cast<float>(rope_frequency(
            rope, pair, rotary_dim, positions[static_cast<size_t>(axis)]));
        const float angle = static_cast<float>(positions[static_cast<size_t>(axis)]) * frequency;
        cos_values[static_cast<size_t>(pair)] = std::cos(angle);
        sin_values[static_cast<size_t>(pair)] = std::sin(angle);
    }
    const float inv_dim = 1.0f / static_cast<float>(head_dim);
    for (int head = 0; head < heads; ++head) {
        float* row = data + static_cast<size_t>(head) * head_dim;
        double sum = 0.0;
        for (int d = 0; d < head_dim; ++d) sum += static_cast<double>(row[d]) * row[d];
        const float inv = 1.0f / std::sqrt(static_cast<float>(sum * inv_dim) + eps);
        for (int d = 0; d < head_dim; ++d) row[d] *= inv * norm_weight[d];
        for (int pair = 0; pair < pairs; ++pair) {
            const float a = row[pair];
            const float b = row[pairs + pair];
            row[pair] = a * cos_values[static_cast<size_t>(pair)] -
                        b * sin_values[static_cast<size_t>(pair)];
            row[pairs + pair] = b * cos_values[static_cast<size_t>(pair)] +
                                a * sin_values[static_cast<size_t>(pair)];
        }
    }
}

void cpu_rope_mrope(float* data, int heads, int head_dim,
                    const std::array<int32_t, 3>& positions,
                    const std::array<int, 3>& sections,
                    bool interleaved, const RopePositionSpec& rope) {
    if (!data || heads <= 0) throw std::invalid_argument("invalid MRoPE arguments");
    validate_mrope(head_dim, sections, static_cast<float>(rope.theta),
                   static_cast<float>(rope.rotary_fraction));
    const int rotary_dim = static_cast<int>(static_cast<float>(head_dim) * rope.rotary_fraction);
    const int pairs = rotary_dim / 2;
    std::vector<float> cos_values(static_cast<size_t>(pairs));
    std::vector<float> sin_values(static_cast<size_t>(pairs));
    for (int pair = 0; pair < pairs; ++pair) {
        const int axis = mrope_axis_for_pair(pair, sections, interleaved);
        const float frequency = static_cast<float>(rope_frequency(
            rope, pair, rotary_dim, positions[static_cast<size_t>(axis)]));
        const float angle = static_cast<float>(positions[static_cast<size_t>(axis)]) * frequency;
        cos_values[static_cast<size_t>(pair)] = std::cos(angle);
        sin_values[static_cast<size_t>(pair)] = std::sin(angle);
    }
    for (int head = 0; head < heads; ++head) {
        float* row = data + static_cast<size_t>(head) * head_dim;
        for (int pair = 0; pair < pairs; ++pair) {
            const float a = row[pair];
            const float b = row[pairs + pair];
            row[pair] = a * cos_values[static_cast<size_t>(pair)] -
                        b * sin_values[static_cast<size_t>(pair)];
            row[pairs + pair] = b * cos_values[static_cast<size_t>(pair)] +
                                a * sin_values[static_cast<size_t>(pair)];
        }
    }
}

} // namespace celeg
