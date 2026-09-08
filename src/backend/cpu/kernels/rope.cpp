#include "math.hpp"

#include "celeg/backend/cpu/rope.hpp"
#include "celeg/model/position.hpp"

#include <array>
#include <cmath>
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

std::pair<int, int> rope_pair_indices(int pair, int pair_count,
                                     RopePairingKind pairing) {
    if (pairing == RopePairingKind::AdjacentPairs) {
        return {2 * pair, 2 * pair + 1};
    }
    return {pair, pair_count + pair};
}

void validate_qk_norm_rope_arguments(float* data, const float* norm_weight,
                                     int heads, int head_dim, int position) {
    if (!data || !norm_weight || heads <= 0 || head_dim <= 0 ||
        (head_dim % 2) != 0 || position < 0) {
        throw std::invalid_argument("invalid QK norm/RoPE arguments");
    }
}

void build_rope_tables(const RopePositionSpec& rope, int rotary_dim, int position,
                       std::vector<float>& cos_vals,
                       std::vector<float>& sin_vals) {
    const int half = rotary_dim / 2;
    cos_vals.resize(static_cast<size_t>(half));
    sin_vals.resize(static_cast<size_t>(half));
    for (int d = 0; d < half; ++d) {
        const float frequency = static_cast<float>(rope_frequency(
            rope, d, rotary_dim, position));
        const float angle = static_cast<float>(position) * frequency;
        cos_vals[static_cast<size_t>(d)] = std::cos(angle);
        sin_vals[static_cast<size_t>(d)] = std::sin(angle);
    }
}

void apply_qk_norm_rope_scalar(float* data, const float* norm_weight,
                               const float* cos_vals, const float* sin_vals,
                               int heads, int head_dim, int rotary_dim,
                               RopePairingKind pairing, float eps) {
    const int half = rotary_dim / 2;
    for (int head = 0; head < heads; ++head) {
        float* vector = data + static_cast<size_t>(head) * head_dim;
        double sum = 0.0;
        for (int d = 0; d < head_dim; ++d) {
            sum += static_cast<double>(vector[d]) * vector[d];
        }
        const float inv = 1.0f / std::sqrt(static_cast<float>(sum / head_dim) + eps);
        for (int pair = 0; pair < half; ++pair) {
            const auto [first, second] = rope_pair_indices(pair, half, pairing);
            const float a = vector[first] * inv * norm_weight[first];
            const float b = vector[second] * inv * norm_weight[second];
            vector[first] = a * cos_vals[pair] - b * sin_vals[pair];
            vector[second] = b * cos_vals[pair] + a * sin_vals[pair];
        }
    }
}

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

void cpu_qk_norm_rope_scalar_dispatch(float* data, const float* norm_weight,
                                      int heads, int head_dim, int position,
                                      const RopePositionSpec& rope, float eps) {
    validate_qk_norm_rope_arguments(data, norm_weight, heads, head_dim, position);
    const int rotary_dim = static_cast<int>(static_cast<float>(head_dim) * rope.rotary_fraction);
    thread_local std::vector<float> cos_vals;
    thread_local std::vector<float> sin_vals;
    build_rope_tables(rope, rotary_dim, position, cos_vals, sin_vals);
    apply_qk_norm_rope_scalar(data, norm_weight, cos_vals.data(), sin_vals.data(),
                              heads, head_dim, rotary_dim, rope.pairing, eps);
}

#if ((defined(__GNUC__) || defined(__clang__)) && \
     (defined(__x86_64__) || defined(__i386__))) || \
    (defined(_MSC_VER) && CELEG_CPU_X86)
void cpu_qk_norm_rope_avx2_dispatch(float* data, const float* norm_weight,
                                    int heads, int head_dim, int position,
                                    const RopePositionSpec& rope, float eps) {
    validate_qk_norm_rope_arguments(data, norm_weight, heads, head_dim, position);
    const int rotary_dim = static_cast<int>(static_cast<float>(head_dim) * rope.rotary_fraction);
    thread_local std::vector<float> cos_vals;
    thread_local std::vector<float> sin_vals;
    build_rope_tables(rope, rotary_dim, position, cos_vals, sin_vals);
    if (rope.pairing == RopePairingKind::SplitHalf &&
        rope.rotary_fraction == 1.0 &&
        std::holds_alternative<NoRopeScaling>(rope.scaling)) {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
        cpu_qk_norm_rope_avx2(data, norm_weight, cos_vals.data(), sin_vals.data(),
                              heads, head_dim, eps);
#else
        detail::cpu_qk_norm_rope_avx2_msvc(
            data, norm_weight, cos_vals.data(), sin_vals.data(), heads, head_dim, eps);
#endif
        return;
    }
    apply_qk_norm_rope_scalar(data, norm_weight, cos_vals.data(), sin_vals.data(),
                              heads, head_dim, rotary_dim, rope.pairing, eps);
}
#endif

}

namespace detail {

CpuQkNormRopeFunction select_cpu_qk_norm_rope_kernel(CpuIsa isa) {
#if ((defined(__GNUC__) || defined(__clang__)) && \
     (defined(__x86_64__) || defined(__i386__))) || \
    (defined(_MSC_VER) && CELEG_CPU_X86)
    if (isa == CpuIsa::Avx2 || isa == CpuIsa::AvxVnni ||
        isa == CpuIsa::Avx512Vnni || isa == CpuIsa::AmxInt8) {
        return cpu_qk_norm_rope_avx2_dispatch;
    }
#else
    (void)isa;
#endif
    return cpu_qk_norm_rope_scalar_dispatch;
}

}

void CpuMathEngine::qk_norm_rope(float* data, const float* norm_weight,
                                 int heads, int head_dim, int position,
                                 const RopePositionSpec& rope, float eps) const {
    qk_norm_rope_(data, norm_weight, heads, head_dim, position, rope, eps);
}

void cpu_qk_norm_rope(float* data, const float* norm_weight,
                      int heads, int head_dim, int position,
                      const RopePositionSpec& rope, float eps) {
    cpu_qk_norm_rope_scalar_dispatch(
        data, norm_weight, heads, head_dim, position, rope, eps);
}

void cpu_qk_norm_only(float* data, const float* norm_weight,
                      int heads, int head_dim, float eps) {
    if (!data || !norm_weight || heads <= 0 || head_dim <= 0 || !(eps > 0.0f)) {
        throw std::invalid_argument("invalid QK norm arguments");
    }
    for (int head = 0; head < heads; ++head) {
        float* vector = data + static_cast<size_t>(head) * head_dim;
        double sum = 0.0;
        for (int d = 0; d < head_dim; ++d) sum += static_cast<double>(vector[d]) * vector[d];
        const float scale = 1.0f / std::sqrt(static_cast<float>(sum / head_dim) + eps);
        for (int d = 0; d < head_dim; ++d) vector[d] *= scale * norm_weight[d];
    }
}

void cpu_rope(float* data, int heads, int head_dim, int position,
              const RopePositionSpec& rope) {
    if (!data || heads <= 0 || head_dim <= 0 || (head_dim % 2) != 0 ||
        position < 0 || !(rope.theta > 0.0)) {
        throw std::invalid_argument("invalid RoPE arguments");
    }
    const int rotary_dim = static_cast<int>(static_cast<float>(head_dim) * rope.rotary_fraction);
    const int half = rotary_dim / 2;
    std::vector<float> cos_vals(static_cast<size_t>(half));
    std::vector<float> sin_vals(static_cast<size_t>(half));
    for (int pair = 0; pair < half; ++pair) {
        const float frequency = static_cast<float>(rope_frequency(
            rope, pair, rotary_dim, position));
        const float angle = static_cast<float>(position) * frequency;
        cos_vals[static_cast<size_t>(pair)] = std::cos(angle);
        sin_vals[static_cast<size_t>(pair)] = std::sin(angle);
    }
    for (int head = 0; head < heads; ++head) {
        float* row = data + static_cast<size_t>(head) * head_dim;
        for (int pair = 0; pair < half; ++pair) {
            const auto [first, second] = rope_pair_indices(pair, half, rope.pairing);
            const float x0 = row[first];
            const float x1 = row[second];
            row[first] = x0 * cos_vals[static_cast<size_t>(pair)] -
                         x1 * sin_vals[static_cast<size_t>(pair)];
            row[second] = x1 * cos_vals[static_cast<size_t>(pair)] +
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
                    float rope_theta, float rotary_fraction,
                    RopePairingKind pairing) {
    if (head_dim <= 0 || (head_dim % 2) != 0 || !(rope_theta > 0.0f) ||
        !(rotary_fraction > 0.0f) || rotary_fraction > 1.0f ||
        sections[0] <= 0 || sections[1] <= 0 || sections[2] <= 0) {
        throw std::invalid_argument("invalid MRoPE arguments");
    }
    if (pairing != RopePairingKind::SplitHalf) {
        throw std::invalid_argument("M-RoPE currently requires split-half pairing");
    }
    const int rotary_dim = static_cast<int>(static_cast<float>(head_dim) * rotary_fraction);
    if ((rotary_dim % 2) != 0 || sections[0] + sections[1] + sections[2] != rotary_dim / 2) {
        throw std::invalid_argument("MRoPE sections do not match rotary dimension");
    }
}

}

void cpu_qk_norm_rope_mrope(float* data, const float* norm_weight,
                            int heads, int head_dim,
                            const std::array<int32_t, 3>& positions,
                            const std::array<int, 3>& sections,
                            bool interleaved, const RopePositionSpec& rope, float eps) {
    if (!data || !norm_weight || heads <= 0) {
        throw std::invalid_argument("invalid MRoPE QK arguments");
    }
    validate_mrope(head_dim, sections, static_cast<float>(rope.theta),
                   static_cast<float>(rope.rotary_fraction), rope.pairing);
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
                   static_cast<float>(rope.rotary_fraction), rope.pairing);
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

}
