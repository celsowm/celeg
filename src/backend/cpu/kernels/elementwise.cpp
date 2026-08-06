#include "celeg/backend/cpu/kernels.hpp"
#include "celeg/backend/cpu/isa.hpp"

#include <cmath>
#include <algorithm>
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
void cpu_rmsnorm_avx2(const float* input, const float* weight, float* output,
                      size_t width, float eps) {
    double sum = 0.0;
    for (size_t i = 0; i < width; ++i) sum += static_cast<double>(input[i]) * input[i];
    const float inv = 1.0f / std::sqrt(static_cast<float>(sum / width) + eps);
    for (size_t i = 0; i < width; ++i) output[i] = input[i] * inv * weight[i];
}

__attribute__((target("avx2,fma")))
void cpu_residual_add_avx2(float* data, const float* residual, size_t count) {
    for (size_t i = 0; i < count; ++i) data[i] += residual[i];
}

__attribute__((target("avx2,fma")))
void cpu_swiglu_avx2(const float* gate_up, float* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const float gate = gate_up[i];
        const float up = gate_up[count + i];
        output[i] = (gate / (1.0f + std::exp(-gate))) * up;
    }
}
#endif

} // namespace

void cpu_rmsnorm(const float* input, const float* weight, float* output,
                 size_t width, float eps) {
    if (!input || !weight || !output || width == 0 || !(eps > 0.0f)) {
        throw std::invalid_argument("invalid RMSNorm arguments");
    }
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    if (g_has_avx2_fma) {
        cpu_rmsnorm_avx2(input, weight, output, width, eps);
        return;
    }
#elif defined(_MSC_VER) && CELEG_CPU_X86
    if (g_has_avx2_fma) {
        detail::cpu_rmsnorm_avx2_msvc(input, weight, output, width, eps);
        return;
    }
#endif
    double sum = 0.0;
    for (size_t i = 0; i < width; ++i) sum += static_cast<double>(input[i]) * input[i];
    const float inv = 1.0f / std::sqrt(static_cast<float>(sum / width) + eps);
    for (size_t i = 0; i < width; ++i) output[i] = input[i] * inv * weight[i];
}

void cpu_rmsnorm_inplace(float* data, const float* weight, size_t width, float eps) {
    thread_local std::vector<float> temp;
    temp.resize(width);
    cpu_rmsnorm(data, weight, temp.data(), width, eps);
    std::copy(temp.begin(), temp.end(), data);
}

void cpu_residual_add(float* data, const float* residual, size_t count) {
    if (!data || !residual) throw std::invalid_argument("invalid residual arguments");
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    if (g_has_avx2_fma) {
        cpu_residual_add_avx2(data, residual, count);
        return;
    }
#elif defined(_MSC_VER) && CELEG_CPU_X86
    if (g_has_avx2_fma) {
        detail::cpu_residual_add_avx2_msvc(data, residual, count);
        return;
    }
#endif
    for (size_t i = 0; i < count; ++i) data[i] += residual[i];
}

void cpu_swiglu(const float* gate_up, float* output, size_t count) {
    if (!gate_up || !output) throw std::invalid_argument("invalid SwiGLU arguments");
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    if (g_has_avx2_fma) {
        cpu_swiglu_avx2(gate_up, output, count);
        return;
    }
#elif defined(_MSC_VER) && CELEG_CPU_X86
    if (g_has_avx2_fma) {
        detail::cpu_swiglu_avx2_msvc(gate_up, output, count);
        return;
    }
#endif
    for (size_t i = 0; i < count; ++i) {
        const float gate = gate_up[i];
        const float up = gate_up[count + i];
        output[i] = (gate / (1.0f + std::exp(-gate))) * up;
    }
}

void cpu_gated_gelu_tanh(const float* gate_up, float* output, size_t count) {
    constexpr float k = 0.7978845608028654f;
    constexpr float c = 0.044715f;
    for (size_t i = 0; i < count; ++i) {
        const float x = gate_up[i];
        const float gelu = 0.5f * x * (1.0f + std::tanh(k * (x + c * x * x * x)));
        output[i] = gelu * gate_up[count + i];
    }
}

void cpu_relu2(const float* input, float* output, size_t count) {
    if (!input || !output) throw std::invalid_argument("invalid ReLU2 arguments");
    for (size_t i = 0; i < count; ++i) {
        const float value = std::max(input[i], 0.0f);
        output[i] = value * value;
    }
}

void cpu_gelu_tanh(float* data, size_t count) {
    constexpr float k = 0.7978845608028654f;
    constexpr float c = 0.044715f;
    for (size_t i = 0; i < count; ++i) {
        const float x = data[i];
        data[i] = 0.5f * x * (1.0f + std::tanh(k * (x + c * x * x * x)));
    }
}

} // namespace celeg
