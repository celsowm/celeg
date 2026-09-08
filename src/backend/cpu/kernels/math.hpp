#pragma once

#include "celeg/backend/cpu/isa.hpp"
#include "celeg/model/definition.hpp"

#include <cstddef>

namespace celeg {

struct CpuKernelBackend;

/// Resolved CPU math dispatch used by model execution after ISA selection.
class CpuMathEngine {
public:
    explicit CpuMathEngine(const CpuKernelBackend& backend);

    CpuIsa isa() const { return isa_; }

    void rmsnorm(const float* input, const float* weight, float* output,
                 size_t width, float eps) const;
    void rmsnorm_inplace(float* data, const float* weight,
                         size_t width, float eps) const;
    void residual_add(float* data, const float* residual, size_t count) const;
    void swiglu(const float* gate_up, float* output, size_t count) const;
    void qk_norm_rope(float* data, const float* norm_weight,
                      int heads, int head_dim, int position,
                      const RopePositionSpec& rope, float eps) const;

private:
    using RmsNormFunction = void (*)(const float*, const float*, float*, size_t, float);
    using ResidualAddFunction = void (*)(float*, const float*, size_t);
    using SwiGluFunction = void (*)(const float*, float*, size_t);
    using QkNormRopeFunction = void (*)(float*, const float*, int, int, int,
                                        const RopePositionSpec&, float);

    CpuIsa isa_ = CpuIsa::Scalar;
    RmsNormFunction rmsnorm_ = nullptr;
    ResidualAddFunction residual_add_ = nullptr;
    SwiGluFunction swiglu_ = nullptr;
    QkNormRopeFunction qk_norm_rope_ = nullptr;
};

/// Returns the immutable math engine bound to an already-resolved CPU ISA.
const CpuMathEngine& cpu_math_engine(CpuIsa isa);

namespace detail {

using CpuQkNormRopeFunction = void (*)(float*, const float*, int, int, int,
                                       const RopePositionSpec&, float);
CpuQkNormRopeFunction select_cpu_qk_norm_rope_kernel(CpuIsa isa);

}

}
