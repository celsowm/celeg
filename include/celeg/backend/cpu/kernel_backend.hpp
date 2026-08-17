#pragma once

#include "celeg/backend/cpu/isa.hpp"
#include "celeg/backend/cpu/gguf.hpp"
#include "celeg/backend/cpu/kernels.hpp"

namespace celeg {

/// Kernel function pointers bound to one CPU ISA. A null entry means that
/// operation has no native implementation for the ISA (the engine must reject
/// or fall back before dispatching there).
struct CpuKernelTable {
    Q4DotFunction q4_dot = nullptr;
    Q4Q8DotFunction q4_q8_dot = nullptr;
    CpuGgufDotFunction gguf_dot = nullptr;
    CpuGgufDot4Function gguf_dot4 = nullptr;

    bool has_dynamic_q8() const { return q4_q8_dot != nullptr; }
};

/// One row of the single CPU kernel-backend registry. This unifies the four
/// previously divergent notions of an ISA: the enum value (`isa`), what the
/// binary actually compiled (`compiled`), what the host can execute
/// (`supports_hw`), and which kernels exist (`kernels`).
struct CpuKernelBackend {
    CpuIsa isa = CpuIsa::Scalar;
    int priority = 0;
    bool compiled = false;
    CpuKernelTable kernels;

    bool supports_hw(const CpuCapabilities& caps) const;
};

/// Lookup the registry row for a known ISA (never resolves `Auto`).
const CpuKernelBackend& cpu_kernel_backend(CpuIsa isa);

/// Resolve a requested ISA (possibly `Auto`) against the host and the compiled
/// kernel set, throwing a descriptive error when no usable backend exists. This
/// is the single home for ISA selection previously split across
/// `CpuCompiledModel::Shared::resolve_isa`, `CpuLinearEngine`, and the four
/// per-kernel selectors.
const CpuKernelBackend& cpu_resolve_kernel_backend(CpuIsa requested,
                                                   const CpuCapabilities& caps);

}  // namespace celeg
