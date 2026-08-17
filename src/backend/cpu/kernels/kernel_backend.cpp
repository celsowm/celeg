#include "celeg/backend/cpu/kernel_backend.hpp"

#include <array>
#include <stdexcept>

namespace celeg {
namespace {

int backend_priority(CpuIsa isa) {
    switch (isa) {
        case CpuIsa::Avx512Vnni: return 40;
        case CpuIsa::AmxInt8: return 25;
        case CpuIsa::AvxVnni: return 30;
        case CpuIsa::Avx2: return 20;
        case CpuIsa::Neon: return 15;
        case CpuIsa::DotProd: return 14;
        case CpuIsa::I8mm: return 13;
        case CpuIsa::Sve2: return 12;
        case CpuIsa::Sme2: return 11;
        case CpuIsa::Scalar: return 0;
        default: return -1;
    }
}

CpuKernelBackend make_backend(CpuIsa isa) {
    CpuKernelBackend backend;
    backend.isa = isa;
    backend.priority = backend_priority(isa);
    backend.compiled = cpu_isa_compiled(isa);
    backend.kernels.q4_dot = select_q4_dot_kernel(isa);
    backend.kernels.q4_q8_dot = select_q4_q8_dot_kernel(isa);
    backend.kernels.gguf_dot = select_cpu_gguf_dot_kernel(isa);
    backend.kernels.gguf_dot4 = select_cpu_gguf_dot4_kernel(isa);
    return backend;
}

const std::array<CpuKernelBackend, 11>& backend_table() {
    static const std::array<CpuKernelBackend, 11> table = {
        make_backend(CpuIsa::Scalar),    make_backend(CpuIsa::Avx2),
        make_backend(CpuIsa::AvxVnni),   make_backend(CpuIsa::Avx512Vnni),
        make_backend(CpuIsa::AmxInt8),   make_backend(CpuIsa::Neon),
        make_backend(CpuIsa::DotProd),   make_backend(CpuIsa::I8mm),
        make_backend(CpuIsa::Sve2),      make_backend(CpuIsa::Sme2),
        make_backend(CpuIsa::Auto),
    };
    return table;
}

}  // namespace

bool CpuKernelBackend::supports_hw(const CpuCapabilities& caps) const {
    switch (isa) {
        case CpuIsa::Auto: return false;
        case CpuIsa::Scalar: return true;
        case CpuIsa::Avx2: return caps.avx2 && caps.fma;
        case CpuIsa::AvxVnni: return caps.avx_vnni;
        case CpuIsa::Avx512Vnni: return caps.avx512f && caps.avx512_vnni;
        case CpuIsa::AmxInt8: return caps.amx_tile && caps.amx_int8;
        case CpuIsa::Neon: return caps.neon;
        case CpuIsa::DotProd: return caps.dotprod;
        case CpuIsa::I8mm: return caps.i8mm;
        case CpuIsa::Sve2: return caps.sve2;
        case CpuIsa::Sme2: return caps.sme2;
    }
    return false;
}

const CpuKernelBackend& cpu_kernel_backend(CpuIsa isa) {
    for (const CpuKernelBackend& backend : backend_table()) {
        if (backend.isa == isa) return backend;
    }
    static const CpuKernelBackend empty;
    return empty;
}

const CpuKernelBackend& cpu_resolve_kernel_backend(CpuIsa requested,
                                                   const CpuCapabilities& caps) {
    const std::array<CpuKernelBackend, 11>& table = backend_table();
    if (requested == CpuIsa::Auto) {
        const CpuKernelBackend* best = nullptr;
        for (const CpuKernelBackend& backend : table) {
            if (backend.compiled && backend.supports_hw(caps)) {
                if (!best || backend.priority > best->priority) best = &backend;
            }
        }
        if (!best) {
            throw std::runtime_error("no CPU kernel backend is available for this host");
        }
        return *best;
    }
    for (const CpuKernelBackend& backend : table) {
        if (backend.isa == requested) {
            if (!backend.compiled) {
                throw std::invalid_argument(
                    "requested CPU ISA was not compiled into this binary");
            }
            if (!backend.supports_hw(caps)) {
                if (requested == CpuIsa::Avx2) {
                    throw std::invalid_argument("AVX2 CPU backend requires FMA");
                }
                throw std::invalid_argument(
                    "requested CPU ISA is not supported by this host");
            }
            return backend;
        }
    }
    throw std::invalid_argument(
        "requested CPU ISA is detected by the API but its native kernel is not implemented");
}

}  // namespace celeg
