#include "celeg/backend/cpu/isa.hpp"

#include <sstream>
#include <stdexcept>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

#if defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

namespace celeg {
namespace {

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
struct CpuidRegs { uint32_t eax{}, ebx{}, ecx{}, edx{}; };

CpuidRegs cpuid(uint32_t leaf, uint32_t subleaf) {
    CpuidRegs r;
#if defined(_MSC_VER)
    int regs[4]{};
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    r.eax = static_cast<uint32_t>(regs[0]);
    r.ebx = static_cast<uint32_t>(regs[1]);
    r.ecx = static_cast<uint32_t>(regs[2]);
    r.edx = static_cast<uint32_t>(regs[3]);
#else
    __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
#endif
    return r;
}

uint64_t xgetbv0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    uint32_t eax = 0, edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<uint64_t>(edx) << 32) | eax;
#endif
}
#endif

}

const char* cpu_isa_name(CpuIsa isa) {
    switch (isa) {
        case CpuIsa::Auto: return "auto";
        case CpuIsa::Scalar: return "scalar";
        case CpuIsa::Avx2: return "avx2";
        case CpuIsa::AvxVnni: return "avx-vnni";
        case CpuIsa::Avx512Vnni: return "avx512-vnni";
        case CpuIsa::AmxInt8: return "amx-int8";
        case CpuIsa::Neon: return "neon";
        case CpuIsa::DotProd: return "dotprod";
        case CpuIsa::I8mm: return "i8mm";
        case CpuIsa::Sve2: return "sve2";
        case CpuIsa::Sme2: return "sme2";
    }
    return "unknown";
}

CpuIsa parse_cpu_isa(const std::string& value) {
    if (value == "auto") return CpuIsa::Auto;
    for (CpuIsa isa : {CpuIsa::Scalar, CpuIsa::Avx2, CpuIsa::AvxVnni,
                       CpuIsa::Avx512Vnni, CpuIsa::AmxInt8, CpuIsa::Neon,
                       CpuIsa::DotProd, CpuIsa::I8mm, CpuIsa::Sve2,
                       CpuIsa::Sme2}) {
        if (value == cpu_isa_name(isa)) return isa;
    }
    throw std::invalid_argument("unknown CPU ISA: " + value);
}

bool cpu_isa_compiled(CpuIsa isa) {
    switch (isa) {
        case CpuIsa::Auto:
        case CpuIsa::Scalar:
            return true;
        case CpuIsa::Avx2:
#if ((defined(__GNUC__) || defined(__clang__)) && \
     (defined(__x86_64__) || defined(__i386__))) || \
    (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
            return true;
#else
            return false;
#endif
        case CpuIsa::AvxVnni:
        case CpuIsa::Avx512Vnni:
#if (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__))
            return true;
#else
            return false;
#endif
        case CpuIsa::Neon:
#if defined(__aarch64__)
            return true;
#else
            return false;
#endif
        case CpuIsa::AmxInt8:
        case CpuIsa::DotProd:
        case CpuIsa::I8mm:
        case CpuIsa::Sve2:
        case CpuIsa::Sme2:
            return false;
    }
    return false;
}

CpuIsa CpuCapabilities::best_isa() const {
    if (supports(CpuIsa::Avx512Vnni) &&
        cpu_isa_compiled(CpuIsa::Avx512Vnni)) {
        return CpuIsa::Avx512Vnni;
    }
    if (supports(CpuIsa::AvxVnni) && cpu_isa_compiled(CpuIsa::AvxVnni)) {
        return CpuIsa::AvxVnni;
    }
    if (supports(CpuIsa::Avx2) && fma && cpu_isa_compiled(CpuIsa::Avx2)) {
        return CpuIsa::Avx2;
    }
    if (supports(CpuIsa::Neon) && cpu_isa_compiled(CpuIsa::Neon)) {
        return CpuIsa::Neon;
    }
    return CpuIsa::Scalar;
}

bool CpuCapabilities::supports(CpuIsa isa) const {
    switch (isa) {
        case CpuIsa::Auto: return true;
        case CpuIsa::Scalar: return true;
        case CpuIsa::Avx2: return avx2;
        case CpuIsa::AvxVnni: return avx_vnni;
        case CpuIsa::Avx512Vnni:
            return avx2 && avx_vnni && avx512f && avx512bw && avx512vl && avx512_vnni;
        case CpuIsa::AmxInt8: return amx_tile && amx_int8;
        case CpuIsa::Neon: return neon;
        case CpuIsa::DotProd: return dotprod;
        case CpuIsa::I8mm: return i8mm;
        case CpuIsa::Sve2: return sve2;
        case CpuIsa::Sme2: return sme2;
    }
    return false;
}

std::string CpuCapabilities::summary() const {
    std::ostringstream out;
    out << "best-executable=" << cpu_isa_name(best_isa())
        << " x86=" << x86 << " arm64=" << arm64
        << " avx2=" << avx2 << " fma=" << fma
        << " avx-vnni=" << avx_vnni
        << " avx512-f=" << avx512f
        << " avx512-bw=" << avx512bw
        << " avx512-vl=" << avx512vl
        << " avx512-vnni=" << avx512_vnni
        << " amx-int8=" << (amx_tile && amx_int8)
        << " neon=" << neon << " dotprod=" << dotprod
        << " i8mm=" << i8mm << " sve2=" << sve2
        << " sme2=" << sme2;
    return out.str();
}

CpuCapabilities detect_cpu_capabilities() {
    CpuCapabilities c;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    c.x86 = true;
    const CpuidRegs leaf1 = cpuid(1, 0);
    const bool osxsave = (leaf1.ecx & (1u << 27)) != 0;
    const bool avx = (leaf1.ecx & (1u << 28)) != 0;
    c.fma = (leaf1.ecx & (1u << 12)) != 0;
    uint64_t xcr0 = 0;
    if (osxsave) xcr0 = xgetbv0();
    const bool ymm_enabled = (xcr0 & 0x6) == 0x6;
    const bool zmm_enabled = (xcr0 & 0xe6) == 0xe6;
    const CpuidRegs leaf7 = cpuid(7, 0);
    c.avx2 = avx && ymm_enabled && (leaf7.ebx & (1u << 5));
    c.avx512f = zmm_enabled && (leaf7.ebx & (1u << 16));
    c.avx512bw = c.avx512f && (leaf7.ebx & (1u << 30));
    c.avx512vl = c.avx512f && (leaf7.ebx & (1u << 31));
    c.avx512_vnni = c.avx512f && (leaf7.ecx & (1u << 11));
    const CpuidRegs leaf71 = cpuid(7, 1);
    c.avx_vnni = c.avx2 && (leaf71.eax & (1u << 4));
    c.amx_tile = (leaf7.edx & (1u << 24)) != 0;
    c.amx_int8 = (leaf7.edx & (1u << 25)) != 0;
#elif defined(__aarch64__) || defined(_M_ARM64)
    c.arm64 = true;
    c.neon = true;
#if defined(__linux__)
    const unsigned long hwcap = getauxval(AT_HWCAP);
    const unsigned long hwcap2 = getauxval(AT_HWCAP2);
#ifdef HWCAP_ASIMDDP
    c.dotprod = (hwcap & HWCAP_ASIMDDP) != 0;
#endif
#ifdef HWCAP2_I8MM
    c.i8mm = (hwcap2 & HWCAP2_I8MM) != 0;
#endif
#ifdef HWCAP2_SVE2
    c.sve2 = (hwcap2 & HWCAP2_SVE2) != 0;
#endif
#ifdef HWCAP2_SME
    c.sme = (hwcap2 & HWCAP2_SME) != 0;
#endif
#ifdef HWCAP2_SME2
    c.sme2 = (hwcap2 & HWCAP2_SME2) != 0;
#endif
#else
#if defined(__ARM_FEATURE_DOTPROD)
    c.dotprod = true;
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
    c.i8mm = true;
#endif
#if defined(__ARM_FEATURE_SVE2)
    c.sve2 = true;
#endif
#endif
#endif
    return c;
}

}
