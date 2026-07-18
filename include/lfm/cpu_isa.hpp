#pragma once

#include <cstdint>
#include <string>

namespace lfm {

enum class CpuIsa : uint8_t {
    Auto,
    Scalar,
    Avx2,
    AvxVnni,
    Avx512Vnni,
    AmxInt8,
    Neon,
    DotProd,
    I8mm,
    Sve2,
    Sme2,
};

struct CpuCapabilities {
    bool x86 = false;
    bool arm64 = false;
    bool avx2 = false;
    bool fma = false;
    bool avx_vnni = false;
    bool avx512f = false;
    bool avx512_vnni = false;
    bool amx_tile = false;
    bool amx_int8 = false;
    bool neon = false;
    bool dotprod = false;
    bool i8mm = false;
    bool sve2 = false;
    bool sme = false;
    bool sme2 = false;

    CpuIsa best_isa() const;
    bool supports(CpuIsa isa) const;
    std::string summary() const;
};

CpuCapabilities detect_cpu_capabilities();
const char* cpu_isa_name(CpuIsa isa);
CpuIsa parse_cpu_isa(const std::string& value);
bool cpu_isa_compiled(CpuIsa isa);

} // namespace lfm
