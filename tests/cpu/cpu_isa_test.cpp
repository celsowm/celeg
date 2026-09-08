#include "celeg/backend/cpu/isa.hpp"
#include "celeg/backend/cpu/kernel_backend.hpp"
#include "support/assertions.hpp"
#include <iostream>

int main() {
    const celeg::CpuCapabilities caps = celeg::detect_cpu_capabilities();
    CELEG_TEST_CHECK(caps.supports(celeg::CpuIsa::Scalar));
    CELEG_TEST_CHECK(caps.supports(caps.best_isa()));
    CELEG_TEST_CHECK(celeg::cpu_isa_compiled(caps.best_isa()));
    CELEG_TEST_CHECK(!celeg::cpu_isa_compiled(celeg::CpuIsa::AmxInt8));
    CELEG_TEST_CHECK(std::string(celeg::cpu_isa_name(caps.best_isa())).size() > 0);
    CELEG_TEST_CHECK(celeg::parse_cpu_isa("auto") == celeg::CpuIsa::Auto);
    CELEG_TEST_CHECK(celeg::parse_cpu_isa("scalar") == celeg::CpuIsa::Scalar);

    celeg::CpuCapabilities no_fma;
    no_fma.x86 = true;
    no_fma.avx2 = true;
    no_fma.avx_vnni = true;
    no_fma.avx512f = true;
    no_fma.avx512bw = true;
    no_fma.avx512vl = true;
    no_fma.avx512_vnni = true;
    CELEG_TEST_CHECK(no_fma.best_isa() == celeg::CpuIsa::Scalar);
    CELEG_TEST_CHECK(!celeg::cpu_kernel_backend(celeg::CpuIsa::Avx2).supports_hw(no_fma));
    CELEG_TEST_CHECK(!celeg::cpu_kernel_backend(celeg::CpuIsa::AvxVnni).supports_hw(no_fma));
    CELEG_TEST_CHECK(!celeg::cpu_kernel_backend(celeg::CpuIsa::Avx512Vnni).supports_hw(no_fma));

    no_fma.fma = true;
    CELEG_TEST_CHECK(celeg::cpu_kernel_backend(celeg::CpuIsa::Avx2).supports_hw(no_fma));
    CELEG_TEST_CHECK(celeg::cpu_kernel_backend(celeg::CpuIsa::AvxVnni).supports_hw(no_fma));
    CELEG_TEST_CHECK(celeg::cpu_kernel_backend(celeg::CpuIsa::Avx512Vnni).supports_hw(no_fma));

    std::cout << "cpu_isa_test: " << caps.summary() << '\n';
}
