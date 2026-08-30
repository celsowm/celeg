#include "celeg/backend/cpu/isa.hpp"
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
    std::cout << "cpu_isa_test: " << caps.summary() << '\n';
}
