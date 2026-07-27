#include "lfm/backend/cpu/isa.hpp"
#include "support/assertions.hpp"
#include <iostream>

int main() {
    const lfm::CpuCapabilities caps = lfm::detect_cpu_capabilities();
    LFM_TEST_CHECK(caps.supports(lfm::CpuIsa::Scalar));
    LFM_TEST_CHECK(caps.supports(caps.best_isa()));
    LFM_TEST_CHECK(lfm::cpu_isa_compiled(caps.best_isa()));
    LFM_TEST_CHECK(!lfm::cpu_isa_compiled(lfm::CpuIsa::AmxInt8));
    LFM_TEST_CHECK(std::string(lfm::cpu_isa_name(caps.best_isa())).size() > 0);
    LFM_TEST_CHECK(lfm::parse_cpu_isa("auto") == lfm::CpuIsa::Auto);
    LFM_TEST_CHECK(lfm::parse_cpu_isa("scalar") == lfm::CpuIsa::Scalar);
    std::cout << "cpu_isa_test: " << caps.summary() << '\n';
}
