#include "lfm/backend/cpu/isa.hpp"
#include <cassert>
#include <iostream>

int main() {
    const lfm::CpuCapabilities caps = lfm::detect_cpu_capabilities();
    assert(caps.supports(lfm::CpuIsa::Scalar));
    assert(caps.supports(caps.best_isa()));
    assert(lfm::cpu_isa_compiled(caps.best_isa()));
    assert(!lfm::cpu_isa_compiled(lfm::CpuIsa::AmxInt8));
    assert(std::string(lfm::cpu_isa_name(caps.best_isa())).size() > 0);
    assert(lfm::parse_cpu_isa("auto") == lfm::CpuIsa::Auto);
    assert(lfm::parse_cpu_isa("scalar") == lfm::CpuIsa::Scalar);
    std::cout << "cpu_isa_test: " << caps.summary() << '\n';
}
