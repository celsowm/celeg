#include "lfm/cpu_model.hpp"

#include <cassert>
#include <iostream>

int main() {
    assert(std::string(lfm::cpu_kv_cache_mode_name(
        lfm::CpuKvCacheMode::Fp32)) == "fp32");
    assert(std::string(lfm::cpu_kv_cache_mode_name(
        lfm::CpuKvCacheMode::Bf16)) == "bf16");
    assert(lfm::parse_cpu_kv_cache_mode("fp32") == lfm::CpuKvCacheMode::Fp32);
    assert(lfm::parse_cpu_kv_cache_mode("bf16") == lfm::CpuKvCacheMode::Bf16);
    lfm::CpuModelOptions defaults;
    assert(defaults.kv_page_tokens == 32);
    assert(defaults.prefill_chunk_tokens == 256);
    assert(defaults.prefill_chunk_threshold == 16);
    bool threw = false;
    try { (void)lfm::parse_cpu_kv_cache_mode("int8"); }
    catch (...) { threw = true; }
    assert(threw);
    std::cout << "cpu_model_options_test: ok\n";
    return 0;
}
