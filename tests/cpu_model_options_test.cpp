#include "lfm/backend/cpu/model.hpp"
#include "support/assertions.hpp"
#include <iostream>

int main() {
    LFM_TEST_CHECK(std::string(lfm::cpu_kv_cache_mode_name(
        lfm::CpuKvCacheMode::Fp32)) == "fp32");
    LFM_TEST_CHECK(std::string(lfm::cpu_kv_cache_mode_name(
        lfm::CpuKvCacheMode::Bf16)) == "bf16");
    LFM_TEST_CHECK(lfm::parse_cpu_kv_cache_mode("fp32") == lfm::CpuKvCacheMode::Fp32);
    LFM_TEST_CHECK(lfm::parse_cpu_kv_cache_mode("bf16") == lfm::CpuKvCacheMode::Bf16);
    lfm::CpuModelOptions defaults;
    LFM_TEST_CHECK(defaults.kv_page_tokens == 32);
    LFM_TEST_CHECK(defaults.prefill_chunk_tokens == 256);
    LFM_TEST_CHECK(defaults.prefill_chunk_threshold == 16);
    bool threw = false;
    try { (void)lfm::parse_cpu_kv_cache_mode("int8"); }
    catch (...) { threw = true; }
    LFM_TEST_CHECK(threw);
    std::cout << "cpu_model_options_test: ok\n";
    return 0;
}
