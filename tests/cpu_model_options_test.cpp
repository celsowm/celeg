#include "celeg/backend/cpu/model.hpp"
#include "support/assertions.hpp"
#include <iostream>

int main() {
    CELEG_TEST_CHECK(std::string(celeg::cpu_kv_cache_mode_name(
        celeg::CpuKvCacheMode::Fp32)) == "fp32");
    CELEG_TEST_CHECK(std::string(celeg::cpu_kv_cache_mode_name(
        celeg::CpuKvCacheMode::Bf16)) == "bf16");
    CELEG_TEST_CHECK(celeg::parse_cpu_kv_cache_mode("fp32") == celeg::CpuKvCacheMode::Fp32);
    CELEG_TEST_CHECK(celeg::parse_cpu_kv_cache_mode("bf16") == celeg::CpuKvCacheMode::Bf16);
    celeg::CpuModelOptions defaults;
    CELEG_TEST_CHECK(defaults.kv_page_tokens == 32);
    CELEG_TEST_CHECK(defaults.prefill_chunk_tokens == 256);
    CELEG_TEST_CHECK(defaults.prefill_chunk_threshold == 16);
    bool threw = false;
    try { (void)celeg::parse_cpu_kv_cache_mode("int8"); }
    catch (...) { threw = true; }
    CELEG_TEST_CHECK(threw);
    std::cout << "cpu_model_options_test: ok\n";
    return 0;
}
