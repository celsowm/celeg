#include "lfm/backend/cpu/concurrent.hpp"
#include "support/assertions.hpp"
#include <cmath>
#include <iostream>

int main() {
    lfm::CpuConcurrentMetrics metrics;
    metrics.prefill_tokens = 100;
    metrics.cumulative_prefill_ms = 50.0;
    metrics.decode_tokens = 20;
    metrics.cumulative_decode_ms = 40.0;
    metrics.cumulative_ttft_ms = 30.0;
    metrics.ttft_samples = 2;
    metrics.cumulative_itl_ms = 24.0;
    metrics.itl_samples = 3;
    metrics.chunked_prefill_steps = 2;
    metrics.chunked_prefill_tokens = 512;
    metrics.maximum_prefill_chunk = 256;
    LFM_TEST_CHECK(std::abs(metrics.prefill_tokens_per_second() - 2000.0) < 1e-9);
    LFM_TEST_CHECK(std::abs(metrics.decode_tokens_per_second() - 500.0) < 1e-9);
    LFM_TEST_CHECK(std::abs(metrics.average_ttft_ms() - 15.0) < 1e-9);
    LFM_TEST_CHECK(std::abs(metrics.average_itl_ms() - 8.0) < 1e-9);
    LFM_TEST_CHECK(metrics.chunked_prefill_steps == 2);
    LFM_TEST_CHECK(metrics.chunked_prefill_tokens == 512);
    LFM_TEST_CHECK(metrics.maximum_prefill_chunk == 256);
    LFM_TEST_CHECK(std::string(lfm::cpu_request_status_name(
        lfm::CpuRequestStatus::Decoding)) == "decoding");
    std::cout << "cpu_concurrent_metrics_test: ok\n";
    return 0;
}
