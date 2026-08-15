#include "celeg/backend/cpu/concurrent.hpp"
#include "support/assertions.hpp"
#include <cmath>
#include <iostream>

int main() {
    celeg::ConcurrentMetrics metrics;
    metrics.prefill_tokens = 100;
    metrics.cumulative_ragged_prefill_ms = 50.0;
    metrics.decoded_tokens = 20;
    metrics.cumulative_packed_decode_ms = 40.0;
    metrics.cumulative_ttft_ms = 30.0;
    metrics.ttft_samples = 2;
    metrics.cumulative_itl_ms = 24.0;
    metrics.itl_samples = 3;

    CELEG_TEST_CHECK(std::abs(metrics.prefill_tokens * 1000.0 /
                               metrics.cumulative_ragged_prefill_ms - 2000.0) < 1e-9);
    CELEG_TEST_CHECK(std::abs(metrics.decoded_tokens * 1000.0 /
                               metrics.cumulative_packed_decode_ms - 500.0) < 1e-9);
    CELEG_TEST_CHECK(std::abs(metrics.average_ttft_ms() - 15.0) < 1e-9);
    CELEG_TEST_CHECK(std::abs(metrics.average_itl_ms() - 8.0) < 1e-9);

    celeg::CpuConcurrentMetricsExtras extras;
    extras.chunked_prefill_steps = 2;
    extras.cumulative_direct_prefill_ms = 512.0;
    extras.maximum_prefill_chunk = 256;
    extras.attention_parallel_calls = 7;
    CELEG_TEST_CHECK(extras.chunked_prefill_steps == 2);
    CELEG_TEST_CHECK(extras.maximum_prefill_chunk == 256);
    CELEG_TEST_CHECK(extras.attention_parallel_calls == 7);

    CELEG_TEST_CHECK(std::string(celeg::request_status_name(
        celeg::RequestStatus::Decoding)) == "decoding");
    std::cout << "cpu_concurrent_metrics_test: ok\n";
    return 0;
}
