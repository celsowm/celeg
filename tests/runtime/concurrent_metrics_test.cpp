#include "celeg/runtime/concurrency/metrics.hpp"
#include "support/assertions.hpp"
#include <iostream>

int main() {
    celeg::ConcurrentMetrics flat;
    flat.submitted = 9;
    flat.completed = 7;
    flat.ttft_samples = 2;
    flat.cumulative_ttft_ms = 10.0;
    flat.ragged_prefill_tokens = 100;
    flat.packed_decode_tokens = 50;
    flat.prefix_cache_hits = 4;
    flat.logical_pages_used = 3;
    flat.logical_pages_total = 8;

    const auto grouped = celeg::group_concurrent_metrics(flat);
    CELEG_TEST_CHECK(grouped.requests.submitted == 9);
    CELEG_TEST_CHECK(grouped.requests.completed == 7);
    CELEG_TEST_CHECK(grouped.requests.average_ttft_ms == 5.0);
    CELEG_TEST_CHECK(grouped.prefill.ragged_tokens == 100);
    CELEG_TEST_CHECK(grouped.decode.packed_tokens == 50);
    CELEG_TEST_CHECK(grouped.prefix.hits == 4);
    CELEG_TEST_CHECK(grouped.kv.pages_used == 3);
    CELEG_TEST_CHECK(grouped.kv.pages_total == 8);
    std::cout << "concurrent_metrics_test: ok\n";
    return 0;
}
