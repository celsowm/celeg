#include "lfm/runtime/concurrency/metrics.hpp"

#include <cassert>
#include <iostream>

int main() {
    lfm::ConcurrentMetrics flat;
    flat.submitted = 9;
    flat.completed = 7;
    flat.ttft_samples = 2;
    flat.cumulative_ttft_ms = 10.0;
    flat.ragged_prefill_tokens = 100;
    flat.packed_decode_tokens = 50;
    flat.prefix_cache_hits = 4;
    flat.logical_pages_used = 3;
    flat.logical_pages_total = 8;

    const auto grouped = lfm::group_concurrent_metrics(flat);
    assert(grouped.requests.submitted == 9);
    assert(grouped.requests.completed == 7);
    assert(grouped.requests.average_ttft_ms == 5.0);
    assert(grouped.prefill.ragged_tokens == 100);
    assert(grouped.decode.packed_tokens == 50);
    assert(grouped.prefix.hits == 4);
    assert(grouped.kv.pages_used == 3);
    assert(grouped.kv.pages_total == 8);
    std::cout << "concurrent_metrics_test: ok\n";
    return 0;
}
