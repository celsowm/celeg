#include "celeg/backend/cpu/concurrent.hpp"

namespace celeg {

double CpuConcurrentMetrics::prefill_tokens_per_second() const {
    return cumulative_prefill_ms > 0.0
        ? static_cast<double>(prefill_tokens) * 1000.0 / cumulative_prefill_ms
        : 0.0;
}

double CpuConcurrentMetrics::decode_tokens_per_second() const {
    return cumulative_decode_ms > 0.0
        ? static_cast<double>(decode_tokens) * 1000.0 / cumulative_decode_ms
        : 0.0;
}

double CpuConcurrentMetrics::average_ttft_ms() const {
    return ttft_samples ? cumulative_ttft_ms / static_cast<double>(ttft_samples) : 0.0;
}

double CpuConcurrentMetrics::average_itl_ms() const {
    return itl_samples ? cumulative_itl_ms / static_cast<double>(itl_samples) : 0.0;
}

} // namespace celeg
